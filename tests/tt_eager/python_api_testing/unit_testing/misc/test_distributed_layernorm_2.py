# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def reference_layernorm(x, gamma, beta, epsilon, is_rmsnorm):
    if is_rmsnorm:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon) * gamma
    else:
        return torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, epsilon)


def reference_part2(x, gamma, beta, epsilon, stats_gathered, n_devices, is_rmsnorm):
    tile_cols_per_device = 1 if is_rmsnorm else 2
    # reduce mean and mean(x^2) across devices
    global_meanx2 = torch.zeros(x.shape[:-1] + (1,))
    global_mean = torch.zeros(x.shape[:-1] + (1,))
    for i in range(n_devices):
        mm_idx = i * tile_cols_per_device * 32
        global_meanx2 += stats_gathered[..., mm_idx : mm_idx + 1]
        if not is_rmsnorm:
            m_idx = mm_idx + 32
            global_mean += stats_gathered[..., m_idx : m_idx + 1]
        # breakpoint()

    global_meanx2 /= x.shape[-1] * n_devices
    global_mean /= x.shape[-1] * n_devices

    if is_rmsnorm:
        return x * torch.rsqrt(global_meanx2 + epsilon) * gamma
    else:
        var = global_meanx2 - global_mean.pow(2)
        return (x - global_mean) / torch.sqrt(var + epsilon) * gamma + beta


def run_layernorm_part_2(inp_shape, n_devices, is_rmsnorm, dtype, device, fp32_enabled=False):
    kernel_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttnn.experimental.tensor.MathFidelity.HiFi4,  # Highest fidelity
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_enabled,
        packer_l1_acc=False,
    )

    torch.manual_seed(1234)
    tile_cols_per_device = 1 if is_rmsnorm else 2  # layernorm has 2 stats to distribute

    canon_inp = torch.randn(inp_shape) * 4 - 1
    gamma = torch.rand(inp_shape[-1]) * 2 - 1
    beta = torch.rand(inp_shape[-1]) * 2 - 1
    gamma_chunked = gamma.chunk(n_devices, dim=-1)
    beta_chunked = beta.chunk(n_devices, dim=-1)
    # Get per-chunk mean and mean(x^2)
    inp_chunked = canon_inp.chunk(n_devices, dim=-1)
    mean = [x.sum(dim=-1, keepdim=True) for x in inp_chunked]
    meanx2 = [x.pow(2).sum(dim=-1, keepdim=True) for x in inp_chunked]

    stats_tiles = torch.zeros(inp_shape[:-1] + (32 * n_devices * tile_cols_per_device,))
    for idx, (m, mm) in enumerate(zip(mean, meanx2)):
        mm_idx = idx * tile_cols_per_device * 32
        stats_tiles[..., mm_idx : mm_idx + 1] = mm

        if not is_rmsnorm:
            m_idx = mm_idx + 32  # next tile is m
            stats_tiles[..., m_idx : m_idx + 1] = m

    epsilon = 1e-5
    # reference impl
    ref_out = reference_layernorm(canon_inp, gamma, beta, epsilon, is_rmsnorm)
    ref_chunks = ref_out.chunk(n_devices, dim=-1)

    all_pass = True
    # lnp2 reference
    for d in range(n_devices):
        lnp2_out = reference_part2(
            inp_chunked[d], gamma_chunked[d], beta_chunked[d], epsilon, stats_tiles, n_devices, is_rmsnorm
        )

        tt_inp = ttnn.as_tensor(
            inp_chunked[d], dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_gamma = ttnn.as_tensor(
            gamma_chunked[d].reshape(1, 1, -1, 32),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_beta = ttnn.as_tensor(
            beta_chunked[d].reshape(1, 1, -1, 32),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_stats = ttnn.as_tensor(
            stats_tiles,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if is_rmsnorm:
            tt_lnp2_out = ttnn.experimental.operations.primary.rmsnorm_part2(
                tt_inp, tt_stats, epsilon, tt_gamma, compute_kernel_config=kernel_config
            )
        else:
            tt_lnp2_out = ttnn.experimental.operations.primary.layernorm_part2(
                tt_inp, tt_stats, epsilon, tt_gamma, tt_beta, compute_kernel_config=kernel_config
            )

        tt_lnp2_out_cpu = ttnn.to_torch(tt_lnp2_out)
        # logger.debug("Comparing reference part 2 to TT part 2")
        # passing, output_str = comp_allclose(lnp2_out, tt_lnp2_out_cpu, rtol=1e-01, atol=1e-02)
        # logger.debug(f"ref vs tt={output_str}")
        # passing, output_str = comp_allclose(ref_chunks[d], lnp2_out, rtol=1e-01, atol=1e-02)
        passing, output_str = comp_allclose(ref_chunks[d], tt_lnp2_out_cpu, rtol=1e-1, atol=1e-01)
        logger.debug(f"layernorm vs tt={output_str}")
        all_pass = all_pass and passing

    assert all_pass


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "inp_shape",
    [
        (1, 1, 2048, 8192),
        (1, 1, 128, 8192),
        (2, 1, 128, 8192),
    ],
)
@pytest.mark.parametrize(
    "n_devices",
    [4, 8],
)
@pytest.mark.parametrize(
    "is_rmsnorm",
    [True, False],
    ids=["rmsnorm", "layernorm"],
)
@pytest.mark.parametrize(
    "fp32_enabled",
    [True, False],
    ids=["fp32_enabled", "fp32_disabled"],
)
def test_layernorm_part_2(inp_shape, n_devices, is_rmsnorm, dtype, device, fp32_enabled):
    run_layernorm_part_2(inp_shape, n_devices, is_rmsnorm, dtype, device, fp32_enabled)


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "inp_shape",
    [
        (1, 1, 2048, 8192),
        (1, 1, 128, 8192),
        (2, 1, 128, 8192),
    ],
)
@pytest.mark.parametrize(
    "n_devices",
    [4, 8],
)
@pytest.mark.parametrize(
    "is_rmsnorm",
    [True, False],
    ids=["rmsnorm", "layernorm"],
)
def test_layernorm_part_2_with_program_cache(inp_shape, n_devices, is_rmsnorm, dtype, device, use_program_cache):
    dummy_tensors = []

    for i in range(3):
        dummy_tensors.append(
            ttnn.as_tensor(
                torch.randn(inp_shape),
                dtype=dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        run_layernorm_part_2(inp_shape, n_devices, is_rmsnorm, dtype, device)

    assert device.num_program_cache_entries() == 1, "Program cache should have only one entry" + str(
        device.num_program_cache_entries()
    )
