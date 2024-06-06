# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn

from models.utility_functions import tt2torch_tensor, get_devices_for_t3000

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def reference_layernorm(x, gamma, beta, epsilon, is_rmsnorm):
    if is_rmsnorm:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon) * gamma
    else:
        return torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, epsilon)


def run_distributed_layernorm(inp_shape, n_devices, is_rmsnorm, dtype, device, fp32_enabled=False):
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
    inp_chunked = canon_inp.chunk(n_devices, dim=-1)

    epsilon = 1e-5
    # reference impl
    out_torch = reference_layernorm(canon_inp, gamma, beta, epsilon, is_rmsnorm)

    tt_inp = []
    for d in range(n_devices):
        tt_inp.append(
            ttnn.as_tensor(
                inp_chunked[d],
                dtype=dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

    tt_gamma = []
    for d in range(n_devices):
        tt_gamma.append(
            ttnn.as_tensor(
                gamma_chunked[d].reshape(1, 1, -1, 32),
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

    tt_beta = []
    for d in range(n_devices):
        tt_beta.append(
            ttnn.as_tensor(
                beta_chunked[d].reshape(1, 1, -1, 32),
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

    # Run layernorm part 1
    tt_stats = []
    for d in range(n_devices):
        if is_rmsnorm:
            tt_stats.append(
                ttnn.experimental.operations.primary.rmsnorm_part1(tt_inp[d], compute_kernel_config=kernel_config)
            )
        else:
            tt_stats.append(
                ttnn.experimental.operations.primary.layernorm_part1(tt_inp[d], compute_kernel_config=kernel_config)
            )

    # AllGather stats
    # tt_stats = ttnn.experimental.tensor.all_gather(
    #     tt_stats,
    #     dim=3,
    #     num_links=1,
    #     output_mem_config=ttnn.DRAM_MEMORY_CONFIG
    # )
    tt_stats = ttnn.all_gather(tt_stats, num_links=1, dim=3)

    # Run layernorm part 2
    tt_out = []
    for d in range(n_devices):
        if is_rmsnorm:
            tt_out.append(
                ttnn.experimental.operations.primary.rmsnorm_part2(
                    tt_inp[d], tt_stats[d], epsilon, tt_gamma[d], compute_kernel_config=kernel_config
                )
            )
        else:
            tt_out.append(
                ttnn.experimental.operations.primary.layernorm_part2(
                    tt_inp[d], tt_stats[d], epsilon, tt_gamma[d], tt_beta[d], compute_kernel_config=kernel_config
                )
            )

    tt_output_host = torch.concat([tt2torch_tensor(tt_o) for tt_o in tt_out], -1)

    passing, output_str = comp_allclose(tt_output_host, out_torch, rtol=1e-1, atol=1e-01)
    logger.debug(f"torch vs tt poc = {output_str}")
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
def test_distributed_layernorm(inp_shape, n_devices, is_rmsnorm, dtype, device, fp32_enabled):
    print(device)
    # device = get_devices_for_t3000(all_devices, n_devices)
    run_distributed_layernorm(inp_shape, n_devices, is_rmsnorm, dtype, device, fp32_enabled)


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
def test_distributed_layernorm_with_program_cache(inp_shape, n_devices, is_rmsnorm, dtype, device, use_program_cache):
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
        run_distributed_layernorm(inp_shape, n_devices, is_rmsnorm, dtype, device)

    assert device.num_program_cache_entries() == 1, "Program cache should have only one entry" + str(
        device.num_program_cache_entries()
    )
