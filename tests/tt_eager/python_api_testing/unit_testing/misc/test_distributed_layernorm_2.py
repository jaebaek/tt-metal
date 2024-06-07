# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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


def run_layernorm_part_2(inp_shape, n_devices, is_rmsnorm, dtype, device):
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

        passing, output_str = comp_allclose(ref_chunks[d], lnp2_out, rtol=1e-03, atol=1e-06)
        logger.debug(f"Out passing={passing}")
        logger.debug(f"Output pcc={output_str}")

        all_pass = all_pass and passing

    assert all_pass


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
def test_layernorm_part_2(inp_shape, n_devices, is_rmsnorm, dtype, device):
    run_layernorm_part_2(inp_shape, n_devices, is_rmsnorm, dtype, device)


# TODO: program caching test
