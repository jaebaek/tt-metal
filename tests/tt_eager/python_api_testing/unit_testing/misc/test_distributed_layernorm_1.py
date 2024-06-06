# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from models.utility_functions import tt2torch_tensor

import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose


def reference_part1(x, n_devices, is_rmsnorm):
    num_chunks = len(x)
    S = x[0].shape[2]
    B = x[0].shape[0]
    counts = []
    sumxs = []
    sumx2s = []
    # Distributed processing
    for chunk in x:
        count_local = chunk.shape[-1]
        sumx_local = torch.sum(chunk, dim=-1, keepdim=True)
        sumx2_local = torch.sum(torch.square(chunk), dim=-1, keepdim=True)

        counts.append(count_local)
        sumxs.append(sumx_local)
        sumx2s.append(sumx2_local)

    # pad with zeros as for tiles
    output = []
    for i in range(num_chunks):
        if is_rmsnorm:
            output.append(torch.concat([sumx2s[i], torch.zeros([B, 1, S, 31])], dim=-1))
        else:
            output.append(
                torch.concat([sumx2s[i], torch.zeros([B, 1, S, 31]), sumxs[i], torch.zeros([B, 1, S, 31])], dim=-1)
            )

    return output


def metal_poc_part1(xs, n_devices, is_rmsnorm):
    sumxs = []

    # Each device computes local statistics mean(x) and mean(x^2)
    # sumx = torch.sum(xs, dim=-1, keepdim=True)
    for i in range(n_devices):
        meanx_local = ttnn.experimental.tensor.reduce(
            xs[i], ttnn.experimental.tensor.ReduceOpMath.SUM, ttnn.experimental.tensor.ReduceOpDim.W, scaler=1.0
        )
        sumxs.append(meanx_local)

    # sumx2 = torch.sum(torch.square(xs), dim=-1, keepdim=True)
    sumx2s = []
    for i in range(n_devices):
        x2_local = ttnn.experimental.tensor.pow(xs[i], 2)
        sumx2_local = ttnn.experimental.tensor.reduce(
            x2_local, ttnn.experimental.tensor.ReduceOpMath.SUM, ttnn.experimental.tensor.ReduceOpDim.W, scaler=1.0
        )
        sumx2s.append(sumx2_local)

    output = []
    for i in range(n_devices):
        if is_rmsnorm:
            output.append(ttnn.experimental.tensor.concat([sumx2s[i]], 3))
        else:
            output.append(ttnn.experimental.tensor.concat([sumx2s[i], sumxs[i]], 3))

    return output


def ln_part1_op(xs, n_devices, is_rmsnorm):
    kernel_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttnn.experimental.tensor.MathFidelity.HiFi4,  # Highest fidelity
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    tt_out = []
    for d in range(n_devices):
        if is_rmsnorm:
            tt_out.append(
                ttnn.experimental.operations.primary.rmsnorm_part1(xs[d], compute_kernel_config=kernel_config)
            )
        else:
            tt_out.append(
                ttnn.experimental.operations.primary.layernorm_part1(xs[d], compute_kernel_config=kernel_config)
            )
    return tt_out


def run_layernorm_part_1(inp_shape, n_devices, is_rmsnorm, dtype, device):
    torch.manual_seed(1234)

    # Set print options
    torch.set_printoptions(threshold=100)

    canon_inp = torch.randn(inp_shape) * 4 - 1

    # Get per-chunk inputs
    inp_chunked = canon_inp.chunk(n_devices, dim=-1)

    all_pass = True

    # LNP1 reference
    out_torch = reference_part1(inp_chunked, n_devices, is_rmsnorm)
    out_torch = torch.concat(out_torch, -1)

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

    # LNP1 metal POC implementation
    tt_out = metal_poc_part1(tt_inp, n_devices, is_rmsnorm)

    # print(f"Ref tt output:")
    # print(f"tt_out[0] sum(x**2): {tt2torch_tensor(tt_out[0])[0,0,:,0]}")
    # if not is_rmsnorm:
    #     print(f"tt_out[0] sum(x): {tt2torch_tensor(tt_out[0])[0,0,:,32]}")

    tt_output_host = torch.concat([tt2torch_tensor(tt_o) for tt_o in tt_out], -1)

    passing, output_str = comp_allclose(tt_output_host, out_torch, rtol=1e-1, atol=1e-01)
    logger.debug(f"torch vs tt poc = {output_str}")
    all_pass = all_pass and passing

    # LNP1 OP
    tt_out = ln_part1_op(tt_inp, n_devices, is_rmsnorm)

    # print(f"Test tt output:")
    # print(f"tt_out[0] sum(x**2): {tt2torch_tensor(tt_out[0])[0,0,:,0]}")
    # if not is_rmsnorm:
    #     print(f"tt_out[0] sum(x): {tt2torch_tensor(tt_out[0])[0,0,:,32]}")

    tt_output_host = torch.concat([tt2torch_tensor(tt_o) for tt_o in tt_out], -1)
    passing, output_str = comp_allclose(tt_output_host, out_torch, rtol=1e-1, atol=1e-01)
    logger.debug(f"tt poc vs tt LNP1 op = {output_str}")
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
        (1, 1, 2048, 1024),
    ],
)
@pytest.mark.parametrize(
    "n_devices",
    [1, 4, 8],
)
@pytest.mark.parametrize(
    "is_rmsnorm",
    [True, False],
    ids=["rmsnorm", "layernorm"],
)
def test_layernorm_part_1(inp_shape, n_devices, is_rmsnorm, dtype, device):
    run_layernorm_part_1(inp_shape, n_devices, is_rmsnorm, dtype, device)


# TODO: program caching test
