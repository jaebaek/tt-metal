# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import tt_lib as ttl
from models.utility_functions import comp_allclose, skip_for_wormhole_b0
from loguru import logger

TILE_HEIGHT = 32
TILE_WIDTH = 32


@pytest.mark.parametrize("p", [1.0])
@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [2, 0.1, 0.1],
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 3, 3],
    ],
)
def test_moreh_norm(input_shape, p, dim_rtol_atol, device):
    torch.manual_seed(2024)

    dim, rtol, atol = dim_rtol_atol

    cpu_x = torch.zeros(input_shape, dtype=torch.float32)
    cpu_x[0, 0, 0:3, 0:3] = 1

    print("cpu_x ", cpu_x)

    # actual
    npu_x = (
        ttl.tensor.Tensor(cpu_x.bfloat16(), ttl.tensor.DataType.BFLOAT16)
        .pad_to_tile(float("0"))
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    npu_y = ttl.operations.primary.moreh_norm(npu_x, p=p, dim=dim)

    print("npu_y", npu_y)

    cpu_tensor = npu_y.cpu().to(ttl.tensor.Layout.ROW_MAJOR).unpad_from_tile([1, 1, 7, 7]).to_torch()

    print("cpu_tensor", cpu_tensor)
