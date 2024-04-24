# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib as ttl
import pytest
from models.utility_functions import comp_allclose
from loguru import logger

torch.set_printoptions(threshold=1000000, linewidth=100000000, sci_mode=False)


@pytest.mark.parametrize(
    "shape_dim",
    (((1, 1, 32 * 3, 32 * 1), 2),),
)
def test_logsoftmax_backward_large_algorithm_for_dim_hw(shape_dim, device):
    device.enable_program_cache()
    shape, dim = shape_dim
    torch.manual_seed(0)

    y = torch.ones(size=shape).to(torch.bfloat16) * 0
    dev_y = ttl.tensor.Tensor(y, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)

    dy = torch.ones(size=shape).to(torch.bfloat16) * 1
    dev_dy = ttl.tensor.Tensor(dy, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)

    # y.backward(dy)
    strategy = (
        ttl.operations.primary.MorehSoftmaxBackwardOpParallelizationStrategy.LARGE_W
        if dim == 3
        else ttl.operations.primary.MorehSoftmaxBackwardOpParallelizationStrategy.LARGE_H
    )
    tt_npu = ttl.operations.primary.moreh_logsoftmax_backward(dev_y, dev_dy, dim, None, strategy)

    tt_dev = tt_npu.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)

    rtol = atol = 0.5
    expected = dy - y
    passing, out = comp_allclose(expected, tt_dev, rtol=rtol, atol=atol)
    logger.debug(out)
    if passing == False:
        print("expected", expected)
        print("tt_dev", tt_dev)

    assert passing
