# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import matmul as tt_matmul
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand


def run_matmul_test(input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, device):
    torch.manual_seed(0)

    x = gen_rand(size=input_shape_1, low=-100, high=100)
    y = gen_rand(size=input_shape_2, low=-100, high=100)

    tt_result = tt_matmul(
        x=x,
        y=y,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )


# bottom - 16000000
# top - 16.776.832

test_sweep_args = [
    (
        (1, 1, 16400000, 2),
        (1, 1, 2, 2),
        [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
    ),
]


@pytest.mark.parametrize(
    "input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config",
    (test_sweep_args),
)
def test_matmul_test(input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, device):
    random.seed(0)
    run_matmul_test(input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, device)
