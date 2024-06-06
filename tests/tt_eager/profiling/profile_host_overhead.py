# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import tt_lib
import time
import statistics
from loguru import logger
from models.utility_functions import torch2tt_tensor
from tests.tt_eager.profiling import ops_for_profiling
from tracy import signpost


test_sweep_args = [
    # (
    #     (1, 2, 1024, 1024),
    #     tt_lib.tensor.DataType.BFLOAT16,
    #     tt_lib.tensor.Layout.TILE,
    #     tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
    #     tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
    # ),
    (
        (1, 4, 1024, 1024),
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.TILE,
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
    ),
]

all_num_call_to_stack = [1, 3]  # For 10 and more test  execution spills to dispatch
num_repeats = 10


def measure_host_overhead(op_func, op_name, device, num_call_to_stack, is_warmup):
    if not is_warmup:
        signpost(header=f"start {op_name}")

    start_time = time.time()
    for _ in range(num_call_to_stack):
        op_func()

    tt_lib.device.Synchronize(device)

    duration = 1000 * (time.time() - start_time)
    total_op_time = duration / num_call_to_stack
    logger.info(f"{num_call_to_stack} calls and Synchronize after {duration:.2f}ms ({total_op_time:.2f}ms per call)")

    start_time = time.time()
    for _ in range(num_call_to_stack):
        op_func()

    duration = 1000 * (time.time() - start_time)
    overhead_ms = duration / num_call_to_stack
    logger.info(f"{num_call_to_stack} calls without Synchronize {duration:.2f}ms ({overhead_ms:.2f}ms per call)")

    start_time = time.time()
    tt_lib.device.Synchronize(device)
    duration = 1000 * (time.time() - start_time)
    duration_per_call = duration / num_call_to_stack
    logger.info(f"Synchronize {duration:.2f}ms ({duration_per_call:.2f}ms per call)")

    if not is_warmup:
        signpost(header=f"end {op_name}")

    # Dumping profile info
    logger.info(f"Dumping device profiler data")
    tt_lib.device.DumpDeviceProfiler(device)

    return overhead_ms, total_op_time


def measure_host_overhead_binary(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    device,
    op,
    num_call_to_stack,
    num_repeats,
    shape_func=None,
    is_warmup=False,
):
    input_shape_0 = input_shape
    input_shape_1 = input_shape

    if shape_func is not None:
        input_shape_0, input_shape_1 = shape_func(input_shape)

    x = torch.Tensor(size=input_shape_0).uniform_(-100, 100)
    y = torch.Tensor(size=input_shape_1).uniform_(-100, 100)

    x = torch2tt_tensor(x, device, dlayout, in_mem_config, dtype)
    y = torch2tt_tensor(y, device, dlayout, in_mem_config, dtype)

    def op_func():
        op["op"](x, y)

    result_overhead = []
    result_op = []

    for _ in range(num_repeats):
        overhead_ms, total_op_time = measure_host_overhead(op_func, op["name"], device, num_call_to_stack, is_warmup)
        result_overhead.append(overhead_ms)
        result_op.append(total_op_time)

    return result_overhead, result_op


def measure_host_overhead_unary(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    device,
    op,
    num_call_to_stack,
    num_repeats,
    shape_func=None,
    is_warmup=False,
):
    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    x = torch2tt_tensor(x, device, dlayout, in_mem_config, dtype)

    def op_func():
        op["op"](x)

    result_overhead = []
    result_op = []

    for _ in range(num_repeats):
        overhead_ms, total_op_time = measure_host_overhead(op_func, op["name"], device, num_call_to_stack, is_warmup)
        result_overhead.append(overhead_ms)
        result_op.append(total_op_time)

    return result_overhead, result_op


def measure_host_overhead_ternary(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    device,
    op,
    num_call_to_stack,
    num_repeats,
    shape_func=None,
    is_warmup=False,
):
    input_shape_0 = input_shape
    input_shape_1 = input_shape
    input_shape_2 = input_shape

    if shape_func is not None:
        input_shape_0, input_shape_1, input_shape_2 = shape_func(input_shape)

    x = torch.Tensor(size=input_shape_0).uniform_(-100, 100)
    y = torch.Tensor(size=input_shape_1).uniform_(-100, 100)
    z = torch.Tensor(size=input_shape_2).uniform_(-100, 100)

    x = torch2tt_tensor(x, device, dlayout, in_mem_config, dtype)
    y = torch2tt_tensor(y, device, dlayout, in_mem_config, dtype)
    z = torch2tt_tensor(z, device, dlayout, in_mem_config, dtype)

    def op_func():
        op["op"](x, y, z)

    result_overhead = []
    result_op = []

    for _ in range(num_repeats):
        overhead_ms, total_op_time = measure_host_overhead(op_func, op["name"], device, num_call_to_stack, is_warmup)
        result_overhead.append(overhead_ms)
        result_op.append(total_op_time)

    return result_overhead, result_op


def run_measure_host_overhead(op, device, text_file, measuring_func):
    results_overhead = []
    results_op = []
    op_count = 0

    for input_shape, dtype, dlayout, in_mem_config, out_mem_config in test_sweep_args:
        logger.info("")
        logger.info(f"Profiling op {op['name']} for input shape {input_shape}")

        if "layout" in op and op["layout"] == "ROW_MAJOR":
            dlayout = tt_lib.tensor.Layout.ROW_MAJOR

        shape_func = None if "shape_func" not in op else op["shape_func"]

        # Warmup
        measuring_func(
            input_shape,
            dtype,
            dlayout,
            in_mem_config,
            out_mem_config,
            device,
            op,
            num_call_to_stack=1,
            num_repeats=1,
            shape_func=shape_func,
            is_warmup=True,
        )

        for num_call_to_stack in all_num_call_to_stack:
            overhead_ms, op_ms = measuring_func(
                input_shape,
                dtype,
                dlayout,
                in_mem_config,
                out_mem_config,
                device,
                op,
                num_call_to_stack,
                num_repeats,
                shape_func=shape_func,
            )

            op_count += len(overhead_ms) * num_call_to_stack * 2
            results_overhead += overhead_ms
            results_op += op_ms

    min_val = round(min(results_overhead), 2)
    mean_val = round(statistics.mean(results_overhead), 2)

    # min_val_op = round(min(results_op), 2)
    mean_val_op = round(statistics.mean(results_op), 2)

    logger.info(f"Measure overhead of launching {op['name']} is {min_val:.2f}ms (mean {mean_val:.2f}ms)")
    text_file.write(f"{op['name']},{op_count},{min_val},{mean_val},{mean_val_op}\n")


def test_host_overhead(device, user_input):
    """
    Run witout tracy:
    pytest tests/tt_eager/profiling/profile_host_overhead.py --input-method cli --cli-input host_overhead_profile

    Run with tracy:
    python -m tracy -v -r -p -o host_overhead_profile -m "pytest tests/tt_eager/profiling/profile_host_overhead.py --input-method cli --cli-input host_overhead_profile"

    Run only for one op:
    python -m tracy -v -r -p -o host_overhead_profile -m "pytest tests/tt_eager/profiling/profile_host_overhead.py --input-method cli --cli-input host_overhead_profile::tt_lib.tensor.atan2"
    """

    if "::" in user_input[0]:
        splitted = user_input[0].split("::")
        out_directory = splitted[0]
        op_name = splitted[1]
        out_file_path = os.path.join(out_directory, f"host_overhead_{op_name}.csv")
    else:
        out_directory = user_input[0]
        out_file_path = os.path.join(out_directory, f"host_overhead_profiler_output.csv")
        op_name = ""

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    with open(out_file_path, "w") as text_file:
        text_file.write(f"op,count,overhead min(ms),overhead mean(ms),total mean(ms)\n")

        for op in ops_for_profiling.all_binary_ops:
            if op_name != "":
                if op["name"] != op_name:
                    continue

            run_measure_host_overhead(op, device, text_file, measure_host_overhead_binary)

        for op in ops_for_profiling.all_unary_ops:
            if op_name != "":
                if op["name"] != op_name:
                    continue

            run_measure_host_overhead(op, device, text_file, measure_host_overhead_unary)

        for op in ops_for_profiling.all_ternary_ops:
            if op_name != "":
                if op["name"] != op_name:
                    continue

            run_measure_host_overhead(op, device, text_file, measure_host_overhead_ternary)
