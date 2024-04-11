# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import time
import random
from pathlib import Path
from itertools import chain
from functools import partial

from loguru import logger

from tests.scripts.common import (
    run_single_test,
    run_process_and_get_result,
    report_tests,
    TestEntry,
    error_out_if_test_report_has_failures,
    TestSuiteType,
    get_git_home_dir_str,
    filter_empty,
    void_for_whb0,
)
from tests.scripts.cmdline_args import (
    get_tt_metal_arguments_from_cmdline_args,
    get_cmdline_args,
)

TT_EAGER_COMMON_TEST_ENTRIES = (
    TestEntry("tt_eager/tests/ops/test_eltwise_binary_op", "ops/test_eltwise_binary_op"),
    TestEntry("tt_eager/tests/ops/test_bcast_op", "ops/test_bcast_op"),
    TestEntry("tt_eager/tests/ops/test_reduce_op", "ops/test_reduce_op"),
    TestEntry("tt_eager/tests/ops/test_transpose_op", "ops/test_transpose_op"),
    TestEntry("tt_eager/tests/ops/test_bmm_op", "ops/test_bmm_op"),
    void_for_whb0(TestEntry("tt_eager/tests/ops/test_eltwise_unary_op", "ops/test_eltwise_unary_op")),
    void_for_whb0(
        TestEntry(
            "tt_eager/tests/ops/test_transpose_wh_single_core",
            "ops/test_transpose_wh_single_core",
        )
    ),
    void_for_whb0(
        TestEntry(
            "tt_eager/tests/ops/test_transpose_wh_multi_core",
            "ops/test_transpose_wh_multi_core",
        )
    ),
    void_for_whb0(TestEntry("tt_eager/tests/ops/test_tilize_op", "ops/test_tilize_op")),
    void_for_whb0(
        TestEntry(
            "tt_eager/tests/ops/test_tilize_op_channels_last",
            "ops/test_tilize_op_channels_last",
        )
    ),
    void_for_whb0(
        TestEntry(
            "tt_eager/tests/ops/test_tilize_zero_padding",
            "ops/test_tilize_zero_padding",
        )
    ),
    void_for_whb0(
        TestEntry(
            "tt_eager/tests/ops/test_tilize_zero_padding_channels_last",
            "ops/test_tilize_zero_padding_channels_last",
        )
    ),
    TestEntry("tt_eager/tests/ops/test_layernorm_op", "ops/test_layernorm_op"),
    TestEntry("tt_eager/tests/ops/test_softmax_op", "ops/test_softmax_op"),
    TestEntry("tt_eager/tests/ops/test_average_pool", "ops/test_average_pool"),
    TestEntry("tt_eager/tests/ops/test_multi_queue_api", "ops/test_multi_queue_api"),
    TestEntry(
        "tt_eager/tests/tensors/test_host_device_loopback",
        "tensors/test_host_device_loopback",
    ),
    TestEntry("tt_eager/tests/tensors/test_copy_and_move", "tensors/test_copy_and_move"),
    TestEntry("tt_eager/tests/tensors/test_raw_host_memory_pointer", "tensors/test_raw_host_memory_pointer"),
    TestEntry("tt_eager/tests/tensors/test_async_tensor_apis", "tensors/test_async_tensor_apis"),
    # Integration tests
    # void_for_whb0(TestEntry("tt_eager/tests/integration_tests/test_bert", "integration_tests/test_bert")),
)

TT_EAGER_SLOW_DISPATCH_TEST_ENTRIES = (void_for_whb0(TestEntry("tt_eager/tests/ops/test_sfpu", "ops/test_sfpu")),)


def run_single_tt_eager_test(test_entry, timeout):
    run_test = partial(run_single_test, "tt_eager", timeout=timeout)

    logger.info(f"========= RUNNING TT EAGER CPP TEST - {test_entry}")

    return run_test(test_entry)


def run_tt_cpp_tests(test_entries, timeout, run_single_test):
    make_test_status_entry = lambda test_entry_: (
        test_entry_,
        run_single_test(test_entry_, timeout),
    )

    seed = time.time()

    random.seed(seed)
    random.shuffle(test_entries)
    logger.info(f"SHUFFLED CPP TESTS - Using order generated by seed {seed}")

    test_and_status_entries = map(make_test_status_entry, test_entries)

    return dict(test_and_status_entries)


@filter_empty
def get_tt_eager_fast_dispatch_test_entries():
    return list(TT_EAGER_COMMON_TEST_ENTRIES)


@filter_empty
def get_tt_eager_slow_dispatch_test_entries():
    return list(TT_EAGER_COMMON_TEST_ENTRIES) + list(TT_EAGER_SLOW_DISPATCH_TEST_ENTRIES)


if __name__ == "__main__":
    cmdline_args = get_cmdline_args(TestSuiteType.TT_EAGER)

    timeout, tt_arch, dispatch_mode = get_tt_metal_arguments_from_cmdline_args(cmdline_args)

    if dispatch_mode == "slow":
        tt_eager_test_entries = get_tt_eager_slow_dispatch_test_entries()
    else:
        tt_eager_test_entries = get_tt_eager_fast_dispatch_test_entries()

    eager_test_report = run_tt_cpp_tests(tt_eager_test_entries, timeout, run_single_tt_eager_test)

    test_report = {**eager_test_report}

    report_tests(test_report)

    error_out_if_test_report_has_failures(test_report)
