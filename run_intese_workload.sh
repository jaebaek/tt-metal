#! /usr/bin/env bash
NUM_LOOPS=$1

	WL_NUM_LOOPS=${NUM_LOOPS} pytest -svv tests/tt_eager/python_api_testing/unit_testing/misc/test_matmul_1d_2d.py::test_loop_multi_core_matmul_2d_8x8[False-True-True-3072-2048-4096-None-dtype0-fidelity0-True-False-no_bias]
