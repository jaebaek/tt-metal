#! /usr/bin/env bash
NUM_LOOPS=$1

for ((i=0; i<${NUM_LOOPS}; i++))
do
	pytest -svv tests/tt_eager/python_api_testing/unit_testing/misc/test_matmul_1d_2d.py::test_multi_core_matmul_2d_8x8[False-True-True-2048-2048-4096-None-dtype5-fidelity5-True-False-no_bias]
done
