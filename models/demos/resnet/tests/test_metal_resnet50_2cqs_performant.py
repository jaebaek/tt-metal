# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import tt_lib

from models.demos.resnet.tests.test_metal_resnet50 import run_resnet50_inference, run_2cq_model
from models.utility_functions import skip_for_wormhole_b0


@skip_for_wormhole_b0("This test is not supported on WHB0, please use the TTNN version.")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_hw_cqs": 2}], indirect=True)
@pytest.mark.parametrize("batch_size", [20], ids=["batch_20"])
@pytest.mark.parametrize(
    "weights_dtype",
    [tt_lib.tensor.DataType.BFLOAT8_B],
    ids=["weights_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [tt_lib.tensor.DataType.BFLOAT8_B],
    ids=["activations_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "math_fidelity",
    [tt_lib.tensor.MathFidelity.LoFi],
    ids=["LoFi"],
)
def test_run_resnet50_2cqs_inference(
    device, use_program_cache, batch_size, weights_dtype, activations_dtype, math_fidelity, imagenet_sample_input
):
    run_resnet50_inference(
        device,
        use_program_cache,
        batch_size,
        weights_dtype,
        activations_dtype,
        math_fidelity,
        imagenet_sample_input,
        run_2cq_model,
    )
