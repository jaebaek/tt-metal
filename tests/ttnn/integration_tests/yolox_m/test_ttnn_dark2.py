# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

from ttnn.model_preprocessing import preprocess_model

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


from models.experimental.functional_yolox_m.reference.dark2 import Dark2, Focus
from models.experimental.functional_yolox_m.tt.ttnn_dark2 import TtDark2, TtFocus

import tests.ttnn.integration_tests.yolox_m.custom_preprocessor_d2 as D2
import ttnn


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        # print('debug:',model,ttnn_module_args)
        parameters = D2.custom_preprocessor(device, model, name, ttnn_module_args)
        return parameters

    return custom_preprocessor


import pytest


# @pytest.mark.parametrize("device_l1_small_size", [32768], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@skip_for_wormhole_b0()
def test_dark2(device, reset_seeds, model_location_generator):
    model_path = model_location_generator("models", model_subdir="Yolox")
    if model_path == "models":
        state_dict = torch.load("tests/ttnn/integration_tests/yolox_m/yolox_m.pth", map_location="cpu")
    else:
        weights_pth = str(model_path / "yolox_m.pth")
        state_dict = torch.load(weights_pth)

    state_dict = state_dict["model"]
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(("backbone.backbone.dark2")))}
    torch_model = Dark2()
    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 48, 320, 320)  # Batch size of 1, 128 input channels, 160x160 height and width
    # torch_output_tensor1, torch_output_tensor2, torch_output_tensor3 = torch_model(torch_input_tensor)
    torch_output_tensor = torch_model(torch_input_tensor)
    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtDark2(parameters)

    # Tensor Preprocessing
    #

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #
    output_tensor = ttnn.to_torch(output_tensor)
    print("shape:", torch_output_tensor.shape)
    output_tensor = output_tensor.reshape(1, 160, 160, 96)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)  # PCC = 0.9852007334820249
