import torch
import torch.nn as nn
import ttnn
import tt_lib
import pytest
from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d, preprocess_model

from tests.ttnn.utils_for_testing import assert_with_pcc


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = True  # ttnn_module_args.in_channels < 256
    ttnn_module_args["use_shallow_conv_variant"] = False


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        print("ttnn_module_args", ttnn_module_args)
        parameters = {}
        parameters = TtDark5.custom_preprocessor(device, model, name, ttnn_module_args)
        return parameters

    return custom_preprocessor


class Dark5(nn.Module):
    def __init__(self):
        super().__init__()
        self.c4 = nn.Conv2d(768, 384, kernel_size=1, stride=1, bias=False)
        self.b4 = nn.BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, inputs):
        x4 = self.c4(inputs)
        x4 = self.b4(x4)
        x4 = self.silu(x4)
        return x4


class TtDark5:
    def custom_preprocessor(device, model, name, ttnn_module_args):
        print("We do reach here!")
        parameters = {}
        if isinstance(model, Dark5):
            ttnn_module_args.c4["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c4["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c4["dtype"] = ttnn.bfloat16
            ttnn_module_args.c4["weights_dtype"] = ttnn.bfloat16
            ttnn_module_args.c4["activation"] = None  # Fuse relu with conv1
            ttnn_module_args.c4["deallocate_activation"] = False
            ttnn_module_args.c4["conv_blocking_and_parallelization_config_override"] = None
            conv4_weight, conv4_bias = fold_batch_norm2d_into_conv2d(model.c4, model.b4)
            update_ttnn_module_args(ttnn_module_args.c4)
            parameters["c4"], c4_parallel_config = preprocess_conv2d(
                conv4_weight, conv4_bias, ttnn_module_args.c4, return_parallel_config=True
            )

    def __init__(self, device, parameters) -> None:
        self.device = device
        print("keys in parameters in Dark5_conv are: ", parameters.keys())
        self.c4 = parameters.c4

    def __call__(self, inputs, device):
        inputs = tt_lib.tensor.interleaved_to_sharded(inputs, self.c4.conv.input_sharded_memory_config)
        output_tensor = self.c4(inputs)
        # output_tensor = ttnn.to_torch(output_tensor)
        # output_tensor = ttnn.from_torch(output_tensor, device = device, dtype = ttnn.bfloat16, layout = ttnn.TILE_LAYOUT)
        # output_tensor = ttnn.leaky_relu(output_tensor, slope = True)
        # output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        return output_tensor


@pytest.mark.parametrize("device_l1_small_size", [32768], indirect=True)
def test_unit_head_c11(device, model_location_generator, reset_seeds):
    model_path = model_location_generator("models", model_subdir="Yolox")
    if model_path == "models":
        state_dict = torch.load("tests/ttnn/integration_tests/yolox_m/yolox_m.pth", map_location="cpu")
    else:
        weights_pth = str(model_path / "yolox_m.pth")
        state_dict = torch.load(weights_pth)

    state_dict = state_dict["model"]
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(("backbone.backbone.dark5.2.conv1")))}

    torch_model = Dark5()

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]
    print("new_state_dict", new_state_dict.keys())
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    reader_patterns_cache = {}

    input = ttnn.load_tensor("tests/ttnn/integration_tests/yolox_m/dark5_conv4_inp_torch.pt")
    torch_input = ttnn.to_torch(input)
    torch_input = torch.reshape(torch_input.float(), (1, 20, 20, 768))
    torch_input = torch.permute(torch_input, (0, 3, 1, 2))

    ttnn_input = ttnn.load_tensor("tests/ttnn/integration_tests/yolox_m/dark5_conv4_inp_ttnn.pt")
    ttnn_input = ttnn_input.to(device)
    print("torch_input", torch_input.shape)

    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtDark5(device, parameters)

    torch_output = torch_model(torch_input)
    ttnn_output = ttnn_model(ttnn_input, device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = torch.reshape(ttnn_output, (1, 20, 20, 384))
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))

    assert_with_pcc(torch_output, ttnn_output)  # PCC = 0.09225181659794127
