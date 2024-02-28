# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import math

import torch
import torchvision

import ttnn
from ttnn.model_preprocessing import (
    preprocess_model,
    preprocess_conv2d,
    fold_batch_norm2d_into_conv2d,
    fold_conv7s2_into_conv4s1,
    preprocess_remaining_children_and_parameters,
    convert_torch_model_to_ttnn_model,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0, pad_and_fold_conv_activation_for_unity_stride

from models.experimental.functional_resnet.tt.ttnn_functional_resnet import resnet_basic_block, resnet_bottleneck_block


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels <= 256
    ttnn_module_args["enable_auto_formatting"] = False  ## ttnn_module_args.kernel_size < (7, 7)
    ttnn_module_args["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
    ttnn_module_args["deallocate_activation"] = True if ttnn_module_args.kernel_size == (3, 3) else False


# @skip_for_wormhole_b0()
# def test_basic_block(device):
#     torch.manual_seed(0)

#     torch_model = torchvision.models.resnet.BasicBlock(inplanes=64, planes=64, stride=1).eval()

#     torch_input_tensor = torch.rand((8, 64, 56, 56), dtype=torch.float32)
#     torch_output_tensor = torch_model(torch_input_tensor)

#     reader_patterns_cache = {}
#     parameters = preprocess_model(
#         initialize_model=lambda: torch_model,
#         run_model=lambda model: model(torch_input_tensor),
#         custom_preprocessor=custom_preprocessor,
#         reader_patterns_cache=reader_patterns_cache,
#         device=device,
#     )

#     assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.9997)

#     padded_input_channels = math.ceil(input_tensor.shape[3] / 16) * 16
#     input_tensor = torch.nn.functional.pad(input_tensor, (0, padded_input_channels - input_tensor.shape[3], 0, 0, 0, 0))
#     input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

#     output_tensor = resnet_basic_block(input_tensor, parameters=parameters)

#     output_tensor = ttnn.to_torch(output_tensor)
#     output_tensor = torch.reshape(
#         output_tensor,
#         [
#             torch_input_tensor.shape[0],
#             torch_input_tensor.shape[2],
#             torch_input_tensor.shape[3],
#             torch_input_tensor.shape[1],
#         ],
#     )
#     output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
#     output_tensor = output_tensor.to(torch_input_tensor.dtype)

#     assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.9998)


# @skip_for_wormhole_b0()
# def test_basic_block_with_downsample(device):
#     torch.manual_seed(0)

#     def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
#         """1x1 convolution"""
#         return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#     torch_model = torchvision.models.resnet.BasicBlock(
#         inplanes=64, planes=64, stride=1, downsample=conv1x1(64, 64, 1)
#     ).eval()

#     torch_input_tensor = torch.rand((8, 64, 56, 56), dtype=torch.float32)
#     torch_output_tensor = torch_model(torch_input_tensor)

#     reader_patterns_cache = {}
#     parameters = preprocess_model(
#         initialize_model=lambda: torch_model,
#         run_model=lambda model: model(torch_input_tensor),
#         custom_preprocessor=custom_preprocessor,
#         reader_patterns_cache=reader_patterns_cache,
#         device=device,
#     )

#    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99966)


@skip_for_wormhole_b0()
def test_resnet_conv7s2(device):
    in_planes = 64

    torch_model = torch.nn.Conv2d(3, in_planes, kernel_size=7, stride=2, padding=[3, 3], bias=False)

    torch_input_tensor = torch.rand((8, 3, 224, 224), dtype=torch.float32)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=custom_preprocessor,
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    input_tensor = pad_and_fold_conv_activation_for_unity_stride(
        torch_input_tensor, *torch_model.padding, *torch_model.stride
    )

    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    output_tensor = parameters.copy_input_to_device(input_tensor)
    output_tensor = parameters(output_tensor)
    output_tensor = parameters.copy_output_from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.9998)


@skip_for_wormhole_b0()
def test_resnet(device):
    torch.manual_seed(0)

    torch_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()

    torch_input_tensor = torch.rand((8, 3, 224, 224), dtype=torch.float32)
    torch_output_tensor = torch_model(torch_input_tensor)

    def custom_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        if isinstance(model, torchvision.models.resnet.BasicBlock):
            ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1

            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)

            update_ttnn_module_args(ttnn_module_args.conv1)
            update_ttnn_module_args(ttnn_module_args.conv2)

            parameters["conv1"] = preprocess_conv2d(conv1_weight, conv1_bias, ttnn_module_args.conv1)
            parameters["conv2"] = preprocess_conv2d(conv2_weight, conv2_bias, ttnn_module_args.conv2)

            if model.downsample is not None:
                downsample_weight, downsample_bias = fold_batch_norm2d_into_conv2d(
                    model.downsample[0], model.downsample[1]
                )
                update_ttnn_module_args(ttnn_module_args.downsample[0])
                parameters["downsample"] = preprocess_conv2d(
                    downsample_weight, downsample_bias, ttnn_module_args.downsample[0]
                )
                ttnn_module_args["downsample"] = ttnn_module_args.downsample[0]

        elif isinstance(model, torchvision.models.resnet.ResNet):
            ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
            parameters["conv1"]: ttnn.Conv2d = fold_conv7s2_into_conv4s1(
                conv1_weight, conv1_bias, ttnn_module_args.conv1
            )

            return preprocess_remaining_children_and_parameters(
                model,
                name=name,
                convert_to_ttnn=convert_to_ttnn,
                custom_preprocessor=custom_preprocessor,
                parameters=parameters,
                ttnn_module_args=ttnn_module_args,
                already_preprocessed_children={"conv1", "bn1", "relu1"},
            )

            for child_name, child in tuple(model.named_children()) + named_parameters:
                if child_name in {"conv1", "bn1"}:
                    continue
                parameters[child_name] = convert_torch_model_to_ttnn_model(
                    child,
                    name=name,
                    convert_to_ttnn=convert_to_ttnn,
                    custom_preprocessor=custom_preprocessor,
                    ttnn_module_args=ttnn_module_args.get(child_name, None),
                )
                if child_name == "maxpool":
                    ttnn_module_args.maxpool["parallel_config_override"] = {
                        "grid_size": parameters["conv1"]["parallel_config"],
                        "ncores_nhw": parameters["conv1"]["num_cores_nhw"],
                    }
                    # update_ttnn_module_args(ttnn_module_args.maxpool)

        return parameters

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        reader_patterns_cache=reader_patterns_cache,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    input_tensor = pad_and_fold_conv_activation_for_unity_stride(
        torch_input_tensor, *torch_model.conv1.padding, *torch_model.conv1.stride
    )

    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    output_tensor = parameters.conv1.copy_input_to_device(input_tensor)
    output_tensor = parameters.conv1(output_tensor)
    output_tensor = parameters.maxpool(output_tensor)
    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)

    for basic_block_parameters in parameters.layer1.values():
        output_tensor = resnet_basic_block(output_tensor, parameters=basic_block_parameters)
    for basic_block_parameters in parameters.layer2.values():
        output_tensor = resnet_basic_block(output_tensor, parameters=basic_block_parameters)
    for basic_block_parameters in parameters.layer3.values():
        output_tensor = resnet_basic_block(output_tensor, parameters=basic_block_parameters)
    for basic_block_parameters in parameters.layer4.values():
        output_tensor = resnet_basic_block(output_tensor, parameters=basic_block_parameters)

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.reshape(output_tensor, (8, 1, 49, 512))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.global_avg_pool2d(output_tensor)
    output_tensor = output_tensor @ parameters.fc.weight + parameters.fc.bias

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.reshape(output_tensor, (8, 1000))

    # The check below doesn't work yet
    # assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999)


@skip_for_wormhole_b0()
def test_bottolneck_block(device):
    torch.manual_seed(0)

    torch_model = torchvision.models.resnet.Bottleneck(inplanes=2048, planes=512, stride=1).eval()
    torch_input_tensor = torch.rand((8, 2048, 7, 7), dtype=torch.float32)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=custom_preprocessor,
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    """input preparation and invocation of the Bottolneck class output reshape"""
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, input_tensor.shape[-1]))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = resnet_bottleneck_block(input_tensor, parameters)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.reshape(
        output_tensor,
        [
            torch_output_tensor.shape[0],
            torch_output_tensor.shape[2],
            torch_output_tensor.shape[3],
            torch_output_tensor.shape[1],
        ],
    )
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@skip_for_wormhole_b0()
def test_bottolneck_block_with_downsample(device):
    torch.manual_seed(0)

    def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
        """1x1 convolution"""
        return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

    torch_model = torchvision.models.resnet.Bottleneck(
        inplanes=512, planes=256, stride=2, downsample=conv1x1(512, 512 * 2, stride=2)
    ).eval()
    torch_input_tensor = torch.rand((8, 512, 28, 28), dtype=torch.float32)
    torch_output_tensor = torch_model(torch_input_tensor)
    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=custom_preprocessor,
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )
    # input preparation
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, input_tensor.shape[-1]))

    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # intialization of Bottolneck class and invocation
    output_tensor = resnet_bottleneck_block(input_tensor, parameters)

    # output tensor reshaping and comparison
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.reshape(
        output_tensor,
        [
            torch_output_tensor.shape[0],
            torch_output_tensor.shape[2],
            torch_output_tensor.shape[3],
            torch_output_tensor.shape[1],
        ],
    )
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    # validation of the output
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.98)


@skip_for_wormhole_b0()
def test_resnet_50(device):
    torch.manual_seed(0)

    torch_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).eval()
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    def custom_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        if isinstance(model, torchvision.models.resnet.Bottleneck):
            ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.conv2["activation"] = "relu"  # Fuse relu with conv1
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)
            conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.conv3, model.bn3)

            update_ttnn_module_args(ttnn_module_args.conv1)
            update_ttnn_module_args(ttnn_module_args.conv2)
            update_ttnn_module_args(ttnn_module_args.conv3)

            parameters["conv1"] = preprocess_conv2d(conv1_weight, conv1_bias, ttnn_module_args.conv1)
            parameters["conv2"] = preprocess_conv2d(conv2_weight, conv2_bias, ttnn_module_args.conv2)
            parameters["conv3"] = preprocess_conv2d(conv3_weight, conv3_bias, ttnn_module_args.conv3)

            if model.downsample is not None:
                downsample_weight, downsample_bias = fold_batch_norm2d_into_conv2d(
                    model.downsample[0], model.downsample[1]
                )
                update_ttnn_module_args(ttnn_module_args.downsample[0])
                parameters["downsample"] = preprocess_conv2d(
                    downsample_weight, downsample_bias, ttnn_module_args.downsample[0]
                )
                ttnn_module_args["downsample"] = ttnn_module_args.downsample[0]
        elif isinstance(model, torchvision.models.resnet.ResNet):
            ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
            parameters["conv1"] = fold_conv7s2_into_conv4s1(conv1_weight, conv1_bias, ttnn_module_args.conv1)

            named_parameters = tuple(
                (name, parameter) for name, parameter in model.named_parameters() if "." not in name
            )
            for child_name, child in tuple(model.named_children()) + named_parameters:
                if child_name in {"conv1", "bn1"}:
                    continue
                parameters[child_name] = convert_torch_model_to_ttnn_model(
                    child,
                    name=name,
                    convert_to_ttnn=convert_to_ttnn,
                    custom_preprocessor=custom_preprocessor,
                    ttnn_module_args=ttnn_module_args.get(child_name, None),
                )
        return parameters

    reader_patterns_cache = {}
    torch_input_tensor = torch.rand((8, 3, 224, 224), dtype=torch.float32)
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        reader_patterns_cache=reader_patterns_cache,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    torch_model.to(torch.bfloat16)
    torch_input_tensor_test = torch_input_tensor.to(torch.bfloat16)
    torch_output_tensor = torch_model(torch_input_tensor_test)

    input_tensor = pad_and_fold_conv_activation_for_unity_stride(
        torch_input_tensor, *torch_model.conv1.padding, *torch_model.conv1.stride
    )
    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)
    output_tensor = parameters.conv1.copy_input_to_device(input_tensor)
    output_tensor = parameters.conv1(output_tensor)
    output_tensor = parameters.maxpool(output_tensor)

    """
    1st bottolneck layer. all the blocks implemented by ttnn
    """
    output_tensor = ttnn.reshape(output_tensor, (1, 1, 56 * 56 * 8, 64))
    # output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)

    for bottolneck_block_parameters in list(parameters.layer1.values()):
        logger.debug(f"parameters 1st block {bottolneck_block_parameters}")
        output_tensor = resnet_bottleneck_block(output_tensor, bottolneck_block_parameters)

    """
    2nd bottolneck layer. 1st block implemented by torch rest by ttnn
    """
    for bottolneck_block_parameters in list(parameters.layer2.values()):
        logger.debug(f"parameters 2nd block {bottolneck_block_parameters}")
        output_tensor = resnet_bottleneck_block(output_tensor, bottolneck_block_parameters, device)

    breakpoint()

    """
    3rd bottolneck layer. 1st block implemented by torch rest by ttnn
    """
    for bottolneck_block_parameters in list(parameters.layer3.values()):
        logger.debug(f"parameters 3rd block {bottolneck_block_parameters}")
        output_tensor = resnet_bottleneck_block(output_tensor, bottolneck_block_parameters)

    """
    4th bottolneck layer. 1st block implemented by torch rest by ttnn
    """
    for bottolneck_block_parameters in list(parameters.layer4.values()):
        logger.debug(f"parameters 4th block {bottolneck_block_parameters}")
        output_tensor = resnet_bottleneck_block(output_tensor, bottolneck_block_parameters)

    # """
    # the last layers of the resnet
    # """
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.reshape(output_tensor, (8, 1, 49, 2048))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.global_avg_pool2d(output_tensor)
    output_tensor = output_tensor @ parameters.fc.weight + parameters.fc.bias

    """
    output verify
    """
    output_tensor_test = ttnn.to_torch(output_tensor)
    output_tensor_test = torch.reshape(output_tensor_test, (8, 1000))
    assert_with_pcc(torch_output_tensor, output_tensor_test, pcc=0.98)
