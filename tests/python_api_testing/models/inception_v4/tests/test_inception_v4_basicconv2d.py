from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from torch import nn
from loguru import logger
import tt_lib
from python_api_testing.models.inception_v4.tt.inception_v4_basicconv2d import (
    TtBasicConv2d,
)
import timm
from utility_functions_new import (
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)


def test_basic_conv2d_inference():
    torch.manual_seed(1234)

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    reference_model = timm.create_model("inception_v4", pretrained=True)
    reference_model.eval()

    basic_conv2d_block_id = 0
    torch_model = reference_model.features[basic_conv2d_block_id]
    base_address = "features.0"
    in_channels = torch_model.conv.in_channels
    out_channels = torch_model.conv.out_channels
    kernel_size = torch_model.conv.kernel_size[0]
    stride = torch_model.conv.stride[0]
    padding = torch_model.conv.padding[0]

    tt_module = TtBasicConv2d(
        device=device,
        state_dict=reference_model.state_dict(),
        base_address=base_address,
        in_planes=in_channels,
        out_planes=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    tt_module.eval()

    with torch.no_grad():
        test_input = torch.rand(1, 3, 64, 64)
        pt_out = torch_model(test_input)

        test_input = torch2tt_tensor(test_input, device)
        tt_out = tt_module(test_input)
        tt_out_torch = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_torch, 0.99)
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("TtBasicConv2d Passed!")
    else:
        logger.warning("TtBasicConv2d Failed!")

    assert does_pass
