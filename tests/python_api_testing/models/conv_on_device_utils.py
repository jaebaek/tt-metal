import torch
import numpy as np
from libs import tt_lib
from utility_functions import pad_by_zero, unpad_from_zero, torch2tt_tensor
from python_api_testing.fused_ops.conv import conv as TtConv
from libs.tt_lib.utils import (
    _nearest_32
)
from loguru import logger

def is_conv_supported_on_device(conv_params):
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

    if (C%32 != 0 or K%32 != 0 or dilation != 1 or groups != 1):
        logger.warning("DOES NOT HAVE SUPPORT FOR Conv with following parameters -")
        logger.warning("K="+str(K)+" C="+str(C)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))
        return False

    return True

def can_run_conv_on_device(act_shape, conv_params):
    #return False
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    [N,C,H,W] = act_shape

    logger.info("Conv with following parameters -")
    logger.info("K="+str(K)+" C="+str(C)+" H="+str(H)+" W="+str(W)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))

    if (C % 32 != 0 or K%32 != 0 or dilation != 1 or groups != 1):
        return False

    return True

def run_conv_on_tt_device(x: torch.Tensor, conv_on_tt, conv_params, device, host):
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    [N,C,H,W] = x.shape

    logger.info("Running Conv with following parameters on device -")
    logger.info("K="+str(K)+" C="+str(C)+" H="+str(H)+" W="+str(W)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))

    OH = ((int) ((H - R + 2 * P_H) / U)) + 1
    OW = ((int) ((W - S + 2 * P_W) / V)) + 1
    conv_as_mm_output_shape_unpadded = [1,1,OH*OW,K]
    x = torch2tt_tensor(x, device, tt_lib.tensor.Layout.CHANNELS_LAST, tt_lib.tensor.MemoryConfig(False, 0))

    logger.info("Going to run conv on tt device")
    x = conv_on_tt(x)

    logger.info("conv on tt device done")
    x = unpad_from_zero(x, conv_as_mm_output_shape_unpadded, host)

    # Convert matmul output layout to conv output layout
    x = torch.transpose(x, 2, 3)
    assert(list(x.shape) == [1,1,K,(OH*OW)])

    return x.reshape([1,K,OH,OW])

def run_conv_on_device_wrapper(conv_weight, conv_params, device, host, conv_bias=None, inputs_on_device=False, output_on_device=False):
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    conv_on_device = TtConv(conv_weight, conv_params, device, conv_bias)

    def run_conv_on_device(x):

        if inputs_on_device:
            # copy input to host to convert data to channels layout and pad if required
            logger.info("Starting conv - going to copy conv from device to host")
            x = x.to(host)
            assert len(x.shape()) == 4
            if(x.layout() != tt_lib.tensor.Layout.ROW_MAJOR):
                x = x.to(tt_lib.tensor.Layout.ROW_MAJOR)
            x = torch.Tensor(x.data()).reshape(x.shape())
            logger.info("copied conv input from device to host")
        [N,C,H,W] = x.shape
        OH = ((int) ((H - R + 2 * P_H) / U)) + 1
        OW = ((int) ((W - S + 2 * P_W) / V)) + 1
        logger.info("Running Conv with following parameters on device -")
        logger.info("K="+str(K)+" C="+str(C)+" H="+str(H)+" W="+str(W)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))
        conv_as_mm_output_shape_unpadded = [1,1,OH*OW,K]
        x_shape_channel_padded = [N,_nearest_32(C),H,W]
        x = tt_lib.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            ).pad(x_shape_channel_padded, (0,0,0,0), 0).to(tt_lib.tensor.Layout.CHANNELS_LAST).to(device, tt_lib.tensor.MemoryConfig(False, 0))
        logger.info("Copied conv input after layout transformation from host to device. Going to run conv on tt device")
        x = conv_on_device(x)
        logger.info("conv on tt device done")
        x = unpad_from_zero(x, conv_as_mm_output_shape_unpadded, host)
        logger.info("copied conv output from device to host")
        # Convert matmul output layout to conv output layout
        x = torch.transpose(x, 2, 3)
        assert(list(x.shape) == [1,1,K,(OH*OW)])
        x = x.reshape([1,K,OH,OW])
        logger.info("done conv output layout transformation on host")
        if output_on_device:
            # copy the output from host (has been converted to row major layout above) to device
            x = tt_lib.tensor.Tensor(
                x.reshape(-1).tolist(),
                x.shape,
                tt_lib.tensor.DataType.BFLOAT16,
                tt_lib.tensor.Layout.ROW_MAJOR,
                ).to(device)
            logger.info("copied conv output from host to device")
        return x
    return run_conv_on_device
