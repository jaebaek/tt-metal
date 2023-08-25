import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

import tt_lib as ttl
from tt_lib.utils import (
    tilize_to_list,
    tilize,
    untilize,
    _nearest_32,
    _nearest_y,
    convert_weights_2d_matrix,
)
from models.utility_functions import print_diff_argmax, is_close, comp_pcc
from tests.python_api_testing.conv.conv_unit_test_utils import (
    create_conv_act_tensor,
    create_conv_weight_tensor,
    create_conv_bias_tensor,
    create_conv_weight_tensor_special_padding,
)
import torch


@pytest.mark.parametrize("run_conv_with_address_map", (False,))
@pytest.mark.parametrize("untilize_out", (True, False))
@pytest.mark.parametrize("has_bias", (False, True,))
# @pytest.mark.parametrize("has_bias", (True,))
@pytest.mark.parametrize(
    "K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w",
    (
        # # resnet18 convs
        # (64, 3, 224, 224, 7, 7, 2, 2, 3, 3),
        # (64, 64, 56, 56, 3, 3, 1, 1, 1, 1),
        # #K=128 C=64 H=56 W=56 R=3 S=3 U=2 V=2 PH=1 PW=1 dilation=1 groups=1
        # (128, 64, 56, 56, 3, 3, 2, 2, 1, 1),
        # #K=128 C=128 H=28 W=28 R=3 S=3 U=1 V=1 PH=1 PW=1 dilation=1 groups=1
        # (128, 64, 28, 28, 3, 3, 1, 1, 1, 1),
        # #K=128 C=64 H=56 W=56 R=1 S=1 U=2 V=2 PH=0 PW=0 dilation=1 groups=1
        # (128, 64, 56, 56, 1, 1, 2, 2, 0, 0),
        # #K=128 C=128 H=28 W=28 R=3 S=3 U=1 V=1 PH=1 PW=1 dilation=1 groups=1
        # (128, 128, 28, 28, 3, 3, 1, 1, 1, 1),
        # #K=256 C=128 H=28 W=28 R=3 S=3 U=2 V=2 PH=1 PW=1 dilation=1 groups=1
        # (256, 128, 28, 28, 3, 3, 2, 2, 1, 1),
        # #K=256 C=256 H=14 W=14 R=3 S=3 U=1 V=1 PH=1 PW=1 dilation=1 groups=1
        # (256, 256, 14, 14, 3, 3, 1, 1, 1, 1),
        # #K=256 C=128 H=28 W=28 R=1 S=1 U=2 V=2 PH=0 PW=0 dilation=1 groups=1
        # (256, 128, 28, 28, 1, 1, 2, 2, 0, 0),
        # #K=256 C=256 H=14 W=14 R=3 S=3 U=1 V=1 PH=1 PW=1 dilation=1 groups=1
        # (256, 256, 14, 14, 3, 3, 1, 1, 1, 1),
        # #K=512 C=256 H=14 W=14 R=3 S=3 U=2 V=2 PH=1 PW=1 dilation=1 groups=1
        # (512, 256, 14, 14, 3, 3, 2, 2, 1, 1),
        # #K=512 C=512 H=7 W=7 R=3 S=3 U=1 V=1 PH=1 PW=1 dilation=1 groups=1
        # (512, 512, 7, 7, 3, 3, 1, 1, 1, 1),
        # channels = 3 padding
        (32, 3, 5, 5, 1, 1, 1, 1, 0, 0),
        # w/ conv padding
        (32, 32, 5, 5, 1, 1, 1, 1, 1, 1),
        # Hat = 1, Wat = 1, Wbt = 1
        (32, 32, 5, 5, 1, 1, 1, 1, 0, 0),
        # Hat = 2, Wat = 1, Wbt = 1
        (32, 32, 8, 8, 1, 1, 1, 1, 0, 0),
        # Hat = 1, Wat = 2, Wbt = 1
        (32, 64, 5, 5, 1, 1, 1, 1, 0, 0),
        # Hat = 2, Wat = 2, Wbt = 1
        (32, 64, 8, 8, 1, 1, 1, 1, 0, 0),
        # Hat = 1, Wat = 1, Wbt = 2
        (64, 32, 5, 5, 1, 1, 1, 1, 0, 0),
        # Hat = 1, Wat = 2, Wbt = 2
        (64, 64, 5, 5, 1, 1, 1, 1, 0, 0),
        # Hat = 2, Wat = 1, Wbt = 2
        (64, 32, 8, 8, 1, 1, 1, 1, 0, 0),
        # Hat = 2, Wat = 2, Wbt = 2
        (64, 64, 8, 8, 1, 1, 1, 1, 0, 0),
        # Hat = 8, Wat = 8, Wbt = 8
        (8 * 32, 8 * 32, 16, 16, 1, 1, 1, 1, 0, 0),
        # resnet50 first conv
        (64, 3, 224, 224, 7, 7, 2, 2, 3, 3),
        # num blocks weight w = 4, num blocks act h = 4, num blocks act w = 3
        (16 * 32, 32, 24, 24, 3, 3, 1, 1, 0, 0),
    ),
)
def test_run_conv_as_large_matmul(
    use_program_cache,
    run_conv_with_address_map,
    K,
    C,
    H,
    W,
    R,
    S,
    stride_h,
    stride_w,
    pad_h,
    pad_w, untilize_out,
    has_bias,
    device,
):
    if run_conv_with_address_map and has_bias:
        ## bias is only supported without address map
        pytest.skip()

    if has_bias and untilize_out:
        ## bias is only supported without untilize out
        pytest.skip()

    num_iterations = 1
    # if not run_conv_with_address_map:
    #     num_iterations = (
    #         2  # run twice to test op caching flow for conv op (without address map)
    #     )
    for i in range(num_iterations):
        # torch.set_printoptions(threshold=10000)
        torch.manual_seed(0)
        a_activation_shape = [1, C, H, W]
        # A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
        A_pyt = torch.ones(a_activation_shape, dtype=torch.bfloat16).float()
        b_weights_shape = [K, C, R, S]
        # B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()
        B_pyt = torch.ones(b_weights_shape, dtype=torch.bfloat16).float()
        bias_shape = [1, 1, 1, K]
        # bias_pyt = torch.randn(bias_shape, dtype=torch.bfloat16).float()
        bias_pyt = torch.zeros(bias_shape, dtype=torch.bfloat16).float() * 3.
        # bias_pyt = torch.range(start=0, end=(K - 1), dtype=torch.bfloat16).float()

        # Parameters to define block dims
        act_block_h = 4
        act_block_w = (int)((_nearest_32(_nearest_y(C, 16) * S))/32)
        weight_block_h = act_block_w
        weight_block_w = 2
        out_subblock_h = 2
        out_subblock_w = 2

        OH = ((int)((H - R + 2 * pad_h) / stride_h)) + 1
        OW = ((int)((W - S + 2 * pad_w) / stride_w)) + 1
        conv_output_shape = [1, OH, OW, K]

        # Prepare activations
        A_cl_host = create_conv_act_tensor(A_pyt, 1, C, H, W)
        if run_conv_with_address_map:
            A = A_cl_host.to(device, ttl.tensor.MemoryConfig(False))
        else:
            A = A_cl_host.to(device)

        # Prepare weights
        B_tiled_host = create_conv_weight_tensor_special_padding(
            B_pyt, K, C, R, S, weight_block_h, weight_block_w
        )
        if run_conv_with_address_map:
            B_tiled = B_tiled_host.to(device, ttl.tensor.MemoryConfig(False))
        else:
            B_tiled = B_tiled_host.to(device)

        # Bias
        bias_cl_host = create_conv_bias_tensor(bias_pyt, 1, K, pad = 0)
        bias_device = bias_cl_host.to(device)

        if has_bias:
            bias = torch.flatten(bias_pyt)
        else:
            bias = None

        # Calculate conv result with golden result. Run Pytorch conv
        out_golden = torch.nn.functional.conv2d(
            A_pyt, B_pyt, bias=bias, stride=(stride_h, stride_w), padding=(pad_h, pad_w)
        )

        # Run TT metal OP
        if run_conv_with_address_map:
            untilize_out = True
            out = ttl.tensor.conv_with_address_map(
                A,
                B_tiled,
                [R, S, stride_h, stride_w, pad_h, pad_w],
                act_block_h,
                act_block_w,
                weight_block_w,
                out_subblock_h,
                out_subblock_w,
                K,
            )
        else:
            if not has_bias:
                bias_device = None
            out = ttl.tensor.conv_with_fast_reader(
                A,
                B_tiled,
                bias_device,
                [R, S, stride_h, stride_w, pad_h, pad_w],
                act_block_h,
                act_block_w,
                weight_block_w,
                out_subblock_h,
                out_subblock_w,
                K,
                untilize_out,
                has_bias)
        if not untilize_out:
           out_unpadded_shape = [1, 1, OH*OW, K]
           assert out_unpadded_shape == out.shape_without_padding()
           out = ttl.tensor.format_output_tensor(out, out.shape_without_padding(), device, ttl.tensor.Layout.ROW_MAJOR)
           out = out.reshape(conv_output_shape[0], conv_output_shape[1], conv_output_shape[2], conv_output_shape[3])
        print(f'soduhzjlkfhkn;oiqAHWFNAFNDSJKLF ')
        out = out.cpu()
        print(f'soduhzjlkfhkn;oiqAHWFNAFNDSJKLF ')
        assert out.shape() == conv_output_shape
        assert out.layout() == ttl.tensor.Layout.ROW_MAJOR

        # Copy output to host and convert tt tensor to pytorch tensor
        out_result = out.to_torch()
        out_result = torch.transpose(out_result, 2, 3)
        out_result = torch.transpose(out_result, 1, 2)

        torch.set_printoptions(
            precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32
        )

        # print(f'OUT: {out_result}')
        # print(f'GLD: {out_golden}')

        # Compare against golden
        assert out_result.shape == out_golden.shape
        passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
        print("Passing=", passing_pcc)
        print("Output pcc=", output_pcc)
        assert passing_pcc
