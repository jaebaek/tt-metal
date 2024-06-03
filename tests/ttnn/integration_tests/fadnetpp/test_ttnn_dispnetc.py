# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_fadnetpp.reference.dispnetc import DispNetC
from models.experimental.functional_fadnetpp.reference.extractnet import ExtractNet
from models.experimental.functional_fadnetpp.tt.ttnn_extractnet import ttExtractNet
from models.utility_functions import skip_for_wormhole_b0


import ttnn
from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels <= 256
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["deallocate_activation"] = True
    ttnn_module_args["conv_blocking_and_parallelization_config_override"] = None
    ttnn_module_args["activation"] = "relu"


def create_custom_preprocessor(device, resblock=True):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, DispNetC):
            print(model)
            model = model.extractnet
            ttnn_module_args = ttnn_module_args["extractnet"]

            ttnn_module_args.conv1a["weights_dtype"] = ttnn.bfloat8_b
            conv1a_weight, conv1a_bias = model.conv1a.weight, model.conv1a.bias
            update_ttnn_module_args(ttnn_module_args.conv1a)
            parameters["conv1a"] = {}
            parameters["conv1a"]["weight"] = conv1a_weight
            parameters["conv1a"]["bias"] = conv1a_bias

            print("conv1a")
            ttnn_module_args.conv1b["weights_dtype"] = ttnn.bfloat8_b
            conv1b_weight, conv1b_bias = model.conv1b.weight, model.conv1b.bias
            update_ttnn_module_args(ttnn_module_args.conv1b)
            parameters["conv1b"] = {}
            parameters["conv1b"]["weight"] = conv1b_weight
            parameters["conv1b"]["bias"] = conv1b_bias

            if resblock:
                parameters["conv2"] = {}
                ttnn_module_args["conv2"]["resblock_1_conv1"] = ttnn_module_args["conv2"]["resblock_1_conv1"]
                ttnn_module_args["conv2"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                weight1, bias1 = fold_batch_norm2d_into_conv2d(model.conv2.resblock_1_conv1, model.conv2.resblock_1_bn1)
                update_ttnn_module_args(ttnn_module_args["conv2"]["resblock_1_conv1"])
                parameters["conv2"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    weight1, bias1, ttnn_module_args["conv2"]["resblock_1_conv1"], return_parallel_config=True
                )

                ttnn_module_args["conv2"]["resblock_2_conv2"] = ttnn_module_args["conv2"]["resblock_2_conv2"]
                ttnn_module_args["conv2"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                weight2, bias2 = fold_batch_norm2d_into_conv2d(model.resblock_2_conv2, model.resblock_2_bn2)
                update_ttnn_module_args(ttnn_module_args["conv2"]["resblock_2_conv2"])
                parameters["conv2"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    weight2, bias2, ttnn_module_args["conv2"]["resblock_2_conv2"], return_parallel_config=True
                )

                ttnn_module_args["conv2"]["resblock_sc_conv"] = ttnn_module_args["conv2"]["shortcut_c"]
                ttnn_module_args["conv2"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
                weight3, bias3 = fold_batch_norm2d_into_conv2d(model.shortcut_c, model.shortcut_b)
                update_ttnn_module_args(ttnn_module_args["conv2"]["resblock_sc_conv"])
                parameters["conv2"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    weight3, bias3, ttnn_module_args["conv2"]["resblock_sc_conv"], return_parallel_config=True
                )
                print("resblock_sc_conv")
                parameters["conv3"] = {}
                for block in enumerate(model.res2):
                    conv1 = block[0]
                    bn1 = block[1]
                    conv2 = block[3]
                    bn2 = block[4]
                    conv3 = block[5]
                    bn3 = block[6]

                    ttnn_module_args["conv3"]["resblock_1_conv1"] = ttnn_module_args["conv3"]
                    ttnn_module_args["conv3"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                    weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                    update_ttnn_module_args(ttnn_module_args["conv3"]["resblock_1_conv1"])
                    parameters["conv3"]["resblock_1_conv1"], _ = preprocess_conv2d(
                        weight1, bias1, ttnn_module_args["conv3"]["resblock_1_conv1"], return_parallel_config=True
                    )

                    ttnn_module_args["conv3"]["resblock_2_conv2"] = ttnn_module_args["conv3"]
                    ttnn_module_args["conv3"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                    weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                    update_ttnn_module_args(ttnn_module_args["conv3"]["resblock_2_conv2"])
                    parameters["conv3"]["resblock_2_conv2"], _ = preprocess_conv2d(
                        weight2, bias2, ttnn_module_args["conv3"]["resblock_2_conv2"], return_parallel_config=True
                    )

                    ttnn_module_args["conv3"]["resblock_sc_conv"] = ttnn_module_args["conv3"]
                    ttnn_module_args["conv3"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
                    weight3, bias3 = fold_batch_norm2d_into_conv2d(conv3, bn3)
                    update_ttnn_module_args(ttnn_module_args["conv3"]["resblock_sc_conv"])
                    parameters["conv3"]["resblock_sc_conv"], _ = preprocess_conv2d(
                        weight3, bias3, ttnn_module_args["conv3"]["resblock_sc_conv"], return_parallel_config=True
                    )

            else:
                ttnn_module_args.conv2["weights_dtype"] = ttnn.bfloat8_b
                conv1_weight, conv1_bias = model.conv2, model.b1
                update_ttnn_module_args(ttnn_module_args.conv2)
                parameters["conv1"], conv2_parallel_config = preprocess_conv2d(
                    conv1_weight, conv1_bias, ttnn_module_args.conv2, return_parallel_config=True
                )

                ttnn_module_args.conv3["weights_dtype"] = ttnn.bfloat8_b
                conv1_weight, conv1_bias = model.conv3, model.b1
                update_ttnn_module_args(ttnn_module_args.conv3)
                parameters["conv1"], conv3_parallel_config = preprocess_conv2d(
                    conv1_weight, conv1_bias, ttnn_module_args.conv3, return_parallel_config=True
                )

            if resblock:
                parameters["conv_redir"] = {}
                for block in enumerate(model.res.module_list):
                    conv1 = block[0]
                    bn1 = block[1]
                    conv2 = block[3]
                    bn2 = block[4]
                    conv3 = block[5]
                    bn3 = block[6]

                    ttnn_module_args["conv_redir"]["resblock_1_conv1"] = ttnn_module_args["conv_redir"]
                    ttnn_module_args["conv_redir"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                    weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                    update_ttnn_module_args(ttnn_module_args["conv_redir"]["resblock_1_conv1"])
                    parameters["conv_redir"]["resblock_1_conv1"], _ = preprocess_conv2d(
                        weight1, bias1, ttnn_module_args["conv_redir"]["resblock_1_conv1"], return_parallel_config=True
                    )

                    ttnn_module_args["conv_redir"]["resblock_2_conv2"] = ttnn_module_args["conv_redir"]
                    ttnn_module_args["conv_redir"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                    weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                    update_ttnn_module_args(ttnn_module_args["conv_redir"]["resblock_2_conv2"])
                    parameters["conv_redir"]["resblock_2_conv2"], _ = preprocess_conv2d(
                        weight2, bias2, ttnn_module_args["conv_redir"]["resblock_2_conv2"], return_parallel_config=True
                    )

                    ttnn_module_args["conv_redir"]["resblock_sc_conv"] = ttnn_module_args["conv_redir"]
                    ttnn_module_args["conv_redir"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
                    weight3, bias3 = fold_batch_norm2d_into_conv2d(conv3, bn3)
                    update_ttnn_module_args(ttnn_module_args["conv_redir"]["resblock_sc_conv"])
                    parameters["conv_redir"]["resblock_sc_conv"], _ = preprocess_conv2d(
                        weight3, bias3, ttnn_module_args["conv_redir"]["resblock_sc_conv"], return_parallel_config=True
                    )

                ttnn_module_args.conv_dy["weights_dtype"] = ttnn.bfloat8_b
                ttnn_module_args.conv_dy["activation"] = "relu"
                conv_dy_weight, conv_dy_bias = fold_batch_norm2d_into_conv2d(model.conv_dy, model.bn_dy)
                update_ttnn_module_args(ttnn_module_args.conv_dy)
                parameters["conv_dy"], conv_dy_parallel_config = preprocess_conv2d(
                    conv_dy_weight, conv_dy_bias, ttnn_module_args.conv_dy, return_parallel_config=True
                )

                ttnn_module_args.conv_d2["weights_dtype"] = ttnn.bfloat8_b
                ttnn_module_args.conv_d2["activation"] = "relu"
                conv_d2_weight, conv_d2_bias = fold_batch_norm2d_into_conv2d(model.conv_d2, model.bn_d2)
                update_ttnn_module_args(ttnn_module_args.conv_d2)
                parameters["conv_d2"], conv_d2_parallel_config = preprocess_conv2d(
                    conv_d2_weight, conv_d2_bias, ttnn_module_args.conv_d2, return_parallel_config=True
                )

                ttnn_module_args.conv_dy_1["weights_dtype"] = ttnn.bfloat8_b
                ttnn_module_args.conv_dy_1["activation"] = "relu"
                conv_dy_1_weight, conv_dy_1_bias = fold_batch_norm2d_into_conv2d(model.conv_dy_1, model.bn_dy_1)
                update_ttnn_module_args(ttnn_module_args.conv_dy_1)
                parameters["conv_dy_1"], conv_dy_1_parallel_config = preprocess_conv2d(
                    conv_dy_1_weight, conv_dy_1_bias, ttnn_module_args.conv_dy_1, return_parallel_config=True
                )

                parameters["conv4"] = {}
                for block in enumerate(model.res.module_list):
                    conv1 = block[0]
                    bn1 = block[1]
                    conv2 = block[3]
                    bn2 = block[4]
                    conv3 = block[5]
                    bn3 = block[6]

                    ttnn_module_args["conv4"]["resblock_1_conv1"] = ttnn_module_args["conv4"]
                    ttnn_module_args["conv4"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                    weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                    update_ttnn_module_args(ttnn_module_args["conv4"]["resblock_1_conv1"])
                    parameters["conv4"]["resblock_1_conv1"], _ = preprocess_conv2d(
                        weight1, bias1, ttnn_module_args["conv4"]["resblock_1_conv1"], return_parallel_config=True
                    )

                    ttnn_module_args["conv4"]["resblock_2_conv2"] = ttnn_module_args["conv4"]
                    ttnn_module_args["conv4"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                    weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                    update_ttnn_module_args(ttnn_module_args["conv4"]["resblock_2_conv2"])
                    parameters["conv4"]["resblock_2_conv2"], _ = preprocess_conv2d(
                        weight2, bias2, ttnn_module_args["conv4"]["resblock_2_conv2"], return_parallel_config=True
                    )

                    ttnn_module_args["conv4"]["resblock_sc_conv"] = ttnn_module_args["conv4"]
                    ttnn_module_args["conv4"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
                    weight3, bias3 = fold_batch_norm2d_into_conv2d(conv3, bn3)
                    update_ttnn_module_args(ttnn_module_args["conv4"]["resblock_sc_conv"])
                    parameters["conv4"]["resblock_sc_conv"], _ = preprocess_conv2d(
                        weight3, bias3, ttnn_module_args["conv4"]["resblock_sc_conv"], return_parallel_config=True
                    )

                parameters["conv4_1"] = {}
                for block in enumerate(model.res.module_list):
                    conv1 = block[0]
                    bn1 = block[1]
                    conv2 = block[3]
                    bn2 = block[4]

                    ttnn_module_args["conv4_1"]["resblock_1_conv1"] = ttnn_module_args["conv4_1"]
                    ttnn_module_args["conv4_1"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                    weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                    update_ttnn_module_args(ttnn_module_args["conv4_1"]["resblock_1_conv1"])
                    parameters["conv4_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                        weight1, bias1, ttnn_module_args["conv4_1"]["resblock_1_conv1"], return_parallel_config=True
                    )

                    ttnn_module_args["conv4_1"]["resblock_2_conv2"] = ttnn_module_args["conv4_1"]
                    ttnn_module_args["conv4_1"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                    weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                    update_ttnn_module_args(ttnn_module_args["conv4_1"]["resblock_2_conv2"])
                    parameters["conv4_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                        weight2, bias2, ttnn_module_args["conv4_1"]["resblock_2_conv2"], return_parallel_config=True
                    )

                parameters["conv5"] = {}
                for block in enumerate(model.res.module_list):
                    conv1 = block[0]
                    bn1 = block[1]
                    conv2 = block[3]
                    bn2 = block[4]
                    conv3 = block[5]
                    bn3 = block[6]

                    ttnn_module_args["conv5"]["resblock_1_conv1"] = ttnn_module_args["conv5"]
                    ttnn_module_args["conv5"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                    weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                    update_ttnn_module_args(ttnn_module_args["conv5"]["resblock_1_conv1"])
                    parameters["conv5"]["resblock_1_conv1"], _ = preprocess_conv2d(
                        weight1, bias1, ttnn_module_args["conv5"]["resblock_1_conv1"], return_parallel_config=True
                    )

                    ttnn_module_args["conv5"]["resblock_2_conv2"] = ttnn_module_args["conv5"]
                    ttnn_module_args["conv5"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                    weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                    update_ttnn_module_args(ttnn_module_args["conv5"]["resblock_2_conv2"])
                    parameters["conv5"]["resblock_2_conv2"], _ = preprocess_conv2d(
                        weight2, bias2, ttnn_module_args["conv5"]["resblock_2_conv2"], return_parallel_config=True
                    )

                    ttnn_module_args["conv5"]["resblock_sc_conv"] = ttnn_module_args["conv5"]
                    ttnn_module_args["conv5"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
                    weight3, bias3 = fold_batch_norm2d_into_conv2d(conv3, bn3)
                    update_ttnn_module_args(ttnn_module_args["conv5"]["resblock_sc_conv"])
                    parameters["conv5"]["resblock_sc_conv"], _ = preprocess_conv2d(
                        weight3, bias3, ttnn_module_args["conv5"]["resblock_sc_conv"], return_parallel_config=True
                    )

                parameters["conv5_1"] = {}
                for block in enumerate(model.res.module_list):
                    conv1 = block[0]
                    bn1 = block[1]
                    conv2 = block[3]
                    bn2 = block[4]

                    ttnn_module_args["conv5_1"]["resblock_1_conv1"] = ttnn_module_args["conv5_1"]
                    ttnn_module_args["conv5_1"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                    weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                    update_ttnn_module_args(ttnn_module_args["conv5_1"]["resblock_1_conv1"])
                    parameters["conv5_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                        weight1, bias1, ttnn_module_args["conv5_1"]["resblock_1_conv1"], return_parallel_config=True
                    )

                    ttnn_module_args["conv5_1"]["resblock_2_conv2"] = ttnn_module_args["conv5_1"]
                    ttnn_module_args["conv5_1"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                    weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                    update_ttnn_module_args(ttnn_module_args["conv5_1"]["resblock_2_conv2"])
                    parameters["conv5_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                        weight2, bias2, ttnn_module_args["conv5_1"]["resblock_2_conv2"], return_parallel_config=True
                    )

                parameters["conv6"] = {}
                for block in enumerate(model.res.module_list):
                    conv1 = block[0]
                    bn1 = block[1]
                    conv2 = block[3]
                    bn2 = block[4]
                    conv3 = block[5]
                    bn3 = block[6]

                    ttnn_module_args["conv6"]["resblock_1_conv1"] = ttnn_module_args["conv6"]
                    ttnn_module_args["conv6"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                    weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                    update_ttnn_module_args(ttnn_module_args["conv6"]["resblock_1_conv1"])
                    parameters["conv6"]["resblock_1_conv1"], _ = preprocess_conv2d(
                        weight1, bias1, ttnn_module_args["conv6"]["resblock_1_conv1"], return_parallel_config=True
                    )

                    ttnn_module_args["conv6"]["resblock_2_conv2"] = ttnn_module_args["conv6"]
                    ttnn_module_args["conv6"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                    weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                    update_ttnn_module_args(ttnn_module_args["conv6"]["resblock_2_conv2"])
                    parameters["conv6"]["resblock_2_conv2"], _ = preprocess_conv2d(
                        weight2, bias2, ttnn_module_args["conv6"]["resblock_2_conv2"], return_parallel_config=True
                    )

                    ttnn_module_args["conv6"]["resblock_sc_conv"] = ttnn_module_args["conv6"]
                    ttnn_module_args["conv6"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
                    weight3, bias3 = fold_batch_norm2d_into_conv2d(conv3, bn3)
                    update_ttnn_module_args(ttnn_module_args["conv6"]["resblock_sc_conv"])
                    parameters["conv6"]["resblock_sc_conv"], _ = preprocess_conv2d(
                        weight3, bias3, ttnn_module_args["conv6"]["resblock_sc_conv"], return_parallel_config=True
                    )

                parameters["conv6_1"] = {}
                for block in enumerate(model.res.module_list):
                    conv1 = block[0]
                    bn1 = block[1]
                    conv2 = block[3]
                    bn2 = block[4]

                    ttnn_module_args["conv6_1"]["resblock_1_conv1"] = ttnn_module_args["conv6_1"]
                    ttnn_module_args["conv6_1"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                    weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                    update_ttnn_module_args(ttnn_module_args["conv6_1"]["resblock_1_conv1"])
                    parameters["conv6_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                        weight1, bias1, ttnn_module_args["conv6_1"]["resblock_1_conv1"], return_parallel_config=True
                    )

                    ttnn_module_args["conv6_1"]["resblock_2_conv2"] = ttnn_module_args["conv6_1"]
                    ttnn_module_args["conv6_1"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                    weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                    update_ttnn_module_args(ttnn_module_args["conv6_1"]["resblock_2_conv2"])
                    parameters["conv6_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                        weight2, bias2, ttnn_module_args["conv6_1"]["resblock_2_conv2"], return_parallel_config=True
                    )
            else:
                ttnn_module_args.conv_redir["weights_dtype"] = ttnn.bfloat8_b
                ttnn_module_args.conv_redir["activation"] = "relu"
                conv_redir_weight, conv_redir_bias = model.conv_redir, model.bn_redir
                update_ttnn_module_args(ttnn_module_args.conv_redir)
                parameters["conv_redir"], conv_redir_parallel_config = preprocess_conv2d(
                    conv_redir_weight, conv_redir_bias, ttnn_module_args.conv_redir, return_parallel_config=True
                )

                ttnn_module_args.conv3_1["weights_dtype"] = ttnn.bfloat8_b
                ttnn_module_args.conv3_1["activation"] = "relu"
                conv3_1_weight, conv3_1_bias = model.conv3_1, model.bn_3_1
                update_ttnn_module_args(ttnn_module_args.conv3_1)
                parameters["conv3_1"], conv3_1_parallel_config = preprocess_conv2d(
                    conv3_1_weight, conv3_1_bias, ttnn_module_args.conv3_1, return_parallel_config=True
                )

                ttnn_module_args.conv_4["weights_dtype"] = ttnn.bfloat8_b
                ttnn_module_args.conv_4["activation"] = "relu"
                conv_4_weight, conv_4_bias = model.conv_4, model.bn_4
                update_ttnn_module_args(ttnn_module_args.conv_4)
                parameters["conv_4"], conv_4_parallel_config = preprocess_conv2d(
                    conv_4_weight, conv_4_bias, ttnn_module_args.conv_4, return_parallel_config=True
                )

                ttnn_module_args.conv_4_1["weights_dtype"] = ttnn.bfloat8_b
                ttnn_module_args.conv_4_1["activation"] = "relu"
                conv_4_1_weight, conv_4_1_bias = model.conv_4_1, model.bn_4_1
                update_ttnn_module_args(ttnn_module_args.conv_4_1)
                parameters["conv_4_1"], conv_4_1_parallel_config = preprocess_conv2d(
                    conv_4_1_weight, conv_4_1_bias, ttnn_module_args.conv_4_1, return_parallel_config=True
                )

                ttnn_module_args.conv_5["weights_dtype"] = ttnn.bfloat8_b
                ttnn_module_args.conv_5["activation"] = "relu"
                conv_5_weight, conv_5_bias = model.conv_5, model.bn_5
                update_ttnn_module_args(ttnn_module_args.conv_5)
                parameters["conv_5"], conv_5_parallel_config = preprocess_conv2d(
                    conv_5_weight, conv_5_bias, ttnn_module_args.conv_5, return_parallel_config=True
                )

                ttnn_module_args.conv_5_1["weights_dtype"] = ttnn.bfloat8_b
                ttnn_module_args.conv_5_1["activation"] = "relu"
                conv_5_1_weight, conv_5_1_bias = model.conv_5_1, model.bn_5_1
                update_ttnn_module_args(ttnn_module_args.conv_5_1)
                parameters["conv_5_1"], conv_5_1_parallel_config = preprocess_conv2d(
                    conv_5_1_weight, conv_5_1_bias, ttnn_module_args.conv_5_1, return_parallel_config=True
                )

                ttnn_module_args.conv_6["weights_dtype"] = ttnn.bfloat8_b
                ttnn_module_args.conv_6["activation"] = "relu"
                conv_6_weight, conv_6_bias = model.conv_6, model.bn_6
                update_ttnn_module_args(ttnn_module_args.conv_6)
                parameters["conv_6"], conv_6_parallel_config = preprocess_conv2d(
                    conv_6_weight, conv_6_bias, ttnn_module_args.conv_6, return_parallel_config=True
                )

                ttnn_module_args.conv_6_1["weights_dtype"] = ttnn.bfloat8_b
                ttnn_module_args.conv_6_1["activation"] = "relu"
                conv_6_1_weight, conv_6_1_bias = model.conv_6_1, model.bn_6_1
                update_ttnn_module_args(ttnn_module_args.conv_6_1)
                parameters["conv_6_1"], conv_6_1_parallel_config = preprocess_conv2d(
                    conv_6_1_weight, conv_6_1_bias, ttnn_module_args.conv_6_1, return_parallel_config=True
                )

        ttnn_module_args.pred_flow6["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.pred_flow6["activation"] = "relu"
        pred_flow6_weight, pred_flow6_bias = model.conv_pf_6, model.bn_pf_6
        update_ttnn_module_args(ttnn_module_args.pred_flow6)
        parameters["pred_flow6"], pred_flow6_parallel_config = preprocess_conv2d(
            pred_flow6_weight, pred_flow6_bias, ttnn_module_args.pred_flow6, return_parallel_config=True
        )

        ttnn_module_args.pred_flow5["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.pred_flow5["activation"] = "relu"
        pred_flow5_weight, pred_flow5_bias = model.conv_pf_5, model.bn_pf_5
        update_ttnn_module_args(ttnn_module_args.pred_flow5)
        parameters["pred_flow5"], pred_flow5_parallel_config = preprocess_conv2d(
            pred_flow5_weight, pred_flow5_bias, ttnn_module_args.pred_flow5, return_parallel_config=True
        )

        ttnn_module_args.pred_flow4["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.pred_flow4["activation"] = "relu"
        pred_flow4_weight, pred_flow4_bias = model.conv_pf_4, model.bn_pf_4
        update_ttnn_module_args(ttnn_module_args.pred_flow4)
        parameters["pred_flow4"], pred_flow4_parallel_config = preprocess_conv2d(
            pred_flow4_weight, pred_flow4_bias, ttnn_module_args.pred_flow4, return_parallel_config=True
        )

        ttnn_module_args.pred_flow3["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.pred_flow3["activation"] = "relu"
        pred_flow3_weight, pred_flow3_bias = model.conv_pf_3, model.bn_pf_3
        update_ttnn_module_args(ttnn_module_args.pred_flow3)
        parameters["pred_flow3"], pred_flow3_parallel_config = preprocess_conv2d(
            pred_flow3_weight, pred_flow3_bias, ttnn_module_args.pred_flow3, return_parallel_config=True
        )

        ttnn_module_args.pred_flow2["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.pred_flow2["activation"] = "relu"
        pred_flow2_weight, pred_flow2_bias = model.conv_pf_2, model.bn_pf_2
        update_ttnn_module_args(ttnn_module_args.pred_flow2)
        parameters["pred_flow2"], pred_flow2_parallel_config = preprocess_conv2d(
            pred_flow2_weight, pred_flow2_bias, ttnn_module_args.pred_flow2, return_parallel_config=True
        )

        ttnn_module_args.pred_flow1["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.pred_flow1["activation"] = "relu"
        pred_flow1_weight, pred_flow1_bias = model.conv_pf_1, model.bn_pf_1
        update_ttnn_module_args(ttnn_module_args.pred_flow1)
        parameters["pred_flow1"], pred_flow1_parallel_config = preprocess_conv2d(
            pred_flow1_weight, pred_flow1_bias, ttnn_module_args.pred_flow1, return_parallel_config=True
        )

        ttnn_module_args.pred_flow0["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.pred_flow0["activation"] = "relu"
        pred_flow0_weight, pred_flow0_bias = model.conv_pf_0, model.bn_pf_0
        update_ttnn_module_args(ttnn_module_args.pred_flow0)
        parameters["pred_flow0"], pred_flow0_parallel_config = preprocess_conv2d(
            pred_flow0_weight, pred_flow0_bias, ttnn_module_args.pred_flow0, return_parallel_config=True
        )

        upconv5_weight, upconv5_bias = model.upconv5_c, model.upconv5_b
        parameters["upconv5"] = {}
        parameters["upconv5"]["weight"] = upconv5_weight
        parameters["upconv5"]["bias"] = upconv5_bias

        upconv4_weight, upconv4_bias = model.upconv4_c, model.upconv4_b
        parameters["upconv4"] = {}
        parameters["upconv4"]["weight"] = upconv4_weight
        parameters["upconv4"]["bias"] = upconv4_bias

        upconv3_weight, upconv3_bias = model.upconv3_c, model.upconv3_b
        parameters["upconv3"] = {}
        parameters["upconv3"]["weight"] = upconv3_weight
        parameters["upconv3"]["bias"] = upconv3_bias

        upconv2_weight, upconv2_bias = model.upconv2_c, model.upconv2_b
        parameters["upconv2"] = {}
        parameters["upconv2"]["weight"] = upconv2_weight
        parameters["upconv2"]["bias"] = upconv2_bias

        upconv1_weight, upconv1_bias = model.upconv1_c, model.upconv1_b
        parameters["upconv1"] = {}
        parameters["upconv1"]["weight"] = upconv1_weight
        parameters["upconv1"]["bias"] = upconv1_bias

        upconv0_weight, upconv0_bias = model.upconv0_c, model.upconv0_b
        parameters["upconv0"] = {}
        parameters["upconv0"]["weight"] = upconv0_weight
        parameters["upconv0"]["bias"] = upconv0_bias

        upflow6to5_weight, upflow6to5_bias = model.upflow6to5_c, model.upflow6to5_b
        parameters["upflow6to5"] = {}
        parameters["upflow6to5"]["weight"] = upflow6to5_weight
        parameters["upflow6to5"]["bias"] = upflow6to5_bias

        upflow5to4_weight, upflow5to4_bias = model.upflow5to4_c, model.upflow5to4_b
        parameters["upflow5to4"] = {}
        parameters["upflow5to4"]["weight"] = upflow5to4_weight
        parameters["upflow5to4"]["bias"] = upflow5to4_bias

        upflow4to3_weight, upflow4to3_bias = model.upflow4to3_c, model.upflow4to3_b
        parameters["upflow4to3"] = {}
        parameters["upflow4to3"]["weight"] = upflow4to3_weight
        parameters["upflow4to3"]["bias"] = upflow4to3_bias

        upflow3to2_weight, upflow3to2_bias = model.upflow3to2_c, model.upflow3to2_b
        parameters["upflow3to2"] = {}
        parameters["upflow3to2"]["weight"] = upflow3to2_weight
        parameters["upflow3to2"]["bias"] = upflow3to2_bias

        upflow2to1_weight, upflow2to1_bias = model.upflow2to1_c, model.upflow2to1_b
        parameters["upflow2to1"] = {}
        parameters["upflow2to1"]["weight"] = upflow2to1_weight
        parameters["upflow2to1"]["bias"] = upflow2to1_bias

        upflow1to0_weight, upflow1to0_bias = model.upflow1to0_c, model.upflow1to0_b
        parameters["upflow1to0"] = {}
        parameters["upflow1to0"]["weight"] = upflow1to0_weight
        parameters["upflow1to0"]["bias"] = upflow1to0_bias

        return parameters

    return custom_preprocessor


# @pytest.mark.parametrize("device_l1_small_size", [32768], indirect=True)
@skip_for_wormhole_b0()
def test_dispnetc(device, reset_seeds, model_location_generator):
    torch_model = DispNetC()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    print("keys: ", keys)
    state_dict = torch_model
    # state_dict = state_dict["state_dict"]
    ds_state_dict = {k: v for k, v in torch_model.state_dict().items()}

    values = [parameter for name, parameter in ds_state_dict.items()]
    # print("values: ", values)
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 3, 960, 576)  # Batch size of 1, 64 input channels, 160x160 height and width
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )
    ttnn_model = ttExtractNet(parameters)

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
    output_tensor = output_tensor.reshape(1, 80, 80, 128)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
