# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

import ttnn
import tt_lib
from tt_lib import fallback_ops
from ttnn.model_preprocessing import preprocess_model

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    torch_to_tt_tensor,
    torch2tt_tensor,
    tt_to_torch_tensor,
)


class TtHead:
    def output_preprocessing(self, output_tensor, device):
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = torch_to_tt_tensor_rm(output_tensor, device, put_on_device=True)
        return output_tensor

    def __init__(self, device, parameters) -> None:
        self.device = device
        print("keys in parameters in TtHead are: ", parameters.keys())
        self.c1 = parameters.c1
        self.c2 = fallback_ops.Conv2d(parameters.c2["weight"], parameters.c2["bias"], 256, 255, 1, 1, 0, bias=True)
        self.c3 = parameters.c3
        self.c4 = parameters.c4
        self.c5 = parameters.c5
        self.c6 = parameters.c6
        self.c7 = parameters.c7
        self.c8 = parameters.c8
        self.c9 = parameters.c9
        self.c10 = fallback_ops.Conv2d(parameters.c10["weight"], parameters.c10["bias"], 512, 255, 1, 1, 0, bias=True)
        self.c11 = parameters.c11
        self.c12 = parameters.c12
        self.c13 = parameters.c13
        self.c14 = parameters.c14
        self.c15 = parameters.c15
        self.c16 = parameters.c16
        self.c17 = parameters.c17
        self.c18 = fallback_ops.Conv2d(parameters.c18["weight"], parameters.c18["bias"], 1024, 255, 1, 1, 0, bias=True)

    def __call__(self, device, input_tensors):
        input_tensor = input_tensors[0].to(device, self.c1.conv.input_sharded_memory_config)
        output_tensor = self.c1(input_tensor)
        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c2(output_tensor)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))

        output_tensor_res1 = output_tensor

        input_tensor = input_tensors[0].to(device, self.c3.conv.input_sharded_memory_config)
        output_tensor = self.c3(input_tensor)

        outNeck1 = input_tensors[2].to(device)
        outNeck1 = ttnn.to_layout(outNeck1, ttnn.ROW_MAJOR_LAYOUT)

        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.concat([output_tensor, outNeck1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = self.c4(output_tensor)
        output_tensor = self.c5(output_tensor)
        output_tensor = self.c6(output_tensor)
        output_tensor = self.c7(output_tensor)
        output_tensor = self.c8(output_tensor)
        output_tensor2 = output_tensor

        output_tensor = self.c9(output_tensor)
        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c10(output_tensor)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor_res2 = output_tensor

        output_tensor = self.c11(output_tensor2)

        outNeck2 = input_tensors[1].to(device)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.concat([output_tensor, outNeck2], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.c12(output_tensor)
        output_tensor = self.c13(output_tensor)
        output_tensor = self.c14(output_tensor)
        output_tensor = self.c15(output_tensor)
        output_tensor = self.c16(output_tensor)
        output_tensor = self.c17(output_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c18(output_tensor)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))

        output_tensor_res3 = output_tensor

        return (
            ttnn.from_device(output_tensor_res1),
            ttnn.from_device(output_tensor_res2),
            ttnn.from_device(output_tensor_res3),
        )
