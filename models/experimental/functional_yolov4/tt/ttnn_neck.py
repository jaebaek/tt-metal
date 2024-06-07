# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn

import ttnn
import tt_lib
from tt_lib import fallback_ops
from ttnn.model_preprocessing import preprocess_model


class TtNeck:
    def __init__(
        self,
        device,
        parameters,
    ) -> None:
        self.device = device
        self.c1 = parameters.c1
        self.c2 = parameters.c2
        self.c3 = parameters.c3
        # print("\n\n\nattributes of parameters.c3: ", parameters.c3.__dict__)
        self.c4 = parameters.c4
        self.c5 = parameters.c5
        self.c6 = parameters.c6
        self.c7 = parameters.c7
        self.c7_2 = parameters.c7_2
        self.c7_3 = parameters.c7_3
        self.c7_4 = parameters.c7_4
        self.c7_5 = parameters.c7_5
        self.c8 = parameters.c8
        self.c8_2 = parameters.c8_2
        self.c9 = parameters.c9
        self.c9_2 = parameters.c9_2
        self.c9_3 = parameters.c9_3
        self.c9_4 = parameters.c9_4
        self.c9_5 = parameters.c9_5
        self.c10 = parameters.c10
        self.c10_2 = parameters.c10_2

        self.max_pool_reader_patterns_cache = {}
        max_pool_parallel_config_override = {}

        max_pool_parallel_config_override["grid_size"] = self.c3.conv.grid_size
        max_pool_parallel_config_override["num_cores_nhw"] = self.c3.conv.sliding_window_op_params.num_cores_nhw
        print(max_pool_parallel_config_override)
        print(max_pool_parallel_config_override["num_cores_nhw"])

        self.p1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.p2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
        self.p3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)

    def __call__(self, device, input_tensors):
        input_tensor0 = input_tensors[0].to(device, self.c1.conv.input_sharded_memory_config)

        output_tensor = self.c1(input_tensor0)
        output_tensor = self.c2(output_tensor)
        output_tensor = self.c3(output_tensor)
        output_tensorc3 = output_tensor

        output_tensorc3 = tt_lib.tensor.sharded_to_interleaved(output_tensorc3, ttnn.L1_MEMORY_CONFIG)
        output_tensorc3 = ttnn.to_layout(output_tensorc3, ttnn.ROW_MAJOR_LAYOUT)

        output_tensorc3 = ttnn.from_device(output_tensorc3)
        output_tensorc3 = ttnn.to_torch(output_tensorc3)
        output_tensorc3 = torch.reshape(output_tensorc3, (1, 10, 10, 512))
        output_tensorc3 = torch.permute(output_tensorc3, (0, 3, 1, 2))

        output_tensor = self.p1(output_tensorc3)
        output_tensorp1 = output_tensor
        output_tensor = self.p2(output_tensorc3)
        output_tensorp2 = output_tensor
        output_tensor = self.p3(output_tensorc3)
        output_tensorp3 = output_tensor

        output_tensorp1 = torch.reshape(output_tensorp1, (1, 512, 1, 100))
        output_tensorp2 = torch.reshape(output_tensorp2, (1, 512, 1, 100))
        output_tensorp3 = torch.reshape(output_tensorp3, (1, 512, 1, 100))
        output_tensorc3 = torch.reshape(output_tensorc3, (1, 512, 1, 100))
        output_tensorp1 = torch.permute(output_tensorp1, (0, 2, 3, 1))
        output_tensorp2 = torch.permute(output_tensorp2, (0, 2, 3, 1))
        output_tensorp3 = torch.permute(output_tensorp3, (0, 2, 3, 1))
        output_tensorc3 = torch.permute(output_tensorc3, (0, 2, 3, 1))

        output_tensorp1 = ttnn.from_torch(
            output_tensorp1,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
        )
        output_tensorp2 = ttnn.from_torch(
            output_tensorp2,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
        )
        output_tensorp3 = ttnn.from_torch(
            output_tensorp3,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
        )
        output_tensorc3 = ttnn.from_torch(
            output_tensorc3,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
        )

        # output_tensorp1 = tt_lib.tensor.sharded_to_interleaved(output_tensorp1, ttnn.L1_MEMORY_CONFIG)
        # output_tensorp1 = ttnn.to_layout(output_tensorp1, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.concat(
            [output_tensorp3, output_tensorp2, output_tensorp1, output_tensorc3],
            dim=3,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = output_tensor.to(device, self.c4.conv.input_sharded_memory_config)
        # print("DEBUG:", output_tensor.memory_config())
        output_tensor = self.c4(output_tensor)
        output_tensor = self.c5(output_tensor)
        output_tensor = self.c6(output_tensor)
        output_tensor_9m = output_tensor
        output_tensor_9m = output_tensor_9m.to(device)
        output_tensor = self.c7(output_tensor)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.upsample(output_tensor, (1, 4, 1), memory_config=output_tensor.memory_config())

        # TODO add ttnn tensor here for testing
        #    input_shape = torch_input_tensor.shape
        #    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        #
        #    input_tensor = input_tensor.reshape(
        #        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
        #    )

        outDownSample4 = input_tensors[1].to(device, self.c7_2.conv.input_sharded_memory_config)
        # CBR block for conc2
        outDownSample4_c7 = self.c7_2(outDownSample4)
        #        outDownSample4_b7 = self.b7(outDownSample4_c7)
        #        outDownSample4_r7 = self.relu(outDownSample4_b7)
        #
        # output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        # output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        # outDownSample4_c7 = tt_lib.tensor.sharded_to_interleaved(outDownSample4_c7, ttnn.L1_MEMORY_CONFIG)
        # outDownSample4_c7 = ttnn.to_layout(outDownSample4_c7, layout=ttnn.TILE_LAYOUT)
        outDownSample4_c7 = ttnn.to_torch(outDownSample4_c7)
        outDownSample4_c7 = ttnn.from_torch(
            outDownSample4_c7,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
        )
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
        )
        output_tensor = ttnn.concat([outDownSample4_c7, output_tensor], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c7_3.conv.input_sharded_memory_config)
        output_tensor = self.c7_3(output_tensor)
        output_tensor = self.c8(output_tensor)
        output_tensor = self.c7_4(output_tensor)
        output_tensor = self.c8_2(output_tensor)
        output_tensor = self.c7_5(output_tensor)
        output_tensor_16m = output_tensor
        output_tensor_16m = output_tensor_16m.to(device)
        print(output_tensor.shape)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = output_tensor.to(device, self.c9.conv.input_sharded_memory_config)

        print(self.c9.conv.input_sharded_memory_config)
        print("Last config:", output_tensor.memory_config())
        output_tensor = self.c9(output_tensor)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.upsample(output_tensor, (1, 4, 1), memory_config=output_tensor.memory_config())
        # output_tensor = self.u(output_tensor)
        #        # CBR block for conc3
        #        # TODO add ttnn random tensor here
        outDownSample3 = input_tensors[2].to(device, self.c9_2.conv.input_sharded_memory_config)
        outDownSample3_c9 = self.c9_2(outDownSample3)
        #        outDownSample3_b9 = self.b9(outDownSample3_c9)
        #        outDownSample3_r9 = self.relu(outDownSample3_b9)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        print(outDownSample3_c9.dtype, output_tensor.dtype)
        outDownSample3_c9 = tt_lib.tensor.sharded_to_interleaved(outDownSample3_c9, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([outDownSample3_c9, output_tensor], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = output_tensor.to(device, self.c9_3.conv.input_sharded_memory_config)
        output_tensor = self.c9_3(output_tensor)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c10.conv.input_sharded_memory_config)
        print("out: ", output_tensor.layout)
        # print("c10: ", self.c10.output_layout)
        output_tensor = self.c10(output_tensor)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c9_4.conv.input_sharded_memory_config)
        output_tensor = self.c9_4(output_tensor)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c10_2.conv.input_sharded_memory_config)
        output_tensor = self.c10_2(output_tensor)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c9_5.conv.input_sharded_memory_config)
        output_tensor = self.c9_5(output_tensor)
        #        #        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        #        #        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        #        #        output_tensor = ttnn.concat([output_tensor, output_tensor_c3], dim=3, memory_config = ttnn.L1_MEMORY_CONFIG)
        #
        #        #        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c8.conv.input_sharded_memory_config)
        #        #        output_tensor = self.c8(output_tensor)
        #
        return ttnn.from_device(output_tensor), ttnn.from_device(output_tensor_9m), ttnn.from_device(output_tensor_16m)
