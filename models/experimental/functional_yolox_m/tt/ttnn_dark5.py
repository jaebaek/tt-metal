# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from math import sqrt
import torch
import torch.nn as nn
import ttnn
import tt_lib
import tt_lib.fallback_ops
from models.experimental.functional_yolox_m.tt.ttnn_bottleneck_block import TtBottleneckBlock
from models.utility_functions import torch_to_tt_tensor_rm


class TtDark5:
    def output_preprocessing(self, output_tensor, device):
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        # output_tensor = torch_to_tt_tensor_rm(output_tensor, device, put_on_device=True)
        return output_tensor

    def __init__(
        self,
        device,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c2 = parameters.c2
        self.c3 = parameters.c3
        self.c4 = parameters.c4
        self.c4 = tt_lib.fallback_ops.Conv2d(
            parameters.c4["weight"], parameters.c4["bias"], 768, 384, 1, 1, 0, bias=True
        )
        # self.c5 = parameters.c5
        self.c5 = tt_lib.fallback_ops.Conv2d(
            parameters.c5["weight"], parameters.c5["bias"], 768, 384, 1, 1, 0, bias=True
        )
        self.bblock = TtBottleneckBlock(parameters.bblock, 2, False)
        self.c6 = parameters.c6

        self.max_pool_reader_patterns_cache = {}
        max_pool_parallel_config_override = {}

        max_pool_parallel_config_override["grid_size"] = self.c2.conv.grid_size
        max_pool_parallel_config_override["num_cores_nhw"] = self.c2.conv.sliding_window_op_params.num_cores_nhw

        # self.p1 = ttnn.MaxPool2d(
        #     kernel_size=(5, 5),
        #     stride=(1, 1),
        #     padding=(2, 2),
        #     dilation=(1, 1),
        #     dtype=ttnn.bfloat16,
        #     device=device,
        #     batch_size=1,
        #     input_height=10,
        #     input_width=10,
        #     reader_patterns_cache=self.max_pool_reader_patterns_cache,
        #     deallocate_activation=False,
        #     # parallel_config_override=max_pool_parallel_config_override,
        #     channels=384,
        # )

        # self.p2 = ttnn.MaxPool2d(
        #     kernel_size=(9, 9),
        #     stride=(1, 1),
        #     padding=(4, 4),
        #     dilation=(1, 1),
        #     dtype=ttnn.bfloat16,
        #     device=device,
        #     batch_size=1,
        #     input_height=10,
        #     input_width=10,
        #     reader_patterns_cache=self.max_pool_reader_patterns_cache,
        #     deallocate_activation=False,
        #     # parallel_config_override=max_pool_parallel_config_override,
        #     channels=384,
        # )
        # self.p3 = ttnn.MaxPool2d(
        #     kernel_size=(13, 13),
        #     stride=(1, 1),
        #     padding=(6, 6),
        #     dilation=(1, 1),
        #     dtype=ttnn.bfloat16,
        #     device=device,
        #     batch_size=1,
        #     input_height=10,
        #     input_width=10,
        #     reader_patterns_cache=self.max_pool_reader_patterns_cache,
        #     deallocate_activation=False,
        #     # parallel_config_override=max_pool_parallel_config_override,
        #     channels=384,
        # )

        self.p1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.p2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
        self.p3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.c1.conv.input_sharded_memory_config)

        output_tensor = self.c1(input_tensor)
        output_tensor = ttnn.silu(output_tensor)
        # output_tensor_c1 = output_tensor
        output_tensor = self.c2(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        # return ttnn.from_device(output_tensor)

        output_tensor_c2 = output_tensor
        # output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        # output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        # tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c2.conv.output_sharded_memory_config)
        # output_tensor = output_tensor.to(device, self.c2.conv.output_sharded_memory_config)
        # print(output_tensor.memory_config())
        output_tensor_c2 = ttnn.from_device(output_tensor_c2)
        output_tensor_c2 = ttnn.to_torch(output_tensor_c2)
        output_tensor_c2 = torch.reshape(output_tensor_c2, (1, 20, 20, 384))
        output_tensor_c2 = torch.permute(output_tensor_c2, (0, 3, 1, 2))

        # print(output_tensor.memory_config())
        output_tensor_p1 = self.p1(output_tensor_c2)
        output_tensor_p2 = self.p2(output_tensor_c2)
        output_tensor_p3 = self.p3(output_tensor_c2)

        output_tensor_p1 = torch.reshape(output_tensor_p1, (1, 384, 1, 400))
        output_tensor_p1 = torch.permute(output_tensor_p1, (0, 2, 3, 1))
        output_tensor_p1 = ttnn.from_torch(output_tensor_p1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        output_tensor_p1 = output_tensor_p1.to(device)

        output_tensor_p2 = torch.reshape(output_tensor_p2, (1, 384, 1, 400))
        output_tensor_p2 = torch.permute(output_tensor_p2, (0, 2, 3, 1))
        output_tensor_p2 = ttnn.from_torch(output_tensor_p2, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        output_tensor_p2 = output_tensor_p2.to(device)

        output_tensor_p3 = torch.reshape(output_tensor_p3, (1, 384, 1, 400))
        output_tensor_p3 = torch.permute(output_tensor_p3, (0, 2, 3, 1))
        output_tensor_p3 = ttnn.from_torch(output_tensor_p3, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        output_tensor_p3 = output_tensor_p3.to(device)

        output_tensor_c2 = torch.reshape(output_tensor_c2, (1, 384, 1, 400))
        output_tensor_c2 = torch.permute(output_tensor_c2, (0, 2, 3, 1))
        output_tensor_c2 = ttnn.from_torch(output_tensor_c2, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        output_tensor_c2 = output_tensor_c2.to(device)

        output_tensor = ttnn.concat([output_tensor_c2] + [output_tensor_p1, output_tensor_p2, output_tensor_p3], dim=3)
        output_tensor = self.c3(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        ttnn.dump_tensor("tests/ttnn/integration_tests/yolox_m/dark5_conv4_inp_ttnn.pt", output_tensor)
        output_tensor_c3 = output_tensor

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c4(output_tensor)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.silu(output_tensor)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(
            output_tensor, self.bblock.module_list[0][0].conv.input_sharded_memory_config
        )
        output_tensor_c4 = output_tensor

        output_tensor_c3 = self.output_preprocessing(output_tensor_c3, device)
        output_tensor = self.c5(output_tensor_c3)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.silu(output_tensor)
        output_tensor_c5 = output_tensor

        output_tensor = self.bblock(device, output_tensor_c4)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor_c5 = ttnn.to_torch(output_tensor_c5)
        output_tensor = torch.cat([output_tensor, output_tensor_c5], dim=3)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        # output_tensor = output_tensor.to(device)
        # output_tensor = ttnn.concat([output_tensor, output_tensor_c5], dim=3)
        output_tensor = output_tensor.to(device, self.c6.conv.input_sharded_memory_config)
        output_tensor = self.c6(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        # output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c5.conv.input_sharded_memory_config)
        return ttnn.from_device(output_tensor)
