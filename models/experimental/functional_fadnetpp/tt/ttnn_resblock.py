# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn


class TtResBlock:
    def __init__(self, parameters, n_in, n_out, stride=1) -> None:
        self.module_list = []
        self.conv1 = parameters["resblock_1_conv1"]
        self.conv2 = parameters["resblock_2_conv2"]
        if stride != 1 or n_out != n_in:
            self.sc = True
            self.shortcut = parameters["resblock_sc_conv1"]

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.module_list[0][0].conv.input_sharded_memory_config)
        residual = input_tensor
        if self.sc:
            residual = self.shortcut(input_tensor)
        output_tensor_h = input_tensor
        output_tensor_1 = self.conv1(output_tensor_h)
        output_tensor_h = self.conv2(output_tensor_1)
        output_tensor_h += residual
        output_tensor_h = ttnn.relu(output_tensor_h)
        return ttnn.from_device(output_tensor_h)
