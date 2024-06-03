# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

from models.experimental.functional_fadnetpp.tt.ttnn_resblock import TtResBlock


class ttExtractNet:
    def __init__(self, parameters, resBlock=True) -> None:
        self.resBlock = resBlock
        self.conv1a = parameters.conv1a
        self.conv1b = parameters.conv1b
        if self.resBlock:
            self.conv2 = TtResBlock(parameters.conv2, 32, 64, stride=2)
            self.conv3 = TtResBlock(parameters.conv3, 64, 128, stride=2)
        else:
            self.conv2 = parameters.conv2
            self.conv3 = parameters.conv3

    def __call__(self, device, input_tensor):
        imgs = torch.chunk(input_tensor, 2, dim=3)
        img_left = imgs[0]
        img_right = imgs[1]
        img_left = img_left.to(device, self.c1.conv.input_sharded_memory_config)
        img_right = img_right.to(device, self.c1.conv.input_sharded_memory_config)

        conv1_l = self.conv1a(img_left)
        conv1_r = self.conv1b(img_right)

        conv2_l = self.conv2(conv1_l)
        conv3_l = self.conv3(conv2_l)

        conv2_r = self.conv2(conv1_r)
        conv3_r = self.conv3(conv2_r)

        return conv1_l, conv2_l, conv3_l, conv3_r
