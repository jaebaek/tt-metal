# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
from models.experimental.functional_yolox_m.reference.bottleneck_block import BottleNeckBlock


class Dark4(nn.Module):
    def __init__(self):
        super(Dark4, self).__init__()

        # Initial Conv layer
        self.c1 = nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        # CSP Layer components
        self.c2 = nn.Conv2d(384, 192, kernel_size=1, stride=1, bias=False)
        self.b2 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c3 = nn.Conv2d(384, 192, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c4 = nn.Conv2d(384, 384, kernel_size=1, stride=1, bias=False)
        self.b4 = nn.BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.bblock = BottleNeckBlock(192, 6, True)

        # Activation function
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        # Initial Conv layer
        x1 = self.c1(x)
        x1_b = self.b1(x1)
        x1_m = self.silu(x1_b)

        # CSP Layer
        x2 = self.c2(x1_m)
        x2_b = self.b2(x2)
        x2_m = self.silu(x2_b)

        x3 = self.c3(x1_m)
        x3_b = self.b3(x3)
        x3_m = self.silu(x3_b)

        bblock_out = self.bblock(x2_m)

        conc1 = torch.cat((bblock_out, x3_m), dim=1)
        x4 = self.c4(conc1)
        x4_b = self.b4(x4)
        x4_m = self.silu(x4_b)

        return x4_m
