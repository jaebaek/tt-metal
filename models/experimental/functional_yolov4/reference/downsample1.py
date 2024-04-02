# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.c2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.b2 = nn.BatchNorm2d(64)

        self.c3 = nn.Conv2d(64, 64, 1, 1, 0, bias=False)
        self.b3 = nn.BatchNorm2d(64)

        self.c4 = nn.Conv2d(64, 64, 1, 1, 0, bias=False)
        self.b4 = nn.BatchNorm2d(64)

        self.c5 = nn.Conv2d(64, 32, 1, 1, 0, bias=False)
        self.b5 = nn.BatchNorm2d(32)

        self.c6 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.b6 = nn.BatchNorm2d(64)

        self.c7 = nn.Conv2d(64, 64, 1, 1, 0, bias=False)
        self.b7 = nn.BatchNorm2d(64)

        self.c8 = nn.Conv2d(128, 64, 1, 1, 0, bias=False)
        self.b8 = nn.BatchNorm2d(64)

    def forward(self, input: torch.Tensor):
        x1 = self.c1(input)
        x1_b = self.b1(x1)
        x1_m = self.relu(x1_b)

        x2 = self.c2(x1_m)
        x2_b = self.b2(x2)
        x2_m = self.relu(x2_b)

        x3 = self.c3(x2_m)
        x3_b = self.b3(x3)
        x3_m = self.relu(x3_b)

        x4 = self.c4(x3_m)
        x4_b = self.b4(x4)
        x4_m = self.relu(x4_b)

        x5 = self.c5(x4_m)
        x5_b = self.b5(x5)
        x5_m = self.relu(x5_b)

        x6 = self.c6(x5_m)
        x6_b = self.b6(x6)
        x6_m = self.relu(x6_b)
        x6_m = x6_m + x4_m

        x7 = self.c7(x6_m)
        x7_b = self.b7(x7)
        x7_m = self.relu(x7_b)
        x7_m = torch.cat([x7_m, x3_m], dim=1)

        x8 = self.c8(x7_m)
        x8_b = self.b8(x8)
        x8_m = self.relu(x8_b)

        return x8_m
