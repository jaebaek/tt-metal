# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch.nn as nn

from models.experimental.functional_yolox_m.reference.yolo_pafpn import YOLOPAFPN
from models.experimental.functional_yolox_m.reference.yolo_head import YOLOXHead


class YOLOX(nn.Module):

    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        self.backbone = YOLOPAFPN()
        # self.head = YOLOXHead(80)

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        return fpn_outs
        # outputs = self.head(fpn_outs)

        # return outputs
