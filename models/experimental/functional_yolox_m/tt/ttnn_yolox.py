# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.functional_yolox_m.tt.ttnn_yolopafpn import TtYOLOPAFPN


class TtYOLOX:
    def __init__(
        self,
        device,
        parameters,
    ) -> None:
        self.backbone = TtYOLOPAFPN(device, parameters["backbone"])
        # self.head = TtYOLOXHead(80)

    def __call__(self, device, input_tensor):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(device, input_tensor)
        return fpn_outs
        # outputs = self.head(fpn_outs)

        # return outputs
