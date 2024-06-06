# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_segformer.tt.ttnn_segformer_dwconv import ttnn_SegformerDWConv


class ttnn_SegformerMixFFN:
    def __init__(self, parameters, model):
        super().__init__()
        self.dwconv = ttnn_SegformerDWConv(parameters=parameters, model=model.dwconv)

    def __call__(self, hidden_states, height, width, parameters, device):
        hidden_states = ttnn.linear(
            hidden_states,
            parameters.dense1.weight,
            bias=parameters.dense1.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=12),
        )
        hidden_states = self.dwconv(hidden_states, height, width, device)
        hidden_states = ttnn.gelu(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.linear(
            hidden_states,
            parameters.dense2.weight,
            bias=parameters.dense2.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=12),
        )
        return hidden_states
