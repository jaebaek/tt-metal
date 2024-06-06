# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from torch import nn
import math
import tt_lib


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    # input = ttnn.to_layout(input, layout)
    # input = ttnn.to_device(input, device)
    return input


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


class ttnn_segformer_efficent_selfattention:
    def __init__(self, hidden_size, num_attention_heads, parameters, sequence_reduction_ratio, model):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = model.sr

    def transpose_for_scores(self, hidden_states):
        new_shape = tuple(hidden_states.shape)[:-1] + (self.num_attention_heads, self.attention_head_size)
        device = hidden_states.device()
        hidden_states = ttnn.from_device(hidden_states)
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, new_shape)
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)
        hidden_states = ttnn.to_device(hidden_states, device)

        if len(hidden_states.shape) == 4:
            output = ttnn.permute(hidden_states, (0, 2, 1, 3))
        else:
            output = ttnn.permute(hidden_states, (0, 2, 1))

        return output

    def __call__(
        self,
        hidden_states,
        height,
        width,
        parameters,
        output_attentions=False,
    ):
        device = hidden_states.device()
        query = ttnn.linear(
            hidden_states,
            parameters.query.weight,
            bias=parameters.query.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=12),
        )
        query_layer = self.transpose_for_scores(query)

        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = hidden_states.shape
            # Reshape to (batch_size, num_channels, height, width)
            hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
            hidden_states = ttnn.from_device(hidden_states)
            hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (batch_size, num_channels, height, width))
            # Apply sequence reduction
            # hidden_states = tt_lib.tensor.interleaved_to_sharded(hidden_states, self.sr.conv.input_sharded_memory_config)
            hidden_states = ttnn_to_torch(hidden_states)
            hidden_states = hidden_states.to(torch.float)
            hidden_states = self.sr(hidden_states)
            hidden_states = torch_to_ttnn(hidden_states, device)
            # Reshape back to (batch_size, seq_len, num_channels)
            hidden_states = ttnn.reshape(hidden_states, (batch_size, num_channels, -1))

            hidden_states = ttnn.to_device(hidden_states, device)
            hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
            if batch_size == 1:
                hidden_states = ttnn.reshape(
                    hidden_states, (batch_size, hidden_states.shape[0], hidden_states.shape[1])
                )

            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=parameters.layer_norm.weight,
                bias=parameters.layer_norm.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        key = ttnn.linear(hidden_states, parameters.key.weight, bias=parameters.key.bias)
        key_layer = self.transpose_for_scores(key)
        value = ttnn.linear(
            hidden_states,
            parameters.value.weight,
            bias=parameters.value.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        key_layer = ttnn.to_torch(key_layer)
        key_layer = torch.permute(key_layer, (0, 1, 3, 2))
        key_layer = ttnn.from_torch(key_layer, ttnn.bfloat16)
        key_layer = ttnn.to_layout(key_layer, ttnn.TILE_LAYOUT)
        key_layer = ttnn.to_device(key_layer, device)
        attention_scores = ttnn.matmul(query_layer, key_layer)

        denominator_value = ttnn.ones(
            attention_scores.shape, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        denominator_value = denominator_value * math.sqrt(self.attention_head_size)
        denominator_value = ttnn.reciprocal(denominator_value)
        attention_scores = attention_scores * denominator_value

        # Normalize the attention scores to probabilities.
        attention_probs = ttnn.softmax(attention_scores, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        context_layer = ttnn.matmul(
            attention_probs,
            value_layer,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=12),
        )

        context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = tuple(context_layer.shape)[:-2] + (self.all_head_size,)
        context_layer = ttnn.from_device(context_layer)
        context_layer = ttnn.to_layout(context_layer, layout=ttnn.ROW_MAJOR_LAYOUT)
        context_layer = ttnn.reshape(context_layer, new_context_layer_shape)
        context_layer = ttnn.to_device(context_layer, device)
        context_layer = ttnn.to_layout(context_layer, layout=ttnn.TILE_LAYOUT)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
