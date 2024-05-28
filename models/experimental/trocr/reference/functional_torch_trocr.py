# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch TrOCR decoder model (based on RoBERTa)."""


import copy
import math
from typing import Optional, Tuple, Union
from loguru import logger
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_pcc,
    comp_allclose,
)

# from ...activations import ACT2FN
# from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
# from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
# from ...modeling_utils import PreTrainedModel
# from ...utils import add_start_docstrings, logging, replace_return_docstrings
# from .configuration_trocr import TrOCRConfig
# from models.experimental.trocr.reference.trocr_configuration import Torch_TrOCRConfig
from models.experimental.trocr.reference.activations import ACT2FN
from models.experimental.trocr.reference.trocr_utils import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from transformers import TrOCRConfig, VisionEncoderDecoderModel, TrOCRConfig


def trocr_learned_positional_embedding(
    input_ids: torch.Tensor, num_embeddings: int, embedding_dim: int, past_key_values_length: int = 0
):
    """
    This function learns positional embeddings up to a fixed maximum size.
    """
    offset = 2
    embedding = nn.Embedding(num_embeddings + offset, embedding_dim)

    bsz, seq_len = input_ids.shape[:2]
    positions = torch.arange(
        past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=embedding.weight.device
    ).expand(bsz, -1)

    return embedding(positions + offset)


# input_ids = torch.tensor([[1, 2, 3, 4]])
# num_embeddings = 10
# embedding_dim = 6
# past_key_values_length = 0

# output = trocr_learned_positional_embedding(input_ids, num_embeddings, embedding_dim, past_key_values_length)
# print(output)


def create_position_ids_from_input_ids(input_ids: torch.Tensor, padding_idx: int, past_key_values_length: int = 0):
    """
    This function replaces non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
    symbols are ignored.
    """
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


def trocr_sinusoidal_positional_embedding(
    input_ids: torch.Tensor,
    num_positions: int,
    embedding_dim: int,
    padding_idx: Optional[int] = None,
    past_key_values_length: int = 0,
):
    """This function produces sinusoidal positional embeddings of any length."""
    offset = 2

    # Build sinusoidal embeddings
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(num_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_positions, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_positions, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0

    # Create the position ids from the input token ids. Any padded tokens remain padded.
    position_ids = create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length).to(
        input_ids.device
    )

    # Expand embeddings if needed
    max_pos = padding_idx + 1 + input_ids.size(1)
    if max_pos > emb.size(0):
        # Recompute/expand embeddings if needed
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_pos, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(max_pos, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(max_pos, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

    x = emb.index_select(0, position_ids.view(-1)).view(input_ids.size(0), input_ids.size(1), -1).detach()

    return x


# input_ids = torch.tensor([[1, 2, 3, 4]])
# num_positions = 10
# embedding_dim = 6
# padding_idx = 0
# past_key_values_length = 0

# output = trocr_sinusoidal_positional_embedding(input_ids, num_positions, embedding_dim, padding_idx, past_key_values_length)
# print(output)

import torch
from torch import nn
from typing import Optional, Tuple


def trocr_attention(
    config,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    embed_dim: int = None,
    num_heads: int = None,
    kdim: int = None,
    vdim: int = None,
    dropout: float = 0.0,
    is_decoder: bool = False,
    bias: bool = True,
    is_cross_attention: bool = False,
):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    # Config initialization
    embed_dim = embed_dim if embed_dim is not None else config.hidden_size
    num_heads = num_heads if num_heads is not None else config.num_attention_heads
    kdim = kdim if kdim is not None else config.hidden_size
    vdim = vdim if vdim is not None else config.hidden_size
    dropout = dropout if dropout is not None else config.attention_probs_dropout_prob
    bias = bias if bias is not None else True

    head_dim = embed_dim // num_heads
    scaling = head_dim**-0.5

    bsz, tgt_len, embed_dim = hidden_states.size()

    # get query proj
    q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    query_states = q_proj(hidden_states) * scaling

    # get key, value proj
    if is_cross_attention and past_key_value is not None:
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        k_proj = nn.Linear(kdim, embed_dim, bias=bias)
        v_proj = nn.Linear(vdim, embed_dim, bias=bias)
        key_states = k_proj(key_value_states).view(bsz, -1, num_heads, head_dim).transpose(1, 2).contiguous()
        value_states = v_proj(key_value_states).view(bsz, -1, num_heads, head_dim).transpose(1, 2).contiguous()
    elif past_key_value is not None:
        k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        key_states = k_proj(hidden_states).view(bsz, -1, num_heads, head_dim).transpose(1, 2).contiguous()
        value_states = v_proj(hidden_states).view(bsz, -1, num_heads, head_dim).transpose(1, 2).contiguous()
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        key_states = k_proj(hidden_states).view(bsz, -1, num_heads, head_dim).transpose(1, 2).contiguous()
        value_states = v_proj(hidden_states).view(bsz, -1, num_heads, head_dim).transpose(1, 2).contiguous()

    if is_decoder:
        past_key_value = (key_states, value_states)

    proj_shape = (bsz * num_heads, -1, head_dim)
    query_states = query_states.view(bsz * num_heads, tgt_len, head_dim)
    key_states = key_states.view(bsz * num_heads, -1, head_dim)
    value_states = value_states.view(bsz * num_heads, -1, head_dim)

    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attention_mask is not None:
        attn_weights = attn_weights.view(bsz, num_heads, tgt_len, -1) + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

    attn_probs = nn.functional.dropout(attn_weights, p=dropout, training=True)

    attn_output = torch.bmm(attn_probs, value_states)

    attn_output = (
        attn_output.view(bsz, num_heads, tgt_len, -1).transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
    )

    out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    attn_output = out_proj(attn_output)

    return attn_output, attn_weights, past_key_value


def test_TrOCRAttention():
    batch_size = 1
    seq_length = 3
    embed_dim = 1024
    num_heads = 16
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    config = model.decoder.config
    # Create input tensors
    # hidden_states = torch.randn(batch_size, seq_length, embed_dim)
    hidden_states = torch.load(
        "/home/ubuntu/jayasurya/tt-metal/models/experimental/trocr/reference/ref_hidden_states.pt"
    )
    key_value_states = None
    attention_mask = None
    layer_head_mask = None

    # Call TrOCRAttention function
    output, attn_weights, past_key_value = trocr_attention(
        config=config,
        hidden_states=hidden_states,
        key_value_states=key_value_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=False,
        embed_dim=embed_dim,
        num_heads=num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        is_decoder=True,
        bias=True,
        is_cross_attention=False,
    )

    # logger.info(comp_pcc(output, ref_out))
    logger.info(comp_pcc(attn_weights, ref_weight))
    logger.info(comp_pcc(past_key_value, ref_pastk))

    # logger.info(comp_allclose(model_output, tt_output_torch))
    logger.info(pcc_message)


# Run the test function
test_TrOCRAttention()
