# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model,
)
from transformers import SegformerModel
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_segformer.tt.ttnn_segformer_efficient_selfattention import (
    ttnn_segformer_efficent_selfattention,
)

from models.experimental.functional_segformer.reference.segformer_efficient_selfattention import (
    SegformerEfficientSelfAttention,
)


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, height, width, num_attention_heads, sequence_reduction_ratio, block_i, efficient_self_attention_i",
    [
        (1, 16384, 32, 128, 128, 1, 8, 0, 0),
        (1, 16384, 32, 128, 128, 1, 8, 0, 1),
        (1, 4096, 64, 64, 64, 2, 4, 1, 0),
        (1, 4096, 64, 64, 64, 2, 4, 1, 1),
        (1, 1024, 160, 32, 32, 5, 2, 2, 0),
        (1, 1024, 160, 32, 32, 5, 2, 2, 1),
        (1, 256, 256, 16, 16, 8, 1, 3, 0),
        (1, 256, 256, 16, 16, 8, 1, 3, 1),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_efficient_selfattention(
    device,
    batch_size,
    seq_len,
    hidden_size,
    height,
    width,
    num_attention_heads,
    sequence_reduction_ratio,
    block_i,
    efficient_self_attention_i,
    reset_seeds,
):
    torch_input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    torch_model = SegformerModel.from_pretrained("nvidia/mit-b0")
    config = torch_model.config
    torch_model = torch_model.encoder.block[block_i][efficient_self_attention_i].attention.self

    reference_model = SegformerEfficientSelfAttention(
        config=config,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        sequence_reduction_ratio=sequence_reduction_ratio,
    )

    sd = torch_model.state_dict()
    reference_model.load_state_dict(sd)
    reference_model.eval()

    torch_output = reference_model(torch_input_tensor, height, width)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: reference_model,
        run_model=lambda model: model(torch_input_tensor, height, width),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = ttnn_segformer_efficent_selfattention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        parameters=parameters,
        sequence_reduction_ratio=reference_model.sr_ratio,
        model=reference_model,
    )

    ttnn_output = ttnn_model(ttnn_input_tensor, height, width, parameters=parameters)

    ttnn_final_output = ttnn.to_torch(ttnn_output[0])
    assert_with_pcc(torch_output[0], ttnn_final_output, pcc=0.99)
