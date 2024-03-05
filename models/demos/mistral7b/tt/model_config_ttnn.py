# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from pathlib import Path


class TtModelArgs:
    """Model args for Mistral 7B as provided by the params.json config file"""

    dim = 4096
    n_layers = 32
    head_dim = 128
    hidden_dim = 14336
    n_heads = 32
    n_kv_heads = 8
    norm_eps = 1e-05
    sliding_window = 4096
    vocab_size = 32000

    # Parameters for our use
    max_batch_size = 32
    max_seq_len = 4096

    def __init__(self, model_base_path="/proj_sw/user_dev/hf_data/mistral"):
        self.model_base_path = Path(model_base_path)
        # Some consumers like SentencePiece only accept str not Path for files
        self.consolidated_weights_path = str(self.model_base_path / "mistral-7B-v0.1/consolidated.00.pth")
        self.tokenizer_path = str(self.model_base_path / "mistral-7B-v0.1/tokenizer.model")

    def weight_cache_path(self, dtype):
        return self.model_base_path / {ttnn.bfloat16: "tensor_cache_bf16", ttnn.bfloat8_b: "tensor_cache_bfp8"}[dtype]
