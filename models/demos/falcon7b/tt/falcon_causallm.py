# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import tt_lib
from models.demos.falcon7b.tt.falcon_model import TtFalconModelShared
from models.demos.falcon7b.tt.model_utils import falcon_lm_head_matmul_2d, get_weights_cached


class TtFalconCausalLM(TtFalconModelShared):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
        seq_len,
    ):
        assert base_url == "", "base_url should be empty at the root of the model!"

        super().__init__(
            devices=devices,
            state_dict=state_dict,
            base_url=f"transformer",
            num_layers=num_layers,
            config=config,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            seq_len=seq_len,
        )
        self.num_devices = len(devices)
        self.model_config = model_config
        self.seq_len = seq_len

        lm_head_str = f"lm_head.weight"

        num_slices = 4 if self.seq_len <= 1024 else 8
        PADDING = torch.zeros([64, 254 * 32])

        lm_head_weights = torch.transpose(self.state_dict[f"lm_head.weight"], -2, -1)
        lm_head_weights = torch.chunk(lm_head_weights, num_slices, dim=-1)
        lm_head_weights_padded = [torch.cat([weight, PADDING], 0) for weight in lm_head_weights]

        self.lm_head_weights = [
            get_weights_cached(
                devices,
                model_config,
                tt_cache_path,
                f"lm_head.weight_slice_{i}_of_{num_slices}",
                weight_config_str="LM_HEAD_MM_WEIGHTS",
                weights_to_cache=lm_head_weights_padded[i],
            )
            for i in range(num_slices)
        ]

    def forward(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        llm_mode: str,
        attention_mask: tt_lib.tensor.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[tt_lib.tensor.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> tt_lib.tensor.Tensor:
        hidden_states, presents = super().forward(
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            llm_mode=llm_mode,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            use_cache=use_cache,
        )

        lm_logits = [
            falcon_lm_head_matmul_2d(
                hidden_states[device_id],
                # list of lists [[1, a], [2, b], [3, c]] - get device_id elements from each list
                [weights[device_id] for weights in self.lm_head_weights],
                num_slices=4 if self.seq_len <= 1024 else 8,
                in0_mem_config=self.model_config["LM_HEAD_MM_INPUT_MEMCFG"],
                in0_dtype=self.model_config["LM_HEAD_MM_INPUT_DTYPE"],
                out_mem_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
                out_dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
            )
            for device_id in range(self.num_devices)
        ]

        return lm_logits, presents
