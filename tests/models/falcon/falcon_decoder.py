# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from torch import nn
from typing import Optional, Tuple

import tt_lib

from tests.models.falcon.falcon_attention import TtFalconAttention
from tests.models.falcon.falcon_mlp import TtFalconMLP
from models.utility_functions import pad_by_zero
from models.utility_functions import tt2torch_tensor, dump_tensor

class TtFalconDecoderLayer(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.state_dict = state_dict
        self.base_url = base_url
        self.device = device
        self.layer_num = layer_num
        self.max_position_embeddings = max_position_embeddings
        self.model_config = model_config

        assert (
            config.parallel_attn
        ), "Path for config.parallel_attn=False is not implemented in TtFalconDecoderLayer!"

        self.self_attn = TtFalconAttention(
            device=device,
            state_dict=state_dict,
            base_url=base_url,
            layer_num=layer_num,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )

        self.mlp = TtFalconMLP(
            device=device,
            state_dict=state_dict,
            base_url=base_url,
            layer_num=layer_num,
            hidden_size=config.hidden_size,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )

        layer_name = f"{base_url}.{layer_num}"

        layernorm_weights_str = f"{layer_name}.input_layernorm.weight"
        layernorm_bias_str = f"{layer_name}.input_layernorm.bias"
        if tt_cache_path is not None:
            self.layernorm_gamma = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{layernorm_weights_str}_{self.model_config['INPUT_LAYERNORM_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["INPUT_LAYERNORM_WEIGHTS_MEMCFG"])
            self.layernorm_beta = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{layernorm_bias_str}_{self.model_config['INPUT_LAYERNORM_BIAS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["INPUT_LAYERNORM_BIAS_MEMCFG"])
        else:
            self.layernorm_gamma = pad_by_zero(
                self.state_dict[layernorm_weights_str],
                device,
                tt_memory_config=self.model_config["INPUT_LAYERNORM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["INPUT_LAYERNORM_WEIGHTS_DTYPE"],
            )[0]
            self.layernorm_beta = pad_by_zero(
                self.state_dict[layernorm_bias_str],
                device,
                tt_memory_config=self.model_config["INPUT_LAYERNORM_BIAS_MEMCFG"],
                tt_dtype=self.model_config["INPUT_LAYERNORM_BIAS_DTYPE"],
            )[0]
        self.layernorm_eps = config.layer_norm_epsilon

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        tt_lib.tensor.Tensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """Input shape: [batch, 1, seq_len, hidden_size]"""

        assert (
            not output_attentions
        )  # hf_reference Falcon Attention doesn't support this
        dump_tensor("decoder_input", "tt", tt2torch_tensor(hidden_states))

        # layernorm_output = tt_lib.tensor.layernorm(
        #     hidden_states,
        #     self.layernorm_eps,  # These don't fit: self.layernorm_gamma, self.layernorm_beta
        #     output_mem_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"],
        #     # output_dtype=self.model_config["INPUT_LAYERNORM_OUTPUT_DTYPE"], # Not currently supported
        # )
        # layernorm_output = tt_lib.tensor.bcast(
        #     layernorm_output,
        #     self.layernorm_gamma,
        #     tt_lib.tensor.BcastOpMath.MUL,
        #     tt_lib.tensor.BcastOpDim.H,
        #     output_mem_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"],
        #     # output_dtype=self.model_config["INPUT_LAYERNORM_OUTPUT_DTYPE"], # Not currently supported
        # )
        # layernorm_output = tt_lib.tensor.bcast(
        #     layernorm_output,
        #     self.layernorm_beta,
        #     tt_lib.tensor.BcastOpMath.ADD,
        #     tt_lib.tensor.BcastOpDim.H,
        #     output_mem_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"],
        #     # output_dtype=self.model_config["INPUT_LAYERNORM_OUTPUT_DTYPE"], # Not currently supported
        # )

        layernorm_output = nn.functional.layer_norm(tt2torch_tensor(hidden_states).to(torch.float32), (hidden_states.shape()[-1],), weight=tt2torch_tensor(self.layernorm_gamma).to(torch.float32)[0, 0, 0], bias=tt2torch_tensor(self.layernorm_beta).to(torch.float32)[0, 0, 0])
        layernorm_output = tt_lib.tensor.Tensor(layernorm_output, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(hidden_states.device())
        dump_tensor("attention_layernorm_out", "tt", tt2torch_tensor(layernorm_output))

        residual = hidden_states

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=layernorm_output,
            alibi=alibi,
            attention_mask=attention_mask,
            llm_mode=llm_mode,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attention_output, outputs = attn_outputs[0], attn_outputs[1:]

        # config.parallel_attn=False not implemented
        # See: tests/models/falcon/hf_reference_falcon_model.py
        # residual and layernorm_output would be modified here

        # MLP
        # mlp will deallocate layernorm_output
        dump_tensor("mlp_input", "tt", tt2torch_tensor(layernorm_output))
        mlp_output = self.mlp(layernorm_output)
        dump_tensor("mlp_output", "tt", tt2torch_tensor(mlp_output))

        # config.parallel_attn=True
        output = tt_lib.tensor.add(
            mlp_output,
            attention_output,
            output_mem_config=self.model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["PARALLEL_ATTN_ADD_OUTPUT_DTYPE"], # Not currently supported
        )
        mlp_output.deallocate()
        attention_output.deallocate()
        dump_tensor("mlp_plus_attention_output", "tt", tt2torch_tensor(output))

        # dropout_add
        # For inference, this is just add
        output = tt_lib.tensor.add(
            output,
            residual,
            output_mem_config=self.model_config["DROPOUT_ADD_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["DROPOUT_ADD_OUTPUT_DTYPE"], # Not currently supported
        )
        residual.deallocate()
        dump_tensor("decoder_output", "tt", tt2torch_tensor(output))

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (
                output,
                (),
            )  # Outputs should be empty if we ignore layer_past as well

        return outputs
