# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

import torch
import tt_lib
from models.demos.falcon7b.tt.model_utils import get_weights_cached
from models.utility_functions import is_wormhole_b0, nearest_32, pad_by_zero, torch2tt_tensor, tt2torch_tensor
from torch import nn


class TtFalconRotaryEmbedding(torch.nn.Module):
    """
    See FalconRotaryEmbedding from hf_modeling_falcon.py
    """

    def __init__(
        self,
        tt_devices,
        dim,
        base_url,
        layer_num,
        max_position_embeddings=2048,
        base=10000,
        model_config=None,
        tt_cache_path=None,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        self.max_seq_len_cached = max_position_embeddings
        self.model_config = model_config
        t = torch.arange(
            self.max_seq_len_cached,
            device=inv_freq.device,
            dtype=inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        layer_name = f"{base_url}.{layer_num}.rotary_embedding"
        cos_str = f"{layer_name}.cos_cached"
        sin_str = f"{layer_name}.sin_cached"

        overwrite_cos, overwrite_sin = False, False

        for _ in range(2):
            self.tt_cos_cached = get_weights_cached(
                tt_devices,
                model_config,
                tt_cache_path,
                cos_str,
                weight_config_str="COS_CACHED_WEIGHTS",
                weights_to_cache=emb.cos()[None, None, :, :],
                overwrite=overwrite_cos,
            )
            overwrite_cos = (
                tt2torch_tensor(self.tt_cos_cached[0]).shape[-2] != self.max_seq_len_cached
            )  # Verify cached tensor has same max seq len
            if not overwrite_cos:
                break

        for _ in range(2):
            self.tt_sin_cached = get_weights_cached(
                tt_devices,
                model_config,
                tt_cache_path,
                sin_str,
                weight_config_str="SIN_CACHED_WEIGHTS",
                weights_to_cache=emb.sin()[None, None, :, :],
                overwrite=overwrite_sin,
            )
            overwrite_sin = (
                tt2torch_tensor(self.tt_sin_cached[0]).shape[-2] != self.max_seq_len_cached
            )  # Verify cached tensor has same max seq len
            if not overwrite_sin:
                break

    def forward(self, layer: tt_lib.tensor.Tensor, token_idx: Optional[int] = None) -> tt_lib.tensor.Tensor:
        seq_len = layer[0].get_legacy_shape()[2]
        assert seq_len <= self.max_seq_len_cached, "seq_len exceeds max_seq_len_cached in RotaryEmbedding!"

        output = []
        for i in range(len(layer)):
            output.append(
                tt_lib.tensor.rotary_embedding(
                    layer[i],
                    self.tt_cos_cached[i],
                    self.tt_sin_cached[i],
                    token_idx,
                    output_mem_config=self.model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"],
                )
            )
        return output


class TtFalconAttention(nn.Module):
    """Mulit-Query Attention: https://arxiv.org/pdf/1911.02150.pdf"""

    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int = 2048,
        model_config=None,
        tt_cache_path=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
        self.devices = devices
        self.num_devices = len(devices)
        self.state_dict = state_dict
        self.model_config = model_config

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )

        layer_name = f"{base_url}.{layer_num}.self_attention"
        query_key_value_str = f"{layer_name}.query_key_value.weight"
        selfout_str = f"{layer_name}.dense.weight"

        self.query_key_value_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            query_key_value_str,
            weight_config_str="FUSED_QKV_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[query_key_value_str], -2, -1) if state_dict else None),
        )
        self.dense_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            selfout_str,
            weight_config_str="SELFOUT_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[selfout_str], -2, -1) if state_dict else None),
        )

        self.rotary_embedding = TtFalconRotaryEmbedding(
            self.devices,
            self.head_dim,
            base_url,
            layer_num,
            max_position_embeddings=self.max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )

        self.scalar = [pad_by_zero(torch.Tensor([1 / math.sqrt(self.head_dim)]), device)[0] for device in devices]

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[tt_lib.tensor.Tensor, Optional[Tuple[tt_lib.tensor.Tensor]]]:
        """
        Prefill input shape: [batch, 1, seq_len, hidden_size]
        Decode input shape: [seq_len, 1, batch, hidden_size]
        """

        if llm_mode == "prefill":
            batch = hidden_states[0].get_legacy_shape()[0]
            seq_len = hidden_states[0].get_legacy_shape()[2]
            assert layer_past is not None
        elif llm_mode == "decode":
            batch = hidden_states[0].get_legacy_shape()[2]
            seq_len = hidden_states[0].get_legacy_shape()[0]
            # We always store max_position_embeddings for kv_cache,
            # so we need separate variable to store the actual len of the kv_cache
            assert layer_past is not None
            assert layer_past_len > 0 and layer_past_len <= self.max_position_embeddings
        else:
            raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

        #################
        ### FUSED QKV ###
        #################
        fused_query_key_value = []
        for i in range(self.num_devices):
            fused_query_key_value.append(
                tt_lib.tensor.falcon_fused_qkv_matmul(
                    hidden_states[i],
                    self.query_key_value_weights[i],
                    output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
                )
            )

        ###########
        ### TMs ###
        ###########
        query_layer, key_layer, value_layer = [], [], []
        for i in range(self.num_devices):
            query_layer_i, key_layer_i, value_layer_i = tt_lib.tensor.nlp_create_qkv_heads_falcon7b(
                fused_query_key_value[i],
                output_mem_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
            )
            fused_query_key_value[i].deallocate()
            query_layer.append(query_layer_i)
            key_layer.append(key_layer_i)
            value_layer.append(value_layer_i)

        #########################
        ### ROTARY EMBEDDINGS ###
        #########################
        if llm_mode == "prefill":
            query_layer = self.rotary_embedding(query_layer)
            key_layer = self.rotary_embedding(key_layer)
        elif llm_mode == "decode":
            query_layer = self.rotary_embedding(query_layer, layer_past_len)
            key_layer = self.rotary_embedding(key_layer, layer_past_len)

        ######################
        ### K CACHE UPDATE ###
        ######################
        if llm_mode == "prefill":
            for i in range(self.num_devices):
                tt_lib.tensor.fill_cache(layer_past[i][0], key_layer[i], user_id)

        elif llm_mode == "decode":
            for i in range(self.num_devices):
                # Update kv_cache in place
                tt_lib.tensor.update_cache(layer_past[i][0], key_layer[i], layer_past_len)
            for i in range(self.num_devices):
                # key and value layers will have kv_seq_len padded to nearest 32
                key_layer[i] = tt_lib.tensor.unpad(
                    layer_past[i][0],
                    [0, 0, 0, 0],
                    [batch - 1, 0, nearest_32(layer_past_len + 1) - 1, self.head_dim - 1],
                    output_mem_config=self.model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"],
                )

        ######################
        ### PRE-SOFTMAX MM ###
        ######################
        key_layer_transposed = []
        for i in range(self.num_devices):
            key_layer_transposed.append(
                tt_lib.tensor.transpose(
                    key_layer[i],
                    -2,
                    -1,
                    output_mem_config=self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"],
                )
            )
            key_layer[i].deallocate()

        if llm_mode == "prefill":
            height_sharded_memory_config = tt_lib.tensor.MemoryConfig(
                memory_layout=tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=tt_lib.tensor.BufferType.L1
            )
            dram_interleaved_memory_config = tt_lib.tensor.MemoryConfig(
                memory_layout=tt_lib.tensor.TensorMemoryLayout.INTERLEAVED,
                buffer_type=tt_lib.tensor.BufferType.DRAM,
            )
            grid_size = (8, 8)
            if seq_len == 128:
                num_slices = 1
            elif seq_len == 1024:
                num_slices = 4
            elif seq_len == 2048:
                num_slices = 16
            num_cores = 64

            attention_output_shape = [1, self.num_heads, seq_len, 64]
            torch_attention_output = torch.randn(attention_output_shape).bfloat16().float()
            tiles_per_shard = math.ceil((((self.num_heads * seq_len) / num_cores) / num_slices) / 32)
            mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
            mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

            attention_outputs_concatenated = [
                torch2tt_tensor(
                    torch_attention_output,
                    self.devices[device_id],
                    tt_memory_config=dram_interleaved_memory_config,
                    tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
                )
                for device_id in range(self.num_devices)
            ]

            compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
                math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )

            for i in range(num_slices):
                slices = [
                    tt_lib.tensor.interleaved_to_sharded_partial(
                        query_layer[device_id],
                        grid_size,
                        mm_activations_height_shard_spec,
                        num_slices,  # num_slices
                        i,  # slice_index
                        tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                        tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                    )
                    for device_id in range(self.num_devices)
                ]

                subblock_h = 1
                subblock_w = 1
                if seq_len == 2048:
                    subblock_w = 8  # best option

                # pre_softmax_mm
                program_config = tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=grid_size,
                    in0_block_w=2,
                    per_core_M=tiles_per_shard,
                    per_core_N=seq_len // 32,
                    out_subblock_h=subblock_h,
                    out_subblock_w=subblock_w,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=False,
                )
                mm_slices = [
                    tt_lib.operations.primary.matmul(
                        slices[device_id],
                        key_layer_transposed[device_id],
                        program_config=program_config,
                        output_mem_config=height_sharded_memory_config,
                        output_dtype=tt_lib.tensor.DataType.BFLOAT16,
                        compute_kernel_config=compute_kernel_config,
                    )
                    for device_id in range(self.num_devices)
                ]
                mm_slices = [
                    tt_lib.operations.primary.bcast(
                        mm_slices[device_id],
                        self.scalar[device_id],
                        tt_lib.tensor.BcastOpMath.MUL,
                        tt_lib.tensor.BcastOpDim.HW,
                        output_mem_config=height_sharded_memory_config,
                        in_place=True,
                    )
                    for device_id in range(self.num_devices)
                ]

                attn_mask_slices = [
                    tt_lib.tensor.interleaved_to_sharded_partial(
                        attention_mask[device_id],
                        grid_size,
                        mm_output_height_shard_spec,
                        num_slices,
                        i,
                        tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                        tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                    )
                    for device_id in range(self.num_devices)
                ]
                mm_slices = [
                    tt_lib.operations.primary.add(
                        mm_slices[device_id],
                        attn_mask_slices[device_id],
                        fused_activations=None,
                        output_mem_config=height_sharded_memory_config,
                        output_dtype=tt_lib.tensor.DataType.BFLOAT16,
                        in_place=True,
                    )
                    for device_id in range(self.num_devices)
                ]

                for device_id in range(self.num_devices):
                    attn_mask_slices[device_id].deallocate()

                softmax_program_config = tt_lib.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=grid_size,
                    subblock_w=1,
                    block_h=mm_output_height_shard_spec[0] // 32,
                    block_w=mm_output_height_shard_spec[1] // 32,
                    math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
                    im_data_format=tt_lib.tensor.DataType.BFLOAT16,
                )

                mm_slices = [
                    tt_lib.operations.primary.softmax_in_place(
                        mm_slices[device_id],
                        program_config=softmax_program_config,
                    )
                    for device_id in range(self.num_devices)
                ]

                subblock_w = 2
                subblock_h = 1
                program_config = tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=grid_size,
                    in0_block_w=seq_len // 32,
                    per_core_M=tiles_per_shard,
                    per_core_N=2,
                    out_subblock_h=subblock_h,
                    out_subblock_w=subblock_w,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=False,
                )

                attn_out_slices = [
                    tt_lib.operations.primary.matmul(
                        mm_slices[device_id],
                        value_layer[device_id],
                        program_config=program_config,
                        output_mem_config=height_sharded_memory_config,
                        output_dtype=tt_lib.tensor.DataType.BFLOAT16,
                        compute_kernel_config=compute_kernel_config,
                    )
                    for device_id in range(self.num_devices)
                ]

                for device_id in range(self.num_devices):
                    tt_lib.tensor.sharded_to_interleaved_partial(
                        attn_out_slices[device_id],
                        attention_outputs_concatenated[device_id],
                        num_slices,
                        i,
                        dram_interleaved_memory_config,
                    )

                for device_id in range(self.num_devices):
                    attn_out_slices[device_id].deallocate()
                    mm_slices[device_id].deallocate()
                    slices[device_id].deallocate()

            # V cache update
            for device_id in range(self.num_devices):
                tt_lib.tensor.fill_cache(layer_past[device_id][1], value_layer[device_id], user_id)

            layer_present = layer_past if use_cache else None

        elif llm_mode == "decode":
            if is_wormhole_b0():
                matmul = tt_lib.operations.primary.transformers.attn_matmul
            else:
                matmul = tt_lib.operations.primary.transformers.group_attn_matmul

            attn_weights = [
                matmul(
                    query_layer[i],
                    key_layer_transposed[i],
                    compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                    output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                )
                for i, device in enumerate(self.devices)
            ]
            query_layer[i].deallocate()
            key_layer_transposed[i].deallocate()

            attn_weights = [
                tt_lib.tensor.bcast(
                    attn_weights[device_id],
                    self.scalar[device_id],
                    tt_lib.tensor.BcastOpMath.MUL,
                    tt_lib.tensor.BcastOpDim.HW,
                    output_mem_config=self.model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"],
                )
                for device_id in range(self.num_devices)
            ]

            if attention_mask is not None:
                attn_weights = [
                    tt_lib.tensor.add(
                        attn_weights[device_id],
                        attention_mask[device_id],
                        output_mem_config=self.model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"],
                    )
                    for device_id in range(self.num_devices)
                ]

            attn_weights = [
                tt_lib.operations.primary.softmax_in_place(
                    attn_weights[device_id],
                )
                for device_id in range(self.num_devices)
            ]

            for device_id in range(self.num_devices):
                # Update kv_cache in place
                tt_lib.tensor.update_cache(layer_past[device_id][1], value_layer[i], layer_past_len)

            value_layer = [
                tt_lib.tensor.unpad(
                    layer_past[device_id][1],
                    [0, 0, 0, 0],
                    [batch - 1, 0, nearest_32(layer_past_len + 1) - 1, self.head_dim - 1],
                    output_mem_config=self.model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"],
                )
                for device_id in range(self.num_devices)
            ]

            attention_outputs_concatenated = [
                matmul(
                    attn_weights[device_id],
                    value_layer[device_id],
                    compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                    output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                )
                for device_id, device in enumerate(self.devices)
            ]
            for i in range(self.num_devices):
                attn_weights[i].deallocate()
                value_layer[i].deallocate()

            layer_present = layer_past if use_cache else None

        else:
            raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

        attn_outputs = [
            tt_lib.tensor.nlp_concat_heads(
                attention_outputs_concatenated[device_id],
                output_mem_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
            )
            for device_id in range(self.num_devices)
        ]

        attn_outputs = [
            tt_lib.tensor.falcon_selfout_matmul(
                attn_outputs[device_id],
                self.dense_weights[device_id],
                output_mem_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
            )
            for device_id in range(self.num_devices)
        ]

        return attn_outputs, layer_present

    def old_forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[tt_lib.tensor.Tensor, Optional[Tuple[tt_lib.tensor.Tensor]]]:
        """
        Prefill input shape: [batch, 1, seq_len, hidden_size]
        Decode input shape: [seq_len, 1, batch, hidden_size]
        """

        assert not output_attentions

        if llm_mode == "prefill":
            batch = hidden_states[0].get_legacy_shape()[0]
            q_len = hidden_states[0].get_legacy_shape()[2]
            assert layer_past is not None
        elif llm_mode == "decode":
            batch = hidden_states[0].get_legacy_shape()[2]
            q_len = hidden_states[0].get_legacy_shape()[0]
            # We always store max_position_embeddings for kv_cache,
            # so we need separate variable to store the actual len of the kv_cache
            assert layer_past is not None
            assert layer_past_len > 0 and layer_past_len <= self.max_position_embeddings
        else:
            raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

        #################
        ### FUSED QKV ###
        #################
        fused_query_key_value = []
        for i in range(self.num_devices):
            fused_query_key_value.append(
                tt_lib.tensor.falcon_fused_qkv_matmul(
                    hidden_states[i],
                    self.query_key_value_weights[i],
                    output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
                )
            )

        ###########
        ### TMs ###
        ###########
        query_layer, key_layer, value_layer = [], [], []
        for i in range(self.num_devices):
            query_layer_i, key_layer_i, value_layer_i = tt_lib.tensor.nlp_create_qkv_heads_falcon7b(
                fused_query_key_value[i],
                output_mem_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
            )
            fused_query_key_value[i].deallocate()
            query_layer.append(query_layer_i)
            key_layer.append(key_layer_i)
            value_layer.append(value_layer_i)

        #########################
        ### ROTARY EMBEDDINGS ###
        #########################
        if llm_mode == "prefill":
            query_layer = self.rotary_embedding(query_layer)
            key_layer = self.rotary_embedding(key_layer)
        elif llm_mode == "decode":
            query_layer = self.rotary_embedding(query_layer, layer_past_len)
            key_layer = self.rotary_embedding(key_layer, layer_past_len)

        ######################
        ### K CACHE UPDATE ###
        ######################
        if llm_mode == "prefill":
            for i in range(self.num_devices):
                tt_lib.tensor.fill_cache(layer_past[i][0], key_layer[i], user_id)

        elif llm_mode == "decode":
            for i in range(self.num_devices):
                # Update kv_cache in place
                tt_lib.tensor.update_cache(layer_past[i][0], key_layer[i], layer_past_len)
            for i in range(self.num_devices):
                # key and value layers will have kv_seq_len padded to nearest 32
                key_layer[i] = tt_lib.tensor.unpad(
                    layer_past[i][0],
                    [0, 0, 0, 0],
                    [batch - 1, 0, nearest_32(layer_past_len + 1) - 1, self.head_dim - 1],
                    output_mem_config=self.model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"],
                )

        ######################
        ### PRE-SOFTMAX MM ###
        ######################
        key_layer_transposed = []
        for i in range(self.num_devices):
            key_layer_transposed.append(
                tt_lib.tensor.transpose(
                    key_layer[i],
                    -2,
                    -1,
                    output_mem_config=self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"],
                )
            )
            key_layer[i].deallocate()

        attn_weights = []
        if llm_mode == "prefill":
            for i in range(self.num_devices):
                attn_weights.append(
                    tt_lib.tensor.matmul(
                        query_layer[i],
                        key_layer_transposed[i],
                        output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    )
                )
                query_layer[i].deallocate()
                key_layer_transposed[i].deallocate()

        elif llm_mode == "decode":
            for i, device in enumerate(self.devices):
                # TODO: switch to group_attn_matmul once multiple q heads is supported (issue #5318)
                if is_wormhole_b0():
                    attn_weights.append(
                        tt_lib.operations.primary.transformers.attn_matmul(
                            query_layer[i],
                            key_layer_transposed[i],
                            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                            output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                            output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                        )
                    )
                else:
                    attn_weights.append(
                        tt_lib.operations.primary.transformers.group_attn_matmul(
                            query_layer[i],
                            key_layer_transposed[i],
                            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                            output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                            output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                        )
                    )
                query_layer[i].deallocate()
                key_layer_transposed[i].deallocate()

        for i in range(self.num_devices):
            attn_weights[i] = tt_lib.tensor.bcast(
                attn_weights[i],
                self.scalar[i],
                tt_lib.tensor.BcastOpMath.MUL,
                tt_lib.tensor.BcastOpDim.HW,
                output_mem_config=self.model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"],
            )

        if attention_mask is not None:
            for i in range(self.num_devices):
                attn_weights[i] = tt_lib.tensor.add(
                    attn_weights[i],
                    attention_mask[i],
                    output_mem_config=self.model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"],
                )

        ###############
        ### SOFTMAX ###
        ###############
        # TODO: Replace with scaled_softmax_attention_mask from BERT
        for i in range(self.num_devices):
            attn_weights[i] = tt_lib.operations.primary.softmax_in_place(
                attn_weights[i],
            )

        ######################
        ### V CACHE UPDATE ###
        ######################
        if llm_mode == "prefill":
            for i in range(self.num_devices):
                tt_lib.tensor.fill_cache(layer_past[i][1], value_layer[i], user_id)

        elif llm_mode == "decode":
            for i in range(self.num_devices):
                # Update kv_cache in place
                tt_lib.tensor.update_cache(layer_past[i][1], value_layer[i], layer_past_len)
            for i in range(self.num_devices):
                value_layer[i] = tt_lib.tensor.unpad(
                    layer_past[i][1],
                    [0, 0, 0, 0],
                    [batch - 1, 0, nearest_32(layer_past_len + 1) - 1, self.head_dim - 1],
                    output_mem_config=self.model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"],
                )

        layer_present = layer_past if use_cache else None

        ########################
        ### POST-SOFTMAX MM ###
        ########################
        attn_output = []
        if llm_mode == "prefill":
            for i in range(self.num_devices):
                attn_output.append(
                    tt_lib.tensor.matmul(
                        attn_weights[i],
                        value_layer[i],
                        output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    )
                )
                attn_weights[i].deallocate()
                value_layer[i].deallocate()

        elif llm_mode == "decode":
            for i in range(self.num_devices):
                # TODO: switch to group_attn_matmul once multiple q heads is supported (issue #5318)
                if is_wormhole_b0():
                    attn_output.append(
                        tt_lib.operations.primary.transformers.attn_matmul(
                            attn_weights[i],
                            value_layer[i],
                            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                            output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                            output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                        )
                    )
                else:
                    attn_output.append(
                        tt_lib.operations.primary.transformers.group_attn_matmul(
                            attn_weights[i],
                            value_layer[i],
                            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                            output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                            output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                        )
                    )
                attn_weights[i].deallocate()
                value_layer[i].deallocate()

        #########################
        ### ATTENTION SELFOUT ###
        #########################
        for i in range(self.num_devices):
            attn_output[i] = tt_lib.tensor.nlp_concat_heads(
                attn_output[i],
                output_mem_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
            )

        for i in range(self.num_devices):
            attn_output[i] = tt_lib.tensor.falcon_selfout_matmul(
                attn_output[i],
                self.dense_weights[i],
                output_mem_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
            )

        return attn_output, layer_present
