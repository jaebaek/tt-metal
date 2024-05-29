# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import nn
import tt_lib

from typing import List
from models.utility_functions import torch2tt_tensor


class TtFusedFalconLayernorm:
    def __init__(self, device, gamma1, beta1, gamma2, beta2, model_config, config, tt_cache_path):
        super().__init__()

        self.model_config = model_config

        layer_name = f"transformer.h.0"

        ln_attn_weights_str = f"{layer_name}.ln_attn.weight"
        ln_attn_bias_str = f"{layer_name}.ln_attn.bias"

        ln_mlp_weights_str = f"{layer_name}.ln_mlp.weight"
        ln_mlp_bias_str = f"{layer_name}.ln_mlp.bias"

        ln_attn_weights_path = (
            tt_cache_path / f"{ln_attn_weights_str}_rm_fusedln_{self.model_config['LN_ATTN_WEIGHTS_DTYPE'].name}.bin"
        )
        if (ln_attn_weights_path).exists():
            ln_attn_gamma_host = tt_lib.tensor.load_tensor(str(ln_attn_weights_path))
            self.ln_attn_gamma = ln_attn_gamma_host.to(device, self.model_config["LN_ATTN_WEIGHTS_MEMCFG"])
        else:
            ln_attn_gamma_host = tt_lib.tensor.Tensor(
                gamma1.reshape([1, 1, 1, -1]),
                self.model_config["LN_ATTN_WEIGHTS_DTYPE"],
            )
            self.ln_attn_gamma = ln_attn_gamma_host.to(device, self.model_config["LN_ATTN_WEIGHTS_MEMCFG"])
            tt_lib.tensor.dump_tensor(
                str(ln_attn_weights_path),
                ln_attn_gamma_host,
            )

        ln_attn_bias_path = (
            tt_cache_path / f"{ln_attn_bias_str}_rm_fusedln_{self.model_config['LN_ATTN_BIAS_DTYPE'].name}.bin"
        )
        if (ln_attn_bias_path).exists():
            ln_attn_beta_host = tt_lib.tensor.load_tensor(str(ln_attn_bias_path))
            self.ln_attn_beta = ln_attn_beta_host.to(device, self.model_config["LN_ATTN_BIAS_MEMCFG"])
        else:
            ln_attn_beta_host = tt_lib.tensor.Tensor(
                beta1.reshape([1, 1, 1, -1]),
                self.model_config["LN_ATTN_BIAS_DTYPE"],
            )
            self.ln_attn_beta = ln_attn_beta_host.to(device, self.model_config["LN_ATTN_BIAS_MEMCFG"])
            tt_lib.tensor.dump_tensor(
                str(ln_attn_bias_path),
                ln_attn_beta_host,
            )

        ln_mlp_weights_path = (
            tt_cache_path / f"{ln_mlp_weights_str}_rm_fusedln_{self.model_config['LN_MLP_WEIGHTS_DTYPE'].name}.bin"
        )
        if (ln_mlp_weights_path).exists():
            ln_mlp_gamma_host = tt_lib.tensor.load_tensor(str(ln_mlp_weights_path))
            self.ln_mlp_gamma = ln_mlp_gamma_host.to(device, self.model_config["LN_MLP_WEIGHTS_MEMCFG"])
        else:
            ln_mlp_gamma_host = tt_lib.tensor.Tensor(
                gamma2.reshape([1, 1, 1, -1]),
                self.model_config["LN_MLP_WEIGHTS_DTYPE"],
            )
            self.ln_mlp_gamma = ln_mlp_gamma_host.to(device, self.model_config["LN_MLP_WEIGHTS_MEMCFG"])
            tt_lib.tensor.dump_tensor(
                str(ln_mlp_weights_path),
                ln_mlp_gamma_host,
            )

        ln_mlp_bias_path = (
            tt_cache_path / f"{ln_mlp_bias_str}_rm_fusedln_{self.model_config['LN_MLP_BIAS_DTYPE'].name}.bin"
        )
        if (ln_mlp_bias_path).exists():
            ln_mlp_beta_host = tt_lib.tensor.load_tensor(str(ln_mlp_bias_path))
            self.ln_mlp_beta = ln_mlp_beta_host.to(device, self.model_config["LN_MLP_BIAS_MEMCFG"])
        else:
            ln_mlp_beta_host = tt_lib.tensor.Tensor(
                beta2.reshape([1, 1, 1, -1]),
                self.model_config["LN_MLP_BIAS_DTYPE"],
            )
            self.ln_mlp_beta = ln_mlp_beta_host.to(device, self.model_config["LN_MLP_BIAS_MEMCFG"])
            tt_lib.tensor.dump_tensor(
                str(ln_mlp_bias_path),
                ln_mlp_beta_host,
            )

        self.layernorm_eps = config.layer_norm_epsilon

        shard_spec_cores_grid = tt_lib.tensor.CoreRangeSet(
            {
                tt_lib.tensor.CoreRange(
                    tt_lib.tensor.CoreCoord(0, 0),
                    tt_lib.tensor.CoreCoord(7, 7),
                ),
            }
        )
        self.memconfig = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            tt_lib.tensor.BufferType.L1,
            tt_lib.tensor.ShardSpec(
                shard_spec_cores_grid,
                [
                    32,
                    1024,
                ],
                tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

        self.prg_config = tt_lib.operations.primary.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 8],
            subblock_w=8,
            block_h=1,
            block_w=32,
            inplace=False,
        )

        self.interleaved_memconfig = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED,
            tt_lib.tensor.BufferType.L1,
        )

    def __call__(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        # block sharded
        out2 = tt_lib.operations.primary.layernorm(
            x, eps=self.layernorm_eps, output_mem_config=self.memconfig, program_config=self.prg_config
        )
        out2 = tt_lib.tensor.sharded_to_interleaved(out2, output_mem_config=self.interleaved_memconfig)

        out1 = tt_lib.tensor.bcast(
            out2, self.ln_attn_gamma, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.H
        )
        out1 = tt_lib.tensor.bcast(
            out1, self.ln_attn_beta, math_op=tt_lib.tensor.BcastOpMath.ADD, dim=tt_lib.tensor.BcastOpDim.H
        )

        out2 = tt_lib.tensor.bcast(
            out2, self.ln_mlp_gamma, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.H
        )
        out2 = tt_lib.tensor.bcast(
            out2, self.ln_mlp_beta, math_op=tt_lib.tensor.BcastOpMath.ADD, dim=tt_lib.tensor.BcastOpDim.H
        )

        return out1, out2
