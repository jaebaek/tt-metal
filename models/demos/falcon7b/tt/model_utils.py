# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
import tt_lib
from models.utility_functions import nearest_y, pad_by_zero, torch2tt_tensor


def get_weights_cached(
    devices,
    model_config,
    tt_cache_path,
    weight_cache_str,
    weight_config_str,
    weights_to_cache,
    overwrite=False,
    padzero=False,
    custom_output_shape=None,
):
    """Load cached weights and duplicate per device. Store if not cached."""
    if custom_output_shape is not None:
        custom_output_shape_str = f"{custom_output_shape[-2]}"
    else:
        custom_output_shape_str = ""

    weights_posix_path = (
        tt_cache_path
        / f"{weight_cache_str}_{model_config[f'{weight_config_str}_DTYPE'].name}_{custom_output_shape_str}.bin"
    )

    if not overwrite and weights_posix_path.exists():
        # Load cached weights
        weights_host = tt_lib.tensor.load_tensor(str(weights_posix_path))
        # Duplicate weights on all devices
        weights = [weights_host.to(device, model_config[f"{weight_config_str}_MEMCFG"]) for device in devices]
    else:
        # Duplicate weights on all devices
        if custom_output_shape:
            # pad torch tensor weights_to_cache for optimal matmul performance
            # padding is inversed for torch tensors from last to first dim
            padding = (
                0,
                custom_output_shape[-1] - weights_to_cache.shape[-1],
                0,
                custom_output_shape[-2] - weights_to_cache.shape[-2],
            )
            weights_to_cache = torch.functional.F.pad(weights_to_cache, padding, "constant", 0.0)

        if padzero:
            weights = [
                pad_by_zero(
                    weights_to_cache,
                    device,
                    tt_memory_config=model_config[f"{weight_config_str}_MEMCFG"],
                    tt_dtype=model_config[f"{weight_config_str}_DTYPE"],
                )[0]
                for device in devices
            ]
        else:
            weights = [
                torch2tt_tensor(
                    weights_to_cache,
                    device,
                    tt_memory_config=model_config[f"{weight_config_str}_MEMCFG"],
                    tt_dtype=model_config[f"{weight_config_str}_DTYPE"],
                )
                for device in devices
            ]
        # Store weights (from first device)
        tt_lib.tensor.dump_tensor(str(weights_posix_path), weights[0].cpu())
    return weights


# this function takes activations that are assumed to be non-padded and weights that are assumed to be padded
# (since they are pushed to device at the moment of model initialization);
# it also takes in number of slices since this should be determined at the moment of pushing weights,
# but in general with 512 < seq_len <= 1024 we should use 4 slices, with 1024 < seq_len <= 2048 we should use 8 slices
def falcon_lm_head_matmul_2d(
    hidden_states: tt_lib.tensor.Tensor,
    weights: List[tt_lib.tensor.Tensor],
    num_slices: int,
    in0_mem_config: tt_lib.tensor.MemoryConfig,
    in0_dtype: tt_lib.tensor.DataType,
    out_mem_config: tt_lib.tensor.MemoryConfig,
    out_dtype: tt_lib.tensor.DataType,
):
    seq_len = hidden_states.get_legacy_shape()[-2]

    assert seq_len % 32 == 0, f"Sequence length must be a multiple of 32, instead it is {seq_len}"
    assert seq_len > 512, f"Falcon lm head 2d is only supported for sequence length > 512, instead it is {seq_len}"
    assert seq_len <= 2048, f"Falcon lm head 2d is only supported for sequence length <= 2048, instead it is {seq_len}"

    assert (
        len(weights) == num_slices
    ), f"Weights are expected to be split into {num_slices} slices, instead there are {len(weights)}"
    weights_inner_dim_in_tiles = weights[0].get_legacy_shape()[-2] // 32
    assert (
        weights_inner_dim_in_tiles == 144
    ), f"Weights are expected to be padded to the inner dim 144 in tiles, instead they are {weights_inner_dim_in_tiles}"

    # pad activations to inner dim 144
    padding = torch.zeros([1, 1, seq_len, 64])
    padding_t = (
        tt_lib.tensor.Tensor(padding, in0_dtype)
        .to(tt_lib.tensor.Layout.TILE)
        .to(hidden_states.device(), in0_mem_config)
    )
    hidden_states = tt_lib.tensor.concat([hidden_states, padding_t], -1)

    compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
        math_fidelity=tt_lib.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid = (8, 8)
    activations_m_in_tiles = seq_len // 32
    weights_n_in_tiles = weights[0].get_legacy_shape()[-1] // 32

    # calculate parameters for the given sequence length
    out_subblock_h = 2
    out_subblock_w = 4
    per_core_M = nearest_y(activations_m_in_tiles / grid[1], out_subblock_h)
    per_core_N = nearest_y(weights_n_in_tiles / grid[0], out_subblock_w)
    in0_block_w = num_slices

    program_config = tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )

    out_slices = []
    for i in range(num_slices):
        out_slices.append(
            tt_lib.operations.primary.matmul(
                hidden_states,
                weights[i],
                program_config=program_config,
                output_mem_config=out_mem_config,
                output_dtype=out_dtype,
                compute_kernel_config=compute_kernel_config,
            )
        )

    out = tt_lib.tensor.concat(out_slices, -1)
    for i in range(num_slices):
        out_slices[i].deallocate(True)

    return out
