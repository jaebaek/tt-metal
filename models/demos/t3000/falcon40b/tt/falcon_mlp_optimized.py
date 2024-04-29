# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
import ttnn

from typing import List
from models.utility_functions import torch2tt_tensor
from models.demos.t3000.falcon40b.tt.model_utils import falcon_prefill_matmul


class TtFalconMLP:
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        tt_cache_path,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.hidden_size = hidden_size
        self.model_config = model_config
        self.output = None

        layer_name = f"{base_url}.{layer_num}"

        dense_h_to_4h_str = f"{layer_name}.mlp.dense_h_to_4h.weight"
        dense_4h_to_h_str = f"{layer_name}.mlp.dense_4h_to_h.weight"

        num_devices = len(devices)
        self.dense_h_to_4h_weights = []
        self.dense_4h_to_h_weights = []
        for i in range(num_devices):
            dense_h_to_4h_path = (
                tt_cache_path
                / f"{dense_h_to_4h_str}_optimized_{i}_{num_devices}_{self.model_config['DENSE_H_TO_4H_MM_WEIGHTS_DTYPE'].name}.bin"
            )
            if (dense_h_to_4h_path).exists():
                self.dense_h_to_4h_weights.append(
                    tt_lib.tensor.load_tensor(str(dense_h_to_4h_path)).to(
                        devices[i], self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"]
                    )
                )
            else:
                dense_h_to_4h_weights_host = torch2tt_tensor(
                    torch.transpose(
                        torch.chunk(self.state_dict[dense_h_to_4h_str], num_devices)[i],
                        -2,
                        -1,
                    ),
                    None,
                    tt_memory_config=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"],
                    tt_dtype=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_DTYPE"],
                )
                self.dense_h_to_4h_weights.append(
                    dense_h_to_4h_weights_host.to(devices[i], self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"])
                )
                tt_lib.tensor.dump_tensor(
                    str(dense_h_to_4h_path),
                    dense_h_to_4h_weights_host,
                )
            dense_4h_to_h_path = (
                tt_cache_path
                / f"{dense_4h_to_h_str}_optimized_{i}_{num_devices}_{self.model_config['DENSE_4H_TO_H_MM_WEIGHTS_DTYPE'].name}.bin"
            )
            if (dense_4h_to_h_path).exists():
                self.dense_4h_to_h_weights.append(
                    tt_lib.tensor.load_tensor(str(dense_4h_to_h_path)).to(
                        devices[i], self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"]
                    )
                )
            else:
                # print(f"dense_4h_to_h original shape: {self.state_dict[dense_4h_to_h_str].shape}")
                dense_4h_to_h_weights_host = torch2tt_tensor(
                    torch.transpose(
                        torch.chunk(self.state_dict[dense_4h_to_h_str], num_devices, -1)[
                            i
                        ],  # chunk FF2 weights in the rows
                        -2,
                        -1,
                    ),
                    None,
                    tt_memory_config=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"],
                    tt_dtype=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_DTYPE"],
                )
                self.dense_4h_to_h_weights.append(
                    dense_4h_to_h_weights_host.to(devices[i], self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"])
                )
                tt_lib.tensor.dump_tensor(
                    str(dense_4h_to_h_path),
                    dense_4h_to_h_weights_host,
                )

        # reduction mask to reduce hidden states after FF2
        self.reduction_weights = []
        reduction_weights_torch = (torch.eye(hidden_size, dtype=torch.float32)).repeat(num_devices, 1)
        reduction_weights_torch_chunked = torch.chunk(reduction_weights_torch, num_devices, dim=-1)
        for i in range(num_devices):
            self.reduction_weights.append(
                torch2tt_tensor(
                    reduction_weights_torch_chunked[i],
                    devices[i],
                    tt_memory_config=self.model_config["DRAM_MEMCFG"],
                    tt_dtype=self.model_config["BFP4_DTYPE"],
                )
            )
        self._allocate_output_mlp_tensors()

    def set_model_config(self, model_config):
        self.model_config = model_config

        self._allocate_output_mlp_tensors()

    def __call__(self, x: List[tt_lib.tensor.Tensor], llm_mode: str) -> List[tt_lib.tensor.Tensor]:
        if llm_mode == "prefill":
            return self.fwd_prefill(x)
        elif llm_mode == "decode":
            return self.fwd_decode(x)
        else:
            assert False

    def fwd_decode(self, x: List[tt_lib.tensor.Tensor]) -> List[tt_lib.tensor.Tensor]:  # TODO: fix decode
        hidden_states = []
        for i in range(len(x)):
            hidden_states.append(
                tt_lib.operations.primary.matmul_1d(
                    x[i],
                    self.dense_h_to_4h_weights[i],
                    program_config=self.model_config["DENSE_H_TO_4H_MM_PROGCFG"],
                    output_mem_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                )
            )
            x[i].deallocate(True)
        for i in range(len(hidden_states)):
            hidden_states[i] = tt_lib.tensor.sharded_to_interleaved(
                hidden_states[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            )
        hidden_states = tt_lib.tensor.all_gather(
            hidden_states,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            output_mem_config=self.model_config["DEFAULT_MEMCFG"],
        )
        for i in range(len(hidden_states)):
            hidden_states[i] = tt_lib.tensor.interleaved_to_sharded(
                hidden_states[i], sharded_mem_config=self.model_config["MLP_ALL_GATHER_OUTPUT_MEMCFG"]
            )
        for i in range(len(hidden_states)):
            hidden_states[i] = tt_lib.operations.primary.matmul_1d(
                hidden_states[i],
                self.dense_4h_to_h_weights[i],
                program_config=self.model_config["DENSE_4H_TO_H_MM_PROGCFG"],
                output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
                compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
            )

        # return TT Tensor
        return hidden_states

    def _allocate_output_mlp_tensors(self):
        if self.model_config["LLM_MODE"] == "prefill":
            if self.output is not None:
                for i in range(len(self.devices)):
                    self.output[i].deallocate()

            seq_len = self.model_config["row_height"]

            # prepare output tensor on device
            out_shape = [(1, 1, seq_len, self.dense_4h_to_h_weights[i].shape[-1]) for i in range(len(self.devices))]
            out_tensors = [torch.zeros(out_shape[i]).bfloat16() for i in range(len(self.devices))]

            self.output = [
                ttnn.from_torch(
                    out_tensors[i],
                    ttnn.bfloat8_b,
                    device=self.devices[i],
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=self.model_config["DEFAULT_MEMCFG"],
                )
                for i in range(len(self.devices))
            ]

    def fwd_prefill(self, x: List[tt_lib.tensor.Tensor]) -> List[tt_lib.tensor.Tensor]:
        seq_len = x[0].get_legacy_shape()[2]
        hidden_size = x[0].get_legacy_shape()[3]
        num_slices = self.model_config["MLP_NUM_SLICES"]
        for slice_idx in range(num_slices):
            x_slices = []
            for i in range(len(x)):
                x_slices.append(
                    tt_lib.tensor.interleaved_to_sharded_partial(
                        x[i],
                        (8, 8),
                        [seq_len // num_slices // 8, hidden_size // 8],
                        num_slices,
                        slice_idx,
                        tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                        tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                    )
                )
            hidden_states_slice = []
            for i in range(len(x)):
                hidden_states_slice.append(
                    falcon_prefill_matmul(
                        x_slices[i],
                        self.dense_h_to_4h_weights[i],
                        self.model_config["COMPUTE_KERNEL_CONFIG"],
                        output_mem_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
                        output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
                        act=[tt_lib.tensor.FusibleActivation.GELU, True],
                        overwrite_subblock_w=1,  # Workaround for non deterministic output/hang; issue: 7066
                        overwrite_subblock_h=1,
                    )
                )
                x_slices[i].deallocate(True)
            for i in range(len(hidden_states_slice)):
                hidden_states_slice[i] = falcon_prefill_matmul(
                    hidden_states_slice[i],
                    self.dense_4h_to_h_weights[i],
                    self.model_config["COMPUTE_KERNEL_CONFIG"],
                    output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
                    overwrite_subblock_w=1,  # Workaround for non deterministic output/hang; issue: 7066
                    overwrite_subblock_h=1,
                )

            for i in range(len(hidden_states_slice)):
                tt_lib.tensor.sharded_to_interleaved_partial(
                    hidden_states_slice[i],
                    self.output[i],
                    num_slices,
                    slice_idx,
                    self.model_config["DEFAULT_MEMCFG"],
                )
                hidden_states_slice[i].deallocate()

        # Deallocate input
        for i in range(len(self.devices)):
            x[i].deallocate()

        # Manual reduce scatter by AllGather and matmul reduce
        hidden_states = tt_lib.tensor.all_gather(
            self.output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            output_mem_config=self.model_config["DEFAULT_MEMCFG"],
        )

        for i in range(len(hidden_states)):
            hidden_states[i] = falcon_prefill_matmul(
                hidden_states[i],
                self.reduction_weights[i],
                self.model_config["COMPUTE_KERNEL_CONFIG"],
                output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
                overwrite_subblock_w=1,  # Workaround for non deterministic output/hang; issue: 7066
                overwrite_subblock_h=1,
            )

        # return TT Tensor
        return hidden_states
