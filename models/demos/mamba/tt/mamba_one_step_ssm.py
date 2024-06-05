# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import tt_lib as ttl
from typing import Callable

from models.demos.mamba.reference.args import ModelArgs


class TtMambaSSM(torch.nn.Module):
    def __init__(self, args: ModelArgs, device, configs, load_fn: Callable):
        super().__init__()

        self.device = device
        self.args = args

        # hidden state
        self.batch_size = args.batch_size
        self.hidden_size = args.d_inner
        self.configs = configs
        self.n = 32
        self.rank = self.args.dt_rank

        """
        We need to split up the x_proj weights because in the reference
        implementation they perform the linear operation for dt, B, and C in a
        single step. Here we can't do that because it would involve fallback op
        slicing, so we break up the weights ahead of time and do the linear ops
        separately.
        """

        x_proj_weight_name = "mixer.x_proj.weight"

        # delta_t_proj_weights
        self.delta_t_proj_weights = load_fn(
            x_proj_weight_name,
            lambda x: x[: self.args.dt_rank, :].transpose(-1, -2),
            postfix="delta_t",
            tt_dtype=ttnn.bfloat8_b,
        )

        # B_proj_weights
        def preprocess_B(x):
            x = x[self.args.dt_rank : (self.args.dt_rank + self.args.d_state), :]
            x = x.transpose(-1, -2)
            x = torch.nn.functional.pad(x, (0, 16), "constant", 0)
            return x

        self.B_proj_weights = load_fn(
            x_proj_weight_name,
            tm_fn=preprocess_B,
            postfix="B_proj",
            tt_dtype=ttnn.bfloat8_b,
        )

        # C_proj_weights
        def preprocess_C(x):
            x = x[(self.args.dt_rank + self.args.d_state) :, :].transpose(-1, -2)
            x = torch.nn.functional.pad(x, (0, 16), "constant", 0)
            return x

        self.C_proj_weights = load_fn(x_proj_weight_name, preprocess_C, postfix="C_proj", tt_dtype=ttnn.bfloat8_b)

        # dt_proj_weights
        dt_proj_weight_name = "mixer.dt_proj.weight"
        dt_proj_bias_name = "mixer.dt_proj.bias"
        self.dt_proj_weights = load_fn(dt_proj_weight_name, lambda x: x.transpose(-1, -2), tt_dtype=ttnn.bfloat8_b)
        self.dt_proj_bias = load_fn(dt_proj_bias_name, tt_dtype=ttnn.bfloat8_b)

        A_weight_name = "mixer.A_log"

        def preprocess_A(x):
            x = -torch.exp(x.float())
            # padding with inf
            x = torch.nn.functional.pad(x, (0, 16), "constant", float("-inf"))
            # x = x.reshape(1, self.hidden_size * self.n)  # (1, 2en)
            return x[0, :].repeat(self.batch_size, 1)  # b, n

        self.A = load_fn(A_weight_name, tm_fn=preprocess_A, postfix=f"A_{self.args.batch_size}")

        # D weight
        D_weight_name = "mixer.D"
        self.D = load_fn(
            D_weight_name,
            lambda x: x.repeat(self.args.batch_size, 1),
            postfix=f"D_{self.args.batch_size}",
        )

        # hidden state
        prev_hidden_states = torch.zeros((1, 1, self.batch_size, self.hidden_size * self.n))
        self.tt_hidden_state = load_fn(f"tt_hidden_state_{args.batch_size}", torch_tensor=prev_hidden_states)

        self.compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self.core_grid_row = 5
        self.core_grid_col = 8

    def forward(self, x):
        assert len(x.shape) == 4, "SSM block expects inputs to be rank 4"

        # delta
        delta_t0 = ttnn.linear(
            x,
            self.delta_t_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            use_1d_systolic_array=True,
            core_grid=ttnn.CoreGrid(y=self.core_grid_row, x=self.core_grid_col),
            dtype=self.configs["dtype"]["activations"],
        )

        delta_t1 = ttnn.linear(
            delta_t0,
            self.dt_proj_weights,
            bias=self.dt_proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            use_1d_systolic_array=True,
            core_grid=ttnn.CoreGrid(y=self.core_grid_row, x=self.core_grid_col),
            dtype=self.configs["dtype"]["activations"],
        )
        ttnn.deallocate(delta_t0)

        delta_t2 = ttnn.softplus(
            delta_t1,
            beta=1.0,
            threshold=20.0,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(delta_t1)

        # calculate abar
        abar0 = ttnn.to_memory_config(self.A, memory_config=ttnn.L1_MEMORY_CONFIG)
        abar1 = ttnn.experimental.operations.primary.transformers.ssm_eltwise_mul(
            abar0,
            delta_t2,
            output_mem_config=ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
            ),
            output_dtype=self.configs["dtype"]["activations"],
        )
        ttnn.deallocate(abar0)

        abar2 = ttl.tensor.exp(
            abar1,
            fast_and_approx=True,
            output_mem_config=ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
            ),
        )
        ttnn.deallocate(abar1)

        # multiply abar and hidden_state
        hidden_state0 = ttnn.to_memory_config(self.tt_hidden_state, memory_config=ttnn.L1_MEMORY_CONFIG)
        amulh0 = ttnn.mul(
            abar2, hidden_state0, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=self.configs["dtype"]["activations"]
        )
        ttnn.deallocate(abar2)
        ttnn.deallocate(hidden_state0)

        # B
        B0 = ttnn.linear(
            x,
            self.B_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            use_1d_systolic_array=True,
            core_grid=ttnn.CoreGrid(y=self.core_grid_row, x=self.core_grid_col),
            dtype=self.configs["dtype"]["activations"],
        )

        # bbar
        bbar0 = ttnn.experimental.operations.primary.transformers.ssm_eltwise_mul(
            B0,
            delta_t2,
            output_mem_config=ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
            ),
            output_dtype=self.configs["dtype"]["activations"],
        )
        ttnn.deallocate(delta_t2)
        ttnn.deallocate(B0)

        # bbar * x
        bmulx0 = ttnn.experimental.operations.primary.transformers.ssm_eltwise_mul(
            bbar0,
            x,
            output_mem_config=ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
            ),
            output_dtype=self.configs["dtype"]["activations"],
        )

        # deallocate bbar
        ttnn.deallocate(bbar0)

        # add amulh and bmulx
        hidden_state1 = ttnn.add(
            amulh0, bmulx0, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=self.configs["dtype"]["activations"]
        )
        ttnn.deallocate(self.tt_hidden_state)
        self.tt_hidden_state = ttnn.to_memory_config(hidden_state1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(amulh0)
        ttnn.deallocate(bmulx0)

        # compute C
        C0 = ttnn.linear(
            x,
            self.C_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            use_1d_systolic_array=True,
            core_grid=ttnn.CoreGrid(y=self.core_grid_row, x=self.core_grid_col),
            dtype=self.configs["dtype"]["activations"],
        )  # b,n

        # c * hidden_state
        C1 = ttnn.experimental.operations.primary.transformers.ssm_eltwise_mul(
            C0,
            hidden_state1,
            output_mem_config=ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
            ),
            output_dtype=self.configs["dtype"]["activations"],
        )
        ttnn.deallocate(hidden_state1)
        ttnn.deallocate(C0)

        # Reduction matmul
        C2 = ttnn.experimental.operations.primary.transformers.ssm_1d_sum_reduce(
            C1,
            output_mem_config=ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
            ),
            output_dtype=self.configs["dtype"]["activations"],
        )
        ttnn.deallocate(C1)

        # x * D
        D = ttnn.to_memory_config(self.D, memory_config=ttnn.L1_MEMORY_CONFIG)
        xD = ttnn.mul(x, D, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=self.configs["dtype"]["activations"])
        ttnn.deallocate(x)

        # add xD and x
        output = ttnn.add(xD, C2, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=self.configs["dtype"]["activations"])
        ttnn.deallocate(C2)
        ttnn.deallocate(xD)

        return output
