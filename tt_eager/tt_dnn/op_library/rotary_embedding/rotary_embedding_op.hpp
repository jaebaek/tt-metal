// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/compute_kernel_config.hpp"

namespace tt {

namespace tt_metal {

enum class RotaryEmbeddingOpParallelizationStrategy { MULTI_CORE = 0, SINGLE_CORE = 1 };

operation::ProgramWithCallbacks rotary_embedding_single_core(
    const Tensor &input, const Tensor &cos, const Tensor &sin, Tensor &output, std::optional<uint32_t> token_idx, DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks rotary_embedding_multi_core(
    const Tensor &input, const Tensor &cos, const Tensor &sin, Tensor &output, std::optional<uint32_t> token_idx, DeviceComputeKernelConfig compute_kernel_config);

struct RotaryEmbedding {
    const uint32_t seq_len;
    std::optional<uint32_t> token_idx;
    const MemoryConfig output_mem_config;
    const DeviceComputeKernelConfig compute_kernel_config;

    RotaryEmbeddingOpParallelizationStrategy get_parallelization_strategy(
        const std::vector<Tensor> &input_tensors) const;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;

    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

inline Tensor rotary_embedding(
    const Tensor &input_tensor,
    const Tensor &cos,
    const Tensor &sin,
    std::optional<uint32_t> token_idx = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor, cos, sin}))};
    operation::launch_op(
        [token_idx, output_mem_config, compute_kernel_config](
            std::vector<Tensor> input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors) mutable -> std::vector<Tensor> {
            auto& input_tensor = input_tensors.at(0);
            auto& cos = input_tensors.at(1);
            auto& sin = input_tensors.at(2);
            TT_FATAL(input_tensor.get_legacy_shape()[-1] % (TILE_WIDTH * 2) == 0, "Input X dim must be divisible into tiles");
            uint32_t seq_len = input_tensor.get_legacy_shape()[-2];
            uint32_t B = input_tensor.get_legacy_shape()[0];
            uint32_t X = input_tensor.get_legacy_shape()[-1];
            TT_FATAL(cos.get_legacy_shape() == sin.get_legacy_shape(), "Cos and Sin dims must match");
            TT_FATAL(cos.get_legacy_shape()[0] == 1 && cos.get_legacy_shape()[1] == 1 && cos.get_legacy_shape()[-1] == X, "Cos dims must match input dims");
            if (token_idx.has_value()) {
                seq_len = input_tensor.get_legacy_shape()[0];
                TT_FATAL(seq_len == 1);
                TT_FATAL(cos.get_legacy_shape()[-2] >= token_idx, "Cos dims must match input dims");
            } else {
                TT_FATAL(cos.get_legacy_shape()[-2] >= seq_len, "Cos dims must match input dims");
            }

            auto arch = input_tensor.device()->arch();
            auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);

            return operation::run(
                RotaryEmbedding{seq_len, token_idx, output_mem_config, kernel_config_val}, {input_tensor, cos, sin});
        },
        {input_tensor, cos, sin},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace tt_metal

}  // namespace tt
