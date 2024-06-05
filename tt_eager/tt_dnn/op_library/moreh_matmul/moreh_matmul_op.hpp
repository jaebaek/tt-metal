/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

#include <optional>

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

void get_tensor_dim(std::vector<uint32_t> &dim, const Shape& shape);
std::vector<int64_t> find_reduce_dim(const Shape& a_shape, const Shape& b_shape);
bool is_same_batch_dim(const Tensor &tensor_a, const Tensor &tensor_b);

operation::ProgramWithCallbacks moreh_matmul_multi_core(
    const Tensor &input,
    const Tensor &other,
    const Tensor &output,
    const std::optional<const Tensor> &bias,
    bool transpose_input,
    bool transpose_other,
    const DeviceComputeKernelConfig &compute_kernel_config);

struct MorehMatmul {
    const MemoryConfig output_mem_config;
    bool transpose_input;
    bool transpose_other;
    const DeviceComputeKernelConfig compute_kernel_config;
    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
    static constexpr auto attribute_names = std::make_tuple("transpose_input", "transpose_other", "output_mem_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->transpose_input), std::cref(this->transpose_other), std::cref(this->output_mem_config), std::cref(this->compute_kernel_config));
    }
};

Tensor moreh_matmul(
    const Tensor &input,
    const Tensor &other,
    bool transpose_input = false,
    bool transpose_other = false,
    const std::optional<const Tensor> output = std::nullopt,
    const std::optional<const Tensor> bias = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary

}  // namespace operations

}  // namespace tt
