/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_dnn/op_library/operation.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

////////////////////////////////////////////////////////////////////////////
//                         MorehBiasAddBackward
////////////////////////////////////////////////////////////////////////////
// TODO: Move bias backward code
operation::ProgramWithCallbacks moreh_bias_backward_multi_core_h(const Tensor &output_grad, const Tensor &bias_grad, const DeviceComputeKernelConfig &compute_kernel_config);

operation::ProgramWithCallbacks moreh_bias_backward_single_core_hw(const Tensor &output_grad, const Tensor &bias_grad, const DeviceComputeKernelConfig &compute_kernel_config);

struct MorehBiasAddBackward {
    MemoryConfig bias_grad_mem_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    static constexpr auto attribute_names = std::make_tuple("bias_grad_mem_config", "compute_kernel_config");
    const auto attribute_values() const { return std::make_tuple(std::cref(this->bias_grad_mem_config), std::cref(this->compute_kernel_config)); }
};

std::vector<std::optional<Tensor>> moreh_linear_backward(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &weight,
    const std::vector<bool> &are_required_outputs,
    std::optional<const Tensor> bias = std::nullopt,
    std::optional<const Tensor> input_grad = std::nullopt,
    std::optional<const Tensor> weight_grad = std::nullopt,
    std::optional<const Tensor> bias_grad = std::nullopt,
    const MemoryConfig &input_grad_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const MemoryConfig &weight_grad_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const MemoryConfig &bias_grad_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary
}  // namespace operations
}  // namespace tt
