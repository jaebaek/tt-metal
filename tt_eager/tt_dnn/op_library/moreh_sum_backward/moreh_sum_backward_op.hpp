/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <vector>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/compute_kernel_config.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct MorehSumBackward {
    std::vector<int64_t> dims;
    bool keep_batch_dim;
    MemoryConfig input_grad_mem_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    static constexpr auto attribute_names = std::make_tuple("dims", "input_grad_mem_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->dims), std::cref(this->input_grad_mem_config), std::cref(this->compute_kernel_config));
    }
};

operation::ProgramWithCallbacks moreh_sum_backward_impl(const Tensor &output_grad, const Tensor &input_grad, const std::vector<int64_t> &dims, const bool &keep_batch_dim, const DeviceComputeKernelConfig &compute_kernel_config);

Tensor moreh_sum_backward(
    const Tensor &output_grad,
    const Tensor &input,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim = std::nullopt,
    const bool keep_batch_dim = false,
    const std::optional<const Tensor> input_grad = std::nullopt,
    const MemoryConfig &input_grad_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary

}  // namespace operations

}  // namespace tt
