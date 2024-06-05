// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <variant>
#include <vector>
#include <tuple>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

inline
std::tuple<uint32_t, uint32_t, uint32_t> extract_spatial_dims(const Shape& shape) {
    const auto rank = shape.rank();

    TT_FATAL(rank >= 2, "Shape must have at least two dims.");
    uint32_t W = shape[-1];
    uint32_t H = shape[-2];

    uint32_t other_dims_product = 1;
    for (auto i = 0; i < rank - 2; ++i) {
        other_dims_product *= shape[i];
    }

    return { W, H, other_dims_product};
}

inline void initialize_dims_with_range(std::vector<int64_t>& dims, uint32_t input_rank) {
    dims.resize(input_rank);
    std::iota(dims.begin(), dims.end(), 0);
}

inline std::vector<int64_t> get_dim(
    const std::optional<std::variant<int64_t, std::vector<int64_t>>>& dim,
    uint32_t input_rank
) {
    std::vector<int64_t> dims;
    if (!dim.has_value()) {
        initialize_dims_with_range(dims, input_rank);
    }
    else if (std::holds_alternative<int64_t>(dim.value())) {
        auto d = std::get<int64_t>(dim.value());
        dims.push_back(d);
    }
    else {
        dims = std::get<std::vector<int64_t>>(dim.value());
        if (dims.empty()) {
            initialize_dims_with_range(dims, input_rank);
        }
    }
    return dims;
}


struct MorehSum {
    int64_t dim;
    bool keep_batch_dim;
    MemoryConfig output_mem_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) const;
    stl::reflection::Attributes attributes() const;
    static constexpr auto attribute_names = std::make_tuple("dim", "output_mem_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->dim), std::cref(this->output_mem_config), std::cref(this->compute_kernel_config));
    }
};

operation::ProgramWithCallbacks moreh_sum_nc_impl(const Tensor &input, const Tensor &output, int64_t dim, const DeviceComputeKernelConfig &compute_kernel_config);
// revised from reduce_op
operation::ProgramWithCallbacks moreh_sum_w_impl(const Tensor &a, const Tensor &output, const DeviceComputeKernelConfig &compute_kernel_config);
operation::ProgramWithCallbacks moreh_sum_h_impl(const Tensor &a, const Tensor &output, const DeviceComputeKernelConfig &compute_kernel_config);

Tensor moreh_sum(
    const Tensor &input,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim = std::nullopt,
    const bool keep_batch_dim = false,
    const std::optional<const Tensor> output = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary

}  // namespace operations

}  // namespace tt
