// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_op.hpp"
#include "tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"

namespace tt {

using namespace constants;

namespace operations {
namespace primary {

namespace {

inline void check_tensor(const Tensor &tensor, const std::string &op_name) {
    TT_FATAL(tensor.get_layout() == Layout::TILE, "{} only supports tiled layout.", op_name);
    TT_FATAL(tensor.get_dtype() == DataType::BFLOAT16, "{} only supports bfloat16.", op_name);
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE, "Operands to {} need to be on device!", op_name);
    TT_FATAL(
        tensor.buffer() != nullptr, "Operands to {} need to be allocated in buffers on device!", op_name);
}

inline void check_tensor(std::optional<Tensor> tensor, const std::string &op_name) {
    if (!tensor.has_value()) {
        return;
    }
    check_tensor(tensor.value(), op_name);
}

}  // namespace

////////////////////////////////////////////////////////////////////////////
//                         MorehSumBackward
////////////////////////////////////////////////////////////////////////////
void MorehSumBackward::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    const auto &output_grad = input_tensors.at(0);
    const auto &input = input_tensors.at(1);
    auto &input_grad = output_tensors.at(0);

    // validate tensor
    check_tensor(output_grad, "moreh_sum_backward output_grad");
    check_tensor(input, "moreh_sum_backward input");
    check_tensor(input_grad, "moreh_sum_backward input_grad");

    const auto &input_shape = input.get_legacy_shape();
    auto input_shape_wo_padding = input_shape.without_padding();
    auto input_rank = input_shape.rank();
    auto output_grad_shape_wo_padding = output_grad.get_legacy_shape().without_padding();

    // validate output_grad shape
    if (this->keep_batch_dim) {
        for (int i = 0; i < input_rank; ++i) {
            TT_FATAL(input_shape_wo_padding[i] >= output_grad_shape_wo_padding[i]);
        }
    } else {
        std::vector<uint32_t> expected_output_grad_shape;
        std::vector<uint32_t> reduced_dims(input_rank, 0);
        for (auto dim : this->dims) {
            TT_FATAL(dim < input_rank, "dim {} < input_rank {}", dim, input_rank);
            reduced_dims[dim] = 1;
        }

        TT_FATAL(input_rank >= 2, "at least input_rank {} >= 2", input_rank);
        for (int i = 0; i < input_rank; ++i) {
            log_debug(LogOp, "reduced_dims[{}] = {}", i, reduced_dims[i]);
            bool is_tile_dim = (i == input_rank - 1 || i == input_rank -2);
            // batch dims
            if (reduced_dims[i] && !is_tile_dim)
                continue;
            uint32_t s = input_shape_wo_padding[i];
            // tile dims are not reduced
            if (reduced_dims[i] && is_tile_dim) {
                s = 1;
            }
            expected_output_grad_shape.push_back(s);
        }

        uint32_t expected_rank = expected_output_grad_shape.size();
        uint32_t rank = output_grad_shape_wo_padding.rank();
        TT_FATAL(expected_rank == rank, "expected_rank {} == rank {}", expected_rank, rank);
        for (int i = 0; i < rank; ++i) {
            TT_FATAL(expected_output_grad_shape[i] >= output_grad_shape_wo_padding[i]);
            log_debug(LogOp, "rank {} expected_output_grad_shape {}, output_grad_shape_wo_padding {}", i, expected_output_grad_shape[i], output_grad_shape_wo_padding[i]);
        }
    }

    // validate input_grad shape
    if (input_grad.has_value()) {
        const auto &input_grad_shape = input_grad.value().get_legacy_shape();
        TT_FATAL(input_shape == input_grad_shape, "both shape between input and input_grad should be the same");
    }
}

std::vector<Shape> MorehSumBackward::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    return {input_tensors.at(1).get_legacy_shape()};
}

std::vector<Tensor> MorehSumBackward::create_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(1).get_dtype(), Layout::TILE, this->input_grad_mem_config);
}

operation::ProgramWithCallbacks MorehSumBackward::create_program(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) const {
    auto &output_grad = inputs.at(0);
    auto &input_grad = outputs.at(0);

    return moreh_sum_backward_impl(output_grad, input_grad, this->dims, this->keep_batch_dim, this->compute_kernel_config);
}

Tensor moreh_sum_backward(
    const Tensor &output_grad,
    const Tensor &input,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keep_batch_dim,
    const std::optional<const Tensor> input_grad,
    const MemoryConfig &input_grad_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {

    std::vector<int64_t> dims = get_dim(dim, input.get_legacy_shape().rank());
    std::sort(dims.begin(), dims.end());

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({output_grad, input}))};
    auto kernel_config_val = init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4);
    operation::launch_op(
        [dims, keep_batch_dim, input_grad_mem_config, kernel_config_val](
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehSumBackward{.dims = dims, .keep_batch_dim=keep_batch_dim, .input_grad_mem_config = input_grad_mem_config, .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {output_grad, input},
        output_tensors,
        {},
        {input_grad});

    return output_tensors.at(0);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
