// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_eager/tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "ttnn/cpp/ttnn/op_library/reduction/reduction_op.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/validation.hpp"

namespace ttnn {

namespace operations {

namespace reduction {

enum class ReduceType {
    Sum,
    Mean,
    Max,
    Min,
    Std,
    Var,
};

template <ReduceType reduce_type>
struct Reduce {
    static const std::array<ttnn::TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2, 4, {ttnn::bfloat8_b, ttnn::bfloat16}, {ttnn::TILE_LAYOUT}, true, false, false, false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor_arg,
        const std::optional<std::variant<int, std::vector<int>>>& dim_arg,
        const bool keepdim,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt) {
        if (not keepdim) {
            TT_THROW("keepdim=False is not supported");
        }

        auto input_shape = input_tensor_arg.get_shape();
        auto rank = input_shape.size();
        auto memory_config = memory_config_arg.value_or(input_tensor_arg.memory_config());

        std::vector<int> dim{};
        if (dim_arg.has_value()) {
            if (not std::holds_alternative<std::vector<int>>(dim_arg.value())) {
                auto dim_as_int = std::get<int>(dim_arg.value());
                dim = std::vector<int>({dim_as_int});
            } else {
                dim = std::get<std::vector<int>>(dim_arg.value());
            }
        } else {
            dim = std::vector<int>(rank);
            for (int i = 0; i < rank; i++) {
                dim[i] = i;
            }
        }

        for (int& axis : dim) {
            if (axis < 0) {
                axis += rank;
            }
            if (axis >= rank) {
                TT_THROW("Invalid dim");
            }
        }
        std::sort(dim.begin(), dim.end());

        std::vector<uint32_t> output_shape;
        std::vector<uint32_t> padded_output_shape;
        for (int axis = 0; axis < input_shape.size(); axis++) {
            if (std::find(dim.begin(), dim.end(), axis) != dim.end()) {
                if (keepdim) {
                    output_shape.push_back(1);
                    padded_output_shape.push_back(axis >= rank - 2 ? ttnn::TILE_SIZE : 1);
                }
            } else {
                output_shape.push_back(input_shape[axis]);
                padded_output_shape.push_back(input_shape[axis]);
            }
        }

        auto input_tensor = ttnn::unsqueeze_to_4D(input_tensor_arg);

        Tensor output_tensor;
        if (!dim_arg.has_value()) {
            if constexpr (reduce_type == ReduceType::Mean) {
                output_tensor = tt::tt_metal::global_mean(input_tensor, memory_config);
            } else if constexpr (reduce_type == ReduceType::Sum) {
                output_tensor = tt::tt_metal::global_sum(input_tensor, memory_config);
            } else if constexpr (reduce_type == ReduceType::Max) {
                output_tensor = tt::tt_metal::global_max(input_tensor, memory_config);
            } else if constexpr (reduce_type == ReduceType::Min) {
                output_tensor = tt::tt_metal::global_min(input_tensor, memory_config);
            } else {
                TT_THROW("Unsupported reduction operation");
            }
        } else {
            tt::tt_metal::ReduceOpDim reduce_op_dim;
            if (dim.size() == 1 and dim[0] == rank - 1) {
                reduce_op_dim = tt::tt_metal::ReduceOpDim::W;
            } else if (dim.size() == 1 and dim[0] == rank - 2) {
                reduce_op_dim = tt::tt_metal::ReduceOpDim::H;
            } else if (dim.size() == 2 and dim[0] == rank - 2 and dim[1] == rank - 1) {
                reduce_op_dim = tt::tt_metal::ReduceOpDim::HW;
            } else {
                TT_THROW("Unsupported dim");
            }

            int reduced_volume = 1;
            for (int axis : dim) {
                reduced_volume *= input_shape[axis];
            }

            if constexpr (reduce_type == ReduceType::Sum) {
                output_tensor = tt::tt_metal::reduce(
                    input_tensor, tt::tt_metal::ReduceOpMath::SUM, reduce_op_dim, 1.0, memory_config);
            } else if constexpr (reduce_type == ReduceType::Mean) {
                output_tensor = tt::tt_metal::reduce(
                    input_tensor, tt::tt_metal::ReduceOpMath::SUM, reduce_op_dim, 1.0 / reduced_volume, memory_config);
            } else if constexpr (reduce_type == ReduceType::Max) {
                output_tensor = tt::tt_metal::reduce(
                    input_tensor, tt::tt_metal::ReduceOpMath::MAX, reduce_op_dim, 1.0, memory_config);
            } else if constexpr (reduce_type == ReduceType::Min) {
                output_tensor = tt::tt_metal::reduce(
                    input_tensor, tt::tt_metal::ReduceOpMath::MIN, reduce_op_dim, 1.0, memory_config);
            } else if constexpr (reduce_type == ReduceType::Var or reduce_type == ReduceType::Std) {
                auto mean_tensor = tt::tt_metal::reduce(
                    input_tensor, tt::tt_metal::ReduceOpMath::SUM, reduce_op_dim, 1.0 / reduced_volume, memory_config);
                auto mean_square_tensor = tt::tt_metal::reduce(
                    tt::tt_metal::pow(input_tensor, 2.0f, memory_config),
                    tt::tt_metal::ReduceOpMath::SUM,
                    reduce_op_dim,
                    1.0 / reduced_volume,
                    memory_config);
                output_tensor = tt::tt_metal::sub(
                    mean_square_tensor,
                    tt::tt_metal::pow(mean_tensor, 2.0f, memory_config),
                    std::nullopt,
                    memory_config);
                if constexpr (reduce_type == ReduceType::Std) {
                    output_tensor = tt::tt_metal::sqrt(output_tensor, memory_config);
                }
            } else {
                TT_THROW("Unsupported reduction operation");
            }
        }

        output_tensor =
            ttnn::reshape(output_tensor, ttnn::Shape{tt::tt_metal::Shape{output_shape, padded_output_shape}});

        return output_tensor;
    }
};

struct ExecuteArgMax {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{4, 4, {ttnn::bfloat16}, {ttnn::ROW_MAJOR_LAYOUT}, true, false, false, false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(uint8_t queue_id, const Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    static ttnn::Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const std::optional<int> dim = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return operation::run(
                   ArgMax{tt::tt_metal::DataType::UINT32, dim, memory_config.value_or(input_tensor.memory_config())},
                   {input_tensor}, {}, {optional_output_tensor}, queue_id)
            .at(0);
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const std::optional<int> dim = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return execute_on_worker_thread(DefaultQueueId, input_tensor, dim, memory_config, optional_output_tensor);
    }
};

}  // namespace reduction
}  // namespace operations

// Generic reductions
constexpr auto sum =
    ttnn::register_operation<ttnn::operations::reduction::Reduce<ttnn::operations::reduction::ReduceType::Sum>>(
        "ttnn::sum");

constexpr auto mean =
    ttnn::register_operation<ttnn::operations::reduction::Reduce<ttnn::operations::reduction::ReduceType::Mean>>(
        "ttnn::mean");

constexpr auto max =
    ttnn::register_operation<ttnn::operations::reduction::Reduce<ttnn::operations::reduction::ReduceType::Max>>(
        "ttnn::max");

constexpr auto min =
    ttnn::register_operation<ttnn::operations::reduction::Reduce<ttnn::operations::reduction::ReduceType::Min>>(
        "ttnn::min");

constexpr auto std =
    ttnn::register_operation<ttnn::operations::reduction::Reduce<ttnn::operations::reduction::ReduceType::Std>>(
        "ttnn::std");

constexpr auto var =
    ttnn::register_operation<ttnn::operations::reduction::Reduce<ttnn::operations::reduction::ReduceType::Var>>(
        "ttnn::var");

// Special reductions
constexpr auto argmax = ttnn::register_operation<ttnn::operations::reduction::ExecuteArgMax>("ttnn::argmax");

}  // namespace ttnn
