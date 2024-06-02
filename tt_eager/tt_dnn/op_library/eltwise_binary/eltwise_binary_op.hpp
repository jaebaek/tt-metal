// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/repeat/repeat_op.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

enum class BinaryOpType {

    ADD,
    SUB,
    MUL,
    GT,
    LT,
    LTE,
    GTE,
    EQ,
    NE,
    SQUARED_DIFFERENCE,
    BIAS_GELU,
    LOGADDEXP,
    LOGICAL_AND,
    LOGICAL_OR,
    LDEXP,
    LOGADDEXP2,
    DIV_FAST
};

enum class BinaryOpParallelizationStrategy { MULTI_CORE };

operation::ProgramWithCallbacks eltwise_binary_multi_core(
    const Tensor &a,
    const Tensor &b,
    const Tensor &output_tensor,
    BinaryOpType op_type,
    const std::optional<std::vector<UnaryWithParam>> fused_activations);

struct EltwiseBinary {
    const BinaryOpType op_type;
    const std::optional<std::vector<UnaryWithParam>> fused_activations;
    const std::optional< MemoryConfig> output_mem_config;
    const DataType output_dtype;
    const bool in_place;

    BinaryOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    void validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::OpPerformanceModel create_op_performance_model(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const {
        // GS specific parameters
        // 80 B/cycle unpacker BW shared
        // 128 datums per cycle math, but unpacker cant keep up
        constexpr int num_cores = 9 * 12;

        int total_bytes = 0;
        for (const auto &t : input_tensors) {
            total_bytes += t.volume() * t.element_size();
        }
        int ideal_eltwise_cycles = total_bytes / 80 / num_cores;

        operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_eltwise_cycles);
#if 0
        tt::log_info(tt::LogOp, "Eltwise PerfModel:");
        tt::log_info(tt::LogOp, "\t Data (Bytes): {}", total_bytes);
        tt::log_info(tt::LogOp, "\t ideal_eltwise_cycles: {}", ideal_eltwise_cycles);
#endif
        return result;
    }
    static constexpr auto attribute_names =
        std::make_tuple("op_type", "fused_activations", "output_mem_config", "output_dtype", "in_place");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->op_type),
            std::cref(this->fused_activations),
            std::cref(this->output_mem_config),
            std::cref(this->output_dtype),
            std::cref(this->in_place));
    }

    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

template <BinaryOpType binary_op_type>
struct make_eltwise_binary {
    Tensor operator()(
        const Tensor &input_tensor_a,
        const Tensor &input_tensor_b,
        std::optional<std::vector<UnaryWithParam>> fused_activations = std::nullopt,
        std::optional<MemoryConfig> output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        std::optional<const DataType> output_dtype = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt) const {
        if(output_tensor.has_value()){
            output_mem_config = output_tensor.value().memory_config();
        }
        std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor_a, input_tensor_b}))};

        operation::launch_op(
            [fused_activations, output_mem_config, output_dtype, output_tensor] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                Tensor in_a = input_tensors.at(0);
                Tensor in_b = input_tensors.at(1);
                Shape shape_a = in_a.get_legacy_shape();
                Shape shape_b = in_b.get_legacy_shape();
                if (shape_a[0] != shape_b[0])
                {
                    if (shape_a[0] > shape_b[0])
                    {
                        Shape shape ({shape_a[0],1,1,1});
                        in_b = repeat(in_b, shape, operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
                    }
                    else
                    {
                        Shape shape ({shape_b[0],1,1,1});
                        in_a = repeat(in_a, shape, operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
                    }
                }
                TT_FATAL(
                    (in_a.get_legacy_shape() == in_b.get_legacy_shape()) or
                    (in_a.get_legacy_shape().without_padding() == in_b.get_legacy_shape().without_padding()),
                    "Input shapes must be the same!");
                return operation::run(
                        EltwiseBinary{
                            binary_op_type,
                            fused_activations,
                            output_mem_config.value(),
                            output_dtype.value_or(in_a.get_dtype()),
                            false},
                        {in_a, in_b}, {}, {output_tensor});
            },
        {input_tensor_a, input_tensor_b}, output_tensors, {}, {output_tensor});
        return output_tensors.at(0);
    }
};

// arithmetic binary ops
constexpr auto add = make_eltwise_binary<BinaryOpType::ADD>{};
constexpr auto sub = make_eltwise_binary<BinaryOpType::SUB>{};
constexpr auto mul = make_eltwise_binary<BinaryOpType::MUL>{};
constexpr auto squared_difference = make_eltwise_binary<BinaryOpType::SQUARED_DIFFERENCE>{};
constexpr auto bias_gelu = make_eltwise_binary<BinaryOpType::BIAS_GELU>{};
constexpr auto logaddexp = make_eltwise_binary<BinaryOpType::LOGADDEXP>{};
constexpr auto ldexp = make_eltwise_binary<BinaryOpType::LDEXP>{};
constexpr auto logaddexp2 = make_eltwise_binary<BinaryOpType::LOGADDEXP2>{};
constexpr auto div_fast = make_eltwise_binary<BinaryOpType::DIV_FAST>{};

// comparative binary ops
constexpr auto lt = make_eltwise_binary<BinaryOpType::LT>{};
constexpr auto gt = make_eltwise_binary<BinaryOpType::GT>{};
constexpr auto lte = make_eltwise_binary<BinaryOpType::LTE>{};
constexpr auto gte = make_eltwise_binary<BinaryOpType::GTE>{};
constexpr auto eq = make_eltwise_binary<BinaryOpType::EQ>{};
constexpr auto ne = make_eltwise_binary<BinaryOpType::NE>{};

// logical ops
constexpr auto logical_and = make_eltwise_binary<BinaryOpType::LOGICAL_AND>{};
constexpr auto logical_or = make_eltwise_binary<BinaryOpType::LOGICAL_OR>{};
}  // namespace tt_metal

namespace operations {

namespace primary {
// TODO: in_place should not take output args
inline Tensor add(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<std::vector<UnaryWithParam>> fused_activations = std::nullopt,
    std::optional< MemoryConfig> output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt,
    bool in_place = false,
    std::optional<Tensor> output_tensor = std::nullopt) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor_a, input_tensor_b}))};
    if(output_tensor.has_value()){
        output_mem_config = output_tensor.value().memory_config();
    }
    operation::launch_op(
        [fused_activations, output_mem_config, output_dtype, in_place, output_tensor] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto& input_tensor_a = input_tensors.at(0);
            auto& input_tensor_b = input_tensors.at(1);

            Shape shape_a = input_tensor_a.get_legacy_shape();
            Shape shape_b = input_tensor_b.get_legacy_shape();
            Tensor in_a = input_tensor_a;
            Tensor in_b = input_tensor_b;
            if (shape_a[0] != shape_b[0])
            {
                if (shape_a[0] > shape_b[0])
                {
                    Shape shape ({shape_a[0],1,1,1});
                    in_b = repeat(input_tensor_b, shape, operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
                }
                else
                {
                    Shape shape ({shape_b[0],1,1,1});
                    in_a = repeat(input_tensor_a, shape, operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
                }
            }
            TT_FATAL(
                (input_tensor_a.get_legacy_shape() == input_tensor_b.get_legacy_shape()) or
                (input_tensor_a.get_legacy_shape().without_padding() == input_tensor_b.get_legacy_shape().without_padding()),
                "Input shapes must be the same!");
            auto add_result = operation::run(
                EltwiseBinary{
                    BinaryOpType::ADD,
                    fused_activations,
                    output_mem_config.value(),
                    output_dtype.value_or(in_a.get_dtype()),
                    in_place},
                {in_a, in_b}, {}, {output_tensor});
            if (in_place) {
                return {in_a};
            }
            return add_result;
        }, {input_tensor_a, input_tensor_b}, output_tensors, {}, {output_tensor});

        return output_tensors.at(0);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt

namespace eltwise_binary_op_utils {
using namespace tt::tt_metal;

std::map<string, string> get_defines(
    BinaryOpType op_typee, const std::optional<std::vector<UnaryWithParam>> fused_activations);

}  // namespace eltwise_binary_op_utils
