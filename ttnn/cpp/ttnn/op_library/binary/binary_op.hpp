// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <variant>

#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tensor/host_buffer/functions.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_eager/tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/op_library/binary/element_wise_multi_core.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations {

namespace binary {

using BinaryOpType = tt::tt_metal::BinaryOpType;

constexpr uint8_t DefaultQueueId = 0;

struct Binary {
    struct operation_attributes_t {
        BinaryOpType binary_op_type;
        bool in_place;
        const std::optional<std::vector<std::string>> activations;
        const MemoryConfig memory_config;
        const DataType dtype;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;

        static constexpr auto attribute_names = std::forward_as_tuple(
            "binary_op_type", "in_place", "activations", "memory_config", "dtype", "compute_kernel_config");
        const auto attribute_values() const {
            return std::forward_as_tuple(
                this->binary_op_type,
                this->in_place,
                this->activations,
                this->memory_config,
                this->dtype,
                this->compute_kernel_config);
        }
    };
    struct tensor_args_t {
        const Tensor& input_tensor_a;
        const Tensor& input_tensor_b;
        std::optional<Tensor> output_tensor;

        static constexpr auto attribute_names =
            std::forward_as_tuple("input_tensor_a", "input_tensor_b", "output_tensor");
        const auto attribute_values() const {
            return std::forward_as_tuple(this->input_tensor_a, this->input_tensor_b, this->output_tensor);
        }
    };
    using shape_return_t = ttnn::Shape;
    using tensor_return_value_t = Tensor;

    struct ElementWiseMultiCore {
        static auto create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return) {
            return element_wise_multi_core::create(operation_attributes, tensor_args, tensor_return);
        }
        static void override_runtime_arguments(
            auto& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return) {
            element_wise_multi_core::override_runtime_arguments(
                cached_program, operation_attributes, tensor_args, tensor_return);
        }
    };

    struct BroadcastWidthMultiCore {
        static auto create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
            return device_operation::CachedProgram{tt::tt_metal::Program()};
        }
        static void override_runtime_arguments(
            auto& cached_program,
            const operation_attributes_t&,
            const tensor_args_t&,
            tensor_return_value_t&) {}
    };

    struct BroadcastHeightMultiCore {
        static auto create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
            return device_operation::CachedProgram{tt::tt_metal::Program()};
        }
        static void override_runtime_arguments(
            auto& cached_program,
            const operation_attributes_t&,
            const tensor_args_t&,
            tensor_return_value_t&) {}
    };

    struct BroadcastHeightAndWidthMultiCore {
        static auto create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
            return device_operation::CachedProgram{tt::tt_metal::Program()};
        }
        static void override_runtime_arguments(
            auto& cached_program,
            const operation_attributes_t&,
            const tensor_args_t&,
            tensor_return_value_t&) {}
    };

    using program_manager_t = std::variant<
        ElementWiseMultiCore,
        BroadcastWidthMultiCore,
        BroadcastHeightMultiCore,
        BroadcastHeightAndWidthMultiCore>;

    static program_manager_t select_program_manager(const operation_attributes_t&, const tensor_args_t&);

    static void validate(const program_manager_t&, const operation_attributes_t&, const tensor_args_t&);

    static shape_return_t compute_output_shapes(const program_manager_t&, const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const program_manager_t&, const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(
        const program_manager_t&, const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace binary

}  // namespace operations

}  // namespace ttnn
