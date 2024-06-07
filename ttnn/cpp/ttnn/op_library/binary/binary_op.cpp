// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/op_library/binary/binary_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_eager/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

namespace ttnn {

namespace operations {

namespace binary {
/* static */ void Binary::validate(
    const program_manager_t& program_manager,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& output_tensor = tensor_args.output_tensor;

    const auto& input_shape_a = input_tensor_a.get_shape();
    const auto& input_shape_b = input_tensor_b.get_shape();

    auto batch_size_0_a = input_shape_a.rank() >= 4 ? input_shape_a[-4] : 1;
    auto batch_size_1_a = input_shape_a.rank() >= 3 ? input_shape_a[-3] : 1;
    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];

    auto batch_size_0_b = input_shape_b.rank() >= 4 ? input_shape_b[-4] : 1;
    auto batch_size_1_b = input_shape_b.rank() >= 3 ? input_shape_b[-3] : 1;
    auto height_b = input_shape_b[-2];
    auto width_b = input_shape_b[-1];

    // Input shape b must be the same as or broadcastable to input shape a
    if (batch_size_0_a != batch_size_0_b) {
        TT_ASSERT(
            batch_size_0_a > batch_size_0_b and batch_size_0_b == 1,
            "ttnn::operations::binary::Binary: batch size mismatch");
    }
    if (batch_size_1_a != batch_size_1_b) {
        TT_ASSERT(
            batch_size_1_a > batch_size_1_b and batch_size_1_b == 1,
            "ttnn::operations::binary::Binary: batch size mismatch");
    }
    if (height_a != height_b) {
        TT_ASSERT(height_a > height_b and height_b == 1, "ttnn::operations::binary::Binary: height mismatch");
    }
    if (width_a != width_b) {
        TT_ASSERT(width_a > width_b and width_b == 1, "ttnn::operations::binary::Binary: width mismatch");
    }

    TT_FATAL(
        input_tensor_a.device() == input_tensor_b.device(),
        "Operands to eltwise binary need to be on the same device!");
    TT_FATAL(
        (input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE),
        "Inputs to eltwise binary must be tilized");
    if (attributes.in_place) {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == attributes.memory_config.memory_layout);
        TT_FATAL(input_tensor_a.memory_config().buffer_type == attributes.memory_config.buffer_type);
        TT_FATAL(input_tensor_a.get_dtype() == attributes.dtype);
    }
    if (input_tensor_a.memory_config().is_sharded()) {
        if (input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
            // If we aren't height sharded, we require all sharding schemes to match until we add blocked
            // reader/writers for width and block sharding
            TT_FATAL((input_tensor_b.memory_config().is_sharded()));
            TT_FATAL(input_tensor_a.shard_spec().value().grid.ranges().size() == 1);
        }
        if (input_tensor_b.memory_config().is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().memory_layout == input_tensor_b.memory_config().memory_layout);
            TT_FATAL(input_tensor_a.shard_spec().value() == input_tensor_b.shard_spec().value());
        }
        if (attributes.memory_config.is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().memory_layout == attributes.memory_config.memory_layout);
        } else {
            TT_FATAL(attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    } else if (input_tensor_b.memory_config().is_sharded()) {
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (attributes.memory_config.is_sharded()) {
            TT_FATAL(input_tensor_b.memory_config().memory_layout == attributes.memory_config.memory_layout);
        } else {
            TT_FATAL(attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (attributes.memory_config.is_sharded()) {
            TT_FATAL(attributes.memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
            uint32_t num_blocks = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
            auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
            uint32_t num_cores = core_grid.x * core_grid.y;
            TT_FATAL(num_blocks < num_cores or num_blocks % num_cores == 0);

        } else {
            TT_FATAL(attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    }

    std::visit(
        [&attributes](auto&& program_manager) {
            if constexpr (std::is_same_v<decltype(program_manager), ElementWiseMultiCore>) {
                TT_FATAL(not attributes.activations.has_value());
            }
        },
        program_manager);

    if (output_tensor.has_value()) {
        TT_FATAL(
            not attributes.in_place,
            "Operation is configured as in_place. First input is used as output. Provided output tensor is "
            "ignored");
    }
}

/* static */ Binary::shape_return_t Binary::compute_output_shapes(
    const program_manager_t&, const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    return input_tensor_a.shape();
}

/* static */ Binary::tensor_return_value_t Binary::create_output_tensors(
    const program_manager_t& program_manager,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    auto output_shape = compute_output_shapes(program_manager, operation_attributes, tensor_args);
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& output_tensor = tensor_args.output_tensor;
    if (operation_attributes.in_place) {
        return {input_tensor_a};
    } else {
        if (output_tensor.has_value()) {
            return output_tensor.value();
        }

        if (std::holds_alternative<ElementWiseMultiCore>(program_manager)) {
            if (operation_attributes.memory_config.is_sharded()) {
                ShardSpec shard_spec{CoreRangeSet({}), {0, 0}};
                if (input_tensor_a.memory_config().is_sharded()) {
                    shard_spec = input_tensor_a.shard_spec().value();
                } else if (input_tensor_b.memory_config().is_sharded()) {
                    shard_spec = input_tensor_b.shard_spec().value();
                } else {
                    uint32_t num_blocks = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
                    auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
                    uint32_t num_grid_cores = core_grid.x * core_grid.y;
                    uint32_t target_num_cores = num_blocks < num_grid_cores ? num_blocks : num_grid_cores;
                    shard_spec.grid = num_cores_to_corerange_set(target_num_cores, core_grid, true);
                    shard_spec.shape = {
                        num_blocks / target_num_cores * TILE_HEIGHT, input_tensor_a.get_legacy_shape()[-1]};
                    shard_spec.orientation = ShardOrientation::ROW_MAJOR;
                }
                auto memory_config = operation_attributes.memory_config;
                memory_config.shard_spec = shard_spec;
                return create_device_tensor(
                    output_shape,
                    operation_attributes.dtype,
                    Layout::TILE,
                    input_tensor_a.device(),
                    operation_attributes.memory_config);
            }
        } else {
            if (operation_attributes.memory_config.is_sharded()) {
                ShardSpec shard_spec{CoreRangeSet({}), {0, 0}};
                if (input_tensor_a.memory_config().is_sharded()) {
                    // Derive output shard_spec based on input
                    shard_spec = input_tensor_a.shard_spec().value();
                }
                auto memory_config = operation_attributes.memory_config;
                memory_config.shard_spec = shard_spec;
                return create_device_tensor(
                    output_shape,
                    operation_attributes.dtype,
                    Layout::TILE,
                    input_tensor_a.device(),
                    operation_attributes.memory_config);
            }
        }
        return create_device_tensor(
            output_shape,
            operation_attributes.dtype,
            Layout::TILE,
            input_tensor_a.device(),
            operation_attributes.memory_config);
    }
}

/* static */ Binary::program_manager_t Binary::select_program_manager(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    const auto& input_shape_a = input_tensor_a.get_shape();
    const auto& input_shape_b = input_tensor_b.get_shape();

    auto batch_size_0_a = input_shape_a.rank() >= 4 ? input_shape_a[-4] : 1;
    auto batch_size_1_a = input_shape_a.rank() >= 3 ? input_shape_a[-3] : 1;
    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];

    auto batch_size_0_b = input_shape_b.rank() >= 4 ? input_shape_b[-4] : 1;
    auto batch_size_1_b = input_shape_b.rank() >= 3 ? input_shape_b[-3] : 1;
    auto height_b = input_shape_b[-2];
    auto width_b = input_shape_b[-1];

    if (batch_size_0_a == batch_size_0_b and batch_size_1_a == batch_size_1_b and height_a == height_b and
        width_a == width_b) {
        return ElementWiseMultiCore{};
    } else if (height_b == 1 or width_b == 1) {
        if (operation_attributes.dtype != input_tensor_a.get_dtype()) {
            TT_THROW("ttnn::operations::binary::Binary: cannot change dtype when broadcasting");
        }
        if (height_b == 1 and width_b == 1) {
            // fmt::print("BinaryProgramType::BroadcastHeightAndWidthMultiCore\n");
            return BroadcastHeightAndWidthMultiCore{};
        } else if (height_b == 1) {
            // fmt::print("BinaryProgramType::BroadcastHeightMultiCore\n");
            return BroadcastHeightMultiCore{};
        } else if (width_b == 1) {
            // fmt::print("BinaryProgramType::BroadcastWidthMultiCore\n");
            return BroadcastWidthMultiCore{};
        }
    }
    TT_THROW("ttnn::operations::binary::Binary: unsupported broadcast");
}

/* static */ tt::stl::hash::hash_t Binary::compute_program_hash(
    const program_manager_t& program_manager,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    operation::Hash hash = tt::stl::hash::hash_objects_with_default_seed(
        typeid(Binary).hash_code(),
        attributes,
        std::visit([](auto&& program_manager) { return typeid(program_manager).hash_code(); }, program_manager),
        input_tensor_a.dtype(),
        std::get<DeviceStorage>(input_tensor_a.storage()).memory_config(),
        input_tensor_b.dtype(),
        std::get<DeviceStorage>(input_tensor_b.storage()).memory_config());
    return hash;
}

}  // namespace binary

}  // namespace operations

}  // namespace ttnn
