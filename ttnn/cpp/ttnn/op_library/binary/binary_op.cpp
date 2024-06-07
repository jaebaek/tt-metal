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

enum class BinaryProgramType {
    ElementWiseMultiCore,
    BroadcastWidthMultiCore,
    BroadcastHeightMultiCore,
    BroadcastHeightAndWidthMultiCore,
};

inline BinaryProgramType get_program_type(const Binary& operation, const std::vector<Tensor>& input_tensors) {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

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

    /*
    fmt::print("input_shape_a: {}, input_shape_b: {}\n", input_shape_a, input_shape_b);
    fmt::print(
        "batch_size_0_a: {}, batch_size_1_a: {}, height_a: {}, width_a: {}\n",
        batch_size_0_a,
        batch_size_1_a,
        height_a,
        width_a);
    fmt::print(
        "batch_size_0_b: {}, batch_size_1_b: {}, height_b: {}, width_b: {}\n",
        batch_size_0_b,
        batch_size_1_b,
        height_b,
        width_b);
    */

    if (batch_size_0_a == batch_size_0_b and batch_size_1_a == batch_size_1_b and height_a == height_b and
        width_a == width_b) {
        return BinaryProgramType::ElementWiseMultiCore;
    } else if (height_b == 1 or width_b == 1) {
        if (operation.dtype != input_tensor_a.get_dtype()) {
            TT_THROW("ttnn::operations::binary::Binary: cannot change dtype when broadcasting");
        }
        if (height_b == 1 and width_b == 1) {
            // fmt::print("BinaryProgramType::BroadcastHeightAndWidthMultiCore\n");
            return BinaryProgramType::BroadcastHeightAndWidthMultiCore;
        } else if (height_b == 1) {
            // fmt::print("BinaryProgramType::BroadcastHeightMultiCore\n");
            return BinaryProgramType::BroadcastHeightMultiCore;
        } else if (width_b == 1) {
            // fmt::print("BinaryProgramType::BroadcastWidthMultiCore\n");
            return BinaryProgramType::BroadcastWidthMultiCore;
        }
    }
    TT_THROW("ttnn::operations::binary::Binary: unsupported broadcast");
}

void Binary::validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    auto program_type = get_program_type(*this, input_tensors);

    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

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
    if (this->in_place) {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == this->memory_config.memory_layout);
        TT_FATAL(input_tensor_a.memory_config().buffer_type == this->memory_config.buffer_type);
        TT_FATAL(input_tensor_a.get_dtype() == this->dtype);
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
        if (this->memory_config.is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().memory_layout == this->memory_config.memory_layout);
        } else {
            TT_FATAL(this->memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    } else if (input_tensor_b.memory_config().is_sharded()) {
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (this->memory_config.is_sharded()) {
            TT_FATAL(input_tensor_b.memory_config().memory_layout == this->memory_config.memory_layout);
        } else {
            TT_FATAL(this->memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (this->memory_config.is_sharded()) {
            TT_FATAL(this->memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
            uint32_t num_blocks = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
            auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
            uint32_t num_cores = core_grid.x * core_grid.y;
            TT_FATAL(num_blocks < num_cores or num_blocks % num_cores == 0);

        } else {
            TT_FATAL(this->memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    }

    if (program_type != BinaryProgramType::ElementWiseMultiCore) {
        TT_FATAL(not this->activations.has_value());
    }

    if (!output_tensors.empty()) {
        TT_FATAL(output_tensors.size() == 1, "Must have 1 output tensors");

        if(output_tensors.at(0).has_value()) {
            TT_FATAL(!this->in_place, "Operation is configured as in_place. First input is used as output. Provided output tensor is ignored");
        }
    }
}

std::vector<tt::tt_metal::Shape> Binary::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    if (input_tensor_a.get_shape().rank() >= input_tensor_b.get_shape().rank()) {
        return {input_tensor_a.get_legacy_shape()};
    }
    return {input_tensor_b.get_legacy_shape()};
}

std::vector<Tensor> Binary::create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    if (this->in_place) {
        return {input_tensor_a};
    } else {
        if (!output_tensors.empty() && output_tensors.at(0).has_value()) {
            return {output_tensors.at(0).value()};
        }

        auto program_type = get_program_type(*this, input_tensors);

        if (program_type == BinaryProgramType::ElementWiseMultiCore) {
            if (this->memory_config.is_sharded()) {
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
                auto memory_config = this->memory_config;
                memory_config.shard_spec = shard_spec;
                return {create_device_tensor(
                    this->compute_output_shapes(input_tensors).at(0),
                    this->dtype,
                    Layout::TILE,
                    input_tensor_a.device(),
                    memory_config)};
            }
        } else {
            if (this->memory_config.is_sharded()) {
                ShardSpec shard_spec{CoreRangeSet({}), {0, 0}};
                if (input_tensor_a.memory_config().is_sharded()) {
                    // Derive output shard_spec based on input
                    shard_spec = input_tensor_a.shard_spec().value();
                }
                auto memory_config = this->memory_config;
                memory_config.shard_spec = shard_spec;
                return {create_device_tensor(
                    this->compute_output_shapes(input_tensors).at(0),
                    this->dtype,
                    Layout::TILE,
                    input_tensor_a.device(),
                    memory_config)};
            }
        }
        return operation::generic_create_output_tensors(
            *this, input_tensors, this->dtype, Layout::TILE, this->memory_config);
    }
}

const std::optional<tt::tt_metal::BcastOpMath> binary_op_type_to_bcast_op_math(const BinaryOpType binary_op_type) {
    switch (binary_op_type) {
        case BinaryOpType::ADD: return tt::tt_metal::BcastOpMath::ADD;
        case BinaryOpType::SUB: return tt::tt_metal::BcastOpMath::SUB;
        case BinaryOpType::MUL: return tt::tt_metal::BcastOpMath::MUL;
        default: return std::nullopt;
    }
}

operation::ProgramWithCallbacks Binary::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& output_tensor = output_tensors.at(0);

    std::vector<UnaryWithParam> activations;
    if (this->activations.has_value()) {
        const auto activations_as_strings = this->activations.value();
        std::transform(
            activations_as_strings.begin(),
            activations_as_strings.end(),
            std::back_inserter(activations),
            [](const std::string& activation) { return string_to_unary_with_param(activation); });
    }

    auto program_type = get_program_type(*this, input_tensors);
    auto bcast_op_math = binary_op_type_to_bcast_op_math(this->binary_op_type);
    if (bcast_op_math.has_value()) {
        switch (program_type) {
            case BinaryProgramType::ElementWiseMultiCore:
                return eltwise_binary_multi_core(
                    input_tensor_a, input_tensor_b, output_tensor, this->binary_op_type, activations);
            case BinaryProgramType::BroadcastHeightAndWidthMultiCore:
                return bcast_multi_core_hw(
                    input_tensor_a, input_tensor_b, output_tensor, bcast_op_math.value(), false /* in-place */);
            case BinaryProgramType::BroadcastHeightMultiCore:
                return bcast_multi_core_h(input_tensor_a, input_tensor_b, output_tensor, bcast_op_math.value());
            case BinaryProgramType::BroadcastWidthMultiCore:
                return bcast_multi_core_w(input_tensor_a, input_tensor_b, output_tensor, bcast_op_math.value());
            default: TT_THROW("Invalid program type");
        }
    } else {
        switch (program_type) {
            case BinaryProgramType::ElementWiseMultiCore:
                return eltwise_binary_multi_core(
                    input_tensor_a, input_tensor_b, output_tensor, this->binary_op_type, activations);
            default: TT_THROW("Invalid program type");
        }
    }
}

const operation::Hash Binary::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto program_type = get_program_type(*this, input_tensors);
    operation::Hash hash = tt::stl::hash::hash_objects_with_default_seed(
        typeid(*this).hash_code(),
        this,
        program_type,
        input_tensor_a.dtype(),
        std::get<DeviceStorage>(input_tensor_a.storage()).memory_config(),
        input_tensor_b.dtype(),
        std::get<DeviceStorage>(input_tensor_b.storage()).memory_config());
    return hash;
}

operation::OpPerformanceModel Binary::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    // GS specific parameters
    // 80 B/cycle unpacker BW shared
    // 128 datums per cycle math, but unpacker cant keep up
    constexpr int num_cores = 9 * 12;

    int total_bytes = 0;
    for (const auto& t : input_tensors) {
        total_bytes += t.volume() * t.element_size();
    }
    int ideal_eltwise_cycles = total_bytes / 80 / num_cores;

    operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_eltwise_cycles);
#if 0
        tt::log_info(tt::LogOp, "Binary PerfModel:");
        tt::log_info(tt::LogOp, "\t Data (Bytes): {}", total_bytes);
        tt::log_info(tt::LogOp, "\t ideal_eltwise_cycles: {}", ideal_eltwise_cycles);
#endif
    return result;
}

}  // namespace binary

}  // namespace operations

}  // namespace ttnn
