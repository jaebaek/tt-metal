// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt::tt_metal {

enum class ScanOpParallelizationStrategy { SHARDED_MULTI_CORE = 0 };

enum class ScanOpDirection { ROWS, COLS, ROWS_REVERSED, COLS_REVERSED };

struct ScanBase {
    void validate(const std::vector<Tensor> &input_tensors) const;

    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
        return {};  // In-place
    }

    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const {
        return {};  // In-place
    }

    ScanOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
        return ScanOpParallelizationStrategy::SHARDED_MULTI_CORE;
    }
};

struct Scan : ScanBase {
    ScanOpDirection direction = ScanOpDirection::COLS;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("direction");

    const auto attribute_values() const { return std::make_tuple(direction); }
};

struct RetileToRowMajor : ScanBase {
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple();

    const auto attribute_values() const { return std::make_tuple(); }
};

struct UndoRetileToRowMajor : ScanBase {
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple();

    const auto attribute_values() const { return std::make_tuple(); }
};

struct ScanOnly : ScanBase {
    ScanOpDirection direction = ScanOpDirection::COLS;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("direction");

    const auto attribute_values() const { return std::make_tuple(direction); }
};

struct ScanCommunicate : ScanBase {
    ScanOpDirection direction = ScanOpDirection::COLS;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("direction");

    const auto attribute_values() const { return std::make_tuple(direction); }
};

Tensor scan(Tensor &a);

Tensor retile_to_row_major(Tensor &a);

Tensor undo_retile_to_row_major(Tensor &a);

Tensor scan_only(Tensor &a);

Tensor scan_communicate(Tensor &a);

}  // namespace tt::tt_metal
