// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/ccl_common.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/ccl_host_datastructures.hpp"
#include "tt_eager/tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"

namespace tt {
namespace tt_metal {

struct ReduceScatter {
    const BinaryOpType binary_op_type;
    const uint32_t scatter_dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const std::optional<chip_id_t> receiver_device_id;
    const std::optional<chip_id_t> sender_device_id;
    const MemoryConfig output_mem_config;
    const ccl::Topology topology;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "binary_op_type",
        "scatter_dim",
        "num_links",
        "ring_size",
        "ring_index",
        "receiver_device_id",
        "sender_device_id",
        "output_mem_config",
        "topology");

    const auto attribute_values() const {
        return std::forward_as_tuple(
            this->binary_op_type,
            this->scatter_dim,
            this->num_links,
            this->ring_size,
            this->ring_index,
            this->receiver_device_id,
            this->sender_device_id,
            this->output_mem_config,
            this->topology);
    };
};

std::vector<Tensor> reduce_scatter(
    const std::vector<Tensor> &input_tensors,
    const uint32_t scatter_split_dim,
    ReduceOpMath reduce_op  = ReduceOpMath::SUM,
    const uint32_t num_links = 1,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

namespace ccl {
namespace reduce_scatter_detail {
operation::ProgramWithCallbacks reduce_scatter_with_workers(
    const std::vector<Tensor>& input_tensors,
    const std::vector<Tensor>& output_tensors,
    BinaryOpType reduce_op,
    const uint32_t scatter_split_dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    ccl::Topology topology);
}
}; // namespace ccl

};  // namespace tt_metal
};  // namespace tt
