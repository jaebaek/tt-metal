// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "ttnn/operations/core.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks layernorm_part1_multi_core(
    const Tensor &a,
    Tensor& output,
    LayerNormType norm_type,
    DeviceComputeKernelConfig compute_kernel_config);



struct LayerNormPart1 {
    LayerNormType norm_type;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

}  // namespace metal

namespace operations {

namespace primary {

template <LayerNormType layernorm_type>
struct make_layernorm_part1 {
    Tensor operator()(
        const Tensor& a,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) const {
        std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({a}))};
        log_debug("layernorm_part1: before launch_op");
        operation::launch_op(
            [compute_kernel_config] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                const auto& a = input_tensors.at(0);
                auto arch = a.storage_type() == StorageType::DEVICE ? a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
                auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, false, false, false);
                return operation::run(
                        LayerNormPart1{
                            .norm_type = layernorm_type,
                            .compute_kernel_config = kernel_config_val},
                        {a});
            }, {a}, output_tensors);
        return output_tensors.at(0);
    }
};

constexpr auto layernorm_part1 = make_layernorm_part1<LayerNormType::LAYERNORM>{};
constexpr auto rmsnorm_part1 = make_layernorm_part1<LayerNormType::RMSNORM>{};


}  // namespace primary

}  // namespace operations

}  // namespace tt
