// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

// got from layernorm_op.hpp
// enum class LayerNormType {
//     LAYERNORM, RMSNORM
// };

// TODO: Do we need a program config?

// struct LayerNormDefaultProgramConfig{
//     tt::stl::reflection::Attributes attributes() const { return {}; };
// };
// struct LayernormDistributedProgramConfig {
//     CoreCoord compute_with_storage_grid_size;

//     tt::stl::reflection::Attributes attributes() const {
//         return {
//             {"compute_with_storage_grid_size", compute_with_storage_grid_size}
//         };
//     };
// };

// using LayerNormProgramConfig = std::variant<
//     LayerNormDefaultProgramConfig,
//     LayerNormShardedMultiCoreProgramConfig
// >;

// operation::ProgramWithCallbacks layernorm_multi_core(
//     const Tensor &a,
//     const std::optional<const Tensor> b,
//     const std::optional<const Tensor> gamma,
//     const std::optional<const Tensor> beta,
//     Tensor& output,
//     LayerNormType norm_type,
//     float eps,
//     DeviceComputeKernelConfig compute_kernel_config
// );

operation::ProgramWithCallbacks layernorm_part2_multi_core(
    const Tensor &a,
    const Tensor &stats,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    LayerNormType norm_type,
    float eps,
    DeviceComputeKernelConfig compute_kernel_config);



struct LayerNormPart2 {
    LayerNormType norm_type;
    float eps;
    MemoryConfig output_mem_config;
    // LayerNormProgramConfig program_config;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    tt::stl::reflection::Attributes attributes() const;
};

}  // namespace metal

namespace operations {

namespace primary {

template <LayerNormType layernorm_type>
struct make_layernorm_part2 {
    Tensor operator()(
        const Tensor& a,
        const Tensor& stats,
        float eps,
        std::optional<const Tensor> gamma = std::nullopt,
        std::optional<const Tensor> beta = std::nullopt,
        const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        // const LayerNormProgramConfig& program_config = LayerNormDefaultProgramConfig{},
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) const {
        std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({a}))};
        operation::launch_op(
            [eps, mem_config,
            // program_config,
            compute_kernel_config] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                const auto& a = input_tensors.at(0);
                const auto& stats = input_tensors.at(1);
                const auto& gamma = optional_input_tensors.at(0);
                const auto& beta = optional_input_tensors.at(1);
                auto arch = a.storage_type() == StorageType::DEVICE ? a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
                auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
                return operation::run(
                        LayerNormPart2{
                            .norm_type = layernorm_type,
                            .eps = eps,
                            .output_mem_config = mem_config,
                            // .program_config = program_config,
                            .compute_kernel_config = kernel_config_val},
                        {a, stats},
                        {gamma, beta});
            }, {a}, output_tensors, {gamma, beta});
        return output_tensors.at(0);
    }
};

constexpr auto layernorm_part2 = make_layernorm_part2<LayerNormType::LAYERNORM>{};
constexpr auto rmsnorm_part2 = make_layernorm_part2<LayerNormType::RMSNORM>{};

// template <LayerNormType layernorm_type>
// struct make_add_layernorm {
//     Tensor operator()(
//         const Tensor &a, const Tensor& b, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, const LayerNormProgramConfig& program_config = LayerNormDefaultProgramConfig{}, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) const {
//         std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({a, b}))};
//         operation::launch_op(
//             [eps, mem_config, program_config, compute_kernel_config] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
//                 const auto& a = input_tensors.at(0);
//                 const auto& b = input_tensors.at(1);
//                 const auto& gamma = optional_input_tensors.at(0);
//                 const auto& beta = optional_input_tensors.at(1);
//                 auto arch = a.storage_type() == StorageType::DEVICE ? a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
//                 auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
//                 return operation::run(
//                    LayerNorm{
//                        .norm_type = layernorm_type,
//                        .eps = eps,
//                        .output_mem_config = mem_config,
//                        .program_config = program_config,
//                        .compute_kernel_config = kernel_config_val},
//                    {a},
//                    {b, gamma, beta});
//             }, {a, b}, output_tensors, {gamma, beta});
//         return output_tensors.at(0);
//     }
// };



// // computes layernorm(a+b)*gamma+beta
// constexpr auto add_layernorm = make_add_layernorm<LayerNormType::LAYERNORM>{};
// constexpr auto add_rmsnorm = make_add_layernorm<LayerNormType::RMSNORM>{};

}  // namespace primary

}  // namespace operations

}  // namespace tt
