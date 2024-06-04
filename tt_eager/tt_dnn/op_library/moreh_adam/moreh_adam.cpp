// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <utility>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_adam/moreh_adam_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace operations {
namespace primary {

operation::ProgramWithCallbacks moreh_adam_(
    const Tensor& param_in,
    const Tensor& grad,
    const Tensor& exp_avg_in,
    const Tensor& exp_avg_sq_in,

    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    uint32_t step,
    bool amsgrad,

    const std::optional<const Tensor> max_exp_avg_sq_in,
    const Tensor& param_out,
    const Tensor& exp_avg_out,
    const Tensor& exp_avg_sq_out,
    const std::optional<const Tensor> max_exp_avg_sq_out,
    const DeviceComputeKernelConfig compute_kernel_config) {
    uint32_t num_tiles = param_in.volume() / TILE_HW;

    Program program{};

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Device* device = param_in.device();
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt_metal::split_work_to_cores(grid, num_tiles);

    auto arch = param_in.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    auto data_format = tt_metal::datatype_to_dataformat_converter(param_in.get_dtype());
    auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;
    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 1},  // param_in
            {CB::c_in1, 1},  // grad
            {CB::c_in2, 1},  // exp_avg_in
            {CB::c_in3, 1},  // exp_avg_sq_in
            {CB::c_in4, 1},  // max_exp_avg_sq_in (optional)
            {CB::c_in5, 5},  // lr, beta1, beta2, eps, weight_decay
            {CB::c_in6, 1},  // 1.0f

            {CB::c_intermed0, 1, intermed_cb_format},  // tmp_grad
            {CB::c_intermed1, 1, intermed_cb_format},  // tmp_exp_avg
            {CB::c_intermed2, 1, intermed_cb_format},  // tmp_exp_avg_sq
            {CB::c_intermed3, 1, intermed_cb_format},  // tmp_max_exp_avg_sq
            {CB::c_intermed4, 1, intermed_cb_format},  //
            {CB::c_intermed5, 1, intermed_cb_format},  //
            {CB::c_intermed6, 1, intermed_cb_format},  // tmp1
            {CB::c_intermed7, 1, intermed_cb_format},  // tmp2

            {CB::c_out0, 1},  // param_out
            {CB::c_out1, 1},  // exp_avg_out
            {CB::c_out2, 1},  // exp_avg_sq_out
            {CB::c_out3, 1},  // max_exp_avg_sq_out (optional)
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////

    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(param_in)),
        static_cast<uint32_t>(is_dram(grad)),
        static_cast<uint32_t>(is_dram(exp_avg_in)),
        static_cast<uint32_t>(is_dram(exp_avg_sq_in)),
        static_cast<uint32_t>(is_dram(max_exp_avg_sq_in))};

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(is_dram(param_out)),
        static_cast<uint32_t>(is_dram(exp_avg_out)),
        static_cast<uint32_t>(is_dram(exp_avg_sq_out)),
        static_cast<uint32_t>(is_dram(max_exp_avg_sq_out))};

    const auto reader_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_adam/kernels/"
        "reader_moreh_adam.cpp";
    const auto writer_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_adam/kernels/"
        "writer_moreh_adam.cpp";

    std::map<std::string, std::string> data_movement_defines{};
    if (amsgrad) {
        data_movement_defines["AMSGRAD"] = "1";
    }
    const auto reader_kernel_id =
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, data_movement_defines);
    const auto writer_kernel_id =
        CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args, data_movement_defines);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    if (amsgrad) {
        compute_defines["AMSGRAD"] = "1";
    }

    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    const std::vector<uint32_t> compute_args_group_1{num_tiles_per_core_group_1};

    const auto compute_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_adam/kernels/"
        "moreh_adam.cpp";

    auto compute_kernel_1_id = CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_tiles_per_core_group_1, compute_args_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);
    KernelHandle compute_kernel_2_id = -1;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{num_tiles_per_core_group_2};

        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_tiles_per_core_group_2, compute_args_group_2},
            compute_defines,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto param_in_addr = param_in.buffer()->address();
    const auto grad_addr = grad.buffer()->address();
    const auto exp_avg_in_addr = exp_avg_in.buffer()->address();
    const auto exp_avg_sq_in_addr = exp_avg_sq_in.buffer()->address();
    const auto max_exp_avg_sq_in_addr =
        max_exp_avg_sq_in.has_value() ? max_exp_avg_sq_in.value().buffer()->address() : 0;

    const auto param_out_addr = param_out.buffer()->address();
    const auto exp_avg_out_addr = exp_avg_out.buffer()->address();
    const auto exp_avg_sq_out_addr = exp_avg_sq_out.buffer()->address();
    const auto max_exp_avg_sq_out_addr =
        max_exp_avg_sq_out.has_value() ? max_exp_avg_sq_out.value().buffer()->address() : 0;

    union {
        float f;
        uint32_t u;
    } f2u_lr, f2u_beta1, f2u_beta2, f2u_eps, f2u_weight_decay;
    f2u_lr.f = lr;
    f2u_beta1.f = beta1;
    f2u_beta2.f = beta2;
    f2u_eps.f = eps;
    f2u_weight_decay.f = weight_decay;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }

        const std::vector<uint32_t> reader_runtime_args{
            param_in_addr,
            grad_addr,
            exp_avg_in_addr,
            exp_avg_sq_in_addr,
            max_exp_avg_sq_in_addr,
            f2u_lr.u,
            f2u_beta1.u,
            f2u_beta2.u,
            f2u_eps.u,
            f2u_weight_decay.u,
            step,
            static_cast<uint32_t>(amsgrad),
            num_tiles_per_core,
            tile_offset};
        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        const std::vector<uint32_t> writer_runtime_args{
            param_out_addr,
            exp_avg_out_addr,
            exp_avg_sq_out_addr,
            max_exp_avg_sq_out_addr,
            num_tiles_per_core,
            tile_offset};
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        if (core_group_1.core_coord_in_core_ranges(core)) {
            tt_metal::SetRuntimeArgs(program, compute_kernel_1_id, core, {step});
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            tt_metal::SetRuntimeArgs(program, compute_kernel_2_id, core, {step});
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }

        tile_offset += num_tiles_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_args_callback = [reader_kernel_id = reader_kernel_id,
                                           writer_kernel_id = writer_kernel_id,
                                           num_cores = num_cores,
                                           num_cores_y = num_cores_y](
                                              const Program& program,
                                              const std::vector<Buffer*>& input_buffers,
                                              const std::vector<Buffer*>& output_buffers) {
        auto param_in_buffer = input_buffers.at(0);
        auto grad_in_buffer = input_buffers.at(1);
        auto exp_avg_in_buffer = input_buffers.at(2);
        auto exp_avg_sq_in_buffer = input_buffers.at(3);
        auto max_exp_avg_sq_in_buffer = input_buffers.at(4);

        auto param_out_buffer = output_buffers.at(0);
        auto exp_avg_out_buffer = output_buffers.at(1);
        auto exp_avg_sq_out_buffer = output_buffers.at(2);
        auto max_exp_avg_sq_out_buffer = output_buffers.at(3);

        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = param_in_buffer->address();
                runtime_args[1] = grad_in_buffer->address();
                runtime_args[2] = exp_avg_in_buffer->address();
                runtime_args[3] = exp_avg_sq_in_buffer->address();
                if (max_exp_avg_sq_in_buffer != nullptr) {
                    runtime_args[4] = max_exp_avg_sq_in_buffer->address();
                }
            }

            {
                auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = param_out_buffer->address();
                runtime_args[1] = exp_avg_out_buffer->address();
                runtime_args[2] = exp_avg_sq_out_buffer->address();
                if (max_exp_avg_sq_out_buffer != nullptr) {
                    runtime_args[3] = max_exp_avg_sq_out_buffer->address();
                }
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
