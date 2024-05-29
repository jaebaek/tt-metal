// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

#include <vector>
namespace tt {
using namespace constants;
namespace operations {

namespace primary {

namespace {

void get_tensor_dim(std::vector<uint32_t> &dim, const Shape& shape) {
    const auto rank = shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;

        // last 2-dim
        if (idx == rank - 1 || idx == rank - 2) {
            dim[i] = shape[idx] / TILE_HEIGHT;
        }
        else {
            dim[i] = shape[idx];
        }
    }

    log_debug(LogOp, "rank {}", rank);
    for (auto i = 0; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        log_debug(LogOp, "dim[{}] = {}", i, dim[i]);
    }
}

}

operation::ProgramWithCallbacks moreh_sum_backward_impl(const Tensor &output_grad, const Tensor &input_grad) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto *device = output_grad.device();
    auto program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output_grad.get_dtype());
    const auto single_tile_size = detail::TileSize(cb_data_format);

    const auto &output_grad_shape = output_grad.get_legacy_shape();
    const auto &output_grad_shape_wo_padding = output_grad_shape.without_padding();
    const auto output_grad_rank = output_grad_shape.rank();

    std::vector<uint32_t> output_grad_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    log_debug(LogOp, "output_grad");
    get_tensor_dim(output_grad_dim, output_grad_shape);

    const auto &input_grad_shape = input_grad.get_legacy_shape();
    const auto &input_grad_shape_wo_padding = input_grad_shape.without_padding();
    const auto input_grad_rank = input_grad_shape.rank();

    std::vector<uint32_t> input_grad_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    log_debug(LogOp, "input_grad");
    get_tensor_dim(input_grad_dim, input_grad_shape);

    std::vector<uint32_t> need_bcast_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 0);
    for (auto i = 0; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        // TODO: both rank can be different when keepdim=false
        auto idx = input_grad_rank - 1 - i;

        // last 2-dim
        if (idx == input_grad_rank - 1 || idx == input_grad_rank - 2) {
            need_bcast_dim[i] = (output_grad_shape_wo_padding[idx] != input_grad_shape_wo_padding[idx]);
        } else {
            need_bcast_dim[i] = (output_grad_shape[idx] != input_grad_shape[idx]);
        }
    }
    const auto num_input_grad_tiles = input_grad.volume() / TILE_HW;

    for (auto i = 0; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        log_debug(LogOp, "need_bcast_dim [{}] = {}", i, need_bcast_dim[i]);
    }
    log_debug(LogOp, "num_input_grad_tiles {}", num_input_grad_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const uint32_t in0_t = 2;   // input
    const uint32_t in1_t = 1;   // zero
    const uint32_t out0_t = 2;  // output
    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_input_grad_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},    // input
            {CB::c_in1, in1_t},    // zero
            {CB::c_out0, out0_t},  // output
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> writer_compile_time_args;
    const auto reader_kernel_file = "tt_eager/tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_impl/kernels/reader_moreh_sum_backward.cpp";
    const auto writer_kernel_file = "tt_eager/tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_impl/kernels/writer_moreh_sum_backward.cpp";
    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{num_cols_per_core_group_1};
    std::map<string, string> compute_defines;
    const auto compute_kernel_file = "tt_eager/tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_impl/kernels/moreh_sum_backward.cpp";
    const auto compute_kernel_1_id = CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_cols_per_core_group_1, compute_args_group_1}, compute_defines);

    std::optional<KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{num_cols_per_core_group_2};
        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_cols_per_core_group_2, compute_args_group_2},
            compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        std::vector<uint32_t> reader_rt_args;
        reader_rt_args.push_back(output_grad.buffer()->address());
        reader_rt_args.push_back(num_tiles_per_core);
        reader_rt_args.push_back(tile_offset);
        reader_rt_args.push_back(static_cast<uint32_t>(is_dram(output_grad)));
        reader_rt_args.insert(reader_rt_args.end(), output_grad_dim.begin(), output_grad_dim.end());
        reader_rt_args.insert(reader_rt_args.end(), input_grad_dim.begin(), input_grad_dim.end());
        reader_rt_args.insert(reader_rt_args.end(), need_bcast_dim.begin(), need_bcast_dim.end());

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            reader_rt_args
        );

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {input_grad.buffer()->address(),
             num_tiles_per_core,
             tile_offset,
             static_cast<uint32_t>(is_dram(input_grad))});

        std::vector<uint32_t> compute_rt_args;
        compute_rt_args.push_back(num_tiles_per_core);
        compute_rt_args.insert(compute_rt_args.end(), need_bcast_dim.begin(), need_bcast_dim.end());

        if (core_group_1.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(
                program,
                compute_kernel_1_id,
                core,
                {num_tiles_per_core, need_bcast_dim[0], need_bcast_dim[1]});
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            TT_ASSERT(compute_kernel_2_id.has_value());
            SetRuntimeArgs(
                program,
                compute_kernel_2_id.value(),
                core,
                {num_tiles_per_core, need_bcast_dim[0], need_bcast_dim[1]});
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }
        tile_offset += num_tiles_per_core;
    }

    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_y](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        log_debug(LogOp, "{}:{} args_callback ", __func__, __LINE__);
        const auto *output_grad_buffer = input_tensors.at(0).buffer();
        const auto *input_grad_buffer = output_tensors.at(0).buffer();
        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = output_grad_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = input_grad_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
