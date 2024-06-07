// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "tt_eager/tt_dnn/op_library/layernorm_distributed/layernorm_part1_op.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include <optional>
#include <variant>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

inline bool is_dram(const Tensor& input_tensor) { return input_tensor.memory_config().buffer_type == BufferType::DRAM; }
inline bool is_dram(const std::optional<const Tensor> input_tensor) {
     return input_tensor.has_value() ? is_dram(input_tensor.value()) : true;
}
inline bool is_dram(const Buffer* b) { return b->buffer_type() == BufferType::DRAM; }

operation::ProgramWithCallbacks layernorm_part1_multi_core(
    const Tensor &a,
    Tensor& output,
    LayerNormType norm_type,
    DeviceComputeKernelConfig compute_kernel_config
) {
    const bool is_rmsnorm = norm_type == LayerNormType::RMSNORM;
    const auto shape = a.get_legacy_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H*W;
    const uint32_t NC = a.volume() / HW;


    // Kernels are configured to support BFLOAT8_B, but bad pcc so we need mixed precision support in compute
    const auto& a_dtype = a.get_dtype();

    const uint32_t Wt = W/TILE_WIDTH;
    const uint32_t Ht = H/TILE_HEIGHT;
    const uint32_t tile_cols_per_device = is_rmsnorm ? 1 : 2;

    uint32_t num_tile_rows = NC * Ht;

    log_debug("is_rmsnorm: {}", is_rmsnorm);
    log_debug("W: {}", W);
    log_debug("H: {}", H);
    log_debug("num_tile_rows: {}", num_tile_rows);
    log_debug("Wt: {}", Wt);
    log_debug("Ht: {}", Ht);


    ////////////////////////////////////////////////////////////////////////////
    //                       Device Setup
    //////////////////////////////////////////////////////////////////////////
    Device *device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                Circular Buffer Data Format Setup
    //////////////////////////////////////////////////////////////////////////
    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;

    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = false;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::WORMHOLE_B0, "kernel config is not for wormhole_b0");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = tt_metal::datatype_to_dataformat_converter(a.get_dtype()) == tt::DataFormat::Float32 ? true : compute_kernel_config.fp32_dest_acc_en;
        } else {
            TT_FATAL("arch not supported");
        }

    }, compute_kernel_config);

    uint32_t block_size = 1; // find_max_divisor(Wt, 8);
    uint32_t writer_block_size = 1;

    tt::DataFormat in_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat out_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t in_single_tile_size = tt_metal::detail::TileSize(in_data_format);
    uint32_t out_single_tile_size = tt_metal::detail::TileSize(out_data_format);
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    uint32_t bfloat16_tile_size = tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    log_debug("in_data_format: {}", in_data_format);
    log_debug("out_data_format: {}", out_data_format);

    tt::DataFormat inb_data_format = tt::DataFormat::Invalid;
    uint32_t inb_single_tile_size = 0;

    auto a_addr = a.buffer()->address();
    auto dst_addr = output.buffer()->address();

    uint32_t num_tiles = a.volume()/TILE_HW;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    /*
    in0_cb: a
    in1_cb: 1 (reduction scalar)

    output CB is packed such that the first tile is for x**2 stats, second tile is for x stats
    in RMSNorm, only first tile has valid data.

    intermed0_cb: xˆ2
    out0_cb: [sum(xˆ2), sum(x)]  # For layernorm
    out0_cb: [sum(xˆ2)]  # RMSNorm

    */

    const uint32_t in0_tiles = Wt;
    const uint32_t in1_tiles = 1; // reduce scalar

    const uint32_t intermed0_tiles = Wt; // xˆ2
    uint32_t out0_tiles = 1;
    if (!is_rmsnorm) {
        out0_tiles = 2;
    }

    TT_ASSERT(W <= TILE_WIDTH*in0_tiles && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");
    TT_ASSERT(in0_tiles % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(intermed0_tiles % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");


    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tile_rows_per_core_group_1, num_tile_rows_per_core_group_2] = split_work_to_cores(grid_size, num_tile_rows, true);

    log_debug("num_cores: {}", num_cores);
    log_debug("grid_size: {}", grid_size);
    log_debug("core_group_1: {}", core_group_1.str());
    log_debug("num_tile_rows_per_core_group_1: {}", num_tile_rows_per_core_group_1);
    log_debug("core_group_2: {}", core_group_2.str());
    log_debug("num_tile_rows_per_core_group_2: {}", num_tile_rows_per_core_group_2);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = CreateProgram();

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) is_dram(a),
        (std::uint32_t) block_size,
    };

    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) is_dram(output),
        (std::uint32_t) writer_block_size
    };


    bool tile_dtype_is_bfloat16 = a.get_dtype() == tt::tt_metal::DataType::BFLOAT16;
    std::map<string, string> compute_defines;

    if (is_rmsnorm) {
        compute_defines["RMSNORM"] = "1";
    }

    auto reader_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/layernorm_distributed/kernels/dataflow/reader_unary_interleaved_lnp1_rm_gb.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args)
    );

    auto writer_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/layernorm/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args)
    );

    vector<uint32_t> compute_args = { Wt, block_size };

    auto compute_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/layernorm_distributed/kernels/compute/layernorm_part1.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = compute_args, .defines = compute_defines}
    );

    // Create circular buffers
    // c_in0 -> a
    CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_tiles*in_single_tile_size, {{CB::c_in0, in_data_format}}).set_page_size(CB::c_in0, in_single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_src0_config );
    // c_in1 -> reduce scalar
    CircularBufferConfig cb_reduce_config = CircularBufferConfig(in1_tiles*bfloat16_tile_size, {{CB::c_in1, DataFormat::Float16_b}}).set_page_size(CB::c_in1, bfloat16_tile_size);
    CreateCircularBuffer( program, all_cores, cb_reduce_config );

    // LN and RMS shared intermediates //
    // c_intermed0 -> xˆ2
    CircularBufferConfig cb_intermed0_config = CircularBufferConfig(intermed0_tiles*single_tile_size, {{CB::c_intermed0, cb_data_format}}).set_page_size(CB::c_intermed0, single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_intermed0_config );

    CircularBufferConfig cb_out0_config = CircularBufferConfig(out0_tiles*out_single_tile_size, {{CB::c_out0, out_data_format}}).set_page_size(CB::c_out0, out_single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_out0_config );

    // Log all circular buffers with program.circular_buffers_on_corerange(all_cores), which returns std::vector<std::shared_ptr<CircularBuffer>>

    for (const auto& cb : program.circular_buffers_on_corerange(*all_cores.ranges().begin())) {
        for (const auto index : cb->buffer_indices()) {
            log_debug("cb_id {}", index);
            log_debug("page_size: {}", cb->page_size(index));
            log_debug("num_pages: {}", cb->num_pages(index));
            log_debug("data_format: {}", cb->data_format(index));
        }
    }

    uint32_t curr_row = 0;
    float winv =  1.0f;
    bfloat16 bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        uint32_t in_tile_offset = curr_row * Wt;
        uint32_t out_tile_offset = curr_row * out0_tiles;

        SetRuntimeArgs(program, reader_kernels_id, core,
            { a_addr, num_tile_rows_per_core, Wt, in_tile_offset, packed_winv_value }
        );
        SetRuntimeArgs(program, compute_kernels_id, core, { num_tile_rows_per_core });
        SetRuntimeArgs(program, writer_kernels_id, core, { dst_addr, num_tile_rows_per_core * out0_tiles, out_tile_offset } );
        curr_row += num_tile_rows_per_core;
    }

    auto override_runtime_args_callback = [
            reader_kernel_id=reader_kernels_id,
            writer_kernel_id=writer_kernels_id,
            num_cores,
            grid_size
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {

        const auto& input_tensor = input_tensors.at(0);

        const auto input_addr = input_tensor.buffer()->address();

        const auto& output_tensor = output_tensors.at(0);
        const auto output_addr = output_tensor.buffer()->address();

        auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
        auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);

        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord core = {i % grid_size.x, i / grid_size.x};

            {
                auto& reader_args = reader_runtime_args_by_core.at(core.x).at(core.y);

                reader_args[0] = input_addr;
            }

            {
                auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
                writer_args[0] = output_addr;
            }
        }
    };

    return {std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}


}  // namespace tt_metal

}  // namespace tt
