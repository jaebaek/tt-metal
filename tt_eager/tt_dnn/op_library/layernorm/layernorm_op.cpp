// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include <optional>

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

// computes layernorm(a+*b)*gamma + beta
// if b is nullptr it's treated as zero (no addition)
operation::ProgramWithCallbacks layernorm_(
    const Tensor &a,
    const std::optional<const Tensor> b,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    float eps,
    bool rms_norm = false
) {

    const auto shape = a.shape();
    uint32_t W = shape[3], H = shape[2];
    uint32_t HW = H*W;
    uint32_t NC = a.volume() / HW;

    // Kernels are configured to support BFLOAT8_B, but bad pcc so we need mixed precision support in compute
    const auto& a_dtype = a.dtype();

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = a.volume() / TILE_HW;

    uint32_t block_size = find_max_divisor(Wt, 8);

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    uint32_t bfloat16_tile_size = tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    auto a_addr = a.buffer()->address();
    auto b_dram_addr = b ? b.value().buffer()->address() : 0;
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;

    uint32_t num_tiles = a.volume()/TILE_HW;
    uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().volume()/TILE_HW : 0;
    uint32_t num_beta_tiles = beta.has_value() ? beta.value().volume()/TILE_HW : 0;

    // For bert, tensor is packed as RM with width 32
    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        num_gamma_tiles = gamma.has_value() ? gamma.value().volume()/TILE_WIDTH : 0;
    }
    if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        num_beta_tiles = beta.has_value() ? beta.value().volume()/TILE_WIDTH : 0;
    }


    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    Device *device = a.device();
    auto dst_addr = output.buffer()->address();


    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    // TODO(AP): this will not work for all Wts possibly, but should work for Wt=8, 12, 16, 32
    // TODO(AP): can also add support for block_size=7 -> 63, 28
    uint32_t WtB    =  div_up(Wt, block_size)*block_size; // Wt padded to be divisible by block size
    uint32_t in0_t  =  WtB; // cb_x for no pre-add variant, x=a+b for fused pre-add, extra space for some buffering
    uint32_t in1_t  =  block_size*2; // buffer for fused pre-add b tensor
    uint32_t out0_t =  block_size*2;
    uint32_t im0_t  =  WtB; // buffer for saving xmm
    uint32_t im3_t  =  WtB; // buffer for xmm^2
    uint32_t in5_t  =  WtB; // buffer for gamma
    uint32_t in6_t  =  WtB; // buffer for beta
    uint32_t im6_t  =  block_size*2; // x=a+b reuse for x-E[x] computation plus a bit extra for buffering
    if (b) {
        im6_t = WtB;
        //cout << "im6_t=WtB=" << WtB << endl;
        in0_t = 2*block_size;
    }
    uint32_t im5_t  =  2*block_size; // for buffering to/from *gamma/+beta
    uint32_t im4_t  =  8; // 8 just in case, 4 would prob suffice
    uint32_t im1_t  =  2;
    uint32_t in2_t  =  2; // scaler for reduce coming from reader
    uint32_t in3_t  =  2; // epsilon coming from reader
    uint32_t im2_t  =  2; //

    TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");
    TT_ASSERT(in0_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in1_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(out0_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(im0_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(im3_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in5_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in6_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(im6_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(Wt % block_size == 0);
    TT_ASSERT(num_gamma_tiles % block_size == 0);
    TT_ASSERT(num_beta_tiles % block_size == 0);

    uint32_t num_tile_rows = NC * Ht;
    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tile_rows_per_core_group_1, num_tile_rows_per_core_group_2] = split_work_to_cores(grid_size, num_tile_rows, true);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = CreateProgram();

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) is_dram(a),
        (std::uint32_t) is_dram(b),
        (std::uint32_t) is_dram(gamma),
        (std::uint32_t) is_dram(beta),
        (std::uint32_t) block_size
    };

    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().shape()[3] * gamma.value().element_size();
        bool gamma_stick_size_is_power_of_two = is_power_of_two_at_least_32(gamma_stick_size);
        reader_compile_time_args.push_back((std::uint32_t) gamma_stick_size_is_power_of_two);
        if (gamma_stick_size_is_power_of_two) {
            uint32_t gamma_log2_stick_size = gamma_stick_size_is_power_of_two ? (std::uint32_t)log2(gamma_stick_size) : 0;
            reader_compile_time_args.push_back((std::uint32_t) gamma_log2_stick_size);
        } else {
            reader_compile_time_args.push_back(gamma_stick_size);
        }
    } else if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().shape()[3] * beta.value().element_size();
        bool beta_stick_size_is_power_of_two = is_power_of_two_at_least_32(beta_stick_size);
        reader_compile_time_args.push_back((std::uint32_t) beta_stick_size_is_power_of_two);
        if (beta_stick_size_is_power_of_two) {
            uint32_t beta_log2_stick_size = beta_stick_size_is_power_of_two ? (std::uint32_t)log2(beta_stick_size) : 0;
            reader_compile_time_args.push_back((std::uint32_t) beta_log2_stick_size);
        } else {
            reader_compile_time_args.push_back(beta_stick_size);
        }
    } else {
        reader_compile_time_args.push_back(0);
        reader_compile_time_args.push_back(0);
    }

    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) is_dram(output),
        (std::uint32_t) block_size
    };


    bool tile_dtype_is_bfloat16 = a.dtype() == tt::tt_metal::DataType::BFLOAT16;
    std::map<string, string> reader_defines;
    std::map<string, string> eltwise_binary_defines;
    if (b) {
        reader_defines["FUSE_PRE_ADD"] = "1";
        eltwise_binary_defines["FUSE_PRE_ADD"] = "1";
    }
    if (gamma.has_value()) {
        reader_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_defines["FUSE_BETA"] = "1";
    }

    auto use_row_major_kernel = (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) or (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR);
    auto reader_kernels_id = CreateKernel(
        program,
        use_row_major_kernel ? "tt_eager/tt_dnn/op_library/layernorm/kernels/reader_unary_interleaved_ln_rm_gb.cpp" : "tt_eager/tt_dnn/op_library/layernorm/kernels/reader_unary_interleaved_ln.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args, .defines = reader_defines}
    );

    auto writer_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args}
    );

    vector<uint32_t> compute_args = { Wt, block_size, gamma.has_value(), beta.has_value() };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = true;
    auto compute_kernels_id = CreateKernel(
        program,
        rms_norm ? "tt_eager/tt_dnn/kernels/compute/rmsnorm.cpp" : "tt_eager/tt_dnn/kernels/compute/layernorm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = compute_args, .defines = eltwise_binary_defines}
    );

    // Create circular buffers
    CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_t*single_tile_size, {{CB::c_in0, cb_data_format}}).set_page_size(CB::c_in0, single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_src0_config );
    CircularBufferConfig cb_out0_config = CircularBufferConfig(out0_t*single_tile_size, {{CB::c_out0, cb_data_format}}).set_page_size(CB::c_out0, single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_out0_config );
    CircularBufferConfig cb_intermed1_config = CircularBufferConfig(im1_t*single_tile_size, {{CB::c_intermed1, cb_data_format}}).set_page_size(CB::c_intermed1, single_tile_size);
    CreateCircularBuffer( program, all_cores,  cb_intermed1_config );
    CircularBufferConfig cb_in2_config = CircularBufferConfig(in2_t*bfloat16_tile_size, {{CB::c_in2, DataFormat::Float16_b}}).set_page_size(CB::c_in2, bfloat16_tile_size);
    CreateCircularBuffer( program, all_cores, cb_in2_config );
    CircularBufferConfig cb_in3_config = CircularBufferConfig(in3_t*bfloat16_tile_size, {{CB::c_in3, DataFormat::Float16_b}}).set_page_size(CB::c_in3, bfloat16_tile_size);
    CreateCircularBuffer( program, all_cores, cb_in3_config );
    CircularBufferConfig cb_intermed2_config = CircularBufferConfig(im2_t*single_tile_size, {{CB::c_intermed2, cb_data_format}}).set_page_size(CB::c_intermed2, single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_intermed2_config );
    if (!rms_norm) {
        CircularBufferConfig cb_intermed0_config = CircularBufferConfig(im0_t*single_tile_size, {{CB::c_intermed0, cb_data_format}}).set_page_size(CB::c_intermed0, single_tile_size);
        CreateCircularBuffer( program, all_cores, cb_intermed0_config );
    }
    CircularBufferConfig c_intermed3_config = CircularBufferConfig(im3_t*single_tile_size, {{CB::c_intermed3, cb_data_format}}).set_page_size(CB::c_intermed3, single_tile_size);
    CreateCircularBuffer( program, all_cores, c_intermed3_config );
    CircularBufferConfig c_intermed4_config = CircularBufferConfig(im4_t*single_tile_size, {{CB::c_intermed4, cb_data_format}}).set_page_size(CB::c_intermed4, single_tile_size);
    CreateCircularBuffer( program, all_cores, c_intermed4_config );
    if (gamma.has_value() || beta.has_value()) {
        CircularBufferConfig c_intermed5_config = CircularBufferConfig(im5_t*single_tile_size, {{CB::c_intermed5, cb_data_format}}).set_page_size(CB::c_intermed5, single_tile_size);
        CreateCircularBuffer( program, all_cores, c_intermed5_config );
    }
    if (gamma.has_value()) {
        uint32_t c_in5_page_size = gamma.value().layout() == Layout::ROW_MAJOR ? bfloat16_tile_size : single_tile_size;
        DataFormat c_in5_df = gamma.value().layout() == Layout::ROW_MAJOR ? DataFormat::Float16_b : cb_data_format;
        CircularBufferConfig c_in5_config = CircularBufferConfig(in5_t * c_in5_page_size, {{CB::c_in5, c_in5_df}})
            .set_page_size(CB::c_in5, c_in5_page_size);
        CreateCircularBuffer( program, all_cores, c_in5_config );
    }
    if (beta.has_value()) {
        uint32_t c_in6_page_size = beta.value().layout() == Layout::ROW_MAJOR ? bfloat16_tile_size : single_tile_size;
        DataFormat c_in6_df = beta.value().layout() == Layout::ROW_MAJOR ? DataFormat::Float16_b : cb_data_format;
        CircularBufferConfig c_in6_config = CircularBufferConfig(in6_t * c_in6_page_size, {{CB::c_in6, c_in6_df}})
            .set_page_size(CB::c_in6, c_in6_page_size);
        CreateCircularBuffer( program, all_cores, c_in6_config );
    }
    if (b) {
        // x = a+b in this notation
        // result = ln(x)*gamma + beta
        // if there's no pre-add we use cb_in0 for x, otherwise a is pre-buffered into in0, added into im6, then im6 is used as x
        // b is buffered into c_in1
        CircularBufferConfig c_intermed6_config = CircularBufferConfig(im6_t*single_tile_size, {{CB::c_intermed6, cb_data_format}}).set_page_size(CB::c_intermed6, single_tile_size);
        CreateCircularBuffer( program, all_cores, c_intermed6_config );
        // c_in1 is input buffer for b
        CircularBufferConfig c_in1_config = CircularBufferConfig(in1_t*single_tile_size, {{CB::c_in1, cb_data_format}}).set_page_size(CB::c_in1, single_tile_size);
        CreateCircularBuffer( program, all_cores, c_in1_config);
    }

    uint32_t curr_row = 0;
    float winv = 1.0f / W; // bcast-w scaler
    bfloat16 bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    union { float f; uint32_t u; } e; e.f = eps; // epsilon
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t num_tile_rows_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        uint32_t tile_offset = curr_row * Wt;

        SetRuntimeArgs(program, reader_kernels_id, core,
            { a_addr, num_tile_rows_per_core, Wt, tile_offset, packed_winv_value, e.u, // 0-5
            gamma_dram_addr, beta_dram_addr, b_dram_addr } // 6-8
        );
        SetRuntimeArgs(program, compute_kernels_id, core, { num_tile_rows_per_core });
        SetRuntimeArgs(program, writer_kernels_id, core, { dst_addr, num_tile_rows_per_core * Wt, tile_offset } );
        curr_row += num_tile_rows_per_core;
    }

    auto override_runtime_args_callback = [
            reader_kernel_id=reader_kernels_id,
            writer_kernel_id=writer_kernels_id,
            num_cores,
            grid_size
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_a_dram_buffer = input_buffers.at(0);
        auto src_b_dram_buffer = input_buffers.at(1);
        auto gamma_dram_buffer = input_buffers.at(2);
        auto beta_dram_buffer = input_buffers.at(3);

        auto dst_dram_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};

            {
                auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_a_dram_buffer->address();
                if (src_b_dram_buffer != nullptr) {
                    runtime_args[8] = src_b_dram_buffer->address();
                }
                if (gamma_dram_buffer != nullptr) {
                    runtime_args[6] = gamma_dram_buffer->address();
                }
                if (beta_dram_buffer != nullptr) {
                    runtime_args[7] = beta_dram_buffer->address();
                }
                SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_dram_buffer->address();
                SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void LayerNorm::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {

    TT_FATAL(input_tensors.size() == 1 and optional_input_tensors.size() <= 3, "Must have between 1 to 4 input tensors");
    auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    TT_FATAL(a.layout() == Layout::TILE);
    TT_FATAL(a.dtype() == DataType::BFLOAT16 or a.dtype() == DataType::BFLOAT8_B);
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");

    if (b.has_value()) {
        TT_FATAL(b.value().layout() == Layout::TILE);
        TT_FATAL(a.shape() == b.value().shape());
        TT_FATAL(a.device() == b.value().device());
        TT_FATAL(b.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(gamma.value().layout() == beta.value().layout(), "Gamma and beta must have the same layout!");
    }

    if (gamma.has_value()) {
        if (gamma.value().layout() == Layout::TILE) {
            TT_FATAL(a.shape()[3] == gamma.value().shape()[3], fmt::format("{} != {}", a.shape()[3], gamma.value().shape()[3]));
            TT_FATAL(a.device() == gamma.value().device());
            TT_FATAL(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(gamma.value().shape()[2] == TILE_HEIGHT);
        } else {
            TT_FATAL(gamma.value().layout() == Layout::ROW_MAJOR);
            TT_FATAL((gamma.value().shape()[3] == TILE_WIDTH && gamma.value().volume() / TILE_WIDTH == a.shape()[3] / TILE_WIDTH));
            TT_FATAL(a.device() == gamma.value().device());
            TT_FATAL(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(gamma.value().dtype() == DataType::BFLOAT16);
        }
    }

    if (beta.has_value()) {
        if (beta.value().layout() == Layout::TILE) {
            TT_FATAL(a.shape()[3] == beta.value().shape()[3]);
            TT_FATAL(a.device() == beta.value().device());
            TT_FATAL(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(beta.value().shape()[2] == TILE_HEIGHT);
        } else {
            TT_FATAL(beta.value().layout() == Layout::ROW_MAJOR);
            TT_FATAL((beta.value().shape()[3] == TILE_WIDTH && beta.value().volume() / TILE_WIDTH == a.shape()[3] / TILE_WIDTH));
            TT_FATAL(a.device() == beta.value().device());
            TT_FATAL(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(beta.value().dtype() == DataType::BFLOAT16);
        }
    }

}

std::vector<Shape> LayerNorm::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> LayerNorm::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks LayerNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);
    return layernorm_(a, b, gamma, beta, output_tensor, this->eps);

}

tt::stl::reflection::Attributes LayerNorm::attributes() const {
    return {
        {"eps", this->eps},
        {"output_mem_config", this->output_mem_config},
    };
}


void RMSNorm::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 1 and optional_input_tensors.size() <= 3, "Must have between 1 to 4 input tensors");
    auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    TT_FATAL(a.layout() == Layout::TILE);
    TT_FATAL(a.dtype() == DataType::BFLOAT16 or a.dtype() == DataType::BFLOAT8_B);
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    if (b.has_value()) {
        TT_FATAL(b.value().layout() == Layout::TILE);
        TT_FATAL(a.shape() == b.value().shape());
        TT_FATAL(a.device() == b.value().device());
        TT_FATAL(b.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    }
    if (gamma.has_value()) {
        TT_FATAL(gamma.value().layout() == Layout::TILE);
        TT_FATAL(a.shape()[3] == gamma.value().shape()[3]);
        TT_FATAL(a.device() == gamma.value().device());
        TT_FATAL(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(gamma.value().shape()[2] == TILE_HEIGHT);
    }
    if (beta.has_value()) {
        TT_FATAL(beta.value().layout() == Layout::TILE);
        TT_FATAL(a.shape()[3] == beta.value().shape()[3]);
        TT_FATAL(a.device() == beta.value().device());
        TT_FATAL(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(beta.value().shape()[2] == TILE_HEIGHT);
    }

}

std::vector<Shape> RMSNorm::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> RMSNorm::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks RMSNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);
    return layernorm_(a, b, gamma, beta, output_tensor, this->eps, true);

}

tt::stl::reflection::Attributes RMSNorm::attributes() const {
    return {
        {"eps", this->eps},
        {"output_mem_config", this->output_mem_config},
    };
}

}  // namespace ll_buda



namespace operations {

using namespace tt_metal;

namespace primary {

operation::ProgramWithCallbacks layernorm_sharded_(
    const Tensor &a,
    const std::optional<const Tensor> b,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    float eps,
    MathFidelity fidelity,
    DataType data_format,
    CoreCoord grid_size,
    uint32_t subblock_wt,
    uint32_t block_ht,
    uint32_t block_wt
) {
    // convert data format
    tt::DataFormat io_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(data_format);
    // tile sizes
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    // tensor shape
    const auto shape = a.shape();
    uint32_t M = shape[2];
    uint32_t K = shape[3];
    uint32_t Mt = M / TILE_WIDTH;
    uint32_t Kt = K / TILE_WIDTH;
    // block
    uint32_t block_w = block_wt * TILE_WIDTH;
    uint32_t block_h = block_ht * TILE_WIDTH;
    uint32_t num_blocks = grid_size.y;
    uint32_t num_subblocks_w = block_wt / subblock_wt;
    // check dims
    TT_ASSERT(block_wt % subblock_wt == 0 && "block_w must be divisible by subblock_w.");
    TT_ASSERT(M % TILE_WIDTH == 0 && "M must be divisible by tile width.");
    TT_ASSERT(K % TILE_WIDTH == 0 && "K must be divisible by tile width.");
    TT_ASSERT(Kt / grid_size.y == block_wt && "block_w must equal to K / num_cores_r.");
    TT_ASSERT(Mt / grid_size.x == block_ht && "block_h must equal to M / num_cores_c.");
    // get sharded addr
    auto in0_addr = a.buffer()->address();
    uint32_t in1_addr;
    bool b_sharded;
    if (b) {
        in1_addr = b.value().buffer()->address();
    } else {
        in1_addr = 0;
    }
    auto out_addr = output.buffer()->address();
    // b, gamma, beta addr
    auto in1_dram_addr = b ? b.value().buffer()->address() : 0;
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;
    // num tiles for a, gamma, beta
    uint32_t num_tiles = a.volume()/TILE_HW;
    uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().volume()/TILE_HW : 0;
    uint32_t num_beta_tiles = beta.has_value() ? beta.value().volume()/TILE_HW : 0;

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    Device *device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t in0_block_tiles = block_wt * block_ht;
    uint32_t in0_CB_tiles = in0_block_tiles;
    uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
    // block size for in1 (tensor b)
    uint32_t in1_CB_size = in0_CB_size;
    // in2 - scaler
    uint32_t in2_CB_size = single_tile_size;
    // in3 - eps
    uint32_t in3_CB_size = single_tile_size;
    // gamma
    uint32_t in5_CB_size = in0_CB_size / block_ht;
    // beta
    uint32_t in6_CB_size = in0_CB_size / block_ht;
    // itermediate buffers change later
    uint32_t x_CB_size = in0_CB_size;
    uint32_t xmm_CB_size = in0_CB_size;
    uint32_t ex_partial_CB_size = in0_CB_size / block_wt;
    uint32_t ex_CB_size = ex_partial_CB_size;
    uint32_t ex_external_CB_size = Kt / block_wt * single_tile_size;
    uint32_t xmm2_CB_size = in0_CB_size / block_ht;
    uint32_t ex2pe_CB_size = 2 * single_tile_size;
    uint32_t fusion_CB_size = 2 * in0_CB_size / block_ht;
    // output buffer size
    uint32_t out_CB_size = in0_CB_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = Program();
    // define core ranges
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = grid_size.x;
    uint32_t num_cores_r = grid_size.y;
    uint32_t num_cores = num_cores_c * num_cores_r;
    CoreRange all_cores{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1}};
    CoreRange top_row{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y}};
    CoreRange all_except_top_row{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y + 1},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1}};
    CoreRange mcast_sender = top_row;
    CoreRange mcast_receiver = all_except_top_row;
    // Mcast args
    auto reduce_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto reduce_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    // reader defines
    std::map<string, string> reader_mcast_sender_defines;
    std::map<string, string> reader_mcast_receiver_defines;
    if (b) {
        if (b.value().memory_config().is_sharded()) {
            reader_mcast_sender_defines["IN1_SHARDED"] = "1";
            reader_mcast_receiver_defines["IN1_SHARDED"] = "1";
        }
        reader_mcast_sender_defines["FUSE_PRE_ADD"] = "1";
        reader_mcast_receiver_defines["FUSE_PRE_ADD"] = "1";
    }
    if (gamma.has_value()) {
        reader_mcast_sender_defines["FUSE_GAMMA"] = "1";
        reader_mcast_receiver_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_mcast_sender_defines["FUSE_BETA"] = "1";
        reader_mcast_receiver_defines["FUSE_BETA"] = "1";
    }
    // reader compile time args
    std::vector<uint32_t> reader_mcast_sender_compile_time_args = {
        (std::uint32_t) reduce_receiver_semaphore,
        (std::uint32_t) reduce_sender_semaphore,
        (std::uint32_t) num_blocks,
        (std::uint32_t) block_ht,
        (std::uint32_t) block_ht * single_tile_size
    };
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args = {
        (std::uint32_t) reduce_receiver_semaphore,
        (std::uint32_t) reduce_sender_semaphore,
        (std::uint32_t) block_ht
    };
    // reader kernel
    auto reader_mcast_sender_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/layernorm/kernels/dataflow/reader_mcast_sender_unary_sharded_ln.cpp",
        top_row,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_mcast_sender_compile_time_args, .defines = reader_mcast_sender_defines}
    );
    auto reader_mcast_receiver_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/layernorm/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp",
        all_except_top_row,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_mcast_receiver_compile_time_args, .defines = reader_mcast_receiver_defines}
    );
    // writer defines
    std::map<string, string> writer_defines;
    // writer compile time args
    std::vector<uint32_t> writer_mcast_sender_compile_time_args = {
        1,
        (std::uint32_t) gamma.has_value(),
        (std::uint32_t) beta.has_value(),
        (std::uint32_t) is_dram(gamma),
        (std::uint32_t) is_dram(beta),
        (std::uint32_t) block_wt
    };
    std::vector<uint32_t> writer_mcast_receiver_compile_time_args = {
        0,
        (std::uint32_t) gamma.has_value(),
        (std::uint32_t) beta.has_value(),
        (std::uint32_t) is_dram(gamma),
        (std::uint32_t) is_dram(beta),
        (std::uint32_t) block_wt
    };
    // writer kernel
    auto writer_mcast_sender_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/layernorm/kernels/dataflow/writer_unary_sharded_ln.cpp",
        top_row,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_mcast_sender_compile_time_args, .defines = writer_defines}
    );
    auto writer_mcast_receiver_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/layernorm/kernels/dataflow/writer_unary_sharded_ln.cpp",
        all_except_top_row,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_mcast_receiver_compile_time_args, .defines = writer_defines}
    );
    // defines
    std::map<string, string> eltwise_binary_defines;
    if (b) {
        eltwise_binary_defines["FUSE_PRE_ADD"] = "1";
    }
    // compute kernel compile time args
    std::vector<uint32_t> top_row_compute_compile_time_args = {
        1,
        gamma.has_value(),
        beta.has_value(),
        num_blocks,
        block_ht,
        block_wt,
        subblock_wt,
        num_subblocks_w,
    };
    std::vector<uint32_t> all_except_top_compute_compile_time_args = {
        0,
        gamma.has_value(),
        beta.has_value(),
        num_blocks,
        block_ht,
        block_wt,
        subblock_wt,
        num_subblocks_w,
    };
    // compute kernel
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = true;
    auto compute_kernels_top_row_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/layernorm/kernels/compute/layernorm.cpp",
        top_row,
        tt_metal::ComputeConfig{.math_fidelity = fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = top_row_compute_compile_time_args, .defines = eltwise_binary_defines}
    );
    auto compute_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/layernorm/kernels/compute/layernorm.cpp",
        all_except_top_row,
        tt_metal::ComputeConfig{.math_fidelity = fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = all_except_top_compute_compile_time_args, .defines = eltwise_binary_defines}
    );
    // Create circular buffers
    // in0 sharded
    uint32_t in0_cb_index = CB::c_in0;
    tt_metal::CircularBufferConfig in0_cb_config = tt_metal::CircularBufferConfig(in0_CB_size, {{in0_cb_index, io_format}})
		.set_page_size(in0_cb_index, single_tile_size).set_globally_allocated_address(*a.buffer());
    auto cb_in0 = tt_metal::CreateCircularBuffer(program, all_cores, in0_cb_config);
    // in1 sharded
    uint32_t in1_cb_index = CB::c_in1;
    CircularBufferID cb_in1 = 0;
    if (b) {
        if (b.value().memory_config().is_sharded()) {
            tt_metal::CircularBufferConfig in1_cb_config = tt_metal::CircularBufferConfig(in1_CB_size, {{in1_cb_index, io_format}})
                .set_page_size(in1_cb_index, single_tile_size).set_globally_allocated_address(*b.value().buffer());
            cb_in1 = tt_metal::CreateCircularBuffer(program, all_cores, in1_cb_config);
        } else {
            tt_metal::CircularBufferConfig in1_cb_config = tt_metal::CircularBufferConfig(in1_CB_size, {{in1_cb_index, cb_data_format}})
                .set_page_size(in1_cb_index, single_tile_size);
            cb_in1 = tt_metal::CreateCircularBuffer(program, all_cores, in1_cb_config);
        }
    }
    // in2 scaler
    uint32_t in2_cb_index = CB::c_in2;
    tt_metal::CircularBufferConfig in2_cb_config = tt_metal::CircularBufferConfig(in2_CB_size, {{in2_cb_index, cb_data_format}})
		.set_page_size(in2_cb_index, single_tile_size);
    auto cb_in2 = tt_metal::CreateCircularBuffer(program, all_cores, in2_cb_config);
    // in4 scaler-c
    uint32_t in4_cb_index = CB::c_in4;
    tt_metal::CircularBufferConfig in4_cb_config = tt_metal::CircularBufferConfig(in2_CB_size, {{in4_cb_index, cb_data_format}})
		.set_page_size(in4_cb_index, single_tile_size);
    auto cb_in4 = tt_metal::CreateCircularBuffer(program, all_cores, in4_cb_config);
    // in3 eps
    uint32_t in3_cb_index = CB::c_in3;
    tt_metal::CircularBufferConfig in3_cb_config = tt_metal::CircularBufferConfig(in3_CB_size, {{in3_cb_index, cb_data_format}})
		.set_page_size(in3_cb_index, single_tile_size);
    auto cb_in3 = tt_metal::CreateCircularBuffer(program, all_cores, in3_cb_config);
    // gamma
    if (gamma.has_value()) {
        uint32_t in5_cb_index = CB::c_in5;
        tt_metal::CircularBufferConfig in5_cb_config = tt_metal::CircularBufferConfig(in5_CB_size, {{in5_cb_index, cb_data_format}})
            .set_page_size(in5_cb_index, single_tile_size);
        auto cb_in5 = tt_metal::CreateCircularBuffer(program, all_cores, in5_cb_config);
    }
    // beta
    if (beta.has_value()) {
        uint32_t in6_cb_index = CB::c_in6;
        tt_metal::CircularBufferConfig in6_cb_config = tt_metal::CircularBufferConfig(in6_CB_size, {{in6_cb_index, cb_data_format}})
            .set_page_size(in6_cb_index, single_tile_size);
        auto cb_in6 = tt_metal::CreateCircularBuffer(program, all_cores, in6_cb_config);
    }
    // x
    uint32_t x_cb_index;
    x_cb_index = CB::c_intermed0;
    tt_metal::CircularBufferConfig x_cb_config = tt_metal::CircularBufferConfig(x_CB_size, {{x_cb_index, cb_data_format}})
        .set_page_size(x_cb_index, single_tile_size);
    auto cb_x = tt_metal::CreateCircularBuffer(program, all_cores, x_cb_config);
    // xmm
    uint32_t xmm_cb_index;
    xmm_cb_index = CB::c_intermed1;
    tt_metal::CircularBufferConfig xmm_cb_config = tt_metal::CircularBufferConfig(xmm_CB_size, {{xmm_cb_index, cb_data_format}})
        .set_page_size(xmm_cb_index, single_tile_size);
    auto cb_xmm = tt_metal::CreateCircularBuffer(program, all_cores, xmm_cb_config);
    // ex_partial
    uint32_t ex_cb_partial_index = CB::dataflow0;
    tt_metal::CircularBufferConfig ex_cb_partial_config = tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex_cb_partial_index, cb_data_format}})
		.set_page_size(ex_cb_partial_index, single_tile_size);
    auto cb_ex_partial = tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_partial_config);
    // ex
    uint32_t ex_cb_index = CB::dataflow1;
    tt_metal::CircularBufferConfig ex_cb_config = tt_metal::CircularBufferConfig(ex_CB_size, {{ex_cb_index, cb_data_format}})
		.set_page_size(ex_cb_index, single_tile_size);
    auto cb_ex = tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_config);
    // ex_external
    uint32_t ex_cb_external_index = CB::dataflow2;
    tt_metal::CircularBufferConfig ex_cb_external_config = tt_metal::CircularBufferConfig(ex_external_CB_size, {{ex_cb_external_index, cb_data_format}})
		.set_page_size(ex_cb_external_index, single_tile_size);
    auto cb_ex_external = tt_metal::CreateCircularBuffer(program, top_row, ex_cb_external_config);
    // ex_partial2
    uint32_t ex_cb_partial2_index = CB::dataflow3;
    tt_metal::CircularBufferConfig ex_cb_partial2_config = tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex_cb_partial2_index, cb_data_format}})
		.set_page_size(ex_cb_partial2_index, single_tile_size);
    auto cb_ex_partial2 = tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_partial2_config);
    // ex2
    uint32_t ex2_cb_index = CB::dataflow4;
    tt_metal::CircularBufferConfig ex2_cb_config = tt_metal::CircularBufferConfig(ex_CB_size, {{ex2_cb_index, cb_data_format}})
		.set_page_size(ex2_cb_index, single_tile_size);
    auto cb_ex2 = tt_metal::CreateCircularBuffer(program, all_cores, ex2_cb_config);
    // ex_external2
    uint32_t ex_cb_external2_index = CB::dataflow5;
    tt_metal::CircularBufferConfig ex_cb_external2_config = tt_metal::CircularBufferConfig(ex_external_CB_size, {{ex_cb_external2_index, cb_data_format}})
		.set_page_size(ex_cb_external2_index, single_tile_size);
    auto cb_ex_external2 = tt_metal::CreateCircularBuffer(program, top_row, ex_cb_external2_config);
    // xmm2
    uint32_t xmm2_cb_index;
    xmm2_cb_index = CB::c_intermed2;
    tt_metal::CircularBufferConfig xmm2_cb_config = tt_metal::CircularBufferConfig(xmm2_CB_size, {{xmm2_cb_index, cb_data_format}})
        .set_page_size(xmm2_cb_index, single_tile_size);
    auto cb_xmm2 = tt_metal::CreateCircularBuffer(program, all_cores, xmm2_cb_config);
    // ex2pe
    uint32_t cb_ex2pe_index;
    cb_ex2pe_index = CB::c_intermed3;
    tt_metal::CircularBufferConfig ex2pe_cb_config = tt_metal::CircularBufferConfig(ex2pe_CB_size, {{cb_ex2pe_index, cb_data_format}})
        .set_page_size(cb_ex2pe_index, single_tile_size);
    auto cb_ex2pe = tt_metal::CreateCircularBuffer(program, all_cores, ex2pe_cb_config);
    // fusion
    uint32_t cb_fusion_index;
    cb_fusion_index = CB::c_intermed4;
    tt_metal::CircularBufferConfig fusion_cb_config = tt_metal::CircularBufferConfig(fusion_CB_size, {{cb_fusion_index, cb_data_format}})
        .set_page_size(cb_fusion_index, single_tile_size);
    auto cb_fusion = tt_metal::CreateCircularBuffer(program, all_cores, fusion_cb_config);
    // out
    uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
    tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, {{output_cb_index, io_format}})
		.set_page_size(output_cb_index, single_tile_size).set_globally_allocated_address(*output.buffer());
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
    // NoC coords
    uint32_t diff_start_coord;
    uint32_t diff_end_coord;
    std::vector<uint32_t> in0_mcast_noc_y;
    diff_start_coord = device->worker_core_from_logical_core({0, start_core_y}).y;
    diff_end_coord = device->worker_core_from_logical_core({0, start_core_y + num_cores_r - 1}).y;
    in0_mcast_noc_y.reserve(num_cores_r);
    // mcast but not to self
    for(uint32_t core_idx_y = 0; core_idx_y < num_cores_r; ++core_idx_y) {
        in0_mcast_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
    }

    // Runtime Args
    std::vector<KernelID> writer_kernel_ids;
    float winv = 1.0f / block_w; // bcast-w scaler
    float cinv = 1.0f / num_cores_r; // bcast-cores scaler
    bfloat16 bfloat_cinv_value = bfloat16(cinv);
    uint32_t packed_cinv_value = pack_two_bfloat16_into_uint32({bfloat_cinv_value, bfloat_cinv_value});
    bfloat16 bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    union { float f; uint32_t u; } e; e.f = eps;

    for(int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for(int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};
            CoreCoord top_core = {(std::size_t) core.x, (std::size_t) start_core_y};
            CoreCoord top_core_plus_one = {(std::size_t) core.x, (std::size_t) start_core_y + 1};
            CoreCoord bottom_core = {(std::size_t) core.x, (std::size_t) start_core_y + num_cores_r - 1};

            auto core_physical = device->worker_core_from_logical_core(core);
            auto top_core_physical = device->worker_core_from_logical_core(top_core);
            auto top_core_plus_one_physical = device->worker_core_from_logical_core(top_core_plus_one);
            auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);

            auto mcast_sender = top_core_physical;
            auto mcast_start = bottom_core_physical;
            auto mcast_end = top_core_plus_one_physical;

            uint32_t in1_tile_start_id = (core_idx_x * block_ht * Kt) + (core_idx_y * block_wt);
            uint32_t gamma_tile_start_id = core_idx_y * block_wt;
            uint32_t beta_tile_start_id = core_idx_y * block_wt;

            // top core reader
            if (core_idx_y == 0) {
                uint32_t worker_shard_same_coord;
                std::vector<uint32_t> mcast_sender_args;
                worker_shard_same_coord = device->worker_core_from_logical_core(core).x;
                mcast_sender_args.push_back(mcast_start.x);
                mcast_sender_args.push_back(mcast_start.y);
                mcast_sender_args.push_back(mcast_end.x);
                mcast_sender_args.push_back(mcast_end.y);
                mcast_sender_args.push_back(worker_shard_same_coord);
                mcast_sender_args.insert(mcast_sender_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
                tt_metal::SetRuntimeArgs(program, reader_mcast_sender_kernels_id, core, mcast_sender_args);

                std::vector<uint32_t> writer_mcast_sender_args;
                writer_mcast_sender_args.push_back(packed_cinv_value);
                writer_mcast_sender_args.push_back(packed_winv_value);
                writer_mcast_sender_args.push_back(e.u);
                writer_mcast_sender_args.push_back(gamma_dram_addr);
                writer_mcast_sender_args.push_back(beta_dram_addr);
                writer_mcast_sender_args.push_back(gamma_tile_start_id);
                writer_mcast_sender_args.push_back(beta_tile_start_id);
                tt_metal::SetRuntimeArgs(program, writer_mcast_sender_kernels_id, core, writer_mcast_sender_args);
                writer_kernel_ids.push_back(writer_mcast_sender_kernels_id);
            } else {
                uint32_t worker_shard_same_coord;
                uint32_t worker_shard_diff_coord;
                std::vector<uint32_t> mcast_receiver_args;
                worker_shard_same_coord = device->worker_core_from_logical_core(core).x;
                worker_shard_diff_coord = device->worker_core_from_logical_core(top_core).y;
                mcast_receiver_args.push_back(core_physical.x);
                mcast_receiver_args.push_back(core_physical.y);
                mcast_receiver_args.push_back(worker_shard_same_coord);
                mcast_receiver_args.push_back(worker_shard_diff_coord);
                tt_metal::SetRuntimeArgs(program, reader_mcast_receiver_kernels_id, core, mcast_receiver_args);

                std::vector<uint32_t> writer_mcast_receiver_args;
                writer_mcast_receiver_args.push_back(packed_cinv_value);
                writer_mcast_receiver_args.push_back(packed_winv_value);
                writer_mcast_receiver_args.push_back(e.u);
                writer_mcast_receiver_args.push_back(gamma_dram_addr);
                writer_mcast_receiver_args.push_back(beta_dram_addr);
                writer_mcast_receiver_args.push_back(gamma_tile_start_id);
                writer_mcast_receiver_args.push_back(beta_tile_start_id);
                tt_metal::SetRuntimeArgs(program, writer_mcast_receiver_kernels_id, core, writer_mcast_receiver_args);
                writer_kernel_ids.push_back(writer_mcast_receiver_kernels_id);
            }
        }
    }

    auto override_runtime_args_callback = [
            writer_kernel_ids,
            cb_in0,
            cb_in1,
            cb_output,
            num_cores,
            grid_size
        ]
    (
        const void* operation,
        Program &program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_buffer_a = input_tensors.at(0).buffer();
        auto b_tensor = optional_input_tensors.at(0);
        auto gamma_tensor = optional_input_tensors.at(1);
        auto beta_tensor = optional_input_tensors.at(2);
        auto dst_buffer = output_tensors.at(0).buffer();

        auto& in0_cb_config = GetCircularBufferConfig(program, cb_in0);
            in0_cb_config.set_globally_allocated_address(*src_buffer_a);

        if (b_tensor.has_value()) {
            auto& in1_cb_config = GetCircularBufferConfig(program, cb_in1);
                in1_cb_config.set_globally_allocated_address(*b_tensor.value().buffer());
        }

        auto& output_cb_config = GetCircularBufferConfig(program, cb_output);
            output_cb_config.set_globally_allocated_address(*dst_buffer);

        int index=0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};

            auto writer_kernel_id = writer_kernel_ids.at(index);

            auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);

            if (gamma_tensor.has_value()) {
                runtime_args[3] = gamma_tensor.value().buffer()->address();
            }
            if (beta_tensor.has_value()) {
                runtime_args[4] = beta_tensor.value().buffer()->address();
            }
            index++;
        }
    };

    return {std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

void LayerNorm::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(input_tensors.size() == 1 and optional_input_tensors.size() <= 3, "Must have between 1 to 4 input tensors");
    auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    TT_ASSERT(a.layout() == Layout::TILE);
    TT_ASSERT(a.dtype() == DataType::BFLOAT16 or a.dtype() == DataType::BFLOAT8_B);
    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");

    if (b.has_value()) {
        TT_ASSERT(b.value().layout() == Layout::TILE, "layot is not tile!");
        TT_ASSERT(a.shape() == b.value().shape(), "shape is not same!");
        TT_ASSERT(a.device() == b.value().device(), "device is not same!");
        TT_ASSERT(b.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    }

    if (gamma.has_value()) {
        if (gamma.value().layout() == Layout::TILE) {
            TT_ASSERT(a.shape()[3] == gamma.value().shape()[3], fmt::format("{} != {}", a.shape()[3], gamma.value().shape()[3]));
            TT_ASSERT(a.device() == gamma.value().device());
            TT_ASSERT(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_ASSERT(gamma.value().shape()[2] == TILE_HEIGHT);
        } else {
            TT_ASSERT(gamma.value().layout() == Layout::ROW_MAJOR);
            TT_ASSERT((gamma.value().shape()[3] == TILE_WIDTH && gamma.value().volume() / TILE_WIDTH == a.shape()[3] / TILE_WIDTH));
            TT_ASSERT(a.device() == gamma.value().device());
            TT_ASSERT(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_ASSERT(gamma.value().dtype() == DataType::BFLOAT16);
        }
    }

    if (beta.has_value()) {
        if (beta.value().layout() == Layout::TILE) {
            TT_ASSERT(a.shape()[3] == beta.value().shape()[3]);
            TT_ASSERT(a.device() == beta.value().device());
            TT_ASSERT(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_ASSERT(beta.value().shape()[2] == TILE_HEIGHT);
        } else {
            TT_ASSERT(beta.value().layout() == Layout::ROW_MAJOR);
            TT_ASSERT((beta.value().shape()[3] == TILE_WIDTH && beta.value().volume() / TILE_WIDTH == a.shape()[3] / TILE_WIDTH));
            TT_ASSERT(a.device() == beta.value().device());
            TT_ASSERT(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_ASSERT(beta.value().dtype() == DataType::BFLOAT16);
        }
    }
}
std::vector<Shape> LayerNorm::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}
std::vector<Tensor> LayerNorm::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return std::visit(
        [&](const auto& program_config) -> std::vector<Tensor> {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                std::is_same_v<ProgramConfigType, tt::operations::primary::LayerNormShardedMultiCoreProgramConfig>
            ) {
                uint32_t M = input_tensor.volume() / input_tensor.shape()[-1] / TILE_HEIGHT;
                uint32_t K = input_tensor.shape()[-1] / TILE_WIDTH;
                uint32_t num_cores_x = program_config.compute_with_storage_grid_size.x;
                uint32_t num_cores_y = program_config.compute_with_storage_grid_size.y;
                uint32_t per_core_M = M / num_cores_x;
                uint32_t per_core_N = K / num_cores_y;

                CoreRangeSet all_cores({});
                ShardOrientation shard_orientation;
                all_cores = CoreRangeSet({CoreRange{.start={0, 0}, .end={num_cores_x - 1, num_cores_y - 1}}});
                shard_orientation = ShardOrientation::COL_MAJOR;
                ShardSpec shard_spec = ShardSpec{.shard_grid=all_cores, .shard_shape={per_core_M * TILE_HEIGHT, per_core_N * TILE_WIDTH}, .shard_orientation=shard_orientation};
                return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensor.dtype(), Layout::TILE, input_tensor.device(), this->output_mem_config, shard_spec)};
            } else {
                return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
            }
        },
        this->program_config
    );
}
operation::ProgramWithCallbacks LayerNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);

    return std::visit(
        [&](const auto& program_config) -> operation::ProgramWithCallbacks {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                std::is_same_v<ProgramConfigType, tt::operations::primary::LayerNormShardedMultiCoreProgramConfig>
            ) {
                MathFidelity fidelity = program_config.math_fidelity;
                uint32_t num_cores_x = program_config.compute_with_storage_grid_size.x;
                uint32_t num_cores_y = program_config.compute_with_storage_grid_size.y;
                CoreCoord grid_size = CoreCoord(num_cores_x, num_cores_y);

                return layernorm_sharded_(
                                            a, b, gamma, beta, output_tensor, this->eps,
                                            fidelity,
                                            program_config.data_format,
                                            grid_size,
                                            program_config.subblock_w,
                                            program_config.block_h,
                                            program_config.block_w
                                            );
            } else {
                return layernorm_(a, b, gamma, beta, output_tensor, this->eps);
            }
        },
        this->program_config
    );
}
tt::stl::reflection::Attributes LayerNorm::attributes() const {
    return {
        {"eps", this->eps},
        {"output_mem_config", this->output_mem_config},
    };
}

} // namespace primary

} // namespace operations

}  // namespace tt
