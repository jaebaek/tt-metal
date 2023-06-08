#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/test_utils/env_vars.hpp"
#include "bfloat16.hpp"
#include "tests/tt_metal/tt_metal/sfpu_helper/sfpu_helper.hpp"
#include "catch.hpp"

using namespace tt;


// Reader reads from single dram to core, writer synchronizes with datacopy kernel and writes to dram
// DRAM --> (Reader Core CB using reader RISCV)
// Reader Core --> Datacopy --> Reader Core
// Reader Core --> Writes to Dram
bool single_core_sfpu(
    tt_metal::Device* device,
    const size_t& num_tiles,
    const size_t& tile_byte_size,
    const size_t& output_dram_channel,
    const size_t& output_dram_byte_address,
    const size_t& input_dram_channel,
    const size_t& input_dram_byte_address,
    const size_t& local_core_input_byte_address,
    const tt::DataFormat& local_core_input_data_format,
    const size_t& local_core_output_byte_address,
    const tt::DataFormat& local_core_output_data_format,
    const CoreCoord& core,
    const string& sfpu_op
) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const size_t byte_size = num_tiles*tile_byte_size;
    tt_metal::Program program = tt_metal::Program();
    auto input_dram_buffer = tt_metal::Buffer(
        device, byte_size, input_dram_byte_address, input_dram_channel, byte_size, tt_metal::BufferType::DRAM);
    auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();
    auto output_dram_buffer = tt_metal::Buffer(
        device, byte_size, output_dram_byte_address, output_dram_channel, byte_size, tt_metal::BufferType::DRAM);
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    auto l1_input_cb = tt_metal::CreateCircularBuffer(
        program,
        device,
        0,
        core,
        num_tiles,
        byte_size,
        local_core_input_byte_address,
        local_core_input_data_format
        );
    auto l1_output_cb = tt_metal::CreateCircularBuffer(
        program,
        device,
        16,
        core,
        num_tiles,
        byte_size,
        local_core_output_byte_address,
        local_core_output_data_format
        );

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        uint32_t(num_tiles), // per_core_block_cnt
        1 // per_core_block_cnt
    };
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto sfpu_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );
    sfpu_kernel->add_define("SFPU_OP_AND_PACK", sfpu_op_to_hlk_op_name.at(sfpu_op));
    bool is_relu = (sfpu_op == "relu");
    sfpu_kernel->add_define("INIT_RELU", is_relu ? "pack_relu_config(1);" : "");
    sfpu_kernel->add_define("DEINIT_RELU", is_relu ? "pack_relu_config(0);" : "");
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CompileProgram(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> inputs = sfpu_op_to_init_func.at(sfpu_op)(
        byte_size, std::chrono::system_clock::now().time_since_epoch().count());
    tt_metal::WriteToBuffer(input_dram_buffer, inputs);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::WriteRuntimeArgsToDevice(
        device,
        reader_kernel,
        core,
        {
            (uint32_t)input_dram_byte_address,
            (uint32_t)input_dram_noc_xy.x,
            (uint32_t)input_dram_noc_xy.y,
            (uint32_t)num_tiles,
        }
    );
    pass &= tt_metal::WriteRuntimeArgsToDevice(
        device,
        writer_kernel,
        core,
        {
            (uint32_t)output_dram_byte_address,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)num_tiles,
        }
    );
    pass &= tt_metal::LaunchKernels(device, program);


    std::vector<uint32_t> golden = sfpu(inputs, sfpu_op_to_function.at(sfpu_op));
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= packed_uint32_t_vector_comparison(dest_buffer_data, golden, sfpu_op_to_comparison_function.at(sfpu_op));
    return pass;
}


TEST_CASE(
    "single_core_relu", "[compute][single_core][sfpu]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    REQUIRE(
        single_core_sfpu(
            device,
            1,
            2*32*32,
            1,
            0,
            0,
            16*1024,
            UNRESERVED_BASE,
            tt::DataFormat::Float16_b,
            UNRESERVED_BASE + 16*32*32,
            tt::DataFormat::Float16_b,
            {.x=0, .y=0},
            "relu"
        )
    );
    tt_metal::CloseDevice(device);
}
TEST_CASE(
    "single_core_exponential", "[compute][single_core][sfpu]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    REQUIRE(
        single_core_sfpu(
            device,
            1,
            2*32*32,
            1,
            0,
            0,
            16*1024,
            UNRESERVED_BASE,
            tt::DataFormat::Float16_b,
            UNRESERVED_BASE + 16*32*32,
            tt::DataFormat::Float16_b,
            {.x=0, .y=0},
            "exponential"
        )
    );
    tt_metal::CloseDevice(device);
}

TEST_CASE(
    "single_core_reciprocal", "[compute][single_core][sfpu]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    REQUIRE(
        single_core_sfpu(
            device,
            1,
            2*32*32,
            1,
            0,
            0,
            16*1024,
            UNRESERVED_BASE,
            tt::DataFormat::Float16_b,
            UNRESERVED_BASE + 16*32*32,
            tt::DataFormat::Float16_b,
            {.x=0, .y=0},
            "reciprocal"
        )
    );
    tt_metal::CloseDevice(device);
}

TEST_CASE(
    "single_core_gelu", "[compute][single_core][sfpu]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    REQUIRE(
        single_core_sfpu(
            device,
            1,
            2*32*32,
            1,
            0,
            0,
            16*1024,
            UNRESERVED_BASE,
            tt::DataFormat::Float16_b,
            UNRESERVED_BASE + 16*32*32,
            tt::DataFormat::Float16_b,
            {.x=0, .y=0},
            "gelu"
        )
    );
    tt_metal::CloseDevice(device);
}

TEST_CASE(
    "single_core_sqrt", "[compute][single_core][sfpu]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    REQUIRE(
        single_core_sfpu(
            device,
            1,
            2*32*32,
            1,
            0,
            0,
            16*1024,
            UNRESERVED_BASE,
            tt::DataFormat::Float16_b,
            UNRESERVED_BASE + 16*32*32,
            tt::DataFormat::Float16_b,
            {.x=0, .y=0},
            "sqrt"
        )
    );
    tt_metal::CloseDevice(device);
}

TEST_CASE(
    "single_core_sigmoid", "[compute][single_core][sfpu]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    REQUIRE(
        single_core_sfpu(
            device,
            1,
            2*32*32,
            1,
            0,
            0,
            16*1024,
            UNRESERVED_BASE,
            tt::DataFormat::Float16_b,
            UNRESERVED_BASE + 16*32*32,
            tt::DataFormat::Float16_b,
            {.x=0, .y=0},
            "sigmoid"
        )
    );
    tt_metal::CloseDevice(device);
}

TEST_CASE(
    "single_core_log", "[compute][single_core][sfpu]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    REQUIRE(
        single_core_sfpu(
            device,
            1,
            2*32*32,
            1,
            0,
            0,
            16*1024,
            UNRESERVED_BASE,
            tt::DataFormat::Float16_b,
            UNRESERVED_BASE + 16*32*32,
            tt::DataFormat::Float16_b,
            {.x=0, .y=0},
            "log"
        )
    );
    tt_metal::CloseDevice(device);
}

TEST_CASE(
    "single_core_tanh", "[compute][single_core][sfpu]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    REQUIRE(
        single_core_sfpu(
            device,
            1,
            2*32*32,
            1,
            0,
            0,
            16*1024,
            UNRESERVED_BASE,
            tt::DataFormat::Float16_b,
            UNRESERVED_BASE + 16*32*32,
            tt::DataFormat::Float16_b,
            {.x=0, .y=0},
            "tanh"
        )
    );
    tt_metal::CloseDevice(device);
}
