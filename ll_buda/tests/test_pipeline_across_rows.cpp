#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"
#include "hostdevcommon/common_values.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();

        // set up the program

        // saturate DRAM
        // uint32_t num_cores = 10;
        // uint32_t num_tiles = 2048;
        // uint32_t block_size_tiles = 16;
        // uint32_t num_blocks_in_CB = 2;
        // uint32_t IO_data_in_dram = true;
        // uint32_t num_repetitions = 4;

        // saturate L1
        // uint32_t num_cores = 10;
        // uint32_t num_tiles = 384;
        // uint32_t block_size_tiles = 16;
        // uint32_t num_blocks_in_CB = 2;
        // uint32_t IO_data_in_dram = false;
        // uint32_t num_repetitions = 64;

        uint32_t num_cores = 10;
        uint32_t num_tiles = 384;
        uint32_t block_size_tiles = 1;
        uint32_t num_blocks_in_CB = 2;
        uint32_t IO_data_in_dram = false;
        uint32_t num_repetitions = 128;

        TT_ASSERT(num_cores >= 2 && num_cores <= 12); // grayskull
        TT_ASSERT(num_tiles % block_size_tiles == 0);

        std::vector<tt_xy_pair> cores;
        for (uint32_t i = 0; i < num_cores; i++) {
            cores.push_back({i, 0});
        }

        log_info(LogTest, "num_cores: {}", num_cores);
        log_info(LogTest, "num_tiles: {}", num_tiles);
        log_info(LogTest, "block_size_tiles: {}", block_size_tiles);
        log_info(LogTest, "num_blocks_in_CB: {}", num_blocks_in_CB);
        log_info(LogTest, "IO_data_in_DRAM: {}", IO_data_in_dram);
        log_info(LogTest, "num_repetitions: {}", num_repetitions);

        uint32_t single_tile_size = 2 * 1024;

        // source and destination buffers
        uint32_t buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t total_bytes_moved = buffer_size * num_repetitions;
        log_info(LogTest, "total_bytes_moved: {}", total_bytes_moved);

        // circular buffers in L1
        uint32_t cb_index = 8;
        uint32_t cb_addr = 120 * 1024;
        uint32_t cb_size_tiles = num_blocks_in_CB * block_size_tiles;
        uint32_t cb_size_bytes = cb_size_tiles * single_tile_size;

        for (auto core : cores) {
            auto cb = ll_buda::CreateCircularBuffer(
                program,
                cb_index,
                core,
                cb_size_tiles,
                cb_size_bytes,
                cb_addr,
                tt::DataFormat::Float16_b
            );
        }

        /// used only if IO data in DRAM
        ll_buda::DramBuffer* src_dram_buffer;
        ll_buda::DramBuffer* dst_dram_buffer;

        // used only if IO data in L1
        ll_buda::L1Buffer* src_l1_buffer;
        ll_buda::L1Buffer* dst_l1_buffer;

        uint32_t src_address;
        tt_xy_pair src_noc_xy;
        uint32_t dst_address;
        tt_xy_pair dst_noc_xy;

        if (IO_data_in_dram) {
            uint32_t dram_buffer_addr = 0;
            TT_ASSERT(dram_buffer_addr + buffer_size <= 1024 * 1024 * 1024); // 1GB

            src_dram_buffer = ll_buda::CreateDramBuffer(0, buffer_size, dram_buffer_addr);
            dst_dram_buffer = ll_buda::CreateDramBuffer(7, buffer_size, dram_buffer_addr);

            src_address = src_dram_buffer->address();
            src_noc_xy = src_dram_buffer->noc_coordinates(device);
            dst_address = dst_dram_buffer->address();
            dst_noc_xy = dst_dram_buffer->noc_coordinates(device);
        } else {
            uint32_t l1_buffer_addr = cb_addr + cb_size_bytes;
            TT_ASSERT(l1_buffer_addr + buffer_size <= 1024 * 1024); // 1 MB

            src_l1_buffer = ll_buda::CreateL1Buffer(program, cores[0],           buffer_size, l1_buffer_addr);
            dst_l1_buffer = ll_buda::CreateL1Buffer(program, cores[num_cores-1], buffer_size, l1_buffer_addr);

            src_address = src_l1_buffer->address();
            src_noc_xy = device->worker_core_from_logical_core(src_l1_buffer->core());
            dst_address = dst_l1_buffer->address();
            dst_noc_xy = device->worker_core_from_logical_core(dst_l1_buffer->core());
        }

        // semaphores in L1, 32B aligned for NOC transfers
        uint32_t sender_semaphore_addr = 109600;
        uint32_t receiver_semaphore_addr = 109632;
        TT_ASSERT(sender_semaphore_addr % 32 == 0);
        TT_ASSERT(receiver_semaphore_addr % 32 == 0);

        // create kernels
        vector<ll_buda::DataMovementKernel*> receiver_kernels;
        vector<ll_buda::DataMovementKernel*> sender_kernels;
        for (int core_id = 0; core_id < num_cores; core_id++) {

            string receiver_kernel_name;
            if (core_id == 0) {
                receiver_kernel_name = "kernels/dataflow/reader_first_stage.cpp";
            } else {
                receiver_kernel_name = "kernels/dataflow/receiver_intermediate_stage.cpp";
            }

            ll_buda::DataMovementKernelArgs* receiver_kernel_compile_time_args = ll_buda::InitializeCompileTimeDataMovementKernelArgs(cores[core_id], {cb_index, block_size_tiles});
            receiver_kernels.push_back(ll_buda::CreateDataMovementKernel(
                program,
                receiver_kernel_name,
                cores[core_id],
                receiver_kernel_compile_time_args,
                ll_buda::DataMovementProcessor::RISCV_1,
                ll_buda::NOC::RISCV_1_default));

            string sender_kernel_name;
            if (core_id == num_cores - 1) {
                sender_kernel_name = "kernels/dataflow/writer_last_stage.cpp";
            } else {
                sender_kernel_name = "kernels/dataflow/sender_intermediate_stage.cpp";
            }
            ll_buda::DataMovementKernelArgs* sender_kernel_compile_time_args = ll_buda::InitializeCompileTimeDataMovementKernelArgs(cores[core_id], {cb_index, block_size_tiles});
            sender_kernels.push_back(ll_buda::CreateDataMovementKernel(
                program,
                sender_kernel_name,
                cores[core_id],
                sender_kernel_compile_time_args,
                ll_buda::DataMovementProcessor::RISCV_0,
                ll_buda::NOC::RISCV_0_default));
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        constexpr bool profile_kernel = true;
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc, profile_kernel);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        // send input data to the device
        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

        if (IO_data_in_dram) {
            pass &= ll_buda::WriteToDeviceDRAM(device, src_dram_buffer, src_vec);
        } else {
            pass &= ll_buda::WriteToDeviceL1(device, src_l1_buffer->core(), src_vec, src_l1_buffer->address());
        }
        // host initializes only the sender's semaphores, reciver's semaphores are initialized by the kernel
        std::vector<uint32_t> invalid = {INVALID};
        for (auto core : cores) {
            ll_buda::WriteToDeviceL1(device, core, invalid, sender_semaphore_addr);
        }

        // send run-time kernel arguments
        for (int core_id = 0; core_id < num_cores; core_id++) {
            if (core_id == 0) {
                ll_buda::WriteRuntimeArgsToDevice(
                    device,
                    receiver_kernels[core_id],
                    cores[core_id],
                    {src_address,
                    (uint32_t)src_noc_xy.x,
                    (uint32_t)src_noc_xy.y,
                    (uint32_t)num_tiles,
                    (uint32_t)num_repetitions});
            } else {
                ll_buda::WriteRuntimeArgsToDevice(
                    device,
                    receiver_kernels[core_id],
                    cores[core_id],
                    {(uint32_t)device->worker_core_from_logical_core(cores[core_id-1]).x,
                    (uint32_t)device->worker_core_from_logical_core(cores[core_id-1]).y,
                    (uint32_t)num_tiles,
                    (uint32_t)sender_semaphore_addr,
                    (uint32_t)receiver_semaphore_addr,
                    (uint32_t)num_repetitions});
            }

            if (core_id == num_cores - 1) {
                ll_buda::WriteRuntimeArgsToDevice(
                    device,
                    sender_kernels[core_id],
                    cores[core_id],
                    {dst_address,
                    (uint32_t)dst_noc_xy.x,
                    (uint32_t)dst_noc_xy.y,
                    (uint32_t)num_tiles,
                    (uint32_t)num_repetitions});
            } else {
                ll_buda::WriteRuntimeArgsToDevice(
                    device,
                    sender_kernels[core_id],
                    cores[core_id],
                    {(uint32_t)device->worker_core_from_logical_core(cores[core_id+1]).x,
                    (uint32_t)device->worker_core_from_logical_core(cores[core_id+1]).y,
                    (uint32_t)num_tiles,
                    (uint32_t)sender_semaphore_addr,
                    (uint32_t)receiver_semaphore_addr,
                    (uint32_t)num_repetitions});
            }
        }

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program, profile_kernel);
        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        if (IO_data_in_dram) {
            ll_buda::ReadFromDeviceDRAM(device, dst_dram_buffer, result_vec, dst_dram_buffer->size());
        } else {
            ll_buda::ReadFromDeviceL1(device, dst_l1_buffer->core(), dst_l1_buffer->address(), result_vec, dst_l1_buffer->size());
        }
            ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= (src_vec == result_vec);

        pass &= ll_buda::CloseDevice(device);;

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
