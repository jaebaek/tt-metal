#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt;

bool RunCustomCycle(tt_metal::Device *device, int loop_count, vector<CoreCoord>* worker_cores)
{
    bool pass = true;

    CoreCoord compute_with_storage_size = device->compute_with_storage_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};
    CoreRange all_cores{.start=start_core, .end=end_core};

    tt_metal::Program program = tt_metal::Program();

    constexpr int loop_size = 200;
    std::map<string, string> kernel_defines = {
        {"LOOP_COUNT", std::to_string(loop_count)},
        {"LOOP_SIZE", std::to_string(loop_size)}
    };

    tt_metal::KernelID brisc_kernel = tt_metal::CreateDataMovementKernel(
            program, "tt_metal/programming_examples/profiler/device/grayskull/test_full_buffer/kernels/full_buffer.cpp",
            all_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .defines = kernel_defines});

    tt_metal::KernelID ncrisc_kernel = tt_metal::CreateDataMovementKernel(
            program, "tt_metal/programming_examples/profiler/device/grayskull/test_full_buffer/kernels/full_buffer.cpp",
            all_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .defines = kernel_defines});

    vector<uint32_t> trisc_kernel_args = {};
    tt_metal::KernelID trisc_kernel = tt_metal::CreateComputeKernel(
            program, "tt_metal/programming_examples/profiler/device/grayskull/test_full_buffer/kernels/full_buffer_compute.cpp",
            all_cores,
            tt_metal::ComputeConfig{.compile_args = trisc_kernel_args, .defines = kernel_defines}
            );

    pass &= tt_metal::CompileProgram(device, program);
    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

    pass &= tt_metal::LaunchKernels(device, program);
    pass &= tt_metal::LaunchKernels(device, program);
    tt_metal::detail::DumpDeviceProfileResults(device, program);

    *worker_cores = device->worker_cores_from_logical_cores(program.logical_cores());
    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);

        vector<CoreCoord> worker_cores;

        int loop_count = 20;
        pass &= RunCustomCycle(device, loop_count, &worker_cores);
        //tt_metal::detail::DumpDeviceProfileResults(device, worker_cores);

        //loop_count = 40;
        //pass &= RunCustomCycle(device, loop_count, &worker_cores);

        pass &= tt_metal::CloseDevice(device);

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
