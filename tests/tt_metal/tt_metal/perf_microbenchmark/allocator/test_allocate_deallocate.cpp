// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"

using namespace tt;
using namespace tt::tt_metal;
using std::chrono::duration_cast;
using std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This test measures time after series of allocations and deallocations
//
// Run ./test_allocate_deallocate --help to see usage
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    std::vector<double> alloc_dealloc_times;
    uint32_t num_iters = 10;
    uint32_t seed = 0;
    uint32_t total_num_pages = 50000;

    try {
        // Input arguments parsing
        std::vector<std::string> input_args(argv, argv + argc);

        if (test_args::has_command_option(input_args, "-h") ||
            test_args::has_command_option(input_args, "--help")) {
            log_info(LogTest, "Usage:");
            log_info(LogTest, "  --num-iters: number of iterations (default 10)");
            log_info(LogTest, "  --seed: seed (default 0)");
            log_info(LogTest,
                "  --total-num-pages: total number of pages to be allocated within an iteration. "
                "Each buffer can have max of 16 pages. Each page is 2048B (default 50000)");
            exit(0);
        }

        try {
            std::tie(num_iters, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-iters", 10);

            std::tie(seed, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--seed", 0);

            std::tie(total_num_pages, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--total-num-pages", 50000);

            test_args::validate_remaining_args(input_args);
        } catch (const std::exception& e) {
            log_error(tt::LogTest, "Command line arguments found exception", e.what());
        }

        srand(seed);

        // Device setup
        int device_id = 0;
        tt_metal::Device* device = tt_metal::CreateDevice(device_id);

        // log_info(LogTest,
        //     "Measuring host-to-device bandwidth for "
        //     "copy_mode={} ",
        //     total_transfer_size);

        std::vector<unique_ptr<Buffer>> buffers;

        uint32_t max_num_pages_per_buffer = 16;
        uint32_t page_size = 2048;
        BufferType buftype = BufferType::DRAM;

        for (uint32_t i = 0; i < num_iters; i++) {
            uint32_t num_pages_left = total_num_pages;
            auto t_begin = std::chrono::steady_clock::now();

            unique_ptr<Buffer> buf;
            while (num_pages_left) {
                uint32_t num_pages = std::min(rand() % (max_num_pages_per_buffer) + 1, num_pages_left);
                num_pages_left -= num_pages;
                size_t buf_size = num_pages * page_size;

                try {
                    buf = std::make_unique<Buffer>(device, buf_size, page_size, buftype);
                } catch (...) { // if OOM is hit
                    buffers.clear();
                    buf = std::make_unique<Buffer>(device, buf_size, page_size, buftype);
                }
                buffers.push_back(std::move(buf));
            }

            buffers.clear();

            auto t_end = std::chrono::steady_clock::now();

            auto elapsed_us = duration_cast<microseconds>(t_end - t_begin).count();
            log_info(LogTest, "Total time: {:.3f}ms", elapsed_us / 1000.0);

            alloc_dealloc_times.push_back(elapsed_us);
        }

        tt_metal::CloseDevice(device);

    } catch (const std::exception& e) {
        log_error(LogTest, "{}", e.what());
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    auto avg_time = calculate_average(alloc_dealloc_times);

    log_info(LogTest, "Average time: {:.3f}ms", avg_time / 1000.0);

    return 0;
}
