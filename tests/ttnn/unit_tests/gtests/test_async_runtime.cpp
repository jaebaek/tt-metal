// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/tensor.hpp"
#include "ttnn_multi_command_queue_fixture.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "common/bfloat16.hpp"
#include "ttnn/cpp/ttnn/async_runtime.hpp"
#include "tt_numpy/functions.hpp"
#include <cmath>

using namespace tt;
using namespace tt_metal;
using MultiCommandQueueSingleDeviceFixture = ttnn::MultiCommandQueueSingleDeviceFixture;
using namespace constants;

TEST_F(MultiCommandQueueSingleDeviceFixture, TestAsyncPreallocatedOutputs) {
    Device* device = this->device_;
    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};

    uint32_t input_buf_size_datums = 1024 * 1024;
    uint32_t output_buf_size_datums = 1024 * 32;
    uint32_t datum_size_bytes = 2;
    uint32_t io_cq = 1; // Data reads and writes done through CQ0
    uint32_t workload_dispatch_cq = 0; // Workload dispatched through CQ1

    ttnn::Shape input_shape = ttnn::Shape(Shape({1, 1, 1024, 1024}));
    auto host_data = std::shared_ptr<bfloat16 []>(new bfloat16[input_buf_size_datums]);
    auto readback_data = std::shared_ptr<bfloat16 []>(new bfloat16[output_buf_size_datums]);


    for (int i = 0; i < input_buf_size_datums; i++) {
        host_data[i] = bfloat16(static_cast<float>(1));
    }
    // Create golden data using tt_eager APIs
    Tensor np_tensor = tt::numpy::full<float>(input_shape.value(), static_cast<float>(1), DataType::BFLOAT16).to(Layout::TILE).to(device);
    std::vector<int64_t> reduce_dims = {3};
    Tensor np_out = tt::operations::primary::moreh_sum(np_tensor, reduce_dims);
    Tensor np_out_host = np_out.cpu();
    const bfloat16* golden_output = std::get<owned_buffer::Buffer<bfloat16>>(std::get<OwnedStorage>(np_out_host.get_storage()).buffer).begin();
    // Enable Asynchronous Execution and test ttnn runtime APIs
    device->set_worker_mode(WorkExecutorMode::ASYNCHRONOUS);
    // Events for host - device synchronization
    auto write_event = std::make_shared<Event>();
    auto workload_event = std::make_shared<Event>();
    // Running sum-reduce with preallocated output
    auto op = tt::operations::primary::MorehSum{.dim = 3};
    // Preallocate Input and Output Tensors on Device
    auto input_buffer = ttnn::allocate_buffer_on_device(input_buf_size_datums * datum_size_bytes, device, input_shape, DataType::BFLOAT16, Layout::TILE, mem_cfg);
    auto output_buffer = ttnn::allocate_buffer_on_device(output_buf_size_datums * datum_size_bytes, device, np_out.get_shape(), DataType::BFLOAT16, Layout::TILE, mem_cfg);
    auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};
    auto output_storage = tt::tt_metal::DeviceStorage{output_buffer};
    Tensor input_tensor = Tensor(input_storage, input_shape, DataType::BFLOAT16, Layout::TILE);
    Tensor output_tensor = Tensor(output_storage, np_out.get_shape(), DataType::BFLOAT16, Layout::TILE);
    // Populate input_tensor with data
    ttnn::write_buffer(io_cq, input_tensor, {host_data});
    // Record the completion of the write event
    ttnn::record_event(device->command_queue(io_cq), write_event);
    // Host stalls until write is completed, before sending workload
    ttnn::event_synchronize(write_event);
    // Dispatch workload. Preallocated output_tensor is populated by op/
    ttnn::run_operation(workload_dispatch_cq, op, {input_tensor}, {}, {output_tensor}).at(0);
    // Record completion of workload
    ttnn::record_event(device->command_queue(workload_dispatch_cq), workload_event);
    ttnn::event_synchronize(workload_event);
    // Read output back, once workload is complete
    ttnn::read_buffer(io_cq, output_tensor, {readback_data});
    // Ensure that reference count book keeping is done correctly
    // Tensors only have one reference in the main thread. Ensure this is true.
    EXPECT_EQ(input_tensor.tensor_attributes->main_thread_ref_count, 1);
    EXPECT_EQ(output_tensor.tensor_attributes->main_thread_ref_count, 1);
    // Buffers are currently jointly owned by the original buffer object, the storage object and the tensor (3).
    EXPECT_EQ(input_buffer.use_count(), 3);
    EXPECT_EQ(output_buffer.use_count(), 3);
    // Deallocate tensors (tensor gives up buffer). Done asynchronously, so sync on queue after.
    input_tensor.deallocate();
    output_tensor.deallocate();
    ttnn::queue_synchronize(device->command_queue(io_cq));
    // Buffer only has 2 owners in main thread.
    EXPECT_EQ(input_buffer.use_count(), 2);
    EXPECT_EQ(output_buffer.use_count(), 2);
    for (int i = 0; i  < output_buf_size_datums; i++) {
        EXPECT_EQ(readback_data[i], golden_output[i]);
    }
}

TEST_F(MultiCommandQueueSingleDeviceFixture, TestAsyncRuntimeAllocatedBuffers) {
    Device* device = this->device_;
    device->set_worker_mode(WorkExecutorMode::ASYNCHRONOUS);
    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};

    uint32_t buf_size_datums = 1024 * 1024;
    uint32_t datum_size_bytes = 2;
    std::vector<uint32_t> inputs = {4, 9, 16, 25, 36, 64};
    uint32_t io_cq = 1;
    uint32_t workload_dispatch_cq = 0;
    ttnn::Shape shape = ttnn::Shape(Shape({1, 1, 1024, 1024}));

    auto host_data = std::shared_ptr<bfloat16 []>(new bfloat16[buf_size_datums]);
    auto readback_data = std::shared_ptr<bfloat16 []>(new bfloat16[buf_size_datums]);
    for (int loop = 0; loop < 10; loop++) {
        log_info(LogTest, "Running outer loop {}", loop);
        for (auto input_val : inputs) {
            for (int i = 0; i < buf_size_datums; i++) {
                host_data[i] = bfloat16(static_cast<float>(input_val));
            }

            auto write_event = std::make_shared<Event>();
            auto workload_event = std::make_shared<Event>();
            auto input_buffer = ttnn::allocate_buffer_on_device(buf_size_datums * datum_size_bytes, device, shape, DataType::BFLOAT16, Layout::TILE, mem_cfg);
            auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};
            Tensor input_tensor = Tensor(input_storage, shape, DataType::BFLOAT16, Layout::TILE);
            ttnn::write_buffer(io_cq, input_tensor, {host_data}); // Write using cq 1
            ttnn::record_event(device->command_queue(io_cq), write_event); // Record write on cq 1
            // Wait until cq 1 write is complete
            ttnn::event_synchronize(write_event);
            auto op0 = tt::tt_metal::EltwiseUnary{std::vector{tt::tt_metal::UnaryWithParam{tt::tt_metal::UnaryOpType::SQRT}}};
            auto op1 = tt::tt_metal::EltwiseUnary{std::vector{tt::tt_metal::UnaryWithParam{tt::tt_metal::UnaryOpType::NEG}}};
            // Run operation on cq 0
            Tensor output_tensor = ttnn::run_operation(workload_dispatch_cq, op0, {input_tensor}).at(0);
            auto dummy_buffer_0 = ttnn::allocate_buffer_on_device(buf_size_datums * datum_size_bytes, device, shape, DataType::BFLOAT16, Layout::TILE, mem_cfg);
            output_tensor = ttnn::run_operation(workload_dispatch_cq, op1, {output_tensor}).at(0);
            // Allocate this buffer to stress test async allocation across op execution and explicit allocation
            auto dummy_buffer_1 = ttnn::allocate_buffer_on_device(buf_size_datums * datum_size_bytes, device, shape, DataType::BFLOAT16, Layout::TILE, mem_cfg);
            // Record cq 0 prog execution
            ttnn::record_event(device->command_queue(workload_dispatch_cq), workload_event);
            // Wait until cq 0 prog execution is done
            ttnn::event_synchronize(workload_event);
            // Read using cq 1
            ttnn::read_buffer(io_cq, output_tensor, {readback_data});
            for (int i = 0; i < buf_size_datums; i++) {
                EXPECT_EQ(static_cast<int>(std::floor(bfloat16(readback_data[i]).to_float())), static_cast<int>(-1 * sqrt(input_val)));
            }
        }
    }
}
