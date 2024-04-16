// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/command_queue.hpp"

#include <algorithm>  // for copy() and assign()
#include <iterator>   // for back_inserter
#include <memory>
#include <string>

#include "allocator/allocator.hpp"
#include "debug_tools.hpp"
#include "dev_msgs.h"
#include "tt_metal/common/logger.hpp"
#include "noc/noc_parameters.h"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/third_party/umd/device/tt_xy_pair.h"

using std::map;
using std::pair;
using std::set;
using std::shared_ptr;
using std::unique_ptr;

std::mutex finish_mutex;
std::condition_variable finish_cv;

namespace tt::tt_metal {

uint32_t get_noc_unicast_encoding(CoreCoord coord) { return NOC_XY_ENCODING(NOC_X(coord.x), NOC_Y(coord.y)); }

// EnqueueReadBufferCommandSection
std::vector<uint32_t> EnqueueReadBufferCommand::commands;

EnqueueReadBufferCommand::EnqueueReadBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    Buffer& buffer,
    void* dst,
    SystemMemoryManager& manager,
    uint32_t expected_num_workers_completed,
    uint32_t src_page_index,
    std::optional<uint32_t> pages_to_read) :
    command_queue_id(command_queue_id),
    dst(dst),
    manager(manager),
    buffer(buffer),
    expected_num_workers_completed(expected_num_workers_completed),
    src_page_index(src_page_index),
    pages_to_read(pages_to_read.has_value() ? pages_to_read.value() : buffer.num_pages()) {

    TT_ASSERT(
        buffer.buffer_type() == BufferType::DRAM or buffer.buffer_type() == BufferType::L1,
        "Trying to read an invalid buffer");

    this->device = device;
    this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());

    // Create commands once, subsequent enqueue_read_buffer calls can just update dynamic fields
    if (this->commands.empty()) {
        CQPrefetchCmd relay_wait;
        relay_wait.base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE;
        relay_wait.relay_inline.length = sizeof(CQDispatchCmd);
        static_assert((sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) % HUGEPAGE_ALIGNMENT == 0);
        relay_wait.relay_inline.stride = sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd);

        uint32_t *relay_wait_l1_ptr = (uint32_t *)&relay_wait;
        for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*relay_wait_l1_ptr++);
        }

        CQDispatchCmd wait_cmd;
        wait_cmd.base.cmd_id = CQ_DISPATCH_CMD_WAIT;
        wait_cmd.wait.barrier = true;
        wait_cmd.wait.notify_prefetch = true;
        wait_cmd.wait.addr = DISPATCH_MESSAGE_ADDR;
        wait_cmd.wait.count = 0;

        uint32_t *wait_cmd_ptr = (uint32_t *)&wait_cmd;
        for (int i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*wait_cmd_ptr++);
        }

        CQPrefetchCmd stall_cmd;
        stall_cmd.base.cmd_id = CQ_PREFETCH_CMD_STALL;
        uint32_t *stall_ptr = (uint32_t *)&stall_cmd;
        for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*stall_ptr++);
        }

        uint32_t stall_aligned_size = align(sizeof(CQPrefetchCmd), HUGEPAGE_ALIGNMENT);
        TT_ASSERT(stall_aligned_size == CQ_PREFETCH_CMD_BARE_MIN_SIZE);

        uint32_t padding = stall_aligned_size - sizeof(CQPrefetchCmd);
        for (int i = 0; i < padding / sizeof(uint32_t); i++) {
            this->commands.push_back(0);
        }

        CQPrefetchCmd no_flush;
        no_flush.base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH;
        no_flush.relay_inline.length = sizeof(CQDispatchCmd);
        no_flush.relay_inline.stride = align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), HUGEPAGE_ALIGNMENT);

        uint32_t *no_flush_ptr = (uint32_t *)&no_flush;
        for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*no_flush_ptr++);
        }

        CQDispatchCmd dev_to_host_cmd;
        dev_to_host_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR_HOST;
        dev_to_host_cmd.write_linear_host.length = 0;

        uint32_t *dev_to_host_cmd_ptr = (uint32_t *)&dev_to_host_cmd;
        for (int i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*dev_to_host_cmd_ptr++);
        }

        padding = align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), HUGEPAGE_ALIGNMENT) - (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd));
        for (int i = 0; i < padding / sizeof(uint32_t); i++) {
            this->commands.push_back(0);
        }

        CQPrefetchCmd relay_buffer;
        relay_buffer.base.cmd_id = CQ_PREFETCH_CMD_RELAY_PAGED;

        relay_buffer.relay_paged.is_dram = 0;
        relay_buffer.relay_paged.start_page = 0;
        relay_buffer.relay_paged.base_addr = 0;
        relay_buffer.relay_paged.page_size = 0;
        relay_buffer.relay_paged.pages = 0;

        uint32_t *relay_buffer_ptr = (uint32_t *)&relay_buffer;
        for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*relay_buffer_ptr++);
        }

        padding = align(sizeof(CQPrefetchCmd), HUGEPAGE_ALIGNMENT) - sizeof(CQPrefetchCmd);
        for (int i = 0; i < padding / sizeof(uint32_t); i++) {
            this->commands.push_back(0);
        }

    }
}

const void EnqueueReadBufferCommand::assemble_device_commands(uint32_t dst_address) {
    uint32_t padded_page_size = align(this->buffer.page_size(), 32);

    CQDispatchCmd *wait_cmd = (CQDispatchCmd*)(this->commands.data() + (sizeof(CQPrefetchCmd) / sizeof(uint32_t)));
    wait_cmd->wait.count = this->expected_num_workers_completed;

    uint32_t dev_to_host_cmd_offset = align(
        sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd) + sizeof(CQPrefetchCmd), // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT + CQ_PREFETCH_CMD_STALL
        HUGEPAGE_ALIGNMENT
    );
    dev_to_host_cmd_offset += sizeof(CQPrefetchCmd); // add CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH offset

    uint32_t dev_to_host_cmd_idx = dev_to_host_cmd_offset / sizeof(uint32_t);
    CQDispatchCmd *dev_to_host_cmd = (CQDispatchCmd*)(this->commands.data() + dev_to_host_cmd_idx);
    dev_to_host_cmd->write_linear_host.length = sizeof(CQDispatchCmd) + (this->pages_to_read * padded_page_size);

    uint32_t relay_paged_cmd_offset = dev_to_host_cmd_offset + sizeof(CQDispatchCmd);
    TT_ASSERT(relay_paged_cmd_offset % HUGEPAGE_ALIGNMENT == 0);
    uint32_t relay_buffer_cmd_idx = relay_paged_cmd_offset / sizeof(uint32_t);

    CQPrefetchCmd *relay_buffer = (CQPrefetchCmd*)(this->commands.data() + relay_buffer_cmd_idx);
    relay_buffer->relay_paged.is_dram = (this->buffer.buffer_type() == BufferType::DRAM);
    relay_buffer->relay_paged.start_page = this->src_page_index;
    relay_buffer->relay_paged.base_addr = this->buffer.address();
    relay_buffer->relay_paged.page_size = padded_page_size;
    relay_buffer->relay_paged.pages = this->pages_to_read;
}

void EnqueueReadBufferCommand::process() {
    this->assemble_device_commands(0);

    uint32_t fetch_size_bytes = this->commands.size() * sizeof(uint32_t);

    // move this into the command queue interface
    TT_ASSERT(fetch_size_bytes <= MAX_PREFETCH_COMMAND_SIZE, "Generated prefetcher command exceeds max command size");
    TT_ASSERT((fetch_size_bytes >> PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "FetchQ command too large to represent");

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    this->manager.cq_write(this->commands.data(), fetch_size_bytes, write_ptr);
    this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);

    this->manager.fetch_queue_write(fetch_size_bytes, this->command_queue_id);
}

// EnqueueWriteBufferCommand section
std::vector<uint32_t> EnqueueWriteBufferCommand::commands;

EnqueueWriteBufferCommand::EnqueueWriteBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    const Buffer& buffer,
    const void* src,
    SystemMemoryManager& manager,
    bool issue_wait,
    uint32_t expected_num_workers_completed,
    uint32_t bank_base_address,
    uint32_t dst_page_index,
    std::optional<uint32_t> pages_to_write) :
    command_queue_id(command_queue_id),
    manager(manager),
    issue_wait(issue_wait),
    src(src),
    buffer(buffer),
    expected_num_workers_completed(expected_num_workers_completed),
    bank_base_address(bank_base_address),
    dst_page_index(dst_page_index),
    pages_to_write(pages_to_write.has_value() ? pages_to_write.value() : buffer.num_pages()) {
    TT_ASSERT(
        buffer.buffer_type() == BufferType::DRAM or buffer.buffer_type() == BufferType::L1,
        "Trying to write to an invalid buffer");
    this->device = device;
    this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());

    if (this->commands.empty()) {
        CQPrefetchCmd relay_wait;
        relay_wait.base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE;
        relay_wait.relay_inline.length = sizeof(CQDispatchCmd);
        static_assert((sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) % HUGEPAGE_ALIGNMENT == 0);
        relay_wait.relay_inline.stride = sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd);

        uint32_t *relay_wait_l1_ptr = (uint32_t *)&relay_wait;
        for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*relay_wait_l1_ptr++);
        }

        CQDispatchCmd wait_cmd;
        wait_cmd.base.cmd_id = CQ_DISPATCH_CMD_WAIT;
        wait_cmd.wait.barrier = false;
        wait_cmd.wait.notify_prefetch = false;
        wait_cmd.wait.addr = DISPATCH_MESSAGE_ADDR;
        wait_cmd.wait.count = 0;

        uint32_t *wait_cmd_ptr = (uint32_t *)&wait_cmd;
        for (int i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*wait_cmd_ptr++);
        }

        CQPrefetchCmd relay_write;
        relay_write.base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE;
        // relay_inline attributes set in assemble_device_commands
        relay_write.relay_inline.length = 0;
        relay_write.relay_inline.stride = 0;

        uint32_t *relay_write_ptr = (uint32_t *)&relay_write;
        for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*relay_write_ptr++);
        }

        CQDispatchCmd write_paged_cmd;
        write_paged_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_PAGED;
         // write_paged attributes set in assemble_device_commands
        write_paged_cmd.write_paged.is_dram = 0;
        write_paged_cmd.write_paged.start_page = 0;
        write_paged_cmd.write_paged.base_addr = 0;
        write_paged_cmd.write_paged.page_size = 0;
        write_paged_cmd.write_paged.pages = 0;

        uint32_t *write_paged_cmd_ptr = (uint32_t *)&write_paged_cmd;
        for (int i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*write_paged_cmd_ptr++);
        }

        // no need to add padding
    }
}

const void EnqueueWriteBufferCommand::assemble_device_commands(uint32_t) {
    uint32_t num_pages = this->pages_to_write;
    uint32_t padded_page_size = align(this->buffer.page_size(), 32);

    CQDispatchCmd *wait_cmd = (CQDispatchCmd*)(this->commands.data() + (sizeof(CQPrefetchCmd) / sizeof(uint32_t)));
    wait_cmd->wait.count = this->expected_num_workers_completed;

    uint32_t relay_write_idx = (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) / sizeof(uint32_t);
    CQPrefetchCmd *relay_write_cmd = (CQPrefetchCmd*)(this->commands.data() + relay_write_idx);
    uint32_t payload_size_bytes = sizeof(CQDispatchCmd) + (this->pages_to_write * padded_page_size);
    relay_write_cmd->relay_inline.length = payload_size_bytes;
    relay_write_cmd->relay_inline.stride = align(sizeof(CQPrefetchCmd) + payload_size_bytes, HUGEPAGE_ALIGNMENT);

    uint32_t write_paged_cmd_idx = relay_write_idx + (sizeof(CQPrefetchCmd) / sizeof(uint32_t));
    CQDispatchCmd *write_paged_cmd = (CQDispatchCmd*)(this->commands.data() + write_paged_cmd_idx);
    write_paged_cmd->write_paged.is_dram = uint8_t(this->buffer.buffer_type() == BufferType::DRAM);

    TT_ASSERT(this->dst_page_index <= 0xFFFF, "Page offset needs to fit within range of uint16_t, bank_base_address was computed incorrectly!");

    write_paged_cmd->write_paged.start_page = uint16_t(this->dst_page_index & 0xFFFF);
    write_paged_cmd->write_paged.base_addr = this->bank_base_address;
    write_paged_cmd->write_paged.page_size = padded_page_size;
    write_paged_cmd->write_paged.pages = this->pages_to_write;
}

void EnqueueWriteBufferCommand::process() {
    this->assemble_device_commands(0);

    uint32_t padded_page_size = align(this->buffer.page_size(), 32);
    uint32_t data_size_in_bytes = this->pages_to_write * padded_page_size;
    // TODO (abhullar): Refactor this when implementing device command struct. ATM wait is always in the commands vector
    uint32_t wait_size_bytes = sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd);
    uint32_t commands_size_bytes = this->issue_wait ? (this->commands.size() * sizeof(uint32_t)) : ((this->commands.size() * sizeof(uint32_t)) - wait_size_bytes);
    uint32_t fetch_size_bytes = commands_size_bytes + data_size_in_bytes;

    // TODO: move this into the command queue interface
    TT_ASSERT(fetch_size_bytes <= MAX_PREFETCH_COMMAND_SIZE, "Generated prefetcher command exceeds max command size");
    TT_ASSERT((fetch_size_bytes >> PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "FetchQ command too large to represent");

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);

    uint32_t commands_offset_idx = this->issue_wait ? 0 : (wait_size_bytes / sizeof(uint32_t));
    this->manager.cq_write(this->commands.data() + commands_offset_idx, commands_size_bytes, write_ptr);

    uint32_t data_write_ptr = write_ptr + commands_size_bytes;

    uint32_t buffer_addr_offset = this->bank_base_address - this->buffer.address();
    uint32_t num_banks = this->device->num_banks(this->buffer.buffer_type());
    uint32_t unpadded_src_offset = ( ((buffer_addr_offset/padded_page_size) * num_banks) + this->dst_page_index) * this->buffer.page_size();
    if (this->buffer.page_size() % 32 != 0 and this->buffer.page_size() != this->buffer.size()) {
        // If page size is not 32B-aligned, we cannot do a contiguous write
        uint32_t src_address_offset = unpadded_src_offset;
        for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_in_bytes;
             sysmem_address_offset += padded_page_size) {
            this->manager.cq_write(
                (char*)this->src + src_address_offset,
                this->buffer.page_size(),
                data_write_ptr + sysmem_address_offset);
            src_address_offset += this->buffer.page_size();
        }
    } else {
        this->manager.cq_write((char*)this->src + unpadded_src_offset, data_size_in_bytes, data_write_ptr);
    }

    this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);
    this->manager.fetch_queue_write(fetch_size_bytes, this->command_queue_id);
}

EnqueueProgramCommand::EnqueueProgramCommand(
    uint32_t command_queue_id,
    Device* device,
    const Program& program,
    SystemMemoryManager& manager,
    std::optional<std::reference_wrapper<Trace>> trace) :
    command_queue_id(command_queue_id),
    manager(manager),
    program(program) {
    this->device = device;
    this->trace = trace;
    this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

const void EnqueueProgramCommand::assemble_device_commands(uint32_t host_data_src) {
}

void EnqueueProgramCommand::process() {
}

std::vector<uint32_t> EnqueueRecordEventCommand::commands;

EnqueueRecordEventCommand::EnqueueRecordEventCommand(
    uint32_t command_queue_id, Device* device, SystemMemoryManager& manager, uint32_t event_id, uint32_t expected_num_workers_completed):
    command_queue_id(command_queue_id), device(device), manager(manager), event_id(event_id), expected_num_workers_completed(expected_num_workers_completed) {

    if (this->commands.empty()) {
        CQPrefetchCmd relay_wait;
        relay_wait.base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE;
        relay_wait.relay_inline.length = sizeof(CQDispatchCmd);
        static_assert((sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) % HUGEPAGE_ALIGNMENT == 0);
        relay_wait.relay_inline.stride = sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd);

        uint32_t *relay_wait_l1_ptr = (uint32_t *)&relay_wait;
        for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*relay_wait_l1_ptr++);
        }

        CQDispatchCmd wait_cmd;
        wait_cmd.base.cmd_id = CQ_DISPATCH_CMD_WAIT;
        wait_cmd.wait.barrier = false;
        wait_cmd.wait.notify_prefetch = false;
        wait_cmd.wait.addr = DISPATCH_MESSAGE_ADDR;
        wait_cmd.wait.count = 0;

        uint32_t *wait_cmd_ptr = (uint32_t *)&wait_cmd;
        for (int i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*wait_cmd_ptr++);
        }

        uint32_t dispatch_event_payload = sizeof(CQDispatchCmd) + EVENT_PADDED_SIZE;
        static_assert((sizeof(CQDispatchCmd) + EVENT_PADDED_SIZE) % 16 == 0);
        // Command to write event to L1
        CQPrefetchCmd relay_event_l1;
        relay_event_l1.base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE;
        relay_event_l1.relay_inline.length = dispatch_event_payload;
        relay_event_l1.relay_inline.stride = align(sizeof(CQPrefetchCmd) + dispatch_event_payload, HUGEPAGE_ALIGNMENT);

        uint32_t *relay_event_l1_ptr = (uint32_t *)&relay_event_l1;
        for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*relay_event_l1_ptr++);
        }

        uint8_t num_hw_cqs = this->device->num_hw_cqs();
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
        tt_cxy_pair dispatch_location = dispatch_core_manager::get(num_hw_cqs).dispatcher_core(this->device->id(), channel, this->command_queue_id);
        CoreType core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(this->device->id());
        CoreCoord dispatch_physical_core = get_physical_core_coordinate(dispatch_location, core_type);

        CQDispatchCmd write_event_l1_cmd;
        write_event_l1_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
        write_event_l1_cmd.write_linear.num_mcast_dests = 0;
        write_event_l1_cmd.write_linear.noc_xy_addr = get_noc_unicast_encoding(dispatch_physical_core);
        write_event_l1_cmd.write_linear.addr = CQ_COMPLETION_LAST_EVENT;
        write_event_l1_cmd.write_linear.length = EVENT_PADDED_SIZE;

        uint32_t *write_event_l1_cmd_ptr = (uint32_t *)&write_event_l1_cmd;
        for (int i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*write_event_l1_cmd_ptr++);
        }

        for (int i = 0; i < EVENT_PADDED_SIZE / sizeof(uint32_t); i++) {
            this->commands.push_back(0);
        }

        uint32_t padding = align(sizeof(CQPrefetchCmd) + dispatch_event_payload, HUGEPAGE_ALIGNMENT) - (sizeof(CQPrefetchCmd) + dispatch_event_payload);
        for (int i = 0; i < padding / sizeof(uint32_t); i++) {
            this->commands.push_back(0);
        }

        // Command to write event to completion queue
        CQPrefetchCmd relay_event_host;
        relay_event_host.base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE;
        relay_event_host.relay_inline.length = dispatch_event_payload;
        relay_event_host.relay_inline.stride = align(sizeof(CQPrefetchCmd) + dispatch_event_payload, HUGEPAGE_ALIGNMENT);

        uint32_t *relay_event_host_ptr = (uint32_t *)&relay_event_host;
        for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*relay_event_host_ptr++);
        }

        CQDispatchCmd write_event_host_cmd;
        write_event_host_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR_HOST;
        write_event_host_cmd.write_linear_host.length = sizeof(CQDispatchCmd) + EVENT_PADDED_SIZE;

        uint32_t *write_event_host_cmd_ptr = (uint32_t *)&write_event_host_cmd;
        for (int i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*write_event_host_cmd_ptr++);
        }

        for (int i = 0; i < EVENT_PADDED_SIZE / sizeof(uint32_t); i++) {
            this->commands.push_back(0);
        }

        for (int i = 0; i < padding / sizeof(uint32_t); i++) {
            this->commands.push_back(0);
        }
    }
}

const void EnqueueRecordEventCommand::assemble_device_commands(uint32_t) {
    CQDispatchCmd *wait_cmd = (CQDispatchCmd*)(this->commands.data() + (sizeof(CQPrefetchCmd) / sizeof(uint32_t)));
    wait_cmd->wait.count = this->expected_num_workers_completed;

    uint32_t event_payload_offset = (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) * 2;
    uint32_t write_event_l1_idx = event_payload_offset / sizeof(uint32_t);

    uint32_t *event_l1_location = (uint32_t*)(this->commands.data() + write_event_l1_idx);
    *event_l1_location = this->event_id;

    uint32_t write_host_cmd_offset = align(event_payload_offset + EVENT_PADDED_SIZE, HUGEPAGE_ALIGNMENT);

    uint32_t write_event_host_idx = (write_host_cmd_offset + sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) / sizeof(uint32_t);

    uint32_t *event_host_location = (uint32_t*)(this->commands.data() + write_event_host_idx);
    *event_host_location = this->event_id;
}

void EnqueueRecordEventCommand::process() {
    this->assemble_device_commands(0);

    uint32_t fetch_size_bytes = this->commands.size() * sizeof(uint32_t);

    // move this into the command queue interface
    TT_ASSERT(fetch_size_bytes <= MAX_PREFETCH_COMMAND_SIZE, "Generated prefetcher command exceeds max command size");
    TT_ASSERT((fetch_size_bytes >> PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "FetchQ command too large to represent");

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    this->manager.cq_write(this->commands.data(), fetch_size_bytes, write_ptr);
    this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);

    this->manager.fetch_queue_write(fetch_size_bytes, this->command_queue_id);
}

EnqueueWaitForEventCommand::EnqueueWaitForEventCommand(
    uint32_t command_queue_id, Device* device, SystemMemoryManager& manager, const Event& sync_event):
    command_queue_id(command_queue_id), device(device), manager(manager), sync_event(sync_event) {
        this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
        // Should not be encountered under normal circumstances (record, wait) unless user is modifying sync event ID.
        TT_ASSERT(command_queue_id != sync_event.cq_id,
            "EnqueueWaitForEventCommand cannot wait on it's own event id on the same CQ. CQ ID: {}", command_queue_id);
}

const void EnqueueWaitForEventCommand::assemble_device_commands(uint32_t) {
}

void EnqueueWaitForEventCommand::process() {
}

// HWCommandQueue section
HWCommandQueue::HWCommandQueue(Device* device, uint32_t id) : manager(device->sysmem_manager()), completion_queue_thread{} {
    ZoneScopedN("CommandQueue_constructor");
    this->device = device;
    this->id = id;
    this->num_entries_in_completion_q = 0;
    this->num_completed_completion_q_reads = 0;

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    this->size_B = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / device->num_hw_cqs();

    tt_cxy_pair completion_q_writer_location =
        dispatch_core_manager::get(device->num_hw_cqs()).completion_queue_writer_core(device->id(), channel, this->id);

    this->completion_queue_writer_core = CoreCoord(completion_q_writer_location.x, completion_q_writer_location.y);

    this->exit_condition = false;
    std::thread completion_queue_thread = std::thread(&HWCommandQueue::read_completion_queue, this);
    this->completion_queue_thread = std::move(completion_queue_thread);
    this->expected_num_workers_completed = 0;
}

HWCommandQueue::~HWCommandQueue() {
    ZoneScopedN("HWCommandQueue_destructor");
    if (this->exit_condition) {
        this->completion_queue_thread.join();  // We errored out already prior
    } else {

        // TODO: SEND THE TERMINATE CMD?

        TT_ASSERT(
            this->issued_completion_q_reads.empty(),
            "There should be no reads in flight after closing our completion queue thread");
        TT_ASSERT(
            this->num_entries_in_completion_q == this->num_completed_completion_q_reads,
            "There shouldn't be any commands in flight after closing our completion queue thread. Num uncompleted commands: {}", this->num_entries_in_completion_q - this->num_completed_completion_q_reads);
        this->exit_condition = true;
        this->completion_queue_thread.join();
    }
}

template <typename T>
void HWCommandQueue::enqueue_command(T& command, bool blocking) {
    command.process();
    if (blocking) {
        this->finish();
    }
}

// TODO: Currently converting page ordering from interleaved to sharded and then doing contiguous read/write
//  Look into modifying command to do read/write of a page at a time to avoid doing copy
void * convert_interleaved_to_sharded_on_host(const void* host, const Buffer& buffer) {
    const uint32_t num_pages = buffer.num_pages();
    const uint32_t page_size = buffer.page_size();

    const uint32_t size_in_bytes = num_pages * page_size;

    void* swapped = malloc(size_in_bytes);

    std::set<uint32_t> pages_seen;
    auto buffer_page_mapping = generate_buffer_page_mapping(buffer);
    uint32_t shard_width_in_pages = buffer.shard_spec().tensor_shard_spec.shape[1] / buffer.shard_spec().page_shape[1];
    for (uint32_t page_id = 0; page_id < num_pages; page_id++) {
        uint32_t local_num_pages;
        auto host_page_id = page_id;
        auto dev_page_id = buffer_page_mapping.host_page_to_dev_page_mapping_[host_page_id];
        TT_ASSERT(host_page_id < num_pages and host_page_id >= 0);
        memcpy((char*)swapped + dev_page_id * page_size, (char*)host + host_page_id * page_size, page_size);
    }
    return swapped;
}

void HWCommandQueue::enqueue_read_buffer(std::shared_ptr<Buffer> buffer, void* dst, bool blocking) {
    this->enqueue_read_buffer(*buffer, dst, blocking);
}

// Read buffer command is enqueued in the issue region and device writes requested buffer data into the completion region
void HWCommandQueue::enqueue_read_buffer(Buffer& buffer, void* dst, bool blocking) {
    ZoneScopedN("HWCommandQueue_read_buffer");

    TT_ASSERT(not is_sharded(buffer.buffer_layout()), "Sharded buffer is not supported in FD 2.0");

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());

    uint32_t padded_page_size = align(buffer.page_size(), 32);
    uint32_t pages_to_read = buffer.num_pages();
    uint32_t unpadded_dst_offset = 0;
    uint32_t src_page_index = 0;

    // this is a streaming command so we don't need to break down to multiple
    auto command = EnqueueReadInterleavedBufferCommand(
        this->id, this->device, buffer, dst, this->manager, this->expected_num_workers_completed, src_page_index, pages_to_read);

    this->issued_completion_q_reads.push(
        detail::ReadBufferDescriptor(buffer, padded_page_size, dst, unpadded_dst_offset, pages_to_read, src_page_index)
    );
    this->num_entries_in_completion_q++;

    this->enqueue_command(command, blocking);

    if (not blocking) { // should this be unconditional?
        std::shared_ptr<Event> event = std::make_shared<Event>();
        this->enqueue_record_event(event);
    }
}

void HWCommandQueue::enqueue_write_buffer(std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<const Buffer>> buffer, HostDataType src, bool blocking) {
    // Top level API to accept different variants for buffer and src
    // For shared pointer variants, object lifetime is guaranteed at least till the end of this function
    std::visit ([this, &buffer, &blocking](auto&& data) {
        using T = std::decay_t<decltype(data)>;
        std::visit ([this, &buffer, &blocking, &data](auto&& b) {
            using type_buf = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<T, const void*>) {
                if constexpr (std::is_same_v<type_buf, std::shared_ptr<const Buffer>>) {
                    this->enqueue_write_buffer(*b, data, blocking);
                } else if constexpr (std::is_same_v<type_buf, std::reference_wrapper<Buffer>>) {
                    this->enqueue_write_buffer(b.get(), data, blocking);
                }
            } else {
                if constexpr (std::is_same_v<type_buf, std::shared_ptr<const Buffer>>) {
                    this->enqueue_write_buffer(*b, data.get() -> data(), blocking);
                } else if constexpr (std::is_same_v<type_buf, std::reference_wrapper<Buffer>>) {
                    this->enqueue_write_buffer(b.get(), data.get() -> data(), blocking);
                }
            }
        }, buffer);
    }, src);
}

CoreType HWCommandQueue::get_dispatch_core_type() {
    return dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

void HWCommandQueue::enqueue_write_buffer(const Buffer& buffer, const void* src, bool blocking) {
    ZoneScopedN("HWCommandQueue_write_buffer");

    if (is_sharded(buffer.buffer_layout())) {
        TT_THROW("Sharded buffers are currently unsupported in FD2.0");
    }

    if (buffer.buffer_layout() == TensorMemoryLayout::WIDTH_SHARDED or
        buffer.buffer_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        convert_interleaved_to_sharded_on_host(src, buffer);
    }

    uint32_t padded_page_size = align(buffer.page_size(), 32);
    uint32_t total_pages_to_write = buffer.num_pages();
    const uint32_t num_banks = this->device->num_banks(buffer.buffer_type());

    const uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->id);

    uint32_t dst_page_index = 0;
    uint32_t bank_base_address = buffer.address();
    uint32_t data_offset_bytes = (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)); // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
    while (total_pages_to_write > 0) {
        bool issue_wait = (dst_page_index == 0 and bank_base_address == buffer.address()); // only stall for the first write of the buffer
        if (issue_wait) {
            data_offset_bytes *= 2; // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        }
        uint32_t space_available_bytes = std::min(command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id), MAX_PREFETCH_COMMAND_SIZE);
        int32_t num_pages_available = (int32_t(space_available_bytes) - int32_t(data_offset_bytes)) / int32_t(padded_page_size);

        uint32_t pages_to_write = std::min(total_pages_to_write, (uint32_t)num_pages_available);

        if (dst_page_index > 0xFFFF) {
            // Page offset in CQ_DISPATCH_CMD_WRITE_PAGED is uint16_t
            // To handle larger page offsets move bank base address up and update page offset to be relative to the new bank address
            uint32_t residual = dst_page_index % num_banks;
            uint32_t num_full_pages_written_per_bank = dst_page_index / num_banks;
            bank_base_address += num_full_pages_written_per_bank * padded_page_size;
            dst_page_index = residual;
        }

        tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for channel {}", this->id);

        auto command = EnqueueWriteInterleavedBufferCommand(
            this->id, this->device, buffer, src, this->manager, issue_wait, this->expected_num_workers_completed, bank_base_address, dst_page_index, pages_to_write);
        this->enqueue_command(command, false); // don't block until the entire src data is enqueued in the issue queue

        total_pages_to_write -= pages_to_write;
        dst_page_index += pages_to_write;
    }

    if (blocking) {
        this->finish();
    } else {
        std::shared_ptr<Event> event = std::make_shared<Event>();
        this->enqueue_record_event(event);
    }
}

void HWCommandQueue::enqueue_program(
    Program& program, std::optional<std::reference_wrapper<Trace>> trace, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_program");
}

void HWCommandQueue::enqueue_record_event(std::shared_ptr<Event> event) {
    ZoneScopedN("HWCommandQueue_enqueue_record_event");

    // Populate event struct for caller. When async queues are enabled, this is in child thread, so consumers
    // of the event must wait for it to be ready (ie. populated) here. Set ready flag last. This couldn't be
    // in main thread otherwise event_id selection would get out of order due to main/worker thread timing.
    event->cq_id = this->id;
    event->event_id = this->manager.get_next_event(this->id);
    event->device = this->device;
    event->ready = true; // what does this mean???

    auto command = EnqueueRecordEventCommand(this->id, this->device, this->manager, event->event_id, this->expected_num_workers_completed);
    this->enqueue_command(command, false);

    this->issued_completion_q_reads.push(detail::ReadEventDescriptor(event->event_id));
    this->num_entries_in_completion_q++;
}

void HWCommandQueue::enqueue_wait_for_event(std::shared_ptr<Event> event) {
    ZoneScopedN("HWCommandQueue_enqueue_wait_for_event");
}


void HWCommandQueue::enqueue_trace() {
    ZoneScopedN("HWCommandQueue_enqueue_trace");
    TT_THROW("Not implemented");
}

void HWCommandQueue::copy_into_user_space(const detail::ReadBufferDescriptor &read_buffer_descriptor, uint32_t read_ptr, chip_id_t mmio_device_id, uint16_t channel) {
    const auto& [buffer_layout, page_size, padded_page_size, dev_page_to_host_page_mapping, dst, dst_offset, num_pages_read, cur_host_page_id] = read_buffer_descriptor;

    uint32_t padded_num_bytes = (num_pages_read * padded_page_size) + sizeof(CQDispatchCmd);
    uint32_t contig_dst_offset = dst_offset;
    uint32_t remaining_bytes_to_read = padded_num_bytes;

    // track the amount of bytes read in the last non-aligned page
    uint32_t remaining_bytes_of_nonaligned_page = 0;

    static std::vector<uint32_t> completion_q_data;

    while (remaining_bytes_to_read != 0) {
        this->manager.completion_queue_wait_front(this->id, this->exit_condition);

        if (this->exit_condition) {
            break;
        }

        uint32_t completion_queue_write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(
            this->device->id(), this->id, this->manager.get_cq_size());
        uint32_t completion_q_write_ptr = (completion_queue_write_ptr_and_toggle & 0x7fffffff) << 4;
        uint32_t completion_q_write_toggle = completion_queue_write_ptr_and_toggle >> (31);
        uint32_t completion_q_read_ptr = this->manager.get_completion_queue_read_ptr(this->id);
        uint32_t completion_q_read_toggle = this->manager.get_completion_queue_read_toggle(this->id);

        uint32_t bytes_avail_in_completion_queue;
        if (completion_q_write_ptr > completion_q_read_ptr and completion_q_write_toggle == completion_q_read_toggle) {
            bytes_avail_in_completion_queue = completion_q_write_ptr - completion_q_read_ptr;
        } else {
            // Completion queue write pointer on device wrapped but read pointer is lagging behind.
            //  In this case read up until the end of the completion queue first
            bytes_avail_in_completion_queue = this->manager.get_completion_queue_limit(this->id) - completion_q_read_ptr;
        }

        // completion queue write ptr on device could have wrapped but our read ptr is lagging behind
        uint32_t bytes_xfered = std::min(padded_num_bytes, bytes_avail_in_completion_queue);
        bytes_xfered = std::min(bytes_xfered, remaining_bytes_to_read);
        uint32_t num_pages_xfered = (bytes_xfered + TRANSFER_PAGE_SIZE - 1) / TRANSFER_PAGE_SIZE;

        completion_q_data.resize(bytes_xfered / sizeof(uint32_t));

        tt::Cluster::instance().read_sysmem(
            completion_q_data.data(), bytes_xfered, completion_q_read_ptr, mmio_device_id, channel);

        this->manager.completion_queue_pop_front(num_pages_xfered, this->id);

        remaining_bytes_to_read -= bytes_xfered;

        if (buffer_layout == TensorMemoryLayout::INTERLEAVED or
            buffer_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            void* contiguous_dst = (void*)(uint64_t(dst) + contig_dst_offset);
            uint32_t offset_in_completion_q_data = (contig_dst_offset == 0) ? (sizeof(CQDispatchCmd) / sizeof(uint32_t)) : 0;
            if ((page_size % 32) == 0) {
                uint32_t data_bytes_xfered = (contig_dst_offset == 0) ? (bytes_xfered - sizeof(CQDispatchCmd)) : bytes_xfered;
                memcpy(contiguous_dst, completion_q_data.data() + offset_in_completion_q_data, data_bytes_xfered);
                contig_dst_offset += data_bytes_xfered;
            } else {
                uint32_t src_offset = offset_in_completion_q_data;
                uint32_t dst_offset_bytes = 0;

                uint32_t pad_size_bytes = padded_page_size - page_size;

                while (src_offset < completion_q_data.size()) {

                    uint32_t src_offset_increment = (padded_page_size / sizeof(uint32_t));
                    uint32_t num_bytes_to_copy;
                    if (remaining_bytes_of_nonaligned_page > 0) {
                        // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                        num_bytes_to_copy = remaining_bytes_of_nonaligned_page - pad_size_bytes;
                        remaining_bytes_of_nonaligned_page = 0;
                        src_offset_increment = (num_bytes_to_copy/sizeof(uint32_t)) + (pad_size_bytes/sizeof(uint32_t));
                    } else if (src_offset + src_offset_increment >= completion_q_data.size()) {
                        // Case 2: Last page of data that was popped off the completion queue
                        uint32_t num_bytes_remaining = (completion_q_data.size() - src_offset) * sizeof(uint32_t);
                        num_bytes_to_copy = std::min(num_bytes_remaining, page_size);
                        remaining_bytes_of_nonaligned_page = padded_page_size - num_bytes_to_copy;
                    } else {
                        num_bytes_to_copy = page_size;
                    }

                    memcpy(
                        (char*)(uint64_t(contiguous_dst) + dst_offset_bytes),
                        completion_q_data.data() + src_offset,
                        num_bytes_to_copy
                    );

                    src_offset += src_offset_increment;
                    dst_offset_bytes += num_bytes_to_copy;
                    contig_dst_offset += num_bytes_to_copy;
                }
            }
        } else if (
            buffer_layout == TensorMemoryLayout::WIDTH_SHARDED or
            buffer_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_THROW("Reading width sharded or block sharded buffers is unsupported in FD2.0");
        }
    }
}

void HWCommandQueue::read_completion_queue() {
    tracy::SetThreadName("COMPLETION QUEUE");
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    while (true) {
        if (this->num_entries_in_completion_q > this->num_completed_completion_q_reads) {
            uint32_t num_events_to_read = this->num_entries_in_completion_q - this->num_completed_completion_q_reads;
            for (uint32_t i = 0; i < num_events_to_read; i++) {

                std::variant<detail::ReadBufferDescriptor, detail::ReadEventDescriptor> read_descriptor = *(this->issued_completion_q_reads.pop());

                this->manager.completion_queue_wait_front(this->id, this->exit_condition); // CQ DISPATCHER IS NOT HANDSHAKING WITH HOST RN

                if (this->exit_condition) {  // Early exit
                    return;
                }

                uint32_t completion_queue_write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(
                    this->device->id(), this->id, this->manager.get_cq_size());
                uint32_t completion_q_write_ptr = (completion_queue_write_ptr_and_toggle & 0x7fffffff) << 4;

                uint32_t read_ptr = this->manager.get_completion_queue_read_ptr(this->id);

                std::visit(
                    [&](auto&& read_descriptor)
                    {
                        using T = std::decay_t<decltype(read_descriptor)>;
                        if constexpr (std::is_same_v<T, detail::ReadBufferDescriptor>) {
                            this->copy_into_user_space(read_descriptor, read_ptr, mmio_device_id, channel);
                        }
                        else if constexpr (std::is_same_v<T, detail::ReadEventDescriptor>) {
                            static std::vector<uint32_t> dispatch_cmd_and_event((sizeof(CQDispatchCmd) + EVENT_PADDED_SIZE) / sizeof(uint32_t));
                            tt::Cluster::instance().read_sysmem(
                                dispatch_cmd_and_event.data(), sizeof(CQDispatchCmd) + EVENT_PADDED_SIZE, read_ptr, mmio_device_id, channel);
                            uint32_t event_completed = dispatch_cmd_and_event.at(sizeof(CQDispatchCmd) / sizeof(uint32_t));
                            TT_ASSERT(event_completed == read_descriptor.event_id, "Event Order Issue: expected to read back completion signal for event {} but got {}!", read_descriptor.event_id, event_completed);
                            this->manager.completion_queue_pop_front(1, this->id);
                            this->manager.set_last_completed_event(this->id, event_completed);
                        }
                    },
                    read_descriptor
                );
            }
            this->num_completed_completion_q_reads += num_events_to_read;
        } else if (this->exit_condition) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

void HWCommandQueue::finish() {
    ZoneScopedN("HWCommandQueue_finish");
    tt::log_debug(tt::LogDispatch, "Finish for command queue {}", this->id);
    std::shared_ptr<Event> event = std::make_shared<Event>();
    this->enqueue_record_event(event);

    if (tt::llrt::OptionsG.get_test_mode_enabled()) {
        while (this->num_entries_in_completion_q > this->num_completed_completion_q_reads) {
            if (DPrintServerHangDetected()) {
                // DPrint Server hang. Mark state and early exit. Assert in main thread.
                this->exit_condition = true;
                this->dprint_server_hang = true;
                return;
            } else if (tt::watcher_server_killed_due_to_error()) {
                // Illegal NOC txn killed watcher. Mark state and early exit. Assert in main thread.
                this->exit_condition = true;
                this->illegal_noc_txn_hang = true;
                return;
            }
        }
    } else {
        while (this->num_entries_in_completion_q > this->num_completed_completion_q_reads);
    }
}

volatile bool HWCommandQueue::is_dprint_server_hung() {
    return dprint_server_hang;
}

volatile bool HWCommandQueue::is_noc_hung() {
    return illegal_noc_txn_hang;
}

void EnqueueAddBufferToProgram(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program>> program, bool blocking) {
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ADD_BUFFER_TO_PROGRAM,
        .blocking = blocking,
        .buffer = buffer,
        .program = program,
    });
}

void EnqueueAddBufferToProgramImpl(const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program>> program) {
    std::visit([program] (auto&& b) {
        using buffer_type = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<buffer_type, std::shared_ptr<Buffer>>) {
            std::visit([&b] (auto&& p) {
                using program_type = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<program_type, std::reference_wrapper<Program>>) {
                    p.get().add_buffer(b);
                }
                else {
                    p->add_buffer(b);
                }
            }, program);
        }
    }, buffer);
}

void EnqueueUpdateRuntimeArgs(CommandQueue& cq, const std::shared_ptr<Kernel> kernel, const CoreCoord &core_coord, std::vector<uint32_t> &update_idx, std::shared_ptr<RuntimeArgs> runtime_args_ptr, bool blocking) {
    auto runtime_args_md = RuntimeArgsMetadata {
            .core_coord = core_coord,
            .runtime_args_ptr = runtime_args_ptr,
            .kernel = kernel,
            .update_idx = update_idx,
    };
    cq.run_command( CommandInterface {
        .type = EnqueueCommandType::UPDATE_RUNTIME_ARGS,
        .blocking = blocking,
        .runtime_args_md = runtime_args_md,
    });
}

void EnqueueUpdateRuntimeArgsImpl (const RuntimeArgsMetadata& runtime_args_md) {
    std::vector<uint32_t> resolved_runtime_args = {};
    resolved_runtime_args.reserve((*runtime_args_md.runtime_args_ptr).size());

    for (const auto& arg : *(runtime_args_md.runtime_args_ptr)) {
        std::visit([&resolved_runtime_args] (auto&& a) {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, Buffer*>) {
                resolved_runtime_args.push_back(a -> address());
            } else {
                resolved_runtime_args.push_back(a);
            }
        }, arg);
    }
    auto& kernel_runtime_args = runtime_args_md.kernel->runtime_args(runtime_args_md.core_coord);
    for (const auto& idx : runtime_args_md.update_idx) {
        kernel_runtime_args[idx] = resolved_runtime_args[idx];
    }
}

void EnqueueSetRuntimeArgs(CommandQueue& cq, const std::shared_ptr<Kernel> kernel, const CoreCoord &core_coord, std::shared_ptr<RuntimeArgs> runtime_args_ptr, bool blocking) {
    auto runtime_args_md = RuntimeArgsMetadata {
            .core_coord = core_coord,
            .runtime_args_ptr = runtime_args_ptr,
            .kernel = kernel,
    };
    cq.run_command( CommandInterface {
        .type = EnqueueCommandType::SET_RUNTIME_ARGS,
        .blocking = blocking,
        .runtime_args_md = runtime_args_md,
    });
}

void EnqueueSetRuntimeArgsImpl(const RuntimeArgsMetadata& runtime_args_md) {
    std::vector<uint32_t> resolved_runtime_args = {};
    resolved_runtime_args.reserve((*runtime_args_md.runtime_args_ptr).size());

    for (const auto& arg : *(runtime_args_md.runtime_args_ptr)) {
        std::visit([&resolved_runtime_args] (auto&& a) {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, Buffer*>) {
                resolved_runtime_args.push_back(a -> address());
            } else {
                resolved_runtime_args.push_back(a);
            }
        }, arg);
    }
    runtime_args_md.kernel -> set_runtime_args(runtime_args_md.core_coord, resolved_runtime_args);
}

void EnqueueGetBufferAddr(CommandQueue& cq, uint32_t* dst_buf_addr, const Buffer* buffer, bool blocking) {
    cq.run_command( CommandInterface {
        .type = EnqueueCommandType::GET_BUF_ADDR,
        .blocking = blocking,
        .shadow_buffer = buffer,
        .dst = dst_buf_addr
    });
}

void EnqueueGetBufferAddrImpl(void* dst_buf_addr, const Buffer* buffer) {
    *(static_cast<uint32_t*>(dst_buf_addr)) = buffer -> address();
}
void EnqueueAllocateBuffer(CommandQueue& cq, Buffer* buffer, bool bottom_up, bool blocking) {
    auto alloc_md = AllocBufferMetadata {
        .buffer = buffer,
        .allocator = *(buffer->device()->allocator_),
        .bottom_up = bottom_up,
    };
    cq.run_command(CommandInterface {
        .type = EnqueueCommandType::ALLOCATE_BUFFER,
        .blocking = blocking,
        .alloc_md = alloc_md,
    });
}

void EnqueueAllocateBufferImpl(AllocBufferMetadata alloc_md) {
    Buffer* buffer = alloc_md.buffer;
    uint32_t allocated_addr;
    if(is_sharded(buffer->buffer_layout())) {
        allocated_addr = allocator::allocate_buffer(*(buffer->device()->allocator_), buffer->shard_spec().size() * buffer->num_cores() * buffer->page_size(), buffer->page_size(), buffer->buffer_type(), alloc_md.bottom_up, buffer->num_cores());
    }
    else {
        allocated_addr = allocator::allocate_buffer(*(buffer->device()->allocator_), buffer->size(), buffer->page_size(), buffer->buffer_type(), alloc_md.bottom_up, std::nullopt);
    }
    buffer->set_address(static_cast<uint64_t>(allocated_addr));
}

void EnqueueDeallocateBuffer(CommandQueue& cq, Allocator& allocator, uint32_t device_address, BufferType buffer_type, bool blocking) {
    // Need to explictly pass in relevant buffer attributes here, since the Buffer* ptr can be deallocated a this point
    auto alloc_md = AllocBufferMetadata {
        .allocator = allocator,
        .buffer_type = buffer_type,
        .device_address = device_address,
    };
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::DEALLOCATE_BUFFER,
        .blocking = blocking,
        .alloc_md = alloc_md,
    });
}

void EnqueueDeallocateBufferImpl(AllocBufferMetadata alloc_md) {
    allocator::deallocate_buffer(alloc_md.allocator, alloc_md.device_address, alloc_md.buffer_type);
}

void EnqueueReadBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, vector<uint32_t>& dst, bool blocking){
    // TODO(agrebenisan): Move to deprecated
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    Buffer & b = std::holds_alternative<std::shared_ptr<Buffer>>(buffer) ? *(std::get< std::shared_ptr<Buffer> > ( buffer )) :
                                                                            std::get<std::reference_wrapper<Buffer>>(buffer).get();
    // Only resizing here to keep with the original implementation. Notice how in the void*
    // version of this API, I assume the user mallocs themselves
    std::visit ( [&dst](auto&& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>>) {
            dst.resize(b.get().page_size() * b.get().num_pages() / sizeof(uint32_t));
        } else if constexpr (std::is_same_v<T, std::shared_ptr<Buffer>>) {
            dst.resize(b->page_size() * b->num_pages() / sizeof(uint32_t));
        }
    }, buffer);

    // TODO(agrebenisan): Move to deprecated
    EnqueueReadBuffer(cq, buffer, dst.data(), blocking);
}

void EnqueueWriteBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, vector<uint32_t>& src, bool blocking){
    // TODO(agrebenisan): Move to deprecated
    EnqueueWriteBuffer(cq, buffer, src.data(), blocking);
}

void EnqueueReadBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, void* dst, bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_READ_BUFFER,
        .blocking = blocking,
        .buffer = buffer,
        .dst = dst
    });
}

void EnqueueReadBufferImpl(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, void* dst, bool blocking) {
    std::visit ( [&cq, dst, blocking](auto&& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>> || std::is_same_v<T, std::shared_ptr<Buffer> > ) {
            cq.hw_command_queue().enqueue_read_buffer(b, dst, blocking);
        }
    }, buffer);
}

void EnqueueWriteBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer,
                                          HostDataType src, bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WRITE_BUFFER,
        .blocking = blocking,
        .buffer = buffer,
        .src = src
    });
}

void EnqueueWriteBufferImpl(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer,
                                          HostDataType src, bool blocking) {
    std::visit ( [&cq, src, blocking](auto&& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>> || std::is_same_v<T, std::shared_ptr<Buffer>> ) {
            cq.hw_command_queue().enqueue_write_buffer(b, src, blocking);
        }
    }, buffer);
}

void EnqueueProgram(CommandQueue& cq, std::variant < std::reference_wrapper<Program>, std::shared_ptr<Program> > program, bool blocking) {
    detail::DispatchStateCheck(true);
    TT_THROW("EnqueueProgram currently unsupported in FD2.0");
    if (cq.get_mode() != CommandQueue::CommandQueueMode::TRACE) {
        TT_FATAL(cq.id() == 0, "EnqueueProgram only supported on first command queue on device for time being.");
    }
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_PROGRAM,
        .blocking = blocking,
        .program = program
    });
}

void EnqueueProgramImpl(CommandQueue& cq, std::variant < std::reference_wrapper<Program>, std::shared_ptr<Program> > program, bool blocking) {
    ZoneScoped;
    std::visit ( [&cq, blocking](auto&& program) {
        ZoneScoped;
        using T = std::decay_t<decltype(program)>;
        Device * device = cq.device();
        std::optional<std::reference_wrapper<Trace>> trace;  // TODO TMZ: remove trace from enqueue_program interface
        if constexpr (std::is_same_v<T, std::reference_wrapper<Program>>) {
            detail::CompileProgram(device, program);
            program.get().allocate_circular_buffers();
            detail::ValidateCircularBufferRegion(program, device);
            cq.hw_command_queue().enqueue_program(program, trace, blocking);
            // Program relinquishes ownership of all global buffers its using, once its been enqueued. Avoid mem leaks on device.
            program.get().release_buffers();
        } else if constexpr (std::is_same_v<T, std::shared_ptr<Program>>) {
            detail::CompileProgram(device, *program);
            program->allocate_circular_buffers();
            detail::ValidateCircularBufferRegion(*program, device);
            cq.hw_command_queue().enqueue_program(*program, trace, blocking);
            // Program relinquishes ownership of all global buffers its using, once its been enqueued. Avoid mem leaks on device.
            program->release_buffers();
        }
    }, program);
}

void EnqueueRecordEvent(CommandQueue& cq, std::shared_ptr<Event> event) {
    TT_THROW("EnqueueRecordEvent currently unsupported in FD2.0");
    TT_ASSERT(event->device == nullptr, "EnqueueRecordEvent expected to be given an uninitialized event");
    TT_ASSERT(event->event_id == -1, "EnqueueRecordEvent expected to be given an uninitialized event");
    TT_ASSERT(event->cq_id == -1, "EnqueueRecordEvent expected to be given an uninitialized event");

    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_RECORD_EVENT,
        .blocking = false,
        .event = event,
    });
}

void EnqueueRecordEventImpl(CommandQueue& cq, std::shared_ptr<Event> event) {
    cq.hw_command_queue().enqueue_record_event(event);
}


void EnqueueWaitForEvent(CommandQueue& cq, std::shared_ptr<Event> event) {

    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT,
        .blocking = false,
        .event = event,
    });
}

void EnqueueWaitForEventImpl(CommandQueue& cq, std::shared_ptr<Event> event) {
    event->wait_until_ready(); // Block until event populated. Worker thread.
    log_trace(tt::LogMetal, "EnqueueWaitForEvent() issued on Event(device_id: {} cq_id: {} event_id: {}) from device_id: {} cq_id: {}",
        event->device->id(), event->cq_id, event->event_id, cq.device()->id(), cq.id());
    cq.hw_command_queue().enqueue_wait_for_event(event);
}


void EventSynchronize(std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready(); // Block until event populated. Parent thread.
    log_trace(tt::LogMetal, "Issuing host sync on Event(device_id: {} cq_id: {} event_id: {})", event->device->id(), event->cq_id, event->event_id);

    while (event->device->sysmem_manager().get_last_completed_event(event->cq_id) < event->event_id) {
        if (tt::llrt::OptionsG.get_test_mode_enabled() && tt::watcher_server_killed_due_to_error()) {
            TT_ASSERT(false, "Command Queue could not complete EventSynchronize. See {} for details.", tt::watcher_get_log_file_name());
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
}

bool EventQuery(std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready(); // Block until event populated. Parent thread.
    bool event_completed = event->device->sysmem_manager().get_last_completed_event(event->cq_id) >= event->event_id;
    log_trace(tt::LogMetal, "Returning event_completed: {} for host query on Event(device_id: {} cq_id: {} event_id: {})",
        event_completed, event->device->id(), event->cq_id, event->event_id);
    return event_completed;
}

void Finish(CommandQueue& cq) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::FINISH,
        .blocking = true
    });
    TT_ASSERT(!(cq.device() -> hw_command_queue(cq.id()).is_dprint_server_hung()),
              "Command Queue could not finish: device hang due to unanswered DPRINT WAIT.");
    TT_ASSERT(!(cq.device() -> hw_command_queue(cq.id()).is_noc_hung()),
              "Command Queue could not finish: device hang due to illegal NoC transaction. See {} for details.",
               tt::watcher_get_log_file_name());
}

void FinishImpl(CommandQueue& cq) {
    cq.hw_command_queue().finish();
}

CommandQueue& BeginTrace(Trace& trace) {
    log_debug(LogMetalTrace, "Begin trace capture");
    trace.begin_capture();
    return trace.queue();
}

void EndTrace(Trace& trace) {
    trace.end_capture();
    log_debug(LogMetalTrace, "End trace capture");
}

uint32_t InstantiateTrace(Trace& trace, CommandQueue& cq) {
    uint32_t trace_id = trace.instantiate(cq);
    return trace_id;
}

void ReleaseTrace(uint32_t trace_id) {
    if (trace_id == -1) {
        Trace::release_all();
    } else if (Trace::has_instance(trace_id)) {
        Trace::remove_instance(trace_id);
    }
}

void EnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    detail::DispatchStateCheck(true);
    TT_FATAL(Trace::has_instance(trace_id), "Trace instance " + std::to_string(trace_id) + " must exist on device");
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_TRACE,
        .blocking = blocking
    });
}

void EnqueueTraceImpl(CommandQueue& cq) {
    // STUB: Run the trace in eager mode for now
    // auto& tq = cq.trace()->queue();
    // for (const auto& cmd : tq.worker_queue) {
    //     cq.run_command_impl(cmd);
    // }
    TT_THROW("EnqueueTrace is not yet implemented!");
}

CommandQueue::CommandQueue(Device* device, uint32_t id, CommandQueueMode mode) :
    device_ptr(device),
    cq_id(id),
    mode(mode),
    worker_state(CommandQueueState::IDLE) {
    if (this->async_mode()) {
        num_async_cqs++;
        // The main program thread launches the Command Queue
        parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        this->start_worker();
    } else if (this->passthrough_mode()) {
        num_passthrough_cqs++;
    }
}

CommandQueue::CommandQueue(Trace& trace) :
    device_ptr(nullptr),
    parent_thread_id(0),
    cq_id(-1),
    mode(CommandQueueMode::TRACE),
    worker_state(CommandQueueState::IDLE) {
}

CommandQueue::~CommandQueue() {
    if (this->async_mode()) {
        this->stop_worker();
    }
    if (not this->trace_mode()) {
        TT_FATAL(this->worker_queue.empty(), "{} worker queue must be empty on destruction", this->name());
    }
}

HWCommandQueue& CommandQueue::hw_command_queue() {
    return this->device()->hw_command_queue(this->cq_id);
}

void CommandQueue::dump() {
    int cid = 0;
    log_info(LogMetalTrace, "Dumping {}, mode={}", this->name());
    for (const auto& cmd : this->worker_queue) {
        log_info(LogMetalTrace, "[{}]: {}", cid, cmd.type);
        cid++;
    }
}

std::string CommandQueue::name() {
    if (this->mode == CommandQueueMode::TRACE) {
        return "TraceQueue";
    }
    return "CQ" + std::to_string(this->cq_id);
}

void CommandQueue::wait_until_empty() {
    log_trace(LogDispatch, "{} WFI start", this->name());
    if (this->async_mode()) {
        // Insert a flush token to push all prior commands to completion
        // Necessary to avoid implementing a peek and pop on the lock-free queue
        this->worker_queue.push(CommandInterface{.type = EnqueueCommandType::FLUSH});
    }
    while (true) {
        if (this->worker_queue.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    log_trace(LogDispatch, "{} WFI complete", this->name());
}

void CommandQueue::set_mode(const CommandQueueMode& mode) {
    TT_ASSERT(not this->trace_mode(), "Cannot change mode of a trace command queue, copy to a non-trace command queue instead!");
    if (this->mode == mode) {
        // Do nothing if requested mode matches current CQ mode.
        return;
    }
    this->mode = mode;
    if (this->async_mode()) {
        num_async_cqs++;
        num_passthrough_cqs--;
        // Record parent thread-id and start worker.
        parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        start_worker();
    } else if (this->passthrough_mode()) {
        num_passthrough_cqs++;
        num_async_cqs--;
        // Wait for all cmds sent in async mode to complete and stop worker.
        this->wait_until_empty();
        this->stop_worker();
    }
}

void CommandQueue::start_worker() {
    if (this->worker_state == CommandQueueState::RUNNING) {
        return;  // worker already running, exit
    }
    this->worker_state = CommandQueueState::RUNNING;
    this->worker_thread = std::make_unique<std::thread>(std::thread(&CommandQueue::run_worker, this));
    tt::log_debug(tt::LogDispatch, "{} started worker thread", this->name());
}

void CommandQueue::stop_worker() {
    if (this->worker_state == CommandQueueState::IDLE) {
        return;  // worker already stopped, exit
    }
    this->worker_state = CommandQueueState::TERMINATE;
    this->worker_thread->join();
    this->worker_state = CommandQueueState::IDLE;
    tt::log_debug(tt::LogDispatch, "{} stopped worker thread", this->name());
}

void CommandQueue::run_worker() {
    // forever loop checking for commands in the worker queue
    // Track the worker thread id, for cases where a command calls a sub command.
    // This is to detect cases where commands may be nested.
    worker_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    while (true) {
        if (this->worker_queue.empty()) {
            if (this->worker_state == CommandQueueState::TERMINATE) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        } else {
            std::shared_ptr<CommandInterface> command(this->worker_queue.pop());
            run_command_impl(*command);
        }
    }
}

void CommandQueue::run_command(const CommandInterface& command) {
    log_trace(LogDispatch, "{} received {} in {} mode", this->name(), command.type, this->mode);
    if (not this->passthrough_mode()) {
        if (std::hash<std::thread::id>{}(std::this_thread::get_id()) == parent_thread_id or this->trace_mode()) {
            // Push to worker queue for trace or async mode. In trace mode, store the execution in the queue.
            // In async mode when parent pushes cmd, feed worker through queue.
            this->worker_queue.push(command);
            if (command.blocking.has_value() and *command.blocking == true) {
                TT_ASSERT(not this->trace_mode(), "Blocking commands cannot be traced!");
                this->wait_until_empty();
            }
        }
        else {
            // Handle case where worker pushes command to itself (passthrough)
            TT_ASSERT(std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_thread_id, "Only main thread or worker thread can run commands through the SW command queue");
            run_command_impl(command);
        }
    } else {
        this->run_command_impl(command);
    }
}

void CommandQueue::run_command_impl(const CommandInterface& command) {
    log_trace(LogDispatch, "{} running {}", this->name(), command.type);
    switch (command.type) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER:
            TT_ASSERT(command.dst.has_value(), "Must provide a dst!");
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueReadBufferImpl(*this, command.buffer.value(), command.dst.value(), command.blocking.value());
            break;
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER:
            TT_ASSERT(command.src.has_value(), "Must provide a src!");
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueWriteBufferImpl(*this, command.buffer.value(), command.src.value(), command.blocking.value());
            break;
        case EnqueueCommandType::ALLOCATE_BUFFER:
            TT_ASSERT(command.alloc_md.has_value(), "Must provide buffer allocation metdata!");
            EnqueueAllocateBufferImpl(command.alloc_md.value());
            break;
        case EnqueueCommandType::DEALLOCATE_BUFFER:
            TT_ASSERT(command.alloc_md.has_value(), "Must provide buffer allocation metdata!");
            EnqueueDeallocateBufferImpl(command.alloc_md.value());
            break;
        case EnqueueCommandType::GET_BUF_ADDR:
            TT_ASSERT(command.dst.has_value(), "Must provide a dst address!");
            TT_ASSERT(command.shadow_buffer.has_value(), "Must provide a shadow buffer!");
            EnqueueGetBufferAddrImpl(command.dst.value(), command.shadow_buffer.value());
            break;
        case EnqueueCommandType::SET_RUNTIME_ARGS:
            TT_ASSERT(command.runtime_args_md.has_value(), "Must provide RuntimeArgs Metdata!");
            EnqueueSetRuntimeArgsImpl(command.runtime_args_md.value());
            break;
        case EnqueueCommandType::UPDATE_RUNTIME_ARGS:
            TT_ASSERT(command.runtime_args_md.has_value(), "Must provide RuntimeArgs Metdata!");
            EnqueueUpdateRuntimeArgsImpl(command.runtime_args_md.value());
            break;
        case EnqueueCommandType::ADD_BUFFER_TO_PROGRAM:
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.program.has_value(), "Must provide a program!");
            EnqueueAddBufferToProgramImpl(command.buffer.value(), command.program.value());
            break;
        case EnqueueCommandType::ENQUEUE_PROGRAM:
            TT_ASSERT(command.program.has_value(), "Must provide a program!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueProgramImpl(*this, command.program.value(), command.blocking.value());
            break;
        case EnqueueCommandType::ENQUEUE_TRACE:
            EnqueueTraceImpl(*this);
            break;
        case EnqueueCommandType::ENQUEUE_RECORD_EVENT:
            TT_ASSERT(command.event.has_value(), "Must provide an event!");
            EnqueueRecordEventImpl(*this, command.event.value());
            break;
        case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT:
            TT_ASSERT(command.event.has_value(), "Must provide an event!");
            EnqueueWaitForEventImpl(*this, command.event.value());
            break;
        case EnqueueCommandType::FINISH:
            FinishImpl(*this);
            break;
        case EnqueueCommandType::FLUSH:
            // Used by CQ to push prior commands
            break;
        default:
            TT_THROW("Invalid command type");
    }
    log_trace(LogDispatch, "{} running {} complete", this->name(), command.type);
}

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, EnqueueCommandType const& type) {
    switch (type) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER: os << "ENQUEUE_READ_BUFFER"; break;
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER: os << "ENQUEUE_WRITE_BUFFER"; break;
        case EnqueueCommandType::ENQUEUE_PROGRAM: os << "ENQUEUE_PROGRAM"; break;
        case EnqueueCommandType::ENQUEUE_TRACE: os << "ENQUEUE_TRACE"; break;
        case EnqueueCommandType::ENQUEUE_RECORD_EVENT: os << "ENQUEUE_RECORD_EVENT"; break;
        case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT: os << "ENQUEUE_WAIT_FOR_EVENT"; break;
        case EnqueueCommandType::FINISH: os << "FINISH"; break;
        case EnqueueCommandType::FLUSH: os << "FLUSH"; break;
        default: TT_THROW("Invalid command type!");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, CommandQueue::CommandQueueMode const& type) {
    switch (type) {
        case CommandQueue::CommandQueueMode::PASSTHROUGH: os << "PASSTHROUGH"; break;
        case CommandQueue::CommandQueueMode::ASYNC: os << "ASYNC"; break;
        case CommandQueue::CommandQueueMode::TRACE: os << "TRACE"; break;
        default: TT_THROW("Invalid CommandQueueMode type!");
    }
    return os;
}
