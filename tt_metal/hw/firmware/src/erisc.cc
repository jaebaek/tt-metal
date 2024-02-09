// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"
#include "ethernet/dataflow_api.h"
#include "firmware_common.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "noc_parameters.h"
#include "risc_attribs.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_producer.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void ApplicationHandler(void) __attribute__((__section__(".init")));

#ifdef __cplusplus
}
#endif

namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
}

uint8_t noc_index = 0;  // TODO: remove hardcoding
uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));

FORCE_INLINE
void multicore_eth_cb_wait_front(db_cb_config_t *eth_db_cb_config, int32_t num_pages) {
    DEBUG_STATUS('C', 'R', 'B', 'W');

    uint16_t pages_received;
    do {
        pages_received = uint16_t(eth_db_cb_config->recv - eth_db_cb_config->ack);
        internal_::risc_context_switch(); // AL: hopefully we can remove this...
        erisc_info->unused_arg1 = 0x02130000 + pages_received;
        erisc_info->unused_arg2 = 0x02130000 + num_pages;
    } while (pages_received < num_pages);
    DEBUG_STATUS('C', 'R', 'B', 'D');
}

FORCE_INLINE
void multicore_eth_cb_pop_front(
    db_cb_config_t *eth_db_cb_config,
    const db_cb_config_t *remote_db_cb_config,
    uint64_t producer_noc_encoding,
    uint32_t num_pages) {
    eth_db_cb_config->ack += num_pages;

    uint32_t pages_ack_addr = (uint32_t)(&(remote_db_cb_config->ack));
    noc_semaphore_set_remote((uint32_t)(&(eth_db_cb_config->ack)), producer_noc_encoding | pages_ack_addr);
}

FORCE_INLINE
void eth_db_acquire(volatile uint32_t *semaphore, uint64_t noc_encoding) {
    while (semaphore[0] == 0 and routing_info->routing_enabled and erisc_info->launch_user_kernel == 0) {
        // Without this context switch a src router on R chip of N300 may get configured for FD
        //  and block L chip of N300 from sending config for dst router on R because the path to the dst router is through the src router
        internal_::risc_context_switch();
    }
}

template <uint32_t producer_cmd_base_addr, uint32_t producer_data_buffer_size, uint32_t consumer_cmd_base_addr, uint32_t consumer_data_buffer_size>
FORCE_INLINE
void eth_program_consumer_cb(
    db_cb_config_t* db_cb_config,
    const db_cb_config_t* remote_db_cb_config,
    bool db_buf_switch,
    uint64_t consumer_noc_encoding,
    uint32_t num_pages,
    uint32_t page_size,
    uint32_t cb_size) {
    // This sets up multi-core CB where the writer is an eth core and consumer is tensix core
    // The data is at different L1 addresses in the producer and consumer
    //  but both producer and consumer use the read pointer to determine location of data in their local L1
    // Producer uses the read pointer when sending to consumer and uses the write pointer to determine L1 address of data in consumer
    // Consumer uses the read pointer to read in data
    // To account for differences in data location, the consumer is sent cb config with the "correct" rd pointer from its POV
    //  and after sending, producer sets it back to its local L1 address

    uint32_t cb_start_producer_addr = get_db_buf_addr<producer_cmd_base_addr, producer_data_buffer_size>(db_buf_switch);
    uint32_t cb_start_consumer_addr = get_db_buf_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch);

    db_cb_config->ack = 0;
    db_cb_config->recv = 0;
    db_cb_config->num_pages = num_pages;
    db_cb_config->page_size_16B = page_size >> 4;
    db_cb_config->total_size_16B = cb_size >> 4;
    db_cb_config->rd_ptr_16B = cb_start_consumer_addr >> 4;    // first set the rd_ptr to value that conumer needs to see
    db_cb_config->wr_ptr_16B = cb_start_consumer_addr >> 4;

    noc_async_write(
        (uint32_t)(db_cb_config), consumer_noc_encoding | (uint32_t)(remote_db_cb_config), sizeof(db_cb_config_t));
    noc_async_write_barrier();  // barrier for now
    // db cb config has been sent to consumer, now read address can be set to expected value for eth producer
    db_cb_config->rd_ptr_16B = cb_start_producer_addr >> 4;
    noc_async_write_barrier();  // barrier for now
}

void __attribute__((section("code_l1"))) risc_init() {
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(n, 0, NOC_NODE_ID);
        my_x[n] = noc_id_reg & NOC_NODE_ID_MASK;
        my_y[n] = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    }
}

void __attribute__((section("code_l1"))) router_init() {
    relay_src_noc_encoding = uint32_t(NOC_XY_ENCODING(routing_info->relay_src_x, routing_info->relay_src_y));
    relay_dst_noc_encoding = uint32_t(NOC_XY_ENCODING(routing_info->relay_dst_x, routing_info->relay_dst_y));

    eth_router_noc_encoding = uint32_t(NOC_XY_ENCODING(my_x[0], my_y[0]));
    my_routing_mode = (EthRouterMode)routing_info->routing_mode;
}

void __attribute__((section("erisc_l1_code"))) ApplicationHandler(void) {
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    risc_init();
    noc_init();

    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }
    ncrisc_noc_full_sync();
    while (routing_info->routing_enabled != 1) {
        internal_::risc_context_switch();
    }

    router_init();

    volatile tt_l1_ptr uint32_t *eth_db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(0));

    static constexpr uint32_t command_start_addr = eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;
    static constexpr uint32_t data_buffer_size = MEM_ETH_SIZE - (DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(uint32_t)) - eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;

    bool db_buf_switch = false;
    db_cb_config_t *eth_db_cb_config = get_local_db_cb_config(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE, false);
    const db_cb_config_t *remote_src_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, false);
    const db_cb_config_t *remote_dst_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, true);

    uint32_t num_pages_transferred = 0;

    while (routing_info->routing_enabled) {
        // FD: assume that no more host -> remote writes are pending
        if (erisc_info->launch_user_kernel == 1) {
            kernel_profiler::init_profiler();
            kernel_profiler::mark_time(CC_MAIN_START);
            kernel_init();
            kernel_profiler::mark_time(CC_MAIN_END);
        }
        if (my_routing_mode == EthRouterMode::FD_SRC) {
            eth_db_acquire(eth_db_semaphore_addr, ((uint64_t)eth_router_noc_encoding << 32));
            if (erisc_info->launch_user_kernel == 1) {
                continue;
            }
            if (routing_info->routing_enabled == 0) {
                break;
            }
            noc_semaphore_inc(
                ((uint64_t)eth_router_noc_encoding << 32) | uint32_t(eth_db_semaphore_addr),
                -1);  // Two's complement addition
            noc_async_write_barrier();

            volatile tt_l1_ptr uint32_t *command_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t *>(command_start_addr);
            volatile tt_l1_ptr CommandHeader *header = (CommandHeader *)command_ptr;
            uint32_t num_buffer_transfers = header->num_buffer_transfers;
            uint32_t producer_consumer_transfer_num_pages = header->producer_router_transfer_num_pages;
            command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;

            if (num_buffer_transfers == 0) {
                 // send cmd even if there is no data associated
                internal_::send_fd_packets();
            }

            erisc_info->unused_arg0 = 210;
            for (uint32_t i = 0; i < num_buffer_transfers; i++) {
                erisc_info->unused_arg1 = i;
                const uint32_t num_pages = command_ptr[2];
                uint32_t num_pages_tunneled = 0;
                while (num_pages_tunneled != num_pages) {
                    uint32_t num_to_write = min(num_pages, producer_consumer_transfer_num_pages);
                    erisc_info->unused_arg1 = num_to_write;
                    erisc_info->unused_arg2 = num_pages;
                    multicore_eth_cb_wait_front(eth_db_cb_config, num_to_write);
                    // contains device command, maybe just send pages, and send cmd once at the start
                    internal_::send_fd_packets();
                    multicore_eth_cb_pop_front(
                        eth_db_cb_config, remote_src_db_cb_config, ((uint64_t)relay_src_noc_encoding << 32), num_to_write);
                    num_pages_tunneled += num_to_write;
                }
            }
            noc_semaphore_inc(((uint64_t)relay_src_noc_encoding << 32) | get_semaphore(0), 1);
            noc_async_write_barrier();  // Barrier for now
        } else if (routing_info->routing_mode == EthRouterMode::FD_DST) {
            // Poll until FD_SRC router sends FD packet
            // Each FD packet comprises of command header followed by command data
            internal_::wait_for_fd_packet();
            if (erisc_info->launch_user_kernel == 1) {
                continue;
            }
            if (routing_info->routing_enabled == 0) {
                break;
            }

            volatile tt_l1_ptr uint32_t *command_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t *>(command_start_addr);
            volatile tt_l1_ptr CommandHeader *header = (CommandHeader *)command_ptr;

            // Block until consumer can accept new command
            // `num_pages_transferred` tracks whether the remote command processor received all the data
            while (eth_db_semaphore_addr[0] == 0 and num_pages_transferred == 0) {
                internal_::risc_context_switch();
            } // Check that there is space in consumer to send command

            // Send the full command header
            constexpr uint32_t consumer_cmd_base_addr = L1_UNRESERVED_BASE;
            constexpr uint32_t consumer_data_buffer_size = (MEM_L1_SIZE - (DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(uint32_t)) - L1_UNRESERVED_BASE);
            uint32_t page_size = header->page_size;
            // The consumer is the remote command processor which acts like a producer from the perspective of the dispatch core
            uint32_t consumer_cb_num_pages = header->producer_cb_num_pages;
            uint32_t consumer_cb_size = header->producer_cb_size;
            uint32_t num_buffer_transfers = header->num_buffer_transfers;
            if (num_pages_transferred == 0) {   // new command
                eth_program_consumer_cb<command_start_addr, data_buffer_size, consumer_cmd_base_addr, consumer_data_buffer_size>(
                    eth_db_cb_config,
                    remote_dst_db_cb_config,
                    db_buf_switch,
                    ((uint64_t)relay_dst_noc_encoding << 32),
                    consumer_cb_num_pages,
                    page_size,
                    consumer_cb_size);
                relay_command<command_start_addr, consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch, ((uint64_t)relay_dst_noc_encoding << 32));
                num_relayed[0] = num_relayed[0] + 1;

                update_producer_consumer_sync_semaphores(((uint64_t)eth_router_noc_encoding << 32), ((uint64_t)relay_dst_noc_encoding << 32), eth_db_semaphore_addr, get_semaphore(0));
            }

            // Send the data that was in this packet
            uint32_t total_num_pages = header->num_pages;
            uint32_t num_pages_to_tx = 0;
           // for (uint32_t i = 0; i < num_buffer_transfers; i++) {
              //internal_::wait_for_fd_packet(1+1);
              if (num_pages_transferred < total_num_pages) {
                // AL: change to while loop?
                uint32_t src_addr = eth_db_cb_config->rd_ptr_16B << 4;
                uint64_t dst_noc_addr = ((uint64_t)relay_dst_noc_encoding << 32) | (eth_db_cb_config->wr_ptr_16B << 4);

                uint32_t producer_consumer_transfer_num_pages = header->producer_router_transfer_num_pages;
                command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER; // jump to buffer transfer region
                const uint32_t num_pages = command_ptr[2];
                // producer_consumer_transfer_num_pages is the total num of data pages that could fit in a FD packet
                num_pages_to_tx = min(num_pages, producer_consumer_transfer_num_pages);

                noc_async_write(src_addr, dst_noc_addr, page_size * num_pages_to_tx);
                uint32_t l1_consumer_fifo_limit_16B =
                    (get_db_buf_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch) + consumer_cb_size) >> 4;
                multicore_cb_push_back(
                    eth_db_cb_config,
                    remote_dst_db_cb_config,
                    ((uint64_t)relay_dst_noc_encoding << 32),
                    l1_consumer_fifo_limit_16B,
                    num_pages_to_tx
                );
              }
                num_pages_transferred += num_pages_to_tx;
              if (num_pages_transferred == total_num_pages) {
                // Done sending all data associated with this command, reset `num_pages_transferred` for next command
                  num_pages_transferred = 0;
                  // debug[0] = 3118;
               }
                // Signal to FD_SRC router that packet has been processed
                internal_::ack_fd_packet();
          //  }

        } else {
            internal_::risc_context_switch();
        }
    }
    internal_::disable_erisc_app();
}
