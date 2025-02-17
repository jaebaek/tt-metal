// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "debug/dprint.h"

inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++ page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++ j, ++ ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}
#endif

FORCE_INLINE
void read_channels(uint32_t& l1_write_addr_act, const uint32_t act_l1_read_addr, const uint32_t reader_channel_idx,
        const uint32_t conv_act_c_read_bytes, const uint32_t coalesced_read_bytes, const uint32_t stride_h_bytes) {

    constexpr uint32_t unroll_factor = WINDOW_INNER;
    uint32_t act_l1_read_addr_plus_offset = act_l1_read_addr + (reader_channel_idx * conv_act_c_read_bytes);
    #pragma GCC unroll unroll_factor
    for (uint32_t inner = 0; inner < WINDOW_INNER; inner++) {
        noc_async_read_one_packet_with_state<true>(act_l1_read_addr_plus_offset, l1_write_addr_act);
        l1_write_addr_act += coalesced_read_bytes;
        // +2 is hard-coded, TODO: generalize
        act_l1_read_addr_plus_offset += stride_h_bytes;
    }
}

void kernel_main() {

    constexpr bool act_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t conv_act_size_w = get_compile_time_arg_val(3);
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t window_inner = get_compile_time_arg_val(7);
    constexpr uint32_t act_block_h_datums = get_compile_time_arg_val(8);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(10);
    constexpr uint32_t act_num_blocks_h = get_compile_time_arg_val(14);
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(15);
    constexpr uint32_t act_w_num_outer = get_compile_time_arg_val(16);
    constexpr uint32_t act_mcast_num_dests = get_compile_time_arg_val(17);
    constexpr uint32_t act_mcast_num_cores = get_compile_time_arg_val(18);
    constexpr uint32_t act_mcast_sender_semaphore_addr = get_compile_time_arg_val(19);
    constexpr uint32_t act_mcast_receiver_semaphore_addr = get_compile_time_arg_val(20);
    constexpr uint32_t act_mcast_sender_size_bytes = get_compile_time_arg_val(21);

    constexpr bool transpose_mcast = get_compile_time_arg_val(22) == 1;

    uint32_t i = 0;
    uint32_t noop = get_arg_val<uint32_t>(i); i+=1;

    if (noop) {
        return;
    }

    uint32_t act_mcast_dest_noc_start_x = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_mcast_dest_noc_start_y = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_mcast_dest_noc_end_x   = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_mcast_dest_noc_end_y   = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_mcast_sender_id        = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_mcast_sender_noc_x     = get_arg_val<uint32_t>(i); i+=1;

    volatile tt_l1_ptr uint32_t *act_mcast_sender_noc_y  = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(i));

    constexpr uint32_t cb_id_act = tt::CB::c_in0;
    constexpr uint32_t tilized_in0_cb_id = tt::CB::c_intermed1;
    constexpr uint32_t cb_id_sharded_act = tt::CB::c_in3;
    constexpr uint32_t cb_id_act_row_major_bfloat16 = tt::CB::c_in6;

    constexpr uint32_t cb_reader_indices = tt::CB::c_in4;
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    // L1 array
    constexpr uint32_t cb_l1_array = tt::CB::c_in5;
    volatile tt_l1_ptr uint32_t* l1_array = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_l1_array));
    // Set up local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_valid_addr_ptr = &l1_array[0];
    act_mcast_sender_semaphore_valid_addr_ptr[0] = 1; // Load const 1 to be used as semaphore valid value sent from sender to receivers
    uint32_t act_mcast_sender_semaphore_valid_addr = reinterpret_cast<uint32_t>(&l1_array[0]);

    // Set up remote VALID value
    volatile tt_l1_ptr uint32_t* act_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_receiver_semaphore_addr);
    noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_sender_semaphore_addr);

    uint64_t act_multicast_noc_addr = get_noc_multicast_addr(
        act_mcast_dest_noc_start_x,
        act_mcast_dest_noc_start_y,
        act_mcast_dest_noc_end_x,
        act_mcast_dest_noc_end_y,
        0
    );

    uint64_t act_mcast_receiver_semaphore_noc_addr = act_multicast_noc_addr | act_mcast_receiver_semaphore_addr;
    constexpr uint32_t num_issued_reads_per_block = act_block_h_datums * window_inner;

    // TODO: need to make the read coalescing optimization cleaner
    // currently works for the case of num_coalesced_reads == weight_size_w since these reads are contiguous on both src/dst side
    constexpr uint32_t coalesced_read_bytes = weight_size_w * conv_act_c_read_bytes;


    // Fully create act matrix and tilize it before mcast
    // set_state uses just x/y from the get_noc_addr, addr is ignored
    uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);
    noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);

    // Reset reader_idx to finish act_block_h_datums
    uint32_t reader_idx = 0;
    for (uint32_t nbh = 0; nbh < act_num_blocks_h; nbh++) {
        cb_reserve_back(cb_id_act_row_major_bfloat16, act_block_num_tiles);
        uint32_t l1_write_addr_act = get_write_ptr(cb_id_act_row_major_bfloat16);

        constexpr uint32_t stride_h_bytes = (conv_act_size_w + 2) * conv_act_c_read_bytes;
        static_assert(act_block_h_datums % 2 == 0); // need to be even to read 2 in the body, due to packing of 2 indices in 1 uint32_t word
        // #pragma GCC unroll 4 // didn't seem to help (neutral), manual unroll 2x perf drop
        for (uint32_t bh = 0; bh < act_block_h_datums / 2; bh++) {
            uint32_t two_reader_indices = packed_reader_indices_ptr[reader_idx];
            read_channels(l1_write_addr_act, act_l1_read_addr, two_reader_indices & 0xffff, conv_act_c_read_bytes, coalesced_read_bytes, stride_h_bytes);
            read_channels(l1_write_addr_act, act_l1_read_addr, two_reader_indices >> 16   , conv_act_c_read_bytes, coalesced_read_bytes, stride_h_bytes);

            reader_idx++;
        }
        // incrementing num issued in one shot is actually slower
        // noc_async_read_inc_num_issued(num_issued_reads_per_block); // "false" on read
        noc_async_read_barrier();
        cb_push_back(cb_id_act_row_major_bfloat16, act_block_num_tiles);

        // Round robin self-mcast and receive tilized act matrix in cb_id_act
        // Compute should function like regular mm
        for (uint32_t act_w_outer_i = 0; act_w_outer_i < act_w_num_outer; act_w_outer_i++) {
            cb_reserve_back(cb_id_act, act_block_num_tiles);
            if (act_w_outer_i == act_mcast_sender_id) {
                // MCAST SENDER: send entire tilized input to other cores in column
                // wait until all act mcast destinations have atomically incremented the act semaphore_addr (i.e. its value should be act_mcast_num_dests), then reset
                // the semaphore_addr value back to zero for the next block
                noc_semaphore_wait(act_mcast_sender_semaphore_addr_ptr, act_mcast_num_dests);
                noc_semaphore_set(act_mcast_sender_semaphore_addr_ptr, 0);

                noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, INVALID);

                // compute tilizes and pops cb_id_act and pushes to tilized_in0_cb_id
                cb_wait_front(tilized_in0_cb_id, act_block_num_tiles);

                // Now we have the block in the CB address, we can mcast to dests!
                uint32_t tilized_act_start_address = get_read_ptr(tilized_in0_cb_id);

                uint64_t act_multicast_data_addr = act_multicast_noc_addr | get_write_ptr(cb_id_act);
                // num_dests will source, since we are copying to a different local CB as well
                noc_async_write_multicast_loopback_src(tilized_act_start_address, act_multicast_data_addr, act_mcast_sender_size_bytes, act_mcast_num_cores + 1, false, false);

                // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc, same cmd_buf
                // Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).

                // We should also multicast VALID flag to destinations for receiver semaphore
                noc_semaphore_set_multicast_loopback_src(act_mcast_sender_semaphore_valid_addr, act_mcast_receiver_semaphore_noc_addr, act_mcast_num_cores + 1, false, false);

                noc_semaphore_wait(act_mcast_receiver_semaphore_addr_ptr, VALID);
            } else {
                // MCAST RECEIVER: receive entire tilized input from sender core
                // Set act semaphore value to INVALID
                noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, INVALID);

                // Atomic increment source core counter
                uint64_t act_mcast_sender_semaphore_noc_addr;
                if constexpr (transpose_mcast) {
                    act_mcast_sender_semaphore_noc_addr = get_noc_addr(act_mcast_sender_noc_x, act_mcast_sender_noc_y[act_w_outer_i], act_mcast_sender_semaphore_addr);
                } else {
                    act_mcast_sender_semaphore_noc_addr = get_noc_addr(act_mcast_sender_noc_y[act_w_outer_i], act_mcast_sender_noc_x, act_mcast_sender_semaphore_addr);
                }
                noc_semaphore_inc(act_mcast_sender_semaphore_noc_addr, 1);

                // wait on act semaphore value to become VALID (set by mcast sender after it multicasts data)
                noc_semaphore_wait(act_mcast_receiver_semaphore_addr_ptr, VALID);
            }
            cb_push_back(cb_id_act, act_block_num_tiles);
        } // act_w_num_outer
        cb_pop_front(tilized_in0_cb_id, act_block_num_tiles);
    }
}
