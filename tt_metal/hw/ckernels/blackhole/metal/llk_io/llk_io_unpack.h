// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "llk_unpack_common_api.h"
#include "stream_interface.h"
#include "stream_io_map.h"
#include "tools/profiler/kernel_profiler.hpp"

using namespace ckernel;

// "llk_setup_operands" is the old function name that HLKC emits
inline void llk_setup_operands() {
    volatile tt_l1_ptr std::uint32_t* circular_buffer_config_addr = (volatile uint32_t*)(CIRCULAR_BUFFER_CONFIG_BASE);

    for (uint32_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {
        // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words!
        uint32_t fifo_addr = circular_buffer_config_addr[0];
        uint32_t fifo_size = circular_buffer_config_addr[1];
        uint32_t fifo_num_pages = circular_buffer_config_addr[2];  // not used atm
        uint32_t fifo_page_size = circular_buffer_config_addr[3];

        cb_interface[cb_id].fifo_rd_ptr = fifo_addr;
        cb_interface[cb_id].fifo_size = fifo_size;
        cb_interface[cb_id].fifo_limit = fifo_addr + fifo_size;  // Check if there is overflow
        cb_interface[cb_id].tiles_acked = 0;
        cb_interface[cb_id].fifo_page_size = fifo_page_size;

        circular_buffer_config_addr += UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG;  // move by 3 uint32's
    }
}

// Wait for N tiles available in the incoming stream
inline void llk_wait_tiles(int operand, std::int32_t num_tiles) {
    // TODO(MO): Manually uncomment until issue #6619 is resolved
    // DeviceZoneScopedSumN1("CB-COMPUTE-WAIT-FRONT");
    std::uint32_t input = operand;
    volatile tt_l1_ptr std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(operand);
    std::uint16_t num_tiles_u = (std::uint16_t)num_tiles;

    std::uint16_t tiles_received;

    uint16_t num_tiles_recv;
    do {
        tiles_received = (std::uint16_t)reg_read((std::uint32_t)tiles_received_ptr);
        num_tiles_recv = tiles_received - cb_interface[input].tiles_acked;
    } while (num_tiles_recv < num_tiles_u);
}

// Pop N tiles from the incoming stream
inline void llk_pop_tiles(
    const std::int32_t operand, const std::int32_t num_tiles, const std::int32_t block_c_dim = 0) {
    std::uint32_t input = operand;
    volatile tt_reg_ptr std::uint32_t* tiles_acked_ptr =
        (volatile std::uint32_t*)((((volatile std::uint32_t)get_cb_tiles_acked_ptr(operand)) >> 2) & 0x3ffff);
    std::uint32_t num_words = num_tiles * cb_interface[operand].fifo_page_size;

    cb_interface[input].tiles_acked += num_tiles;
    TT_SETDMAREG(0, cb_interface[input].tiles_acked, 0, LO_16(4));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);
    TT_STOREREG(4, (std::uint32_t)&tiles_acked_ptr[0]);
    cb_interface[input].fifo_rd_ptr += num_words;

    if (cb_interface[input].fifo_rd_ptr >= cb_interface[input].fifo_limit) {
        cb_interface[input].fifo_rd_ptr -= cb_interface[input].fifo_size;
    }
}

inline void llk_wait_blocks(int operand, std::int32_t num_blocks) { llk_wait_tiles(operand, num_blocks); }

// FIXME-WH-UPLIFT
// FIXME: FP32 accumulation --> pop tiles in the operand? just change rd_ptr?
inline void llk_clear_tiles(std::uint32_t operand, std::uint32_t num_tiles) {
    // std::uint32_t input = operand_to_input_index(operand);
    // if (cb_interface[input].accumulation_buffer) {
    //     std::uint32_t num_words = num_tiles * cb_interface[input].fifo_page_size;

    //     cb_interface[input].fifo_rd_ptr += num_words;

    //     if (cb_interface[input].f.fifo_rd_ptr >= operands[input].fifo_limit) {
    //         cb_interface[input].f.fifo_rd_ptr -= operands[input].fifo_size;
    //     }

    //     cb_interface[input].f.fifo_rd_base_ptr = operands[input].fifo_rd_ptr; //inc base ptr

    //     cb_interface[input].curr_iter = 0;
    // }
}
