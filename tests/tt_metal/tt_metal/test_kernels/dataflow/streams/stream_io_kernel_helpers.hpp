// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "dataflow_api.h"
#include "stream_interface.h"
#include "tt_metal/hw/inc/wormhole/noc/noc_overlay_parameters.h"

struct stream_state_t {
    const uint32_t local_data_buffer_base_address;
    const uint32_t local_msg_info_ptr_base_address;

    uint32_t local_stream_id;
    uint32_t remote_stream_id;

    const uint32_t local_start_phase_id;
    uint32_t local_phase_id;
    uint32_t messages_per_phase;
    uint32_t msg_info_wrptr_addr;

    uint32_t num_tiles_sent;
    uint32_t tile_header_num_msgs;

    uint32_t local_buffer_base_addr;
    uint32_t local_buffer_size;
    uint32_t local_msg_info_ptr;
    uint32_t local_buffer_read_offset;

    uint32_t remote_buffer_base_addr;
    uint32_t remote_buffer_size;
    uint32_t remote_msg_info_ptr;
    uint32_t remote_buffer_write_offset;

    uint32_t remote_phase_id;

    uint32_t get_current_local_buffer_address() const {
        return local_data_buffer_base_address + local_buffer_read_offset;
    }
};

struct fabric_sender_stream_state_t {
    // All possible helpers should be hidden behind this struct because it will
    // simplify the enablement of optimization
    const uint32_t local_data_buffer_base_address;
    const uint32_t local_msg_info_ptr_base_address;

    uint32_t local_stream_id;
    uint32_t remote_stream_id;

    const uint32_t local_start_phase_id;
    uint32_t local_phase_id;
    uint32_t messages_per_phase;
    uint32_t msg_info_wrptr_addr;

    uint32_t num_tiles_sent;
    uint32_t tile_header_num_msgs;

    // uint32_t local_buffer_base_addr;
    // uint32_t local_buffer_size;
    uint32_t local_msg_info_ptr;
    // uint32_t local_buffer_read_offset;

    uint32_t remote_buffer_base_addr;
    uint32_t remote_buffer_size;
    uint32_t remote_msg_info_ptr;
    uint32_t remote_buffer_write_offset;

    uint32_t remote_phase_id;
    uint32_t

    uint32_t get_current_local_buffer_address() const {
        return local_data_buffer_base_address + local_buffer_read_offset;
    }
};


struct phase_iterator_t {
    phase_iterator_t(uint32_t start_phase, uint32_t max_phase) :
        phase_id(start_phase), max_phase(max_phase), start_phase(start_phase) {}
    uint32_t phase_id;
    uint32_t max_phase;
    uint32_t start_phase;

    FORCE_INLINE uint32_t get() const { return phase_id; }

    FORCE_INLINE void increment() { phase_id = phase_id == max_phase ? start_phase : phase_id + 1; }
};

struct noc_endpoint_info_t {
    uint32_t data_noc_id;
    uint32_t update_noc_id;
    uint32_t noc_x;
    uint32_t noc_y;
};

#define STREAM_CFG(field, val) ((val) << (field))

#define AUTO_CFG_HEADER(next_phase_num_cfg_reg_writes, curr_phase_num_msgs, phase_num_incr) \
    ((uint32_t)(((next_phase_num_cfg_reg_writes) << 24) | ((curr_phase_num_msgs) << 12) | (phase_num_incr)))

#define STREAM_REMOTE_DEST(dest_x, dest_y, dest_stream_id)                     \
    (((dest_x) << STREAM_REMOTE_DEST_X) | ((dest_y) << STREAM_REMOTE_DEST_Y) | \
     ((dest_stream_id) << STREAM_REMOTE_DEST_STREAM_ID))

#define STREAM_REMOTE_SRC(src_x, src_y, src_stream_id) \
    (((src_x) << STREAM_REMOTE_SRC_X) | ((src_y) << STREAM_REMOTE_SRC_Y) | ((src_stream_id) << REMOTE_SRC_STREAM_ID))

FORCE_INLINE uint32_t
blob_header_dw(uint32_t next_phase_num_cfg_reg_writes, uint32_t curr_phase_num_msgs, uint32_t phase_num_incr) {
    return (next_phase_num_cfg_reg_writes << 24) | (curr_phase_num_msgs << 12) | phase_num_incr;
}

FORCE_INLINE void stream_phase_blob_run(
    uint32_t stream_id, volatile uint32_t *blob_start_addr, uint32_t start_phase_num_cfg_regs) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX, reinterpret_cast<uint32_t>(blob_start_addr));
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, start_phase_num_cfg_regs << NEXT_PHASE_NUM_CFG_REG_WRITES);
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_MISC_CFG_REG_INDEX,
        (0x1 << PHASE_AUTO_CONFIG) | (1 << NEXT_PHASE_SRC_CHANGE) | (1 << NEXT_PHASE_DEST_CHANGE));
}
FORCE_INLINE void stream_phase_blob_run(
    uint32_t stream_id,
    volatile uint32_t *blob_start_addr,
    uint32_t num_messages_per_phase,
    uint32_t start_phase_num_cfg_regs) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX, reinterpret_cast<uint32_t>(blob_start_addr));

    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX,
        blob_header_dw(start_phase_num_cfg_regs, num_messages_per_phase, 1));
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_MISC_CFG_REG_INDEX,
        (0x1 << PHASE_AUTO_ADVANCE) | (0x1 << PHASE_AUTO_CONFIG) | (1 << NEXT_PHASE_SRC_CHANGE) |
            (1 << NEXT_PHASE_DEST_CHANGE));
    NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_ADVANCE_REG_INDEX, 1);
}

FORCE_INLINE uint32_t blob_cfg_dw(uint32_t reg_index, uint32_t reg_val) { return (reg_val << 8) | reg_index; }

FORCE_INLINE uint32_t set_blob_reg_field(uint32_t blob_dw, uint32_t field_width, uint32_t field_offset, uint32_t val) {
    uint32_t mask = ((1 << field_width) - 1) << field_offset;
    return (blob_dw & ~mask) | ((val << field_offset) & mask);
}

FORCE_INLINE uint32_t get_first_available_phase_out_of_reset(uint32_t stream_id) {
    uint32_t stream_phase_coming_out_of_reset = stream_get_curr_phase(stream_id);
    return (
        stream_phase_coming_out_of_reset < 4096   ? 4096 : 1);
}

FORCE_INLINE uint32_t notify_remote_receiver_of_starting_phase(
    uint32_t stream_id, uint32_t local_buffer_addr, uint64_t remote_receiver_noc_addr) {
    uint32_t starting_phase = get_first_available_phase_out_of_reset(stream_id);
    ASSERT(starting_phase > 0);
    *reinterpret_cast<volatile uint32_t *>(local_buffer_addr) = starting_phase;
    noc_async_write(local_buffer_addr, remote_receiver_noc_addr, sizeof(uint32_t));
    // noc_semaphore_set_remote(local_buffer_addr, remote_receiver_noc_addr);
    noc_async_writes_flushed();
    return starting_phase;
}

FORCE_INLINE uint32_t wait_for_remote_source_starting_phase(volatile uint32_t *addr) {
    while (*addr == 0) {
        asm volatile("nop");
    }
    return *addr;
}

////////////////////////////////////////////////
///  Remote Sender Helpers
////////////////////////////////////////////////
FORCE_INLINE uint32_t get_sender_stream_config_reg(uint32_t tx_noc_id, uint32_t rx_src_update_noc, bool drain_after_phase_send) {
    uint32_t stream_cfg_reg = 0;
    bool next_phase_src_dest_change = drain_after_phase_send ? 1 : 0;
    stream_cfg_reg |= STREAM_CFG(OUTGOING_DATA_NOC, tx_noc_id) | STREAM_CFG(REMOTE_SRC_UPDATE_NOC, rx_src_update_noc) |
                      STREAM_CFG(SOURCE_ENDPOINT, 1) | STREAM_CFG(REMOTE_RECEIVER, 1) |
                      STREAM_CFG(NEXT_PHASE_SRC_CHANGE, next_phase_src_dest_change) |
                      STREAM_CFG(NEXT_PHASE_DEST_CHANGE, next_phase_src_dest_change) |
                      STREAM_CFG(PHASE_AUTO_ADVANCE, 0) | STREAM_CFG(DATA_AUTO_SEND, 0) |
                      STREAM_CFG(REG_UPDATE_VC_REG, 1);

    return stream_cfg_reg;
}


FORCE_INLINE void write_message_size_to_message_info_buffer(
    stream_state_t const &stream_state, uint32_t message_size_noc_words) {
    ASSERT((message_size_noc_words << 4) <= stream_state.local_buffer_size);
    if (!((message_size_noc_words << 4) <= stream_state.local_buffer_size)) {
        DPRINT << "YIKES\n";
    }
    *reinterpret_cast<volatile uint32_t *>(stream_state.local_msg_info_ptr) = message_size_noc_words;
}

FORCE_INLINE void reset_stream_message_info_buffer_rdptr(stream_state_t &stream_state, uint32_t stream_id) {
    stream_state.local_msg_info_ptr = stream_state.local_msg_info_ptr_base_address;
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_MSG_INFO_PTR_REG_INDEX, ((uint32_t)(stream_state.local_msg_info_ptr_base_address >> 4)));
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX, (((uint32_t)stream_state.local_msg_info_ptr_base_address >> 4)));
}
FORCE_INLINE void advance_stream_message_info_buffer_wrptr(
    stream_state_t &stream_state, uint32_t stream_id, uint32_t message_size) {
    stream_state.local_msg_info_ptr += (1 << 4);
    stream_state.local_buffer_read_offset += message_size;
    if (stream_state.local_buffer_read_offset >= stream_state.local_buffer_size) {
        stream_state.local_buffer_read_offset -= stream_state.local_buffer_size;
    }
}

FORCE_INLINE void wait_for_stream_write_complete(uint32_t sender_stream_id) {
    while (!stream_phase_advance_wait(sender_stream_id)) {
        asm volatile("nop");
    }
}

// This function is heavily couple with the autonomous looping stream setup. It's *NOT*
// recommended to use this as a generic function for talking to streams unless they are
// setup in this specific looping configuration.
FORCE_INLINE void stream_noc_write(
    uint32_t src_addr,
    uint32_t dest_addr,
    uint32_t size_bytes,
    uint32_t remote_noc_x,
    uint32_t remote_noc_y,
    uint32_t dest_noc_id,
    fabric_sender_stream_state_t &stream_state) {
    const uint32_t tiles_per_phase = stream_state.messages_per_phase;

    // This was taken from the autonomous stream test-bench which already correctly stores
    // the message size in 16B words, in the packet header. However, for packet_(mux|demux),
    // the message size is stored in bytes, so we need to override it to be in 16B words

    uint32_t message_size_noc_words = *reinterpret_cast<volatile uint32_t *>(src_addr);
    ASSERT(size_bytes == message_size_noc_words);
    message_size_noc_words = message_size_noc_words >> 4;
    *reinterpret_cast<volatile uint32_t *>(src_addr) = message_size_noc_words;


    uint32_t dest_noc_reg = 0;
    uint32_t num_tiles = stream_state.num_tiles_sent;
    const bool send_last_message_and_drain = num_tiles == (stream_state.tile_header_num_msgs - 1);

    bool first_message = num_tiles == 0;

    NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_CURR_PHASE_BASE_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_CURR_PHASE_REG_INDEX, stream_state.local_phase_id);

    if (first_message) {
        reset_stream_message_info_buffer_rdptr(stream_state, stream_state.local_stream_id);
        stream_state.local_buffer_read_offset = 0;
    }
    copy_from_cb_to_stream_buffer(stream_state, src_addr, message_size_noc_words);

    // Override this sender stream to point to the src address as its src buffer
    NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_BUF_START_REG_INDEX, src_addr >> 4);
    NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_BUF_SIZE_REG_INDEX, message_size_noc_words);

    uint32_t rx_src_update_noc = 1 - dest_noc_id;
    if (send_last_message_and_drain) {
        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id,
            STREAM_MISC_CFG_REG_INDEX,
            get_sender_stream_config_reg(dest_noc_id, rx_src_update_noc, true));

    } else if (first_message) {

        uint32_t rx_src_update_noc = 1 - dest_noc_id;
        uint32_t translated_remote_noc_x = dest_noc_id == 0 ? remote_noc_x : noc_size_x - 1 - remote_noc_x;
        uint32_t translated_remote_noc_y = dest_noc_id == 0 ? remote_noc_y : noc_size_y - 1 - remote_noc_y;
        uint32_t dest_stream_id = stream_state.remote_stream_id;

        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id,
            STREAM_REMOTE_DEST_REG_INDEX,
            STREAM_REMOTE_DEST(translated_remote_noc_x, translated_remote_noc_y, dest_stream_id));
        NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_HI_REG_INDEX, 0);
        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id, STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX, stream_state.remote_msg_info_ptr >> 4);

        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id, STREAM_REMOTE_DEST_BUF_START_REG_INDEX, stream_state.remote_buffer_base_addr >> 4);
        // Inserting an assert here causes test to pass
        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id,
            STREAM_REMOTE_DEST_BUF_START_HI_REG_INDEX,
            (stream_state.remote_buffer_base_addr / MEM_WORD_WIDTH) >> MEM_WORD_ADDR_WIDTH);
        NOC_STREAM_WRITE_REG_FIELD(
            stream_state.local_stream_id,
            STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX,
            REMOTE_DEST_BUF_SIZE_WORDS,
            stream_state.remote_buffer_size >> 4);

        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id,
            STREAM_MISC_CFG_REG_INDEX,
            get_sender_stream_config_reg(dest_noc_id, rx_src_update_noc, false));
    }

    // Remove this if we want to remove the sync at the end of this function (and instead want to support syncing later)
    ASSERT(
        dest_addr == (NOC_STREAM_READ_REG(stream_state.local_stream_id, STREAM_REMOTE_DEST_BUF_START_REG_INDEX) +
                      NOC_STREAM_READ_REG(stream_state.local_stream_id, STREAM_REMOTE_DEST_WR_PTR_REG_INDEX))
                         << 4);

    write_message_size_to_message_info_buffer(stream_state, message_size_noc_words);
    advance_stream_message_info_buffer_wrptr(stream_state, stream_state.local_stream_id, message_size_noc_words << 4);


    NOC_STREAM_WRITE_REG(
        stream_state.local_stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, AUTO_CFG_HEADER(0, 1 /*tiles_per_phase*/, 1));
    NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_PHASE_ADVANCE_REG_INDEX, 0x1);


    if (first_message) {
        // wait for handshake to complete
        while (!stream_phase_is_active(stream_state.local_stream_id)) {
            asm volatile("");
        }
    }

    if (send_last_message_and_drain) {
        // We only wrap around to 0 when the remote receiver relay stream has finished its second phase. We need to do
        // this to avoid any handshake bugs we might hit if the second phase of relay must sync with phase 1 of the
        // producer (this) since the relay will handshake with phase 1 of the producer (this) stream for relay stream's
        // first phase too
        num_tiles = 0;
        stream_state.remote_phase_id = 3 - stream_state.remote_phase_id;  // will alternate between 1 and 2
        // Remote phase was already updated so the condition is inverted
        stream_state.local_phase_id =
            (stream_state.remote_phase_id == 1) ? stream_state.local_start_phase_id : stream_state.local_phase_id + 1;
    } else {
        num_tiles++;
        stream_state.local_phase_id++;
    }

    stream_relay_tiles(stream_state.local_stream_id, 1, message_size_noc_words);
    wait_for_stream_write_complete(stream_state.local_stream_id);

    stream_state.num_tiles_sent = num_tiles;
}
