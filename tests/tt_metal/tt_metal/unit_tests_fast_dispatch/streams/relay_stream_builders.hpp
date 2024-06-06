// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/common/core_coord.h"

#include <vector>
#include <cstdint>

namespace tt {

namespace tt_metal {

namespace streams {
constexpr uint32_t tile_header_size = 32;  // needs to provide noc word alignment
// constexpr uint32_t tile_header_size = 16;
constexpr uint32_t noc_word_size = 16;
}

struct stream_config_t {
    uint32_t buffer_addr;
    uint32_t buffer_size;  // in bytes
    uint32_t tile_header_buffer_addr;
    uint32_t tile_header_num_msgs;
    uint32_t tile_header_buffer_size;  // in bytes
};

struct stream_builder_spec_t {
    uint32_t buffer_size_bytes;
    uint32_t tile_header_buffer_size_bytes;
};

std::vector<uint32_t> get_relay_rt_args(
    Device* device,
    uint32_t relay_stream_overlay_blob_addr,
    uint32_t relay_done_semaphore,
    CoreCoord const& sender_core,
    CoreCoord const& receiver_core,
    uint32_t sender_noc_id,
    uint32_t receiver_noc_id,
    // stream_config_t const& sender_stream_config,
    stream_config_t const& relay_stream_config,
    stream_config_t const& receiver_stream_config,
    uint32_t remote_src_start_phase_addr,
    uint32_t dest_remote_src_start_phase_addr,
    bool is_first_relay_in_chain);


} // namespace tt_metal
} // namespace tt
