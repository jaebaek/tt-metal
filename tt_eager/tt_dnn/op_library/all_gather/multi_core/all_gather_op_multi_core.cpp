// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include "tt_metal/common/core_coord.h"
#include "eth_l1_address_map.h"
#include "impl/buffers/buffer.hpp"
#include "tensor/tensor_impl.hpp"
#include "tt_dnn/op_library/all_gather/all_gather_op.hpp"
#include "tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/ccl_host_datastructures.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/ccl_common.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include <sstream>
#include <type_traits>

using namespace tt::constants;

namespace tt {

namespace tt_metal {

using namespace ccl;

static std::tuple<CoreRangeSet,CoreRangeSet> select_worker_cores(AllGatherConfig const& all_gather_config, uint32_t num_links, uint32_t link, uint32_t full_send_direction) {
    constexpr uint32_t worker_grid_width = 8;
    const bool fit_sender_and_receiver_workers_on_same_row = (worker_grid_width / 2) >= all_gather_config.get_num_eth_buffers_per_edm();
    std::set<CoreRange> receiver_worker_cores = {};
    std::set<CoreRange> sender_worker_cores = {};
    uint32_t max_cols = 8;
    uint32_t curr_row = link * (((all_gather_config.get_num_eth_buffers_per_edm() * 2 - 1) / max_cols) + 1) +
        (full_send_direction * num_links * (((all_gather_config.get_num_eth_buffers_per_edm() * 2 - 1) / max_cols) + 1));
    uint32_t curr_col = 0;
    for (uint32_t r = 0; r < all_gather_config.get_num_eth_buffers_per_edm(); r++) {
        receiver_worker_cores.insert(CoreRange(CoreCoord(curr_col, curr_row)));
        curr_col ++;
        if (curr_col == max_cols) {
            curr_col = 0;
            curr_row++;
        }
    }
    for (uint32_t s = 0; s < all_gather_config.get_num_eth_buffers_per_edm(); s++) {
        sender_worker_cores.insert(CoreRange(CoreCoord(curr_col, curr_row)));
        curr_col ++;
        if (curr_col == max_cols) {
            curr_col = 0;
            curr_row++;
        }
    }
    return {CoreRangeSet(receiver_worker_cores), CoreRangeSet(sender_worker_cores)};
}




// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)
operation::ProgramWithCallbacks all_gather_multi_core_with_workers(const Tensor& input_tensor, Tensor& output_tensor, const uint32_t dim, const uint32_t num_links, const uint32_t ring_size, const uint32_t ring_index, const std::optional<chip_id_t> receiver_device_id, const std::optional<chip_id_t> sender_device_id, all_gather_op::Topology topology) {
    TT_FATAL(!(receiver_device_id == std::nullopt && sender_device_id == std::nullopt), "At least one of receiver_device_id or sender_device_id must be specified");

    bool is_linear = topology == all_gather_op::Topology::Linear;
    std::unique_ptr<CclOpTensorConfig> input_tensor_config = CclOpTensorConfig::build_all_gather_tensor_config(input_tensor);
    std::unique_ptr<CclOpTensorConfig> output_tensor_config = CclOpTensorConfig::build_all_gather_tensor_config(output_tensor);

    tt_metal::Program program{};
    const auto& device = input_tensor.device();
    auto const& all_gather_config = AllGatherConfig(input_tensor, output_tensor, dim, ring_size, num_links, topology);
    auto const& topology_config = ccl::RingTopology(device, topology, sender_device_id, receiver_device_id, num_links, ring_size, ring_index);

    auto const& sharding_info = ShardedAllGatherConfig(input_tensor, output_tensor, dim);
    bool enable_print = false; // ring_index == 0
    all_gather_config.print();

    bool is_sharded = input_tensor.is_sharded();

    TT_FATAL(input_tensor.buffer()->page_size() <= all_gather_config.get_eth_buffer_size(), "Page size too large");

    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();

    int32_t shard_size_in_bytes = is_sharded ?
        (input_tensor_config->get_page_size() * input_buffer->shard_spec().tensor2d_shape[0] * input_buffer->shard_spec().tensor2d_shape[1]) / input_tensor.shard_spec()->num_cores() :
        -1;
    uint32_t input_page_size = is_sharded ? shard_size_in_bytes : input_tensor_config->get_page_size();
    uint32_t output_page_size = is_sharded ? shard_size_in_bytes : output_tensor_config->get_page_size();
    if (is_sharded) {
        log_trace(tt::LogOp, "input_buffer->page_size: {}", input_tensor_config->get_page_size());
        log_trace(tt::LogOp, "input_buffer->shard_spec().tensor2d_shape[0]: {}", input_buffer->shard_spec().tensor2d_shape[0]);
        log_trace(tt::LogOp, "input_buffer->shard_spec().tensor2d_shape[1]: {}", input_buffer->shard_spec().tensor2d_shape[1]);
    }
    const uint32_t max_buffer_per_chunk = is_sharded ?
        round_down(all_gather_config.get_eth_buffer_size(), shard_size_in_bytes):
        round_down(all_gather_config.get_eth_buffer_size(), input_tensor_config->get_page_size());
    const uint32_t max_pages_per_chunk = is_sharded ?
        max_buffer_per_chunk / shard_size_in_bytes :
        max_buffer_per_chunk / input_page_size;
    log_trace(tt::LogOp, "shard_size_in_bytes: {}", shard_size_in_bytes);
    log_trace(tt::LogOp, "input_page_size: {}", input_page_size);
    log_trace(tt::LogOp, "max_buffer_per_chunk: {}", max_buffer_per_chunk);
    log_trace(tt::LogOp, "max_pages_per_chunk: {}", max_pages_per_chunk);
    bool rm = input_tensor.get_layout() == Layout::ROW_MAJOR;
    bool width = input_tensor.get_legacy_shape().rank() - 1 == dim;
    DataFormat df = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    uint32_t global_num_workers = all_gather_config.get_num_eth_buffers_per_edm() * num_links;

    std::map<string, string> worker_defines;
    if (rm) {
        worker_defines["RM_INTERLEAVED"] = "1";
    } else {
        worker_defines["TILE_INTERLEAVED"] = "1";
    }

    // number of worker cores is 2x this since there is 1 worker for the sender buffer and 1 worker for the receiver buffer
    uint32_t total_worker_core_pairs_used = num_links * all_gather_config.get_num_eth_buffers_per_edm();

    std::vector<KernelHandle> worker_reader_sender_kernels;
    worker_reader_sender_kernels.reserve(total_worker_core_pairs_used);
    std::vector<KernelHandle> worker_writer_sender_kernels;
    worker_writer_sender_kernels.reserve(total_worker_core_pairs_used);

    std::vector<KernelHandle> worker_reader_receiver_kernels;
    worker_reader_receiver_kernels.reserve(total_worker_core_pairs_used);
    std::vector<KernelHandle> worker_writer_receiver_kernels;
    worker_writer_receiver_kernels.reserve(total_worker_core_pairs_used);

    std::vector<CoreCoord> all_worker_sender_cores;
    all_worker_sender_cores.reserve(total_worker_core_pairs_used);
    std::vector<CoreCoord> all_worker_receiver_cores;
    all_worker_receiver_cores.reserve(total_worker_core_pairs_used);


    uint32_t num_input_pages = input_tensor.buffer()->size() / input_page_size;
    uint32_t min_pages_per_link = num_input_pages / num_links;



    const uint32_t num_full_send_directions = topology == all_gather_op::Topology::Linear ? 2 : 1;
    constexpr uint32_t max_num_full_send_directions = 2;

    std::vector<EriscDatamoverBuilder> clockwise_edm_builders;
    std::vector<EriscDatamoverBuilder> counter_clockwise_edm_builders;
    TT_ASSERT(num_full_send_directions > 0);

    // TODO: move to all-gather config
    auto edm_sem_addrs_per_link = std::vector<std::vector<uint32_t>>(num_links);
    auto edm_buffer_addrs_per_link = std::vector<std::vector<uint32_t>>(num_links);
    for (uint32_t link = 0; link < num_links; link++) {
        edm_sem_addrs_per_link.at(link).reserve(all_gather_config.get_num_eth_buffers_per_edm());
        edm_buffer_addrs_per_link.at(link).reserve(all_gather_config.get_num_eth_buffers_per_edm());

        uint32_t edm_sem_addr = all_gather_config.get_eth_sems_l1_base_byte_address();
        uint32_t edm_buffer_addr = all_gather_config.get_eth_buffers_l1_base_byte_address();
        for (uint32_t direction = 0; direction < num_full_send_directions; direction++) {
            for (uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
                edm_sem_addrs_per_link.at(link).push_back(edm_sem_addr);
                edm_sem_addr += all_gather_config.get_semaphore_size();
                edm_buffer_addrs_per_link.at(link).push_back(edm_buffer_addr);
                edm_buffer_addr += all_gather_config.get_eth_buffer_size();
                TT_ASSERT((direction == 0 && b == 0) || (edm_buffer_addrs_per_link.at(link).back() != edm_buffer_addrs_per_link.at(link).front()));
                TT_ASSERT((direction == 0 && b == 0) || (edm_sem_addrs_per_link.at(link).back() != edm_sem_addrs_per_link.at(link).front()));
            }
        }

        clockwise_edm_builders.emplace_back(
            all_gather_config.get_eth_buffer_size(), all_gather_config.get_erisc_handshake_address(), edm_sem_addrs_per_link.at(link), edm_buffer_addrs_per_link.at(link), ccl::EriscDataMoverBufferSharingMode::NOT_SHARED, ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED);
        counter_clockwise_edm_builders.emplace_back(
            all_gather_config.get_eth_buffer_size(), all_gather_config.get_erisc_handshake_address(), edm_sem_addrs_per_link.at(link), edm_buffer_addrs_per_link.at(link), ccl::EriscDataMoverBufferSharingMode::NOT_SHARED, ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED);
    }

    for (uint32_t direction = 0; direction < num_full_send_directions; direction++) {
        // if we're in ring topology, we'll always need to transfer all ring indices (except the last one)
        // but if we are implementing a line topology, the number of transfers will depend on whether we
        // are setting up the forward/clockwise direction or the backward/counter-clockwise direction and also
        // how far we are from the first/last chip, depending on whether we are in forward or  direction
        const uint32_t sender_num_transfers = topology == all_gather_op::Topology::Linear ?
            (direction == 0 ? ring_index + 1: ring_size - ring_index):
            ring_size - 1;
        const uint32_t receiver_num_transfers = topology == all_gather_op::Topology::Linear ?
            sender_num_transfers - 1:
            ring_size - 1;
        std::vector<CoreCoord> eth_sender_cores;
        eth_sender_cores.reserve(num_links);
        std::vector<CoreCoord> eth_receiver_cores;
        eth_receiver_cores.reserve(num_links);
        // If linear topology, the first chip in the chain will not have a "receiver" eth core (or more correctly,
        // it doesn't have an input clockwise our output counter-clockwise connection)
        bool is_first_chip_in_chain = direction == 0 ? ring_index == 0 : ring_index == ring_size - 1;
        // If linear topology, the last chip in the chain will not have a "sender" eth core (or more correctly,
        // it doesn't have an output clockwise our input counter-clockwise connection)
        bool is_last_chip_in_chain = direction == 0 ? ring_index == ring_size - 1 : ring_index == 0;

        uint32_t sender_socket_idx = 0;
        uint32_t receiver_socket_idx = 0;
        if (receiver_device_id == sender_device_id) {
            if (ring_index == 0) {
                receiver_socket_idx = 1;
            } else {
                sender_socket_idx = 1;
            }
        }
        log_trace (tt::LogOp, "--------------------------------------------");
        log_trace (tt::LogOp, "ring_index: {}, ring_size: {}, direction: {}", ring_index, ring_size, direction);
        for (uint32_t l = 0; l < num_links; ++l) {
            // Get the cores for the sender and receiver worker cores
            if (!is_linear || ring_index != ring_size - 1) {
                auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id.value()).at(sender_socket_idx + l);
                eth_sender_cores.push_back(eth_sender_core);
                log_trace(tt::LogOp, "\teth_sender_core on link {}: (x={},y={})", l, eth_sender_core.x, eth_sender_core.y);
            }
            if (!is_linear || ring_index != 0) {
                auto eth_receiver_core = device->get_ethernet_sockets(sender_device_id.value()).at(receiver_socket_idx + l);
                eth_receiver_cores.push_back(eth_receiver_core);
                log_trace(tt::LogOp, "\teth_receiver_core on link {}: (x={},y={})", l, eth_receiver_core.x, eth_receiver_core.y);
            }
        }

        auto is_buffer_in_clockwise_direction = [&all_gather_config,&direction](uint32_t b) {
            TT_ASSERT(direction < max_num_full_send_directions);
            bool in_clockwise_direction = all_gather_config.is_buffer_in_clockwise_ring(b);
            return (direction == 0) ? in_clockwise_direction : !in_clockwise_direction;
        };

        std::vector<uint32_t> pages_per_link(num_links, min_pages_per_link);
        for (uint32_t i = 0; i < num_input_pages % min_pages_per_link; ++i) {
            pages_per_link.at(i)++;
        }

        auto tensor_slicer = ccl::InterleavedRingAllGatherTensorSlicer (
            input_tensor,
            output_tensor,
            dim,
            ring_index
        );

        ///
        /// (counter clockwise sender) < ----- (this chip) < ----- (counter-clockwise receiver)
        ///
        /// (clockwise receiver)       ------> (this chip) ------> (clockwise sender)
        /// So clockwise sender and counter-clockwise receiver are on the same core
        //  and counter-clockwise sender and clockwise receiver are on the same corwe

        for (uint32_t i = 0; i < num_links; ++i) {
            // We can't have overlap between the mcast grid for worker cores for different links since mcasting the semaphore in receiver would corrupt other link semaphores
            // We can have overlap between a link's sender and receiver worker grids if we have the semaphores at different addresses
            auto const& [receiver_workers, sender_workers] = select_worker_cores(all_gather_config, num_links, i, direction);
            uint32_t worker_index = 0;
            uint32_t workers_per_link = all_gather_config.get_num_workers_per_link() / all_gather_config.get_num_eth_buffers_per_edm();

            // Circular Buffer Setup
            uint32_t cb_page_size = is_sharded ? shard_size_in_bytes : input_page_size;
            log_trace(tt::LogOp, "input_page_size: {}", input_page_size);
            uint32_t cb_num_pages = 2 * max_pages_per_chunk;
            log_trace(tt::LogOp, "cb_num_pages: {}", cb_num_pages);
            uint32_t src0_cb_index = CB::c_in0;
            tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(cb_num_pages * cb_page_size, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, cb_page_size);
            CBHandle cb_src0_sender_workers = CreateCircularBuffer(program, sender_workers, cb_src0_config);
            CBHandle cb_src0_receiver_workers = CreateCircularBuffer(program, receiver_workers, cb_src0_config);

            // This semaphore is used by the receiver core to tell workers that data is available to read
            auto receiver_worker_semaphore_addr = tt_metal::CreateSemaphore(program, receiver_workers, 0);
            // This semaphore is used by the receiver core to tell the worker sender writer that sender buffer is available to write to
            auto sender_worker_writer_semaphore_addr = tt_metal::CreateSemaphore(program, sender_workers, 0);
            // This semaphore is used by the worker receiver writer to tell the worker sender reader that data has been committed to memory
            // This is currently a running counter of how many chunks were committed since the sender worker never decrements this buffer
            // Potentially avoid overflow by having it actually decrement (using noc atomic inc with value of -1)
            auto sender_worker_reader_semaphore_addr = tt_metal::CreateSemaphore(program, sender_workers, 0);

            // Rename this the _channel
            std::vector<uint32_t> pages_per_buffer;

            // number of pages that can fit in a single ethernet L1 buffer (not the number of pages sent to this channel)
            std::vector<uint32_t> pages_per_eth_l1_buffer;
            pages_per_buffer.reserve(all_gather_config.get_num_eth_buffers_per_edm());
            uint32_t max_pages_per_eth_l1_sender_buffer = all_gather_config.get_eth_buffer_size() / input_page_size;
            for(uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
                pages_per_buffer.push_back((pages_per_link.at(i) / all_gather_config.get_num_eth_buffers_per_edm()));
                pages_per_eth_l1_buffer.push_back(
                    is_sharded ? std::min(pages_per_buffer.back(), max_pages_per_eth_l1_sender_buffer)
                            : max_pages_per_eth_l1_sender_buffer);
                if (b < pages_per_link.at(i) % all_gather_config.get_num_eth_buffers_per_edm()) {
                    pages_per_buffer.back()++;
                }

                log_trace(tt::LogOp, "pages_per_link[{}]: {}", i, pages_per_link.at(i));
                log_trace(tt::LogOp, "pages_per_buffer[{}]: {}", b, pages_per_buffer.at(b));
                log_trace(tt::LogOp, "max_pages_per_eth_l1_sender_buffer: {}",max_pages_per_eth_l1_sender_buffer);
            }
            TT_ASSERT(std::accumulate(pages_per_buffer.begin(), pages_per_buffer.end(), 0) == pages_per_link.at(i));

            uint32_t bytes_per_chunk = 0, pages_per_chunk = 0, num_full_chunks = 0, rem_bytes = 0, rem_pages = 0;
            uint32_t link_size_bytes = pages_per_link.at(i) * input_page_size;
            if (pages_per_link.at(i) >= max_pages_per_chunk) {
                bytes_per_chunk = max_buffer_per_chunk;
                pages_per_chunk = max_pages_per_chunk;
                TT_ASSERT(max_buffer_per_chunk == max_pages_per_chunk * input_page_size);
                num_full_chunks = link_size_bytes / bytes_per_chunk;
                rem_bytes = link_size_bytes % bytes_per_chunk;
                rem_pages = pages_per_link.at(i) % max_pages_per_chunk;
            } else {
                rem_bytes = link_size_bytes;
                rem_pages = pages_per_link.at(i);
            }

            auto sender_worker_cores = corerange_to_cores(sender_workers, std::nullopt, true);
            auto receiver_worker_cores = corerange_to_cores(receiver_workers, std::nullopt, true);
            all_worker_sender_cores.insert(all_worker_sender_cores.end(), sender_worker_cores.begin(), sender_worker_cores.end());
            all_worker_receiver_cores.insert(all_worker_receiver_cores.end(), receiver_worker_cores.begin(), receiver_worker_cores.end());

            TT_ASSERT(rem_pages < pages_per_chunk || num_full_chunks == 0);
            TT_ASSERT(rem_pages <= max_pages_per_chunk);
            std::vector<uint32_t> num_full_chunks_per_worker(all_gather_config.get_num_eth_buffers_per_edm(), num_full_chunks / all_gather_config.get_num_eth_buffers_per_edm());
            std::vector<bool> is_channel_shrinkable(all_gather_config.get_num_eth_buffers_per_edm(), false);
            std::vector<uint32_t> largest_packets_per_channel(all_gather_config.get_num_eth_buffers_per_edm(), 0);
            std::vector<uint32_t> rem_pages_per_worker(all_gather_config.get_num_eth_buffers_per_edm(), 0);
            {
                uint32_t worker_idx = 0;
                for (worker_idx = 0; worker_idx < num_full_chunks % all_gather_config.get_num_eth_buffers_per_edm(); ++worker_idx) {
                    num_full_chunks_per_worker.at(worker_idx)++;
                }
                if (rem_pages != 0) {
                    rem_pages_per_worker.at(worker_idx % all_gather_config.get_num_eth_buffers_per_edm()) = rem_pages;
                    TT_ASSERT(rem_pages_per_worker.at(worker_idx % all_gather_config.get_num_eth_buffers_per_edm()) * 2 <= cb_num_pages);
                }
            }

            std::vector<uint32_t> clockwise_link_buffer_num_messages_to_send;
            std::vector<uint32_t> counter_clockwise_link_buffer_num_messages_to_send;
            std::vector<uint32_t> edm_semaphores_base_address;
            std::vector<uint32_t> link_buffer_sender_addresses;
            clockwise_link_buffer_num_messages_to_send.reserve(all_gather_config.get_num_eth_buffers_per_edm());
            counter_clockwise_link_buffer_num_messages_to_send.reserve(all_gather_config.get_num_eth_buffers_per_edm());
            edm_semaphores_base_address.reserve(all_gather_config.get_num_eth_buffers_per_edm());
            link_buffer_sender_addresses.reserve(all_gather_config.get_num_eth_buffers_per_edm());
            if (is_sharded) {
                for(uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
                    auto input_tensor_shard_arg_generator = InputTensorShardAddrGenArgGenerator(
                                    device,
                                    dynamic_cast<CclOpShardedTensorConfig*>(input_tensor_config.get()),
                                    ring_index,
                                    ring_size,
                                    global_num_workers,
                                    b + i * all_gather_config.get_num_workers_per_link(),
                                    0,
                                    0,
                                    // We want the input tensor to always be read in forward shard order so we
                                    // always tell it we are in counter-clockwise direction (forward read order)
                                    false
                                );
                    uint32_t max_shards_per_eth_buffer = std::min<uint32_t>(all_gather_config.get_eth_buffer_size() / input_tensor_shard_arg_generator.args_struct.shard_size_in_bytes, input_tensor_shard_arg_generator.args_struct.num_dest_cores);
                    TT_ASSERT(max_shards_per_eth_buffer > 0, "Codepath needs further generalization to support computing multiple sends per shard. Shard size: {}", input_tensor_shard_arg_generator.args_struct.shard_size_in_bytes);
                    log_info(tt::LogOp, "max_shards_per_eth_buffer: {}", max_shards_per_eth_buffer);
                    num_full_chunks_per_worker.at(b) = input_tensor_shard_arg_generator.args_struct.num_dest_cores < max_shards_per_eth_buffer ? 1 : input_tensor_shard_arg_generator.args_struct.num_dest_cores / max_shards_per_eth_buffer;
                    rem_pages_per_worker.at(b) = max_shards_per_eth_buffer > input_tensor_shard_arg_generator.args_struct.num_dest_cores ? 0 : input_tensor_shard_arg_generator.args_struct.num_dest_cores - (num_full_chunks_per_worker.at(b) * max_shards_per_eth_buffer);
                    TT_ASSERT(rem_pages_per_worker.at(b) == 0 || input_tensor_shard_arg_generator.args_struct.num_dest_cores >= num_full_chunks_per_worker.at(b) * max_shards_per_eth_buffer);
                    TT_ASSERT(input_tensor_shard_arg_generator.args_struct.num_dest_cores == rem_pages_per_worker.at(b) + num_full_chunks_per_worker.at(b) * max_shards_per_eth_buffer);

                    uint32_t full_chunk_size_bytes = max_shards_per_eth_buffer * input_tensor_shard_arg_generator.args_struct.shard_size_in_bytes;
                    bool shrinkable = num_full_chunks_per_worker.at(b) == 1 && all_gather_config.get_eth_buffer_size() > full_chunk_size_bytes;
                    is_channel_shrinkable.at(b) = shrinkable;
                    largest_packets_per_channel.at(b) = shrinkable ? full_chunk_size_bytes : all_gather_config.get_eth_buffer_size();
                }
            } else {
                for(uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
                    bool shrinkable = num_full_chunks_per_worker.at(b) == 0;
                    is_channel_shrinkable.at(b) = shrinkable;
                    largest_packets_per_channel.at(b) = shrinkable ? rem_pages_per_worker.at(b) * input_page_size : all_gather_config.get_eth_buffer_size();
                }
            }
            for(uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
                // link num messages
                clockwise_link_buffer_num_messages_to_send.push_back(
                    (num_full_chunks_per_worker.at(b) + (rem_pages_per_worker.at(b) > 0 ? 1 : 0)) *
                    sender_num_transfers);
                counter_clockwise_link_buffer_num_messages_to_send.push_back(
                    (num_full_chunks_per_worker.at(b) + (rem_pages_per_worker.at(b) > 0 ? 1 : 0)) *
                    receiver_num_transfers);
            }
            for(uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
                log_trace(tt::LogOp, "rem_pages_per_worker[{}]: {}", b, rem_pages_per_worker.at(b));
                log_trace(tt::LogOp, "num_full_chunks_per_worker[{}]: {}", b, num_full_chunks_per_worker.at(b));
                log_trace(tt::LogOp, "clockwise_link_buffer_num_messages_to_send[{}]: {}", b, clockwise_link_buffer_num_messages_to_send.at(b));
                log_trace(tt::LogOp, "counter_clockwise_link_buffer_num_messages_to_send[{}]: {}", b, counter_clockwise_link_buffer_num_messages_to_send.at(b));
            }

            std::vector<uint32_t> receiver_semaphores_base_address;
            std::vector<uint32_t> link_buffer_receiver_addresses;
            receiver_semaphores_base_address.reserve(all_gather_config.get_num_eth_buffers_per_edm());
            link_buffer_receiver_addresses.reserve(all_gather_config.get_num_eth_buffers_per_edm());
            for(uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
                receiver_semaphores_base_address.push_back(all_gather_config.get_eth_sems_l1_base_byte_address() + b * all_gather_config.get_semaphore_size());
                link_buffer_receiver_addresses.push_back(all_gather_config.get_eth_buffers_l1_base_byte_address() + b * all_gather_config.get_eth_buffer_size());
            }

            std::vector<uint32_t> sender_eth_sem_addrs; sender_eth_sem_addrs.reserve(all_gather_config.get_num_eth_buffers_per_edm());
            std::vector<uint32_t> sender_eth_buffer_addrs; sender_eth_buffer_addrs.reserve(all_gather_config.get_num_eth_buffers_per_edm());
            std::vector<uint32_t> receiver_eth_sem_addrs; receiver_eth_sem_addrs.reserve(all_gather_config.get_num_eth_buffers_per_edm());
            std::vector<uint32_t> receiver_eth_buffer_addrs; receiver_eth_buffer_addrs.reserve(all_gather_config.get_num_eth_buffers_per_edm());
            for (uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
                uint32_t num_workers_per_eth_buffer = std::min(workers_per_link, all_gather_config.get_num_eth_buffers_per_edm() - worker_index);

                std::vector<ccl::WorkerXY> sender_worker_coords;
                std::vector<ccl::WorkerXY> receiver_worker_coords;
                for (uint32_t w = b * num_workers_per_eth_buffer; w < (b + 1) * num_workers_per_eth_buffer; ++w) {
                    sender_worker_coords.push_back(
                        ccl::WorkerXY(
                            device->worker_core_from_logical_core(sender_worker_cores.at(w)).x,
                            device->worker_core_from_logical_core(sender_worker_cores.at(w)).y));
                    receiver_worker_coords.push_back(
                        ccl::WorkerXY(
                            device->worker_core_from_logical_core(receiver_worker_cores.at(w)).x,
                            device->worker_core_from_logical_core(receiver_worker_cores.at(w)).y));
                }

                bool sender_enabled = (!is_linear || !is_last_chip_in_chain);
                if (sender_enabled) {
                    auto &sender_edm_builder = is_buffer_in_clockwise_direction(b) ? clockwise_edm_builders.at(i) : counter_clockwise_edm_builders.at(i);
                    log_trace(tt::LogOp, "Adding sender EDM channel");
                    EriscDatamoverBuilder::ChannelBufferInterface const& sender_channel_buffer_info =
                        sender_edm_builder.add_sender_channel(sender_worker_writer_semaphore_addr, clockwise_link_buffer_num_messages_to_send.at(b), sender_worker_coords);
                    if (is_channel_shrinkable.at(b)) {
                        TT_ASSERT(largest_packets_per_channel.at(b) > 0);
                        log_trace(tt::LogOp, "\tsetting channel_max_size to {} for channel {}", largest_packets_per_channel.at(b), b);
                        sender_edm_builder.set_max_message_size_bytes(sender_channel_buffer_info.channel, largest_packets_per_channel.at(b));
                    }
                    sender_eth_sem_addrs.push_back(sender_channel_buffer_info.eth_semaphore_l1_address);
                    sender_eth_buffer_addrs.push_back(sender_channel_buffer_info.eth_buffer_l1_address);
                }

                bool receiver_enabled = (!is_linear || !is_first_chip_in_chain);
                if (receiver_enabled) {
                    auto &receiver_edm_builder = is_buffer_in_clockwise_direction(b) ? counter_clockwise_edm_builders.at(i) : clockwise_edm_builders.at(i);
                    log_trace(tt::LogOp, "Adding receiver EDM channel");
                    EriscDatamoverBuilder::ChannelBufferInterface const& receiver_channel_buffer_info =
                        receiver_edm_builder.add_receiver_channel(receiver_worker_semaphore_addr, counter_clockwise_link_buffer_num_messages_to_send.at(b), receiver_worker_coords);
                    if (is_channel_shrinkable.at(b)) {
                        TT_ASSERT(largest_packets_per_channel.at(b) > 0);
                        log_trace(tt::LogOp, "\tsetting channel_max_size to {} for channel {}", largest_packets_per_channel.at(b), b);
                        receiver_edm_builder.set_max_message_size_bytes(receiver_channel_buffer_info.channel, largest_packets_per_channel.at(b));
                    }
                    receiver_eth_sem_addrs.push_back(receiver_channel_buffer_info.eth_semaphore_l1_address);
                    receiver_eth_buffer_addrs.push_back(receiver_channel_buffer_info.eth_buffer_l1_address);
                }
            }


            // 1 Worker per buffer
            for (uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
                uint32_t global_worker_index = all_gather_config.get_num_eth_buffers_per_edm() * i + b;

                bool is_clockwise_direction = is_buffer_in_clockwise_direction(b);

                // Not fully sure about these two
                uint32_t last_output_page_offset = (ring_size - 1) * tensor_slicer.output_page_offset;
                uint32_t last_output_addr_offset = (ring_size - 1) * tensor_slicer.output_addr_offset;

                log_trace(tt::LogOp,"\tlast_output_page_offset={}", last_output_page_offset);
                log_trace(tt::LogOp,"\tlast_output_addr_offset={}", last_output_addr_offset);

                if (!is_linear || !is_last_chip_in_chain) {
                    //// Send Reader
                    auto build_worker_send_reader_ct_args = [&]() {
                        if (is_sharded) {
                            // # Send Reader (CT)
                            // 1) Shard Type
                            // 2) num_transfers
                            std::vector<uint32_t> worker_reader_sender_ct_args = {
                                static_cast<uint32_t>(sharding_info.get_shard_type()),
                                static_cast<uint32_t>(sender_num_transfers)
                            };
                            log_trace(tt::LogOp, "----worker_reader_sender_ct_args size={}", worker_reader_sender_ct_args.size());
                            log_trace(tt::LogOp, "\tsharding_info.get_shard_type(): {}", sharding_info.get_shard_type());
                            log_trace(tt::LogOp, "\tnum_transfers: {}", sender_num_transfers);

                            return worker_reader_sender_ct_args;
                        } else {
                            std::vector<uint32_t> worker_reader_sender_ct_args = {
                                static_cast<uint32_t>(all_gather_config.is_input_dram()),
                                static_cast<uint32_t>(all_gather_config.is_output_dram()),
                                static_cast<uint32_t>(sender_num_transfers),
                                static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                                static_cast<uint32_t>(input_page_size),
                                static_cast<uint32_t>(output_page_size),
                                static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)),
                                static_cast<uint32_t>(rem_pages_per_worker.at(b)),
                                static_cast<uint32_t>(tensor_slicer.input_start_page_idx),
                                static_cast<uint32_t>(tensor_slicer.output_start_page_idx),
                                static_cast<uint32_t>(tensor_slicer.output_start_addr_offset),
                                static_cast<uint32_t>(tensor_slicer.row_idx),
                                static_cast<uint32_t>(tensor_slicer.col_idx),
                                static_cast<uint32_t>(tensor_slicer.row_offset),
                                static_cast<uint32_t>(tensor_slicer.col_offset),
                                static_cast<uint32_t>(tensor_slicer.num_rows),
                                static_cast<uint32_t>(tensor_slicer.num_cols),
                                static_cast<uint32_t>(last_output_page_offset),
                                static_cast<uint32_t>(tensor_slicer.output_page_offset),
                                static_cast<uint32_t>(last_output_addr_offset),
                                static_cast<uint32_t>(tensor_slicer.output_addr_offset),
                                static_cast<uint32_t>(ring_index),
                                static_cast<uint32_t>(sender_worker_reader_semaphore_addr),
                                static_cast<uint32_t>(is_clockwise_direction ? 1 : 0),
                                static_cast<uint32_t>(cb_num_pages / 2),
                                static_cast<uint32_t>(ring_size)
                            };

                            log_trace(tt::LogOp, "Worker {} SR args", b);
                            log_trace(tt::LogOp, "\tall_gather_config.is_input_dram(): {}", all_gather_config.is_input_dram());
                            log_trace(tt::LogOp, "\tall_gather_config.is_output_dram(): {}", all_gather_config.is_output_dram());
                            log_trace(tt::LogOp, "\tsender_num_transfers: {}", sender_num_transfers);
                            log_trace(tt::LogOp, "\tnum_full_chunks_per_worker.at(b): {}", num_full_chunks_per_worker.at(b));
                            log_trace(tt::LogOp, "\tinput_page_size: {}", input_page_size);
                            log_trace(tt::LogOp, "\toutput_page_size: {}", output_page_size);
                            log_trace(tt::LogOp, "\tpages_per_eth_l1_buffer.at(b): {}", pages_per_eth_l1_buffer.at(b));
                            log_trace(tt::LogOp, "\trem_pages_per_worker.at(b): {}", rem_pages_per_worker.at(b));
                            log_trace(tt::LogOp, "\tinput_start_page_idx: {}", tensor_slicer.input_start_page_idx);
                            log_trace(tt::LogOp, "\toutput_start_page_idx: {}", tensor_slicer.output_start_page_idx);
                            log_trace(tt::LogOp, "\toutput_start_addr_offset: {}", tensor_slicer.output_start_addr_offset);
                            log_trace(tt::LogOp, "\trow_idx: {}", tensor_slicer.row_idx);
                            log_trace(tt::LogOp, "\tcol_idx: {}", tensor_slicer.col_idx);
                            log_trace(tt::LogOp, "\trow_offset: {}", tensor_slicer.row_offset);
                            log_trace(tt::LogOp, "\tcol_offset: {}", tensor_slicer.col_offset);
                            log_trace(tt::LogOp, "\tnum_rows: {}", tensor_slicer.num_rows);
                            log_trace(tt::LogOp, "\tnum_cols: {}", tensor_slicer.num_cols);
                            log_trace(tt::LogOp, "\tlast_output_page_offset: {}", last_output_page_offset);
                            log_trace(tt::LogOp, "\toutput_page_offset: {}", tensor_slicer.output_page_offset);
                            log_trace(tt::LogOp, "\tlast_output_addr_offset: {}", last_output_addr_offset);
                            log_trace(tt::LogOp, "\toutput_addr_offset: {}", tensor_slicer.output_addr_offset);
                            log_trace(tt::LogOp, "\tring_index: {}", ring_index);
                            log_trace(tt::LogOp, "\tsender_worker_reader_semaphore_addr: {}", sender_worker_reader_semaphore_addr);
                            log_trace(tt::LogOp, "\tis_clockwise_direction: {}", is_clockwise_direction ? 1 : 0);

                            return worker_reader_sender_ct_args;
                        }
                    };

                    std::vector<uint32_t> const& worker_send_reader_ct_args = build_worker_send_reader_ct_args();

                    auto build_worker_send_reader_rt_args = [&]() {
                        bool is_clockwise = is_buffer_in_clockwise_direction(b);
                        if (is_sharded) {
                            // # Send Reader (RT)
                            // 1) local semaphore address (same as before)
                            // 2) input tensor shard reader
                            // 3) output tensor shard reader
                            auto curr_link = i;

                            TT_ASSERT(all_gather_config.get_num_eth_buffers_per_edm() == 1 || all_gather_config.get_num_eth_buffers_per_edm() == 2 || all_gather_config.get_num_eth_buffers_per_edm() == 4 || all_gather_config.get_num_eth_buffers_per_edm() == 8);
                            TT_ASSERT(input_tensor.buffer() != nullptr);
                            auto input_tensor_shard_arg_generator =
                                InputTensorShardAddrGenArgGenerator(
                                    device,
                                    dynamic_cast<CclOpShardedTensorConfig*>(input_tensor_config.get()),
                                    ring_index,
                                    ring_size,
                                    global_num_workers,
                                    global_worker_index,
                                    0,
                                    0,
                                    // We want the input tensor to always be read in forward shard order so we
                                    // always tell it we are in counter-clockwise direction (forward read order)
                                    false
                                    );
                            auto const& [starting_dest_worker_index, starting_chunk_into_shard] = OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                                all_gather_config, input_tensor, output_tensor,
                                is_clockwise ?
                                    (ring_index == 0 ? ring_size - 1 : ring_index - 1) :
                                    (ring_index == ring_size - 1 ? 0 : ring_index + 1),
                                global_worker_index);

                            log_trace(tt::LogOp, "SendReader {} ring_index: {}, start dest worker index: {}, starting chunk into shard: {}", global_worker_index, ring_index, starting_dest_worker_index, starting_chunk_into_shard);
                            auto output_tensor_shard_arg_generator =
                                OutputTensorShardAddrGenArgGenerator(
                                    all_gather_config,
                                    device,
                                    dynamic_cast<CclOpShardedTensorConfig*>(input_tensor_config.get()),
                                    dynamic_cast<CclOpShardedTensorConfig*>(output_tensor_config.get()),
                                    ring_index,
                                    ring_size,
                                    global_num_workers,
                                    global_worker_index,
                                    starting_dest_worker_index,
                                    starting_chunk_into_shard,
                                    is_clockwise);

                            auto const& input_shard_addr_generator_args = input_tensor_shard_arg_generator.generate();
                            auto const& output_shard_addr_generator_args = output_tensor_shard_arg_generator.generate();
                            std::vector<uint32_t> worker_send_reader_rt_args;
                            worker_send_reader_rt_args.reserve(2 + input_shard_addr_generator_args.size() + output_shard_addr_generator_args.size());
                            worker_send_reader_rt_args.push_back(sender_worker_reader_semaphore_addr);
                            worker_send_reader_rt_args.push_back(pages_per_buffer.at(b));
                            worker_send_reader_rt_args.push_back(pages_per_eth_l1_buffer.at(b));
                            worker_send_reader_rt_args.push_back(cb_num_pages / 2);
                            std::copy(input_shard_addr_generator_args.begin(), input_shard_addr_generator_args.end(), std::back_inserter(worker_send_reader_rt_args));
                            std::copy(output_shard_addr_generator_args.begin(), output_shard_addr_generator_args.end(), std::back_inserter(worker_send_reader_rt_args));

                            log_trace(tt::LogOp, "---worker_send_reader_rt_args.size()={}-----", worker_send_reader_rt_args.size());
                            log_trace(tt::LogOp, "\tsender_worker_reader_semaphore_addr: {}", sender_worker_reader_semaphore_addr);
                            log_trace(tt::LogOp, "\tinput_shard_addr_generator_args:");
                            input_tensor_shard_arg_generator.dump_to_log();
                            log_trace(tt::LogOp, "\toutput_tensor_shard_arg_generator:");
                            output_tensor_shard_arg_generator.dump_to_log();

                            return worker_send_reader_rt_args;
                        } else {
                            std::vector<uint32_t> args = {
                                static_cast<uint32_t>(input_buffer->address()),
                                static_cast<uint32_t>(output_buffer->address())
                            };
                            return args;
                        }
                    };
                    std::vector<uint32_t> const& worker_send_reader_rt_args = build_worker_send_reader_rt_args();

                    std::string const& send_reader_kernel_path = is_sharded ?
                        "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_ring_gather_send_reader.cpp" :
                        "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_send_reader.cpp";
                    KernelHandle worker_reader_sender_kernel_id = tt_metal::CreateKernel(
                        program,
                        send_reader_kernel_path,
                        sender_worker_cores.at(b),
                        tt_metal::ReaderDataMovementConfig(worker_send_reader_ct_args, worker_defines));

                    worker_reader_sender_kernels.push_back(worker_reader_sender_kernel_id);

                    tt_metal::SetRuntimeArgs(
                        program,
                        worker_reader_sender_kernel_id,
                        sender_worker_cores.at(b),
                        worker_send_reader_rt_args);


                    //// Send Writer
                    auto build_worker_sender_writer_ct_args = [&]() {
                        if (is_sharded) {
                            std::vector<uint32_t> worker_sender_writer_ct_args = {
                                static_cast<uint32_t>(sharding_info.get_shard_type())
                            };
                            log_trace(tt::LogOp, "----worker_sender_writer_ct_args size={}", worker_sender_writer_ct_args.size());
                            log_trace(tt::LogOp, "\tsharding_info.get_shard_type(): {}", sharding_info.get_shard_type());

                            return worker_sender_writer_ct_args;
                        } else {
                            CoreCoord const& worker_eth_sender_core = is_clockwise_direction ? eth_sender_cores.at(i) : eth_receiver_cores.at(i);
                            std::vector<uint32_t> worker_writer_sender_ct_args = {
                                static_cast<uint32_t>(all_gather_config.is_output_dram()),
                                static_cast<uint32_t>(sender_num_transfers),
                                static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                                static_cast<uint32_t>(input_page_size),
                                static_cast<uint32_t>(output_page_size),
                                static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)),
                                static_cast<uint32_t>(rem_pages_per_worker.at(b)),
                                static_cast<uint32_t>(tensor_slicer.input_start_page_idx),
                                static_cast<uint32_t>(tensor_slicer.output_start_page_idx),
                                static_cast<uint32_t>(tensor_slicer.output_start_addr_offset),
                                static_cast<uint32_t>(tensor_slicer.row_idx),
                                static_cast<uint32_t>(tensor_slicer.col_idx),
                                static_cast<uint32_t>(tensor_slicer.row_offset),
                                static_cast<uint32_t>(tensor_slicer.col_offset),
                                static_cast<uint32_t>(tensor_slicer.num_rows),
                                static_cast<uint32_t>(tensor_slicer.num_cols),
                                static_cast<uint32_t>(ring_index),

                                // worker local L1 address of semaphore
                                static_cast<uint32_t>(sender_worker_writer_semaphore_addr),
                                static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_sender_core).x),
                                static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_sender_core).y),
                                static_cast<uint32_t>(cb_num_pages / 2),
                            };
                            log_trace(tt::LogOp, "Worker {} SW CT args", b);
                            log_trace(tt::LogOp, "\tall_gather_config.is_output_dram(): {}", all_gather_config.is_output_dram());
                            log_trace(tt::LogOp, "\tsender_num_transfers: {}", sender_num_transfers);
                            log_trace(tt::LogOp, "\tnum_full_chunks_per_worker: {}", num_full_chunks_per_worker.at(b));
                            log_trace(tt::LogOp, "\tinput_page_size: {}", input_page_size);
                            log_trace(tt::LogOp, "\toutput_page_size: {}", output_page_size);
                            log_trace(tt::LogOp, "\tpages_per_eth_l1_buffer: {}", pages_per_eth_l1_buffer.at(b));
                            log_trace(tt::LogOp, "\trem_pages_per_worker: {}", rem_pages_per_worker.at(b));
                            log_trace(tt::LogOp, "\tinput_start_page_idx: {}", tensor_slicer.input_start_page_idx);
                            log_trace(tt::LogOp, "\toutput_start_page_idx: {}", tensor_slicer.output_start_page_idx);
                            log_trace(tt::LogOp, "\toutput_start_addr_offset: {}", tensor_slicer.output_start_addr_offset);
                            log_trace(tt::LogOp, "\trow_idx: {}", tensor_slicer.row_idx);
                            log_trace(tt::LogOp, "\tcol_idx: {}", tensor_slicer.col_idx);
                            log_trace(tt::LogOp, "\trow_offset: {}", tensor_slicer.row_offset);
                            log_trace(tt::LogOp, "\tcol_offset: {}", tensor_slicer.col_offset);
                            log_trace(tt::LogOp, "\tnum_rows: {}", tensor_slicer.num_rows);
                            log_trace(tt::LogOp, "\tnum_cols: {}", tensor_slicer.num_cols);
                            log_trace(tt::LogOp, "\tring_index: {}", ring_index);
                            log_trace(tt::LogOp, "\tsender_worker_writer_semaphore_addr: {}", sender_worker_writer_semaphore_addr);
                            log_trace(tt::LogOp, "\tethernet_core_x: {}", device->ethernet_core_from_logical_core(worker_eth_sender_core).x);
                            log_trace(tt::LogOp, "\tethernet_core_y: {}", device->ethernet_core_from_logical_core(worker_eth_sender_core).y);
                            log_trace(tt::LogOp, "\thalf_cb_num_pages: {}", cb_num_pages / 2);

                            return worker_writer_sender_ct_args;
                        }
                    };

                    std::vector<uint32_t> const& worker_sender_writer_ct_args = build_worker_sender_writer_ct_args();

                    auto build_worker_sender_writer_rt_args = [&]() {
                        if (is_sharded) {
                            // Send Writer Args (RT)
                            // 1) eth_sender_l1_base_addr
                            // 2) eth_sender_l1_sem_addr
                            // 3) eth_sender_noc_x
                            // 4) eth_sender_noc_y
                            // 5) writer_send_sem_addr
                            // 6) num_transfers
                            // 7)

                            bool is_clockwise = is_buffer_in_clockwise_direction(b);
                            auto const& [starting_dest_worker_index, starting_chunk_into_shard] = OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                                all_gather_config,
                                input_tensor,
                                output_tensor,
                                // this writes the input tensor to the first output location
                                ring_index,
                                global_worker_index);
                            auto input_tensor_shard_arg_generator = InputTensorShardAddrGenArgGenerator(
                                    device,
                                    dynamic_cast<CclOpShardedTensorConfig*>(input_tensor_config.get()),
                                    ring_index,
                                    ring_size,
                                    global_num_workers,
                                    global_worker_index,
                                    0,
                                    0,
                                    // We want the input tensor to always be read in forward shard order so we
                                    // always tell it we are in counter-clockwise direction (forward read order)
                                    false
                                );
                            auto output_tensor_shard_arg_generator =
                                OutputTensorShardAddrGenArgGenerator(
                                    all_gather_config,
                                    device,
                                    dynamic_cast<CclOpShardedTensorConfig*>(input_tensor_config.get()),
                                    dynamic_cast<CclOpShardedTensorConfig*>(output_tensor_config.get()),
                                    ring_index,
                                    ring_size,
                                    global_num_workers,
                                    global_worker_index,
                                    starting_dest_worker_index,
                                    starting_chunk_into_shard,
                                    is_buffer_in_clockwise_direction(b)
                                );
                            auto const& output_tensor_shard_addr_gen_args = output_tensor_shard_arg_generator.generate();

                            CoreCoord const& worker_eth_sender_core = is_clockwise_direction ? eth_sender_cores.at(i) : eth_receiver_cores.at(i);
                            std::vector<uint32_t> worker_writer_sender_rt_args = {
                                static_cast<uint32_t>(sender_eth_buffer_addrs.at(b)), // eth_sender_l1_base_addr
                                static_cast<uint32_t>(sender_eth_sem_addrs.at(b)), // eth_sender_l1_sem_addr
                                static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_sender_core).x), // eth_sender_noc_x
                                static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_sender_core).y), // eth_sender_noc_y
                                static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)), //output_tensor_shard_arg_generator.args_struct.num_dest_cores),//pages_per_eth_l1_buffer.at(b)),
                                static_cast<uint32_t>(sender_worker_writer_semaphore_addr), // writer_send_sem_addr
                                static_cast<uint32_t>(sender_num_transfers),
                                static_cast<uint32_t>(input_tensor_shard_arg_generator.args_struct.num_dest_cores),
                                static_cast<uint32_t>(cb_num_pages / 2),
                            };
                            std::copy(output_tensor_shard_addr_gen_args.begin(), output_tensor_shard_addr_gen_args.end(), std::back_inserter(worker_writer_sender_rt_args));

                            // Currently the kernel assumes we don't need to break up the initial local tensor send to EDM into multiple
                            // chunks

                            log_trace(tt::LogOp, "----worker_writer_sender_rt_args size={}", worker_writer_sender_rt_args.size());
                            log_trace(tt::LogOp, "\teth_sender_l1_base_addr: {}", sender_eth_buffer_addrs.at(b));
                            log_trace(tt::LogOp, "\teth_sender_l1_sem_addr: {}", sender_eth_sem_addrs.at(b));
                            log_trace(tt::LogOp, "\teth_sender_noc_x: {}", device->ethernet_core_from_logical_core(worker_eth_sender_core).x);
                            log_trace(tt::LogOp, "\teth_sender_noc_y: {}", device->ethernet_core_from_logical_core(worker_eth_sender_core).y);
                            log_trace(tt::LogOp, "\tpages_per_eth_l1_buffer: {}", pages_per_eth_l1_buffer.at(b));
                            log_trace(tt::LogOp, "\twriter_send_sem_addr: {}", sender_worker_writer_semaphore_addr);
                            log_trace(tt::LogOp, "\tnum_transfers: {}", sender_num_transfers);
                            output_tensor_shard_arg_generator.dump_to_log();

                            return worker_writer_sender_rt_args;
                        } else {
                            std::vector<uint32_t> worker_writer_sender_rt_args = {
                                static_cast<uint32_t>(output_buffer->address()),
                                static_cast<uint32_t>(sender_eth_buffer_addrs.at(b)),
                                static_cast<uint32_t>(sender_eth_sem_addrs.at(b))
                            };
                            log_trace(tt::LogOp, "Worker {} SW rt args", b);
                            log_trace(tt::LogOp, "\toutput_buffer->address(): {}", output_buffer->address());
                            log_trace(tt::LogOp, "\tsender_eth_buffer_addrs: {}", sender_eth_buffer_addrs.at(b));
                            log_trace(tt::LogOp, "\tsender_eth_sem_addrs: {}", sender_eth_sem_addrs.at(b));
                            return worker_writer_sender_rt_args;
                        }
                    };
                    std::vector<uint32_t> const& worker_sender_writer_rt_args = build_worker_sender_writer_rt_args();

                    std::string const& sender_writer_kernel_path = is_sharded ?
                        "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_ring_gather_send_writer.cpp" :
                        "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_send_writer.cpp";
                    KernelHandle worker_sender_writer_kernel_id = tt_metal::CreateKernel(
                        program,
                        sender_writer_kernel_path,
                        sender_worker_cores.at(b),
                        tt_metal::WriterDataMovementConfig(worker_sender_writer_ct_args, worker_defines));

                    worker_writer_sender_kernels.push_back(worker_sender_writer_kernel_id);

                    tt_metal::SetRuntimeArgs(
                        program,
                        worker_sender_writer_kernel_id,
                        sender_worker_cores.at(b),
                        worker_sender_writer_rt_args);

                } // (!is_linear || !is_last_chip_in_chain)
                else {
                    log_trace(tt::LogOp, "Skipping sender workers for linear chain");
                }

                // RECEIVER WORKERS
                if (!is_linear || !is_first_chip_in_chain) {
                    TT_ASSERT(!is_linear ||
                        ((is_clockwise_direction && (ring_index != 0)) || (!is_clockwise_direction && ring_index != ring_size - 1))
                    );
                    // TODO(snijjar): squash before merge
                    uint32_t receiver_ring_index = is_linear?
                        (is_clockwise_direction ? ring_index - 1 : ring_index + 1):
                        (is_clockwise_direction ?
                            (ring_index == 0 ? ring_size - 1 : ring_index - 1):
                            (ring_index == ring_size - 1 ? 0 : ring_index + 1));

                    uint32_t receiver_output_start_addr_offset = receiver_ring_index * tensor_slicer.output_addr_offset;

                    uint32_t receiver_output_start_page_idx = tensor_slicer.output_start_page_idx;
                    if (topology == all_gather_op::Topology::Linear) {
                        if (is_clockwise_direction) {
                            receiver_output_start_page_idx -= tensor_slicer.output_page_offset;
                        } else {
                            receiver_output_start_page_idx += tensor_slicer.output_page_offset;
                        }
                    } else {
                        if (is_clockwise_direction) {
                            bool is_wraparound_ring_index = ring_index == 0;
                            if (is_wraparound_ring_index) {
                                receiver_output_start_page_idx += last_output_page_offset;
                            } else {
                                receiver_output_start_page_idx -= tensor_slicer.output_page_offset;
                            }
                        } else {
                            // counter clockwise direction
                            bool is_wraparound_ring_index = ring_index == ring_size - 1;
                            if (is_wraparound_ring_index) {
                                receiver_output_start_page_idx -= last_output_page_offset;
                            } else {
                                receiver_output_start_page_idx += tensor_slicer.output_page_offset;
                            }
                        }
                    }
                    log_trace(tt::LogOp,"\treceiver_output_start_addr_offset={}", receiver_output_start_addr_offset);
                    log_trace(tt::LogOp,"\treceiver_output_start_page_idx={}", receiver_output_start_page_idx);
                    log_trace(tt::LogOp,"\treceiver_ring_index={}", receiver_ring_index);

                    //// Receive Reader
                    auto build_worker_receiver_reader_ct_args = [&]() {
                        if (is_sharded) {
                            // Receiver Reader Args (CT)
                            std::vector<uint32_t> worker_receiver_reader_ct_args = {
                                static_cast<uint32_t>(sharding_info.get_shard_type())
                            };
                            log_trace(tt::LogOp, "----worker_receiver_reader_ct_args size={}", worker_receiver_reader_ct_args.size());
                            log_trace(tt::LogOp, "\tsharding_info.get_shard_type(): {}", sharding_info.get_shard_type());

                            return worker_receiver_reader_ct_args;
                        } else {
                            CoreCoord const& worker_eth_receiver_core = is_clockwise_direction ? eth_receiver_cores.at(i) : eth_sender_cores.at(i);
                            std::vector<uint32_t> worker_receiver_reader_ct_args = {
                                static_cast<uint32_t>(receiver_num_transfers),
                                static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                                static_cast<uint32_t>(input_page_size),
                                static_cast<uint32_t>(pages_per_chunk),
                                static_cast<uint32_t>(rem_pages_per_worker.at(b)),
                                static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).x),
                                static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).y),
                                static_cast<uint32_t>(receiver_eth_sem_addrs.at(b)),
                                static_cast<uint32_t>(receiver_worker_semaphore_addr),
                                static_cast<uint32_t>(cb_num_pages / 2)
                            };

                            log_trace(tt::LogOp, "Worker {} RR ct args", b);
                            log_trace(tt::LogOp, "\treceiver_num_transfers: {}", receiver_num_transfers);
                            log_trace(tt::LogOp, "\tnum_full_chunks_per_worker: {}", num_full_chunks_per_worker.at(b));
                            log_trace(tt::LogOp, "\tinput_page_size: {}", input_page_size);
                            log_trace(tt::LogOp, "\tpages_per_chunk: {}", pages_per_chunk);
                            log_trace(tt::LogOp, "\trem_pages_per_worker: {}", rem_pages_per_worker.at(b));
                            log_trace(tt::LogOp, "\tethernet_core_x: {}", device->ethernet_core_from_logical_core(worker_eth_receiver_core).x);
                            log_trace(tt::LogOp, "\tethernet_core_y: {}", device->ethernet_core_from_logical_core(worker_eth_receiver_core).y);
                            log_trace(tt::LogOp, "\treceiver_eth_sem_addrs: {}", receiver_eth_sem_addrs.at(b));
                            log_trace(tt::LogOp, "\treceiver_worker_semaphore_addr: {}", receiver_worker_semaphore_addr);
                            log_trace(tt::LogOp, "\thalf_cb_num_pages : {}", cb_num_pages / 2);
                            return worker_receiver_reader_ct_args;
                        }
                    };
                    std::vector<uint32_t> const& worker_receiver_reader_ct_args = build_worker_receiver_reader_ct_args();

                    auto build_worker_receiver_reader_rt_args = [&]() {
                        if (is_sharded) {
                            // Receiver Reader Args (RT)
                            // 1) eth_receiver_noc_x
                            // 2) eth_receiver_noc_y
                            // 3) eth_receiver_l1_base_addr
                            // 4) eth_receiver_l1_semaphore_addr
                            // 5) (local) receiver_read_sem_addr
                            // 6) output tensor shard addr gen
                            auto curr_link = i;
                            bool is_clockwise = is_buffer_in_clockwise_direction(b);
                            auto const& [starting_dest_worker_index, starting_chunk_into_shard] = OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                                all_gather_config,
                                input_tensor,
                                output_tensor,
                                is_clockwise ?
                                    (ring_index == 0 ? ring_size - 1 : ring_index - 1) :
                                    (ring_index == ring_size - 1 ? 0 : ring_index + 1),
                                global_worker_index);
                            CoreCoord const& worker_eth_receiver_core = is_clockwise_direction ? eth_receiver_cores.at(i) : eth_sender_cores.at(i);
                            auto input_tensor_shard_arg_generator = InputTensorShardAddrGenArgGenerator(
                                    device,
                                    dynamic_cast<CclOpShardedTensorConfig*>(input_tensor_config.get()),
                                    ring_index,
                                    ring_size,
                                    global_num_workers,
                                    global_worker_index,
                                    0,
                                    0,
                                    // We want the input tensor to always be read in forward shard order so we
                                    // always tell it we are in counter-clockwise direction (forward read order)
                                    false
                                );
                            auto const& output_tensor_shard_addr_gen_args = input_tensor_shard_arg_generator.generate();
                            std::vector<uint32_t> worker_reader_receiver_rt_args;
                            worker_reader_receiver_rt_args.reserve(7 + output_tensor_shard_addr_gen_args.size());

                            worker_reader_receiver_rt_args.push_back(static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).x)); // eth_receiver_noc_x
                            worker_reader_receiver_rt_args.push_back(static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).y)); // eth_receiver_noc_y
                            worker_reader_receiver_rt_args.push_back(receiver_eth_buffer_addrs.at(b)); // eth_receiver_l1_base_addr
                            worker_reader_receiver_rt_args.push_back(static_cast<uint32_t>(receiver_eth_sem_addrs.at(b))); // eth_receiver_l1_semaphore_addr
                            worker_reader_receiver_rt_args.push_back(receiver_worker_semaphore_addr); // local_receiver_read_sem_addr
                            worker_reader_receiver_rt_args.push_back(pages_per_eth_l1_buffer.at(b)), //output_tensor_shard_arg_generator.args_struct.num_dest_cores), //pages_per_eth_l1_buffer.at(b)); // num_shards_per_eth_buf
                            worker_reader_receiver_rt_args.push_back(receiver_num_transfers); // local_receiver_read_sem_addr
                            worker_reader_receiver_rt_args.push_back(static_cast<uint32_t>(cb_num_pages / 2)); // local_receiver_read_sem_addr
                            std::copy(output_tensor_shard_addr_gen_args.begin(), output_tensor_shard_addr_gen_args.end(), std::back_inserter(worker_reader_receiver_rt_args));

                            log_trace(tt::LogOp, "----worker_receiver_reader_rt_args size={}", worker_reader_receiver_rt_args.size());
                            log_trace(tt::LogOp, "\teth_receiver_noc_x: {}", static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).x));
                            log_trace(tt::LogOp, "\teth_receiver_noc_y: {}", static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).y));
                            log_trace(tt::LogOp, "\teth_receiver_l1_base_addr: {}", receiver_eth_buffer_addrs.at(b));
                            log_trace(tt::LogOp, "\teth_receiver_l1_semaphore_addr: {}", static_cast<uint32_t>(receiver_eth_sem_addrs.at(b)));
                            log_trace(tt::LogOp, "\tlocal_receiver_read_sem_addr: {}", receiver_worker_semaphore_addr);
                            log_trace(tt::LogOp, "\tnum_shards_per_eth_buf: {}", pages_per_eth_l1_buffer.at(b));

                            input_tensor_shard_arg_generator.dump_to_log();

                            return worker_reader_receiver_rt_args;
                        } else {
                            std::vector<uint32_t> worker_reader_receiver_rt_args = {
                                static_cast<uint32_t>(receiver_eth_buffer_addrs.at(b))
                            };

                            log_trace(tt::LogOp, "Worker {} RR rt args", b);
                            log_trace(tt::LogOp, "\treceiver_eth_buffer_addrs: {}", receiver_eth_buffer_addrs.at(b));
                            return worker_reader_receiver_rt_args;
                        }
                    };
                    std::vector<uint32_t> worker_receiver_reader_rt_args = build_worker_receiver_reader_rt_args();

                    std::string const& receiver_reader_kernel_path = is_sharded ?
                        "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_ring_gather_receive_reader.cpp" :
                        "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_receive_reader.cpp";
                    KernelHandle worker_receiver_reader_kernel_id = tt_metal::CreateKernel(
                        program,
                        receiver_reader_kernel_path,
                        receiver_worker_cores.at(b),
                        tt_metal::ReaderDataMovementConfig(worker_receiver_reader_ct_args, worker_defines));

                    worker_reader_receiver_kernels.push_back(worker_receiver_reader_kernel_id);

                    tt_metal::SetRuntimeArgs(
                        program,
                        worker_receiver_reader_kernel_id,
                        receiver_worker_cores.at(b),
                        worker_receiver_reader_rt_args);

                    //// Receive Writer
                    auto build_worker_receive_writer_ct_args = [&]() {
                        if (is_sharded) {
                            // # Receiver Writer (CT)
                            // 1) Shard Type
                            std::vector<uint32_t> worker_receive_writer_ct_args = {
                                static_cast<uint32_t>(sharding_info.get_shard_type())
                            };
                            log_trace(tt::LogOp, "----worker_receive_writer_ct_args size={}", worker_receive_writer_ct_args.size());
                            log_trace(tt::LogOp, "\tsharding_info.get_shard_type(): {}", sharding_info.get_shard_type());

                            return worker_receive_writer_ct_args;
                        } else {
                            std::vector<uint32_t> worker_writer_receiver_ct_args = {
                                static_cast<uint32_t>(all_gather_config.is_output_dram()),
                                static_cast<uint32_t>(receiver_num_transfers),
                                static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                                static_cast<uint32_t>(input_page_size),
                                static_cast<uint32_t>(output_page_size),
                                static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)),
                                static_cast<uint32_t>(rem_pages_per_worker.at(b)),
                                static_cast<uint32_t>(receiver_output_start_page_idx),
                                static_cast<uint32_t>(receiver_output_start_addr_offset),
                                static_cast<uint32_t>(tensor_slicer.row_idx),
                                static_cast<uint32_t>(tensor_slicer.col_idx),
                                static_cast<uint32_t>(tensor_slicer.row_offset),
                                static_cast<uint32_t>(tensor_slicer.col_offset),
                                static_cast<uint32_t>(tensor_slicer.num_rows),
                                static_cast<uint32_t>(tensor_slicer.num_cols),
                                static_cast<uint32_t>(last_output_page_offset),
                                static_cast<uint32_t>(tensor_slicer.output_page_offset),
                                static_cast<uint32_t>(last_output_addr_offset),
                                static_cast<uint32_t>(tensor_slicer.output_addr_offset),
                                static_cast<uint32_t>(receiver_ring_index),
                                static_cast<uint32_t>(sender_worker_reader_semaphore_addr),
                                static_cast<uint32_t>(is_clockwise_direction ? 1 : 0),
                                static_cast<uint32_t>(cb_num_pages / 2),
                                static_cast<uint32_t>(ring_size)
                            };

                            log_trace(tt::LogOp, "Worker {} RW ct args", b);
                            log_trace(tt::LogOp, "\tall_gather_config.is_output_dram(): {}", all_gather_config.is_output_dram());
                            log_trace(tt::LogOp, "\treceiver_num_transfers: {}", receiver_num_transfers);
                            log_trace(tt::LogOp, "\tnum_full_chunks_per_worker.at(b): {}", num_full_chunks_per_worker.at(b));
                            log_trace(tt::LogOp, "\tinput_page_size: {}", input_page_size);
                            log_trace(tt::LogOp, "\toutput_page_size: {}", output_page_size);
                            log_trace(tt::LogOp, "\tpages_per_eth_l1_buffer.at(b): {}", pages_per_eth_l1_buffer.at(b));
                            log_trace(tt::LogOp, "\trem_pages_per_worker.at(b): {}", rem_pages_per_worker.at(b));
                            log_trace(tt::LogOp, "\treceiver_output_start_page_idx: {}", receiver_output_start_page_idx);
                            log_trace(tt::LogOp, "\treceiver_output_start_addr_offset: {}", receiver_output_start_addr_offset);
                            log_trace(tt::LogOp, "\trow_idx: {}", tensor_slicer.row_idx);
                            log_trace(tt::LogOp, "\tcol_idx: {}", tensor_slicer.col_idx);
                            log_trace(tt::LogOp, "\trow_offset: {}", tensor_slicer.row_offset);
                            log_trace(tt::LogOp, "\tcol_offset: {}", tensor_slicer.col_offset);
                            log_trace(tt::LogOp, "\tnum_rows: {}", tensor_slicer.num_rows);
                            log_trace(tt::LogOp, "\tnum_cols: {}", tensor_slicer.num_cols);
                            log_trace(tt::LogOp, "\tlast_output_page_offset: {}", last_output_page_offset);
                            log_trace(tt::LogOp, "\toutput_page_offset: {}", tensor_slicer.output_page_offset);
                            log_trace(tt::LogOp, "\tlast_output_addr_offset: {}", last_output_addr_offset);
                            log_trace(tt::LogOp, "\toutput_addr_offset: {}", tensor_slicer.output_addr_offset);
                            log_trace(tt::LogOp, "\treceiver_ring_index: {}", receiver_ring_index);
                            log_trace(tt::LogOp, "\tsender_worker_reader_semaphore_addr: {}", sender_worker_reader_semaphore_addr);
                            log_trace(tt::LogOp, "\tis_clockwise_direction ? 1 : 0: {}", is_clockwise_direction ? 1 : 0);
                            log_trace(tt::LogOp, "\thalf_cb_num_pages: {}", cb_num_pages / 2);
                            log_trace(tt::LogOp, "\tring_size: {}", ring_size);
                            return worker_writer_receiver_ct_args;
                        }
                    };
                    std::vector<uint32_t> const& worker_receive_writer_ct_args = build_worker_receive_writer_ct_args();

                    auto build_worker_receive_writer_rt_args = [&]() {
                        auto worker_sender_reader = device->worker_core_from_logical_core(sender_worker_cores.at(b));
                        if (is_sharded) {
                            // # Receiver Writer (RT)
                            // 1) Remote sender reader semaphore address
                            // 2) Output tensor Writer shard addr gen
                            bool is_clockwise = is_buffer_in_clockwise_direction(b);

                            auto const& [starting_dest_worker_index, starting_chunk_into_shard] = OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                                all_gather_config,
                                input_tensor,
                                output_tensor,
                                is_clockwise ?
                                    (ring_index == 0 ? ring_size - 1 : ring_index - 1) :
                                    (ring_index == ring_size - 1 ? 0 : ring_index + 1),
                                global_worker_index);
                            log_trace(tt::LogOp, "ReceiverWriter {} ring_index: {}, start dest worker index: {}, starting chunk into shard: {}", global_worker_index, ring_index, starting_dest_worker_index, starting_chunk_into_shard);
                            OutputTensorShardAddrGenArgGenerator output_tensor_shard_arg_generator(
                                all_gather_config,
                                device,
                                dynamic_cast<CclOpShardedTensorConfig*>(input_tensor_config.get()),
                                dynamic_cast<CclOpShardedTensorConfig*>(output_tensor_config.get()),
                                ring_index,
                                ring_size,
                                global_num_workers,
                                all_gather_config.get_num_eth_buffers_per_edm() * i + b,
                                starting_dest_worker_index,
                                starting_chunk_into_shard,
                                is_buffer_in_clockwise_direction(b));
                            auto const& output_shard_addr_generator_args = output_tensor_shard_arg_generator.generate();
                            std::vector<uint32_t> worker_receive_writer_rt_args;
                            worker_receive_writer_rt_args.reserve(5 + output_shard_addr_generator_args.size());
                            worker_receive_writer_rt_args.push_back(static_cast<uint32_t>(worker_sender_reader.x));
                            worker_receive_writer_rt_args.push_back(static_cast<uint32_t>(worker_sender_reader.y));
                            worker_receive_writer_rt_args.push_back(sender_worker_reader_semaphore_addr);

                            worker_receive_writer_rt_args.push_back(output_tensor_shard_arg_generator.args_struct.num_dest_cores), //pages_per_eth_l1_buffer.at(b));
                            worker_receive_writer_rt_args.push_back(receiver_num_transfers);
                            worker_receive_writer_rt_args.push_back(pages_per_buffer.at(b));
                            worker_receive_writer_rt_args.push_back(static_cast<uint32_t>(cb_num_pages / 2));


                            std::copy(output_shard_addr_generator_args.begin(), output_shard_addr_generator_args.end(), std::back_inserter(worker_receive_writer_rt_args));

                            log_trace(tt::LogOp, "----worker_receive_writer_rt_args size={}", worker_receive_writer_rt_args.size());
                            log_trace(tt::LogOp, "\tsender_worker_reader_semaphore_addr: {}", sender_worker_reader_semaphore_addr);
                            output_tensor_shard_arg_generator.dump_to_log();

                            return worker_receive_writer_rt_args;
                        } else {
                            std::vector<uint32_t> worker_writer_receiver_rt_args = {
                                static_cast<uint32_t>(output_buffer->address()),
                                static_cast<uint32_t>(worker_sender_reader.x),
                                static_cast<uint32_t>(worker_sender_reader.y),
                            };

                            log_trace(tt::LogOp, "Worker {} RW rt args", b);
                            log_trace(tt::LogOp, "\toutput_buffer->address(): {}", output_buffer->address());
                            log_trace(tt::LogOp, "\tworker_sender_reader.x: {}", worker_sender_reader.x);
                            log_trace(tt::LogOp, "\tworker_sender_reader.y: {}", worker_sender_reader.y);
                            return worker_writer_receiver_rt_args;
                        }
                    };
                    std::vector<uint32_t> worker_receive_writer_rt_args = build_worker_receive_writer_rt_args();

                    std::string const& receiver_writer_kernel_path = is_sharded ?
                        "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_ring_gather_receive_writer.cpp" :
                        "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_receive_writer.cpp";
                    KernelHandle worker_receive_writer_kernel_id = tt_metal::CreateKernel(
                        program,
                        receiver_writer_kernel_path,
                        receiver_worker_cores.at(b),
                        tt_metal::WriterDataMovementConfig(worker_receive_writer_ct_args, worker_defines));

                    worker_writer_receiver_kernels.push_back(worker_receive_writer_kernel_id);

                    tt_metal::SetRuntimeArgs(
                        program,
                        worker_receive_writer_kernel_id,
                        receiver_worker_cores.at(b),
                        worker_receive_writer_rt_args);
                } // (!is_linear || !is_first_chip_in_chain)
                else {
                    log_trace(tt::LogOp, "Skipping receiver workers for linear chain");
                }

                uint32_t pages_per_worker = num_full_chunks_per_worker.at(b) * pages_per_chunk + rem_pages_per_worker.at(b);
                tensor_slicer.increment(pages_per_worker);

            }


            if (receiver_device_id == sender_device_id) {
                receiver_socket_idx += 2;
                sender_socket_idx += 2;
            } else {
                receiver_socket_idx += 1;
                sender_socket_idx += 1;
            }
        }
    } // num_full_send_directions

    ccl::generate_edm_kernels_for_ring_or_linear_topology(
        program,
        device,
        topology_config,
        clockwise_edm_builders,
        counter_clockwise_edm_builders,
        receiver_device_id,
        sender_device_id);

    auto override_runtime_arguments_callback = [num_links, total_worker_core_pairs_used, worker_reader_sender_kernels, worker_writer_sender_kernels, worker_reader_receiver_kernels, worker_writer_receiver_kernels, all_worker_sender_cores, all_worker_receiver_cores] (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        bool is_sharded = input_tensors[0].is_sharded();
        const auto& input = input_tensors[0];
        const auto& output = output_tensors[0];
        for (uint32_t i = 0; i < total_worker_core_pairs_used; ++i) {
            if (is_sharded) {
                auto &worker_reader_sender_runtime_args = GetRuntimeArgs(program, worker_reader_sender_kernels.at(i), all_worker_sender_cores.at(i));
                worker_reader_sender_runtime_args[7] = input.buffer()->address();
                uint32_t num_dest_cores = worker_reader_sender_runtime_args.at(12);
                worker_reader_sender_runtime_args[12 + num_dest_cores + 4] = output.buffer()->address();
                log_trace(tt::LogOp, "override worker_reader_sender_runtime_args:");
                for (uint32_t j = 0; j < worker_reader_sender_runtime_args.size(); ++j) {
                    log_trace(tt::LogOp, "\tworker_reader_sender_runtime_args[{}]: {}", j, worker_reader_sender_runtime_args.at(j));
                }

                auto &worker_writer_sender_runtime_args = GetRuntimeArgs(program, worker_writer_sender_kernels.at(i), all_worker_sender_cores.at(i));
                worker_writer_sender_runtime_args[12] = output.buffer()->address();
                log_trace(tt::LogOp, "override worker_writer_sender_runtime_args:");
                for (uint32_t j = 0; j < worker_writer_sender_runtime_args.size(); ++j) {
                    log_trace(tt::LogOp, "\tworker_writer_sender_runtime_args[{}]: {}", j, worker_reader_sender_runtime_args.at(j));
                }

                auto &worker_writer_receiver_runtime_args = GetRuntimeArgs(program, worker_writer_receiver_kernels.at(i), all_worker_receiver_cores.at(i));
                worker_writer_receiver_runtime_args[10] = output.buffer()->address();
                log_trace(tt::LogOp, "override worker_writer_receiver_runtime_args:");
                for (uint32_t j = 0; j < worker_writer_receiver_runtime_args.size(); ++j) {
                    log_trace(tt::LogOp, "\tworker_writer_receiver_runtime_args[{}]: {}", j, worker_reader_sender_runtime_args.at(j));
                }
            } else {
                auto &worker_reader_sender_runtime_args = GetRuntimeArgs(program, worker_reader_sender_kernels.at(i), all_worker_sender_cores.at(i));
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                worker_reader_sender_runtime_args[1] = output.buffer()->address();
                auto &worker_writer_sender_runtime_args = GetRuntimeArgs(program, worker_writer_sender_kernels.at(i), all_worker_sender_cores.at(i));
                worker_writer_sender_runtime_args[0] = output.buffer()->address();

                auto &worker_writer_receiver_runtime_args = GetRuntimeArgs(program, worker_writer_receiver_kernels.at(i), all_worker_receiver_cores.at(i));
                worker_writer_receiver_runtime_args[0] = output.buffer()->address();
            }
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
