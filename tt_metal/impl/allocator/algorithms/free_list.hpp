// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "hostdevcommon/common_values.hpp"
#include "tt_metal/impl/allocator/algorithms/allocator_algorithm.hpp"

namespace tt {

namespace tt_metal {

namespace allocator {

class FreeList : public Algorithm<FreeList> {
   public:
    enum class SearchPolicy {
        BEST = 0,
        FIRST = 1
    };

    FreeList(uint64_t max_size_bytes, uint64_t offset_bytes, uint64_t min_allocation_size, uint64_t alignment, SearchPolicy search_policy);

    void init();

    std::vector<std::pair<uint64_t, uint64_t>> available_addresses(uint64_t size_bytes) const;

    std::optional<uint64_t> allocate(uint64_t size_bytes, bool bottom_up=true, uint64_t address_limit=0);

    std::optional<uint64_t> allocate_at_address(uint64_t absolute_start_address, uint64_t size_bytes);

    void deallocate(uint64_t absolute_address);

    void clear();

    Statistics get_statistics() const;

    void dump_blocks(std::ofstream &out) const;

   private:
    struct Block {
        Block(uint64_t address, uint64_t size, bool free) : address(address), size(size), free(free) {}
        Block(uint64_t address, uint64_t size, bool free, std::shared_ptr<Block> prev_block, std::shared_ptr<Block> next_block, std::shared_ptr<Block> prev_free, std::shared_ptr<Block> next_free)
            : address(address), size(size), free(free), prev_block(prev_block), next_block(next_block), prev_free(prev_free), next_free(next_free) {}

        uint64_t address;
        uint64_t size;
        bool free;
        std::shared_ptr<Block> prev_block = nullptr;
        std::shared_ptr<Block> next_block = nullptr;
        std::shared_ptr<Block> prev_free = nullptr;
        std::shared_ptr<Block> next_free = nullptr;
    };

    void dump_block(const std::shared_ptr<Block> block, std::ofstream &out) const;

    std::shared_ptr<Block> search_best(uint64_t size_bytes, bool bottom_up);

    std::shared_ptr<Block> search_first(uint64_t size_bytes, bool bottom_up);

    std::shared_ptr<Block> search(uint64_t size_bytes, bool bottom_up);

    void allocate_entire_free_block(std::shared_ptr<Block> free_block_to_allocate);

    void update_left_aligned_allocated_block_connections(std::shared_ptr<Block> free_block, std::shared_ptr<Block> allocated_block);

    void update_right_aligned_allocated_block_connections(std::shared_ptr<Block> free_block, std::shared_ptr<Block> allocated_block);

    std::shared_ptr<Block> allocate_slice_of_free_block(std::shared_ptr<Block> free_block, uint64_t offset, uint64_t size_bytes);

    void update_lowest_occupied_address();

    void update_lowest_occupied_address(uint64_t address);

    SearchPolicy search_policy_;
    std::shared_ptr<Block> block_head_;
    std::shared_ptr<Block> block_tail_;
    std::shared_ptr<Block> free_block_head_;
    std::shared_ptr<Block> free_block_tail_;

    std::unordered_map<uint64_t, std::shared_ptr<Block>> allocated_address_to_block_;
};

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
