// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>
#include "hostdevcommon/common_values.hpp"

#include "tt_metal/impl/allocator/allocator_types.hpp"

namespace tt {

namespace tt_metal {

namespace allocator {

template <class AllocatorType>
class Algorithm {
   private:
    friend AllocatorType;
    Algorithm(uint64_t max_size_bytes, uint64_t offset_bytes, uint64_t min_allocation_size, uint64_t alignment)
        : max_size_bytes_(max_size_bytes), offset_bytes_(offset_bytes), min_allocation_size_(min_allocation_size), alignment_(alignment), lowest_occupied_address_(std::nullopt) {
        TT_ASSERT(offset_bytes % this->alignment_ == 0, "Offset {} should be {} B aligned", offset_bytes, this->alignment_);
    }

   public:

    ~Algorithm() {}

    uint64_t align(uint64_t address) const {
        uint64_t factor = (address + alignment_ - 1) / alignment_;
        return factor * alignment_;
    }

    uint64_t max_size_bytes() const { return max_size_bytes_; }

    std::optional<uint64_t> lowest_occupied_address() const {
        if (not this->lowest_occupied_address_.has_value()) {
            return this->lowest_occupied_address_;
        }
        return this->lowest_occupied_address_.value() + this->offset_bytes_;
    }

    void init() {
        static_cast<AllocatorType *>(this)->init();
    }

    std::vector<std::pair<uint64_t, uint64_t>> available_addresses(uint64_t size_bytes) const {
        return static_cast<AllocatorType *>(this)->available_addresses(size_bytes);
    }

    // bottom_up=true indicates that allocation grows from address 0
    std::optional<uint64_t> allocate(uint64_t size_bytes, bool bottom_up=true, uint64_t address_limit=0) {
        return static_cast<AllocatorType *>(this)->allocate(size_bytes, bottom_up, address_limit);
    }

    std::optional<uint64_t> allocate_at_address(uint64_t absolute_start_address, uint64_t size_bytes) {
        return static_cast<AllocatorType *>(this)->allocate_at_address(absolute_start_address, size_bytes);
    }

    void deallocate(uint64_t absolute_address) {
        static_cast<AllocatorType *>(this)->deallocate(absolute_address);
    }

    void clear() {
        static_cast<AllocatorType *>(this)->clear();
    }

    Statistics get_statistics() const {
        return static_cast<AllocatorType *>(this)->get_statistics();
    }

    void dump_blocks(std::ofstream &out) const {
        static_cast<AllocatorType *>(this)->dump_block();
    }

   protected:
    uint64_t max_size_bytes_ = 0;
    uint64_t offset_bytes_ = 0;
    uint64_t min_allocation_size_ = 0;
    uint64_t alignment_ = 0;
    std::optional<uint64_t> lowest_occupied_address_ = std::nullopt;
};

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
