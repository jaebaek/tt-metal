#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/core_coord.h"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

// Semaphores are statically allocated withing range [SEMAPHORE_BASE, SEMAPHORE_BASE + SEMAPHORE_SIZE]
class Semaphore {
   public:
    Semaphore(
        Device *device,
        const CoreRangeSet &core_ranges,
        uint32_t address,
        uint32_t initial_value) : device_(device), core_ranges_(core_ranges), address_(address), initial_value_(initial_value) {}

    Semaphore(const Semaphore &other);

    Semaphore& operator=(const Semaphore &other);

    Semaphore(Semaphore &&other);

    Semaphore& operator=(Semaphore &&other);

    constexpr uint32_t size() const { return SEMAPHORE_SIZE / NUM_SEMAPHORES; }

    Device *device() const { return device_; }

    uint32_t address() const { return address_; }

    CoreRangeSet core_ranges() const { return core_ranges_; }

    uint32_t initial_value() const { return initial_value_; }

    bool initialized_on_logical_core(const CoreCoord &logical_core) const;

   private:
    Device *device_;
    CoreRangeSet core_ranges_;             // Ranges of cores where this semaphore is initialized
    uint32_t address_;
    uint32_t initial_value_;              // Initial value of semaphore
};

}  // namespace tt_metal

}  // namespace tt
