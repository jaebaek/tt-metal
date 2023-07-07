#include <stdint.h>
#include "dataflow_kernel_api.h"
#include "hostdevcommon/common_values.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {

    uint32_t sender_noc_x            = dataflow::get_arg_val<uint32_t>(0);
    uint32_t sender_noc_y            = dataflow::get_arg_val<uint32_t>(1);
    uint32_t num_tiles               = dataflow::get_arg_val<uint32_t>(2);
    uint32_t sender_semaphore_addr   = dataflow::get_arg_val<uint32_t>(3);
    uint32_t receiver_semaphore_addr = dataflow::get_arg_val<uint32_t>(4);
    uint32_t num_repetitions         = dataflow::get_arg_val<uint32_t>(5);

    volatile uint32_t* receiver_semaphore_addr_ptr = reinterpret_cast<volatile uint32_t*>(receiver_semaphore_addr);

    constexpr uint32_t cb_id            = get_compile_time_arg_val(0);
    constexpr uint32_t block_size_tiles = get_compile_time_arg_val(1);

    uint32_t block_size_bytes = dataflow::get_tile_size(cb_id) * block_size_tiles;

    uint64_t sender_semaphore_noc_addr = dataflow::get_noc_addr(sender_noc_x, sender_noc_y, sender_semaphore_addr);

    for (uint32_t j = 0; j < num_repetitions; j++) {
        for (uint32_t i = 0; i<num_tiles ; i += block_size_tiles) {
            dataflow::cb_reserve_back(cb_id, block_size_tiles);

            // Reset receiver's own semaphore value to INVALID
            dataflow_internal::noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);

            // Tell sender we're ready -- atomic increment sender's semaphore
            dataflow_internal::noc_semaphore_inc(sender_semaphore_noc_addr, 1);

            // Wait on receiver's own semaphore value to become VALID (set by sender after it sends the data)
            dataflow_internal::noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);

            dataflow::cb_push_back(cb_id, block_size_tiles);
        }
    }

}
