#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    kernel_profiler::mark_time(3);

    uint32_t receiver_noc_x          = get_arg_val<uint32_t>(0);
    uint32_t receiver_noc_y          = get_arg_val<uint32_t>(1);
    uint32_t num_tiles               = get_arg_val<uint32_t>(2);
    uint32_t sender_semaphore_addr   = get_arg_val<uint32_t>(3);
    uint32_t receiver_semaphore_addr = get_arg_val<uint32_t>(4);

    // initialized by the host to 0 before program launch
    volatile uint32_t* sender_semaphore_addr_ptr = reinterpret_cast<volatile uint32_t*>(sender_semaphore_addr);

    constexpr uint32_t cb_id            = get_compile_time_arg_val(0);
    constexpr uint32_t block_size_tiles = get_compile_time_arg_val(1);

    uint32_t block_size_bytes = get_tile_size(cb_id) * block_size_tiles;

    for (uint32_t i = 0; i < num_tiles; i += block_size_tiles) {
        cb_wait_front(cb_id, block_size_tiles);
        uint32_t l1_addr = get_read_ptr(cb_id);

        // wait until receiver has set the sender's semaphore_addr value to 1, which means receiver has reserved space in the CB
        noc_semaphore_wait(sender_semaphore_addr_ptr, 1);
        // set the semaphore value back to zero for the next block
        noc_semaphore_set(sender_semaphore_addr_ptr, 0);

        // Now we have the block in the CB (at l1_addr), we can send to receiver
        uint64_t receiver_data_noc_addr = get_noc_addr(receiver_noc_x, receiver_noc_y, l1_addr);
        noc_async_write(l1_addr, receiver_data_noc_addr, block_size_bytes);

        // We also set the receiver's semaphore, so that it knows that the data has been written to the CB
        uint64_t receiver_semaphore_noc_addr = get_noc_addr(receiver_noc_x, receiver_noc_y, receiver_semaphore_addr);
        noc_semaphore_inc(receiver_semaphore_noc_addr, 1);

        // do we need the barrier? to make sure we've sent the data before we pop the CB? or does sempahore inter-lock already guarantee that?
        noc_async_write_barrier();

        cb_pop_front(cb_id, block_size_tiles);
    }

    kernel_profiler::mark_time(4);
}
