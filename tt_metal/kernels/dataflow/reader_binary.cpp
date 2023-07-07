#include <stdint.h>
#include "dataflow_kernel_api.h"

void kernel_main() {
    uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    uint32_t src0_noc_x = get_arg_val<uint32_t>(1);
    uint32_t src0_noc_y = get_arg_val<uint32_t>(2);
    uint32_t src1_addr  = get_arg_val<uint32_t>(3);
    uint32_t src1_noc_x = get_arg_val<uint32_t>(4);
    uint32_t src1_noc_y = get_arg_val<uint32_t>(5);
    uint32_t num_tiles  = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = dataflow::get_tile_size(cb_id_in0);
    uint32_t ublock_size_bytes_1 = dataflow::get_tile_size(cb_id_in1);
    uint32_t ublock_size_tiles = 1;

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    // read ublocks from src0/src1 to CB0/CB1, then push ublocks to compute (unpacker)
    for (uint32_t i=0; i<num_tiles; i += ublock_size_tiles) {
        uint64_t src0_noc_addr = dataflow::get_noc_addr(src0_noc_x, src0_noc_y, src0_addr);
        uint64_t src1_noc_addr = dataflow::get_noc_addr(src1_noc_x, src1_noc_y, src1_addr);

        dataflow::cb_reserve_back(cb_id_in0, ublock_size_tiles);
        dataflow::cb_reserve_back(cb_id_in1, ublock_size_tiles);
        l1_write_addr_in0 = dataflow::get_write_ptr(cb_id_in0);
        l1_write_addr_in1 = dataflow::get_write_ptr(cb_id_in1);

        dataflow::noc_async_read(src0_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
        dataflow::noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);

        dataflow::noc_async_read_barrier();

        dataflow::cb_push_back(cb_id_in0, ublock_size_tiles);
        dataflow::cb_push_back(cb_id_in1, ublock_size_tiles);

        src0_addr += ublock_size_bytes_0;
        src1_addr += ublock_size_bytes_1;
    }
}
