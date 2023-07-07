#include <stdint.h>
#include "dataflow_kernel_api.h"

//#include "debug_print.h"

void generate_bcast_scaler() {
    constexpr uint32_t cb_in_2 = 2;
    uint32_t scaler = get_arg_val<uint32_t>(8);
    union { float f; uint32_t u; } u; u.u = scaler;
    //DPRINT << "basic Scaler = " << F32(u.f) << ENDL();
    dataflow::cb_reserve_back(cb_in_2, 1);
    auto ptr = reinterpret_cast<uint16_t*>(dataflow::get_write_ptr(cb_in_2));
    for (int j = 0; j < 1024; j++)
        ptr[j] = uint16_t(0);

    for (int k = 0; k < 4; k++)
    for (int j = 0; j < 16; j++)
        ptr[k*256 + j] = uint16_t(u.u>>16);
    dataflow::cb_push_back(cb_in_2, 1);
}

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(3); // same arg index as in reader_unary and in reader_unary_transpose_wh_8bank

    constexpr uint32_t cb_id_in0 = 0, cb_id_in1 = 1;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = dataflow::get_tile_size(cb_id_in0);

    #ifdef KERNEL_COMPILE_TIME_ARG_0
    constexpr bool read_from_dram = get_compile_time_arg_val(0);
    #else
    constexpr bool read_from_dram = true;
    #endif

    const dataflow::InterleavedPow2AddrGen<read_from_dram> src_a = { src_addr, 11 };

    #if GENERATE_BCAST_SCALER
    // TODO(AP): cleanup, probably with named args/param pack/reflection.
    generate_bcast_scaler();
    constexpr uint32_t blk = BLOCK_SIZE;
    #else
    constexpr uint32_t blk = 1; // 1 for correctness for unfused kernels
    #endif

    #ifdef TILE_OFFSET
    uint32_t tile_offset = TILE_OFFSET;
    #else
    constexpr uint32_t tile_offset = 0;
    #endif
    //DPRINT << "Reader Tile offset=" << tile_offset << ENDL();

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t i_tile = 0;
    for (uint32_t i = 0; i<num_tiles; i += blk) {
        uint32_t rem = blk; // (i + blk > num_tiles) ? num_tiles - i : blk;
        dataflow::cb_reserve_back(cb_id_in0, rem);
        uint32_t l1_write_addr = dataflow::get_write_ptr(cb_id_in0);

        for (uint32_t r = 0; r<rem; r++) {
            uint64_t src_noc_addr = dataflow::get_noc_addr(i+r+tile_offset, src_a); // not contiguous for sequential r, can be banked
            auto addr = l1_write_addr + (r<<11);
            dataflow::noc_async_read(src_noc_addr, addr, tile_bytes); // TODO(AP): data type size
        }
        // DPRINT << uint(my_x[loading_noc]) << ", " << uint(my_y[loading_noc]) << ENDL();
        dataflow::noc_async_read_barrier();
        dataflow::cb_push_back(cb_id_in0, rem);
    }
}
