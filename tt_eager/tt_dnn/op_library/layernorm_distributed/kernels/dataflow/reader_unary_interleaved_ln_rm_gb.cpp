// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "debug/assert.h"
#include "debug/dprint.h"

void kernel_main() {
    const uint32_t src_addr  = get_arg_val<uint32_t>(0);
    const uint32_t NCHt      = get_arg_val<uint32_t>(1);
    const uint32_t Wt        = get_arg_val<uint32_t>(2);
    const uint32_t tile_offset = get_arg_val<uint32_t>(3);
    const uint32_t stats_tile_offset = get_arg_val<uint32_t>(4);

    const uint32_t gamma_addr = get_arg_val<uint32_t>(7);
    const uint32_t beta_addr = get_arg_val<uint32_t>(8);
    const uint32_t stats_addr = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_inp = tt::CB::c_in0;
    constexpr uint32_t cb_stats = tt::CB::c_in1;
    constexpr uint32_t cb_gamma = tt::CB::c_in2;
    constexpr uint32_t cb_beta = tt::CB::c_in3;
    constexpr uint32_t cb_eps = tt::CB::c_in4;
    constexpr uint32_t cb_reduce = tt::CB::c_in5;

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(cb_inp);
    const DataFormat src0_data_format = get_dataformat(cb_inp);
    const uint32_t stats_tile_bytes = get_tile_size(cb_stats);
    const DataFormat stats_data_format = get_dataformat(cb_stats);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool stats_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool gamma_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool beta_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t blk = get_compile_time_arg_val(4); // needed for correctness of softmax/LN kernels
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(5);

    const InterleavedAddrGenFast<src0_is_dram> src_a = {
        .bank_base_address = src_addr,
        .page_size = src0_tile_bytes,
        .data_format = src0_data_format
    };

    const InterleavedAddrGenFast<stats_is_dram> src_stats = {
        .bank_base_address = stats_addr,
        .page_size = stats_tile_bytes,
        .data_format = stats_data_format
    };

    // #define stick_size_is_pow2 get_compile_time_arg_val(6) == 1
    constexpr bool stick_size_is_pow2 = get_compile_time_arg_val(6) == 1;
    ASSERT(stick_size_is_pow2);
    // #if (stick_size_is_pow2)
    const uint32_t log_base_2_of_page_size = get_compile_time_arg_val(7);
    // #else
    // const uint32_t page_size = get_compile_time_arg_val(6);
    #ifdef FUSE_GAMMA
    // #if (stick_size_is_pow2)
    const InterleavedPow2AddrGen<gamma_is_dram> addrg = {
        .bank_base_address = gamma_addr,
        .log_base_2_of_page_size = log_base_2_of_page_size
    };
    // #else
    // const InterleavedAddrGen<gamma_is_dram> addrg = {
    //     .bank_base_address = gamma_addr,
    //     .page_size = page_size
    // };
    // #endif
    const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
    #endif
    #ifdef FUSE_BETA
    // #if (stick_size_is_pow2)
    const InterleavedPow2AddrGen<beta_is_dram> addrb = {
        .bank_base_address = beta_addr,
        .log_base_2_of_page_size = log_base_2_of_page_size
    };
    // #else
    // const InterleavedAddrGen<beta_is_dram> addrb = {
    //     .bank_base_address = beta_addr,
    //     .page_size = page_size
    // };
    // #endif
    const uint32_t beta_tile_bytes = get_tile_size(cb_beta);
    #endif



    // Generate constant tiles for layernorm compute
    uint32_t scaler = get_arg_val<uint32_t>(5);
    generate_reduce_scaler(cb_reduce, scaler);
    const uint32_t eps = get_arg_val<uint32_t>(6);
    generate_bcast_col_scalar(cb_eps, eps);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t offs = 0;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        DPRINT << "Reader: ncht = " << ncht << ENDL();
        // Read stats tiles
        cb_reserve_back(cb_stats, stats_tiles_cols);
        uint32_t stats_wr_ptr = get_write_ptr(cb_stats);
        const uint32_t stats_start_tile_idx = stats_tile_offset + ncht * stats_tiles_cols;
        for (uint32_t st = 0; st < stats_tiles_cols; ++st) {
            noc_async_read_tile(stats_start_tile_idx + st, src_stats, stats_wr_ptr);
            stats_wr_ptr += stats_tile_bytes;
            noc_async_read_barrier();
        }
        cb_push_back(cb_stats, stats_tiles_cols);

        // read input tiles
        for (uint32_t wt = 0; wt<Wt; wt += blk) {
            cb_reserve_back(cb_inp, blk);
            uint32_t inp_wr_ptr = get_write_ptr(cb_inp);

            for (uint32_t r = 0; r<blk; r++) {
                noc_async_read_tile(offs+wt+r+tile_offset, src_a, inp_wr_ptr);
                inp_wr_ptr += src0_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_inp, blk);

        } // wt loop

        #if defined FUSE_GAMMA || defined FUSE_BETA
        if (ncht == 0) {
            for (uint32_t wt = 0; wt<Wt; wt += blk) {
                #ifdef FUSE_GAMMA
                {
                    cb_reserve_back(cb_gamma, blk);
                    uint32_t l1_write_addr = get_write_ptr(cb_gamma);
                    for (uint32_t r = 0; r<blk; r++) {
                        uint64_t gamma_noc_addr = get_noc_addr(wt + r, addrg);
                        noc_async_read(gamma_noc_addr, l1_write_addr, 32);
                        gamma_noc_addr += 32;
                        noc_async_read(gamma_noc_addr, l1_write_addr + 512, 32);
                        l1_write_addr += gamma_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_gamma, blk);
                }
                #endif

                #ifdef FUSE_BETA
                {
                    cb_reserve_back(cb_beta, blk);
                    uint32_t l1_write_addr = get_write_ptr(cb_beta);
                    for (uint32_t r = 0; r<blk; r++) {
                         uint64_t beta_noc_addr = get_noc_addr(wt + r, addrb);
                        noc_async_read(beta_noc_addr, l1_write_addr, 32);
                        beta_noc_addr += 32;
                        noc_async_read(beta_noc_addr, l1_write_addr + 512, 32);
                        l1_write_addr += beta_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_beta, blk);
                }
                #endif
            } // wt loop
        }
        #endif
        offs += Wt;
    } // ncht loop
}
