// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "mod_div_lib.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "debug/dprint.h"

#ifdef FUSE_BIAS
#include "compute_kernel_api/bcast.h"
#endif

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#define DEBUG_PRINT 0

// #include "debug_macros.h"

// SliceRange srt = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 8, .ws = 1};
// SliceRange srr = SliceRange{.h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1};
// SliceRange srr1 = SliceRange{.h0 = 1, .h1 = 2, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1};
// SliceRange src = SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};

inline void tilize_in(
    uint32_t in_cb_id,
    uint32_t in_subblock_h,
    uint32_t in_block_w,
    uint32_t in_num_subblocks,
    uint32_t out_cb_id
) {
    tilize_init_short(in_cb_id, in_block_w);
    for (uint32_t in_subblock = 0; in_subblock < in_num_subblocks; ++in_subblock) {
        for (uint32_t h = 0; h < in_subblock_h; ++h) {
            cb_wait_front(in_cb_id, in_block_w);
            cb_reserve_back(out_cb_id, in_block_w);
            tilize_block(in_cb_id, in_block_w, out_cb_id);
            cb_push_back(out_cb_id, in_block_w);
            cb_pop_front(in_cb_id, in_block_w);
        }
    }
    tilize_uninit(in_cb_id);
} // tilize_in()

template <uint32_t out_subblock_w, uint32_t out_block_w>
inline void reblock_and_untilize(
    uint32_t num_out_subblocks_in_col,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    uint32_t interm_cb_id,
    uint32_t out_cb_id) {

    uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles, num_out_subblocks_in_col);
    cb_wait_front(interm_cb_id, num_tiles_in_row_of_subblocks);

    uint32_t within_block_index = 0;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        uint32_t block_offset = 0;

        cb_reserve_back(out_cb_id, out_block_w);
        for (uint32_t n = 0; n < num_out_subblocks_in_col; n++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < out_subblock_w; w++) {
                uint32_t tile_index = block_offset + within_block_index + w;
                copy_tile(interm_cb_id, tile_index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dst<out_subblock_w, out_block_w>(out_cb_id, 1, n);
            tile_regs_release();
            block_offset += out_subblock_num_tiles;
        }
        cb_push_back(out_cb_id, out_block_w);

        within_block_index += out_subblock_w;
    }
    cb_pop_front(interm_cb_id, num_tiles_in_row_of_subblocks);
}

namespace NAMESPACE {
void MAIN {

    constexpr uint32_t in0_block_w            = get_compile_time_arg_val(0); // inner block size in tiles
    constexpr uint32_t in0_num_subblocks      = get_compile_time_arg_val(1); // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles    = get_compile_time_arg_val(2); // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t in0_subblock_h         = get_compile_time_arg_val(4);
    constexpr uint32_t in1_num_subblocks      = get_compile_time_arg_val(5); // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles    = get_compile_time_arg_val(6); //out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_block_w            = get_compile_time_arg_val(7); // out_subblock_w*in1_num_subblocks
    // if these are not defined as volatile, it causes code size for TRISC2 to be too large if num_blocks > 1
    constexpr uint32_t in0_num_blocks_h       = get_compile_time_arg_val(8);
    constexpr uint32_t in0_num_blocks_w       = get_compile_time_arg_val(9);
    constexpr uint32_t in1_num_blocks_w       = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_h         = get_compile_time_arg_val(11); // inner row block size in tiles
    constexpr uint32_t out_subblock_w         = get_compile_time_arg_val(12); // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(13); // out_subblock_h * out_subblock_w;
    constexpr bool tilize_in0                 = get_compile_time_arg_val(14);
    constexpr bool untilize_out               = get_compile_time_arg_val(15);

    constexpr uint32_t out_block_num_tiles    = in0_num_subblocks * in1_num_subblocks * out_subblock_num_tiles;
    constexpr uint32_t out_block_w = in1_block_w;
    constexpr bool spill = in0_num_blocks_w > 1;

    // CB indices
    constexpr uint32_t in0_cb_id                                = tt::CB::c_in0;
    constexpr uint32_t in1_cb_id                                = tt::CB::c_in1;
    constexpr uint32_t in0_pretilize_cb_id                      = tt::CB::c_in6;
    constexpr uint32_t in0_cb_second_reader_id                  = tt::CB::c_in7;
    constexpr uint32_t matmul_partials_cb                       = tt::CB::c_intermed0;
    constexpr uint32_t tilized_in0_cb_id                        = tt::CB::c_intermed1;
    //constexpr uint32_t untilize_mode_reblock_cb                 = tt::CB::c_intermed2;
    constexpr uint32_t out_cb_id                                = tt::CB::c_out0;

    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? matmul_partials_cb : out_cb_id;

    #ifdef FUSE_BIAS
    constexpr uint32_t bias_ntiles_w = get_compile_time_arg_val(16);
    constexpr uint32_t bias_cb_id                           = tt::CB::c_in2;
    uint32_t bias_block_offset = 0;
    constexpr uint32_t mm_out_cb_id = matmul_partials_cb;
    #else
    constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;
    #endif

    constexpr uint32_t mm_in0_cb_id = tilize_in0 ? tilized_in0_cb_id : in0_cb_id;

    #ifdef SPLIT_READER
    constexpr uint32_t in0_num_subblocks_read = in0_num_subblocks / 2;
    #else
    constexpr uint32_t in0_num_subblocks_read = in0_num_subblocks;
    #endif
    //DPRINT << "compute 1 " << ENDL();
    //uint32_t shwetank=1;
    mm_block_init(mm_in0_cb_id, in1_cb_id, out_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
    #ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
    #endif
    //DPRINT << "compute initialization " << in1_num_blocks_w << "    " << in0_num_blocks_h << ENDL();
    // in1 num blocks w is the outer loop. Output blocks are computed in col major order.
    for(uint32_t in1_block_w_i = 0; in1_block_w_i < in1_num_blocks_w; ++in1_block_w_i) {

        for(uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
            //DPRINT << "compute_148 " << in1_block_w_i << "  " <<  in0_block_h_i << ENDL();
            #ifdef PRE_TILIZE
            unpack_reconfig_data_format_srca(in1_cb_id, in0_pretilize_cb_id);

            tilize_in(in0_pretilize_cb_id, in0_subblock_h, in0_block_w, in0_num_subblocks, tilized_in0_cb_id);

            mm_block_init_short_with_dt(in0_cb_id, in1_cb_id, in0_pretilize_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
            #endif

            bool enable_reload = false;

            #ifdef PACK_RELU
            // for each output block we start we relu disabled so that intermediate results are not relu'd
            PACK(( llk_pack_relu_config(ReluType::NO_RELU) ));
            #endif

            UNPACK( const uint32_t partials_cb_read_ptr = cb_interface[matmul_partials_cb].fifo_rd_ptr );
            PACK( const uint32_t partials_cb_write_ptr = cb_interface[matmul_partials_cb].fifo_wr_ptr );
            uint32_t curr_matmul_out_cb = matmul_partials_cb;
            for(uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
                //DPRINT << "compute_168 " << in1_block_w_i << "  " << in0_block_h_i << " " << in0_block_w_i << ENDL();
                bool last_out = (in0_block_w_i == in0_num_blocks_w - 1);
                if constexpr (tilize_in0) {
                    #if defined PACK_RELU and not defined FUSE_BIAS
                    if (last_out) {
                        // if last block we pack the final result with relu enabled
                        PACK(( llk_pack_relu_config(ReluType::NO_RELU) ));
                    }
                    #endif
                    #ifdef PACKER_L1_ACC
                        pack_reconfig_l1_acc(0);
                        pack_reconfig_data_format(curr_matmul_out_cb, tilized_in0_cb_id);
                    #endif

                    unpack_reconfig_data_format_srca(in1_cb_id, in0_cb_id);

                    tilize_in(in0_cb_id, in0_subblock_h, in0_block_w, in0_num_subblocks_read, tilized_in0_cb_id);
                    #ifdef SPLIT_READER
                    tilize_in(in0_cb_second_reader_id, in0_subblock_h, in0_block_w, in0_num_subblocks_read, tilized_in0_cb_id);
                    #endif

                    mm_block_init_short_with_dt(mm_in0_cb_id, in1_cb_id, /*srca_old_operand=*/in0_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
                }
                cb_wait_front(mm_in0_cb_id, in0_block_num_tiles);
                cb_wait_front(in1_cb_id, in1_block_num_tiles);
                //DPRINT << "compute test 1" << ENDL();
                if (last_out) {
                    #if defined PACK_RELU and not defined FUSE_BIAS
                    // if last block we pack the final result with relu enabled
                    PACK(( llk_pack_relu_config(ReluType::ZERO_RELU) ));
                    #endif
                    #ifndef FUSE_BIAS
                    curr_matmul_out_cb = mm_out_cb_id;
                    #endif
                }

                //DPRINT << "compute test 2" << ENDL();
                #ifdef PACKER_L1_ACC
                pack_reconfig_data_format(curr_matmul_out_cb);
                #endif
                uint32_t in0_index_subblock_offset = 0;
                //DPRINT << "compute_212 " << in0_num_subblocks << " " <<  in1_num_subblocks << ENDL();
                for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                    uint32_t in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock_i = 0; in1_subblock_i < in1_num_subblocks; ++in1_subblock_i) {
                        //DPRINT << "compute_212 " << in0_subblock_i << " " << in1_subblock_i << "    " << in0_num_subblocks << " " <<  in1_num_subblocks << ENDL();
                        if (enable_reload) {
                            // Reconfigure input
                            copy_tile_to_dst_init_short_with_dt(in1_cb_id, matmul_partials_cb);
                            cb_wait_front(matmul_partials_cb, out_subblock_num_tiles);
                            tile_regs_acquire();

                            uint32_t start_dst_index = 0;
                            uint32_t start_tile_index = 0;
                            copy_block_matmul_partials(matmul_partials_cb, start_tile_index, start_dst_index, out_subblock_num_tiles);

                            cb_pop_front(matmul_partials_cb, out_subblock_num_tiles);
                            // Reconfigure srcA back
                            mm_block_init_short_with_dt(mm_in0_cb_id, in1_cb_id, matmul_partials_cb, false, out_subblock_w, out_subblock_h, in0_block_w);
                        } else {
                            // just acquire
                            tile_regs_acquire();
                        }

                        // Compute output sub-block
                        uint32_t dst_index = 0; // start at 0, each call to matmul_block internally increments dst_index
                        uint32_t in0_index = in0_index_subblock_offset; // offset into in0 block
                        uint32_t in1_index = in1_index_subblock_offset; // offset into in1 block
                        // inner dim that we accumulate is the inner dim of in0/in1, which is in0_block_w
                        for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; inner_dim_idx++) {
                            // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                            // accumulation is done by iterating matmul_block across inner dim
                            // in0_block_w is passed as innder dim (kt) to matmul_block, interally used to stride in0
                            matmul_block(mm_in0_cb_id, in1_cb_id, in0_index, in1_index, dst_index, false, out_subblock_w, out_subblock_h, in0_block_w);
                            in0_index ++;  // stride right by 1
                            in1_index += in1_block_w; // to stride down by 1 need to stride by in_per_core_w (should be called in1_block_w)
                        }

                        #if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
                        if (last_out) {
                            for (uint32_t i = 0; i < out_subblock_num_tiles; ++ i) {
                                SFPU_OP_FUNC_ACTIVATION
                            }
                        }
                        #endif
                        tile_regs_commit();
                        cb_reserve_back(curr_matmul_out_cb, out_subblock_num_tiles);
                        tile_regs_wait();

                        #ifdef PACKER_L1_ACC
                            // no accumulation for first iteration, last iteration
                            // accumulation happens with copying tiles to dst
                            if (in0_block_w_i == 0) {
                                pack_reconfig_l1_acc(0);
                            } else if(last_out) {
                                //Fuse bias always uses intermediate buffer
                                //no need to spill and reload last iteration
                                #ifdef FUSE_BIAS
                                pack_reconfig_l1_acc(1);
                                #else
                                pack_reconfig_l1_acc(0);
                                #endif
                            } else {
                                pack_reconfig_l1_acc(1);
                            }
                        #endif

                        uint32_t start_dst_index = 0;
                        matmul_pack_tile(start_dst_index, curr_matmul_out_cb, out_subblock_num_tiles);

                        tile_regs_release();
                        cb_push_back(curr_matmul_out_cb, out_subblock_num_tiles);

                        in1_index_subblock_offset += out_subblock_w;
                    } // for in1_num_subblocks
                    in0_index_subblock_offset += in0_subblock_num_tiles;
                }

                //DPRINT << "compute_test_284" << ENDL();
                #ifdef PACKER_L1_ACC
                    #ifdef FUSE_BIAS
                        if (in0_block_w_i < in0_num_blocks_w  - 1) {
                            //Wait for l1 accumulation to populate interm buffer,
                            //then pop to update fifo rd pointer
                            cb_wait_front(matmul_partials_cb, out_block_num_tiles);
                            cb_pop_front(matmul_partials_cb, out_block_num_tiles);
                        }
                        // never reload when with bias, bias uses interm buffer
                        enable_reload = false;
                    #else
                        //Last iteration does spill and reload to output buffer
                        if (in0_block_w_i < in0_num_blocks_w  - 2) {
                            cb_wait_front(matmul_partials_cb, out_block_num_tiles);
                            cb_pop_front(matmul_partials_cb, out_block_num_tiles);
                        }
                        if (in0_block_w_i == in0_num_blocks_w - 2) { enable_reload = true; }
                    #endif
                #else
                    if constexpr (spill) { enable_reload = true; }
                #endif

                if constexpr (spill) {
                    if (!last_out) {
                        UNPACK( cb_interface[matmul_partials_cb].fifo_rd_ptr = partials_cb_read_ptr );
                        PACK( cb_interface[matmul_partials_cb].fifo_wr_ptr = partials_cb_write_ptr );
                    }
                }

                cb_pop_front(mm_in0_cb_id, in0_block_num_tiles);
                cb_pop_front(in1_cb_id, in1_block_num_tiles);
            } // for in0_num_blocks_w
            //DPRINT << " compute_test_321 " << in0_block_h_i << "    " << in1_block_w_i << ENDL();
            if constexpr(matmul_partials_cb == mm_out_cb_id) {
                UNPACK( cb_interface[matmul_partials_cb].fifo_rd_ptr = partials_cb_read_ptr );
            }
            #ifdef FUSE_BIAS
            #ifdef PACK_RELU
            // if last block we pack the final result with relu enabled
            PACK(( llk_pack_relu_config(ReluType::ZERO_RELU) ));
            #endif
            #ifdef PACKER_L1_ACC
            pack_reconfig_l1_acc(0);
            #endif
            pack_reconfig_data_format(matmul_partials_cb, out_cb_id);
            unpack_reconfig_data_format(in1_cb_id, matmul_partials_cb, mm_in0_cb_id, bias_cb_id);
            add_bcast_rows_init_short(matmul_partials_cb, bias_cb_id);

            cb_wait_front(bias_cb_id, bias_ntiles_w);
            cb_wait_front(matmul_partials_cb, out_block_num_tiles);
            //DPRINT << "conv_compute_336 " << in0_num_subblocks << "    " << in1_num_subblocks << ENDL();
            for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                uint32_t in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock_i = 0; in1_subblock_i < in1_num_subblocks; ++in1_subblock_i) {
                    DPRINT << "conv_compute_339 " << in0_subblock_i << " " << in1_subblock_i << ENDL();
                    tile_regs_acquire(); //--> error here
                    uint32_t i = 0;
                    //DPRINT << "conv_compute_349 " << ENDL();
                    for (uint32_t h = 0; h < out_subblock_h; ++ h) {
                        uint32_t bcast_tile_i = bias_block_offset + in1_index_subblock_offset;
                        for (uint32_t w = 0; w < out_subblock_w; ++ w) {
                            //DPRINT << "conv_compute_345 " << h << " " << w << ENDL();
                            add_tiles_bcast_rows(matmul_partials_cb, bias_cb_id, i, bcast_tile_i, i);
                            ++ bcast_tile_i;
                            ++ i;
                        }
                    }

                    #ifdef SFPU_OP_INIT_ACTIVATION
                    for (uint32_t i = 0; i < out_subblock_num_tiles; ++ i) {
                        SFPU_OP_FUNC_ACTIVATION
                    }
                    #endif
                    tile_regs_commit();
                    // do not pop front bias as it may be used again for subsequent blocks
                    //DPRINT << "conv_compute_369 " << ENDL();
                    cb_pop_front(matmul_partials_cb, out_subblock_num_tiles);

                    cb_reserve_back(untilize_mode_out_cb_id, out_subblock_num_tiles);
                    DPRINT << "conv_compute_373 " << ENDL();
                    tile_regs_wait();
                    DPRINT << "conv_compute_373.2 " << ENDL();
                    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                        pack_tile(i, untilize_mode_out_cb_id);
                        DPRINT << "conv_compute_373.3  " << i << ENDL();
                    }
                    DPRINT << "conv_compute_380 " << ENDL();
                    tile_regs_release();
                    //DPRINT << "conv_compute_382 " << ENDL();
                    cb_push_back(untilize_mode_out_cb_id, out_subblock_num_tiles);

                    in1_index_subblock_offset += out_subblock_w;
                    DPRINT << "conv_compute_386 " << ENDL();

                } // for in1_num_subblocks
            } // in0_num_subblocks
            #endif
            if constexpr(untilize_out) {
                //DPRINT << "conv_compute_388 " << ENDL();
                #if defined PACKER_L1_ACC and not defined FUSE_BIAS
                pack_reconfig_l1_acc(0);
                pack_reconfig_data_format(matmul_partials_cb, out_cb_id);
                #endif
                #ifdef PACK_RELU
                PACK(( llk_pack_relu_config(ReluType::NO_RELU) ));
                #endif
                #ifndef FUSE_BIAS
                unpack_reconfig_data_format_srca(in1_cb_id, matmul_partials_cb);
                #endif
                pack_untilize_dst_init_short<out_subblock_w, out_block_w>(out_cb_id);
                copy_tile_to_dst_init_short();
                for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                    //DPRINT << "conv_compute_387 " << in0_subblock_i << ENDL();
                    reblock_and_untilize<out_subblock_w,out_block_w> (
                        in1_num_subblocks,
                        out_subblock_num_tiles,
                        out_subblock_h,
                        matmul_partials_cb,
                        out_cb_id);
                }
                pack_untilize_uninit(matmul_partials_cb);
            }
            if constexpr((in1_num_blocks_w > 1 || in0_num_blocks_h > 1)) {
                #ifdef FUSE_BIAS
                //DPRINT << "FUSE_BIAS_399 " << ENDL();
                unpack_reconfig_data_format(matmul_partials_cb, in1_cb_id, bias_cb_id, mm_in0_cb_id);
                #else
                //DPRINT << "FUSE_BIAS_402 " << ENDL();
                unpack_reconfig_data_format_srca(matmul_partials_cb, in1_cb_id);
                #endif

                if constexpr (!tilize_in0) {
                    mm_block_init_short(mm_in0_cb_id, in1_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
                }
            }
        } // for in0_num_blocks_h
        #ifdef FUSE_BIAS
            //DPRINT << "FUSE_BIAS_412 " << ENDL();
            bias_block_offset += in1_block_w;
        #endif
    } // for in1_num_blocks_w

    //DPRINT << "compute complete " << ENDL();
} // MAIN
} // NAMESPACE
