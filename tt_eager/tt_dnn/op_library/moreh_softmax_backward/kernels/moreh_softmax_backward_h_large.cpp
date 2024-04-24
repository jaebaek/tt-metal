// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    constexpr int dst_mask = 1;

    constexpr auto cb_y = tt::CB::c_in0;
    constexpr auto cb_dy = tt::CB::c_in1;
    constexpr auto cb_bcast_scaler = tt::CB::c_in2;
    constexpr auto cb_mask = tt::CB::c_in3;
    constexpr auto cb_dx = tt::CB::c_out0;

    constexpr auto cb_inter0 = tt::CB::c_intermed0;
    constexpr auto cb_sum = tt::CB::c_intermed1;
    constexpr auto cb_add = tt::CB::c_intermed3;

    binary_op_init_common(cb_y, cb_bcast_scaler);

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {

        // This for loop should not affect the result.
        for (uint32_t h = 0; h < Ht; ++h) {
            if (h == 0) {
                acquire_dst(tt::DstMode::Half);
                cb_reserve_back(cb_add, onetile);
                cb_wait_front(cb_dy, 1);

                copy_tile_init();
                copy_tile(cb_dy, 0, dst0);
                pack_tile(dst0, cb_add);

                cb_pop_front(cb_dy, onetile);
                cb_push_back(cb_add, onetile);
                release_dst(tt::DstMode::Half);
            }
            else {
                if (h == Ht - 1) {
                    acquire_dst(tt::DstMode::Half);

                    cb_reserve_back(cb_inter0, onetile);
                    cb_wait_front(cb_dy, onetile);
                    cb_wait_front(cb_mask, onetile);

                    copy_tile_init();
                    copy_tile(cb_dy, 0, dst0);

                    copy_tile_init();
                    copy_tile(cb_mask, 0, dst_mask);

                    pack_tile(dst0, cb_inter0);

                    cb_pop_front(cb_dy, onetile);

                    cb_push_back(cb_inter0, onetile);

                    release_dst(tt::DstMode::Half);

                    acquire_dst(tt::DstMode::Half);
                    cb_reserve_back(cb_add, onetile);
                    cb_wait_front(cb_add, onetile);
                    cb_wait_front(cb_inter0, onetile);

                    add_tiles_init();
                    add_tiles(cb_add, cb_inter0, 0, 0, dst0);

                    pack_tile(dst0, cb_add);

                    cb_pop_front(cb_add, onetile);
                    cb_pop_front(cb_inter0, onetile);

                    cb_push_back(cb_add, onetile);
                    release_dst(tt::DstMode::Half);
                } else {
                    acquire_dst(tt::DstMode::Half);

                    cb_reserve_back(cb_add, onetile);
                    cb_wait_front(cb_dy, 1);

                    copy_tile_init();
                    copy_tile(cb_dy, 0, dst0);
                    pack_tile(dst0, cb_add);

                    cb_pop_front(cb_dy, onetile);
                    cb_push_back(cb_add, onetile);

                    release_dst(tt::DstMode::Half);
                }
            }
        }

        // This reduce should not affect the result.
        acquire_dst(tt::DstMode::Half);

        cb_reserve_back(cb_sum, onetile);
        cb_wait_front(cb_bcast_scaler, onetile);

        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        for (uint32_t x = 0; x < 1; ++x) {
            cb_wait_front(cb_add, x + 1);  // must be a cumulative wait for correctness

            constexpr uint32_t bcast_scaler0 = 0;  // 0th index from bcast_scaler CB
            reduce_tile(cb_add, cb_bcast_scaler, x, bcast_scaler0, dst0);
        }
        cb_pop_front(cb_add, onetile);

        reduce_revert_delta();
        pack_tile(dst0, cb_sum);
        cb_push_back(cb_sum, onetile);

        release_dst(tt::DstMode::Half);

        // Only this for loop should affect the result.
        for (uint32_t h = 0; h < Ht; ++h) {
            constexpr auto cb_tmp0 = tt::CB::c_intermed0;
            acquire_dst(tt::DstMode::Half);

            cb_reserve_back(cb_tmp0, onetile);
            cb_wait_front(cb_y, 1);

            copy_tile_init();
            copy_tile(cb_y, 0, dst0);
            pack_tile(dst0, cb_tmp0);

            cb_pop_front(cb_y, onetile);
            cb_push_back(cb_tmp0, onetile);

            release_dst(tt::DstMode::Half);

            // dy - y
            acquire_dst(tt::DstMode::Half);

            cb_reserve_back(cb_dx, onetile);
            cb_wait_front(cb_dy, onetile);
            cb_wait_front(cb_tmp0, onetile);

            sub_tiles_init();
            sub_tiles(cb_dy, cb_tmp0, 0, 0, dst0);

            pack_tile(dst0, cb_dx);

            cb_pop_front(cb_dy, onetile);
            cb_pop_front(cb_tmp0, onetile);

            cb_push_back(cb_dx, onetile);

            release_dst(tt::DstMode::Half);
        }

        cb_pop_front(cb_sum, onetile);
    }
}
}  // namespace NAMESPACE
