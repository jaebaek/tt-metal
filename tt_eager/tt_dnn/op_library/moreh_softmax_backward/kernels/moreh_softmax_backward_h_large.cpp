// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL

#include "tt_eager/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

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
                ACQ();
                copy_tile_to_cb(cb_dy, cb_add);
                REL();
            }
            else {
                if (h == Ht - 1) {
                    ACQ();
                    constexpr uint32_t onetile = 1;
                    constexpr int dst0 = 0;
                    constexpr int dst_mask = 1;

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

                    REL();

                    ACQ();
                    add_tiles_to_cb(cb_add, cb_inter0, cb_add);
                    REL();
                } else {
                    ACQ();
                    copy_tile_to_cb(cb_dy, cb_add);
                    REL();
                }
            }
        }

        // This reduce should not affect the result.
        ACQ();
        reduce_tile_to_cb(REDUCE_OP, REDUCE_DIM, cb_add, cb_bcast_scaler, cb_sum, 1, /*pop0=*/1, /*pop1=*/0);
        REL();

        // Only this for loop should affect the result.
        for (uint32_t h = 0; h < Ht; ++h) {
            constexpr auto cb_tmp0 = tt::CB::c_intermed0;
            ACQ();
            copy_tile_to_cb(cb_y, cb_tmp0);
            REL();

            // dy - y
            ACQ();
            sub_tiles_to_cb(cb_dy, cb_tmp0, cb_dx);
            REL();
        }

        cb_pop_front(cb_sum, onetile);
    }
}
}  // namespace NAMESPACE
