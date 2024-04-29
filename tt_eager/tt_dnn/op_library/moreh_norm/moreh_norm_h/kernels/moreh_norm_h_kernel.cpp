// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/kernels/compute/moreh_common.hpp"
#include "debug/dprint.h"

namespace NAMESPACE {

void MAIN {
    int i{0};
    const auto num_cols_per_core = get_arg_val<uint32_t>(i++);
    const auto Ht = get_arg_val<uint32_t>(i++);
    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);

    std::uint8_t input_id{tt::CB::c_in0};
    const auto cb_x = input_id++;                // input

    std::uint8_t output_id{tt::CB::c_out0};
    const auto cb_y = output_id++;  // output

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0);

    constexpr uint32_t TILE_H = 32;

    DPRINT_PACK( DPRINT << "EXECUTE POWER \n";)

    // The following code produces the correct output.
    // ACQ();
    // cb_wait_front(cb_x, onetile);
    // cb_reserve_back(cb_y, onetile);

    // // copy
    // copy_tile_init();
    // copy_tile(cb_x, 0, dst0);

    // // power
    // power_tile_init();
    // power_tile(dst0, p);

    // pack_tile(dst0, cb_y);

    // cb_push_back(cb_y, onetile);

    // cb_pop_front(cb_x, onetile);
    // REL();


    DPRINT_PACK(DPRINT << "num_cols_per_core " << num_cols_per_core << "\n"; )
    DPRINT_PACK(DPRINT << "Ht " << Ht << "\n"; )
    // This loop only runs once.
    for (uint32_t col_idx = 0; col_idx < num_cols_per_core; ++col_idx) {
        // This loop only runs once.
        for (uint32_t row_idx = 0; row_idx < Ht; ++row_idx) {
            DPRINT_PACK( DPRINT << "EXECUTE POWER \n";)
            // The following code is identical to the above but produces incorrect output.
            ACQ();
            cb_wait_front(cb_x, onetile);
            cb_reserve_back(cb_y, onetile);

            // copy
            copy_tile_init();
            copy_tile(cb_x, 0, dst0);

            // power
            power_tile_init();
            power_tile(dst0, p);

            pack_tile(dst0, cb_y);

            cb_push_back(cb_y, onetile);

            cb_pop_front(cb_x, onetile);
            REL();
        }
    }

}  // void MAIN
}  // namespace NAMESPACE
