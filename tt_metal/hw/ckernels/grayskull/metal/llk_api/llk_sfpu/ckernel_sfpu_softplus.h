// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_log.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
sfpi_inline void calculate_softplus(uint param0, uint param1, uint param2) {
    vFloat beta = Converter::to_float(param0);
    vFloat beta_reciprocal = Converter::to_float(param1);
    vFloat threshold = Converter::to_float(param2);
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0] * beta;
        v_if(a >= threshold) {
            dst_reg++;
            return;
        }
        v_endif;

        a = calculate_exponential_body<APPROXIMATION_MODE>(a) + 1.0f;

        dst_reg[0] = a;
        _calculate_log_body_<false>(0);
        a = beta_reciprocal * dst_reg[0];

        dst_reg[0] = a;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
sfpi_inline void softplus_init() {
    exp_init<false>();
}

}  // namespace sfpu
}  // namespace ckernel
