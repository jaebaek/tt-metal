// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp16b_to_uint32()
{
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat in = dst_reg[0];

        // check sign
        v_if (in <= 0) {
            dst_reg[0] = 0;
        } v_else {
            // extract exponent
            vInt exp = exexp(in);

            v_if (exp < 0) {
                dst_reg[0] = 0;
            } v_elseif (exp > 31) {
                // set to uint32 max value in case of overflow
                vInt tmp = 2147483647;
                dst_reg[0] = setsgn(reinterpret<vFloat>(tmp), 1);
            } v_else {
                // extract mantissa
                vInt man = exman8(in);
                // shift the mantissa by (23-exponent) to the right
                vInt shift = exp - 23;
                man = shft(reinterpret<vUInt>(man), shift);
                dst_reg[0] = man;
            } v_endif
        } v_endif

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp16b_to_uint16()
{
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPENCC(0,0,0,0);
        TTI_SFPLOAD(0,0,3,0);
        TTI_SFPSETCC(0,0,0,0);
        TTI_SFPLOADI(0,0,0);
        TTI_SFPENCC(0,0,0,0);
        TTI_SFP_STOCH_RND(0,0,2,0,1,14);
        TTI_SFPSTORE(1,6,3,0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp16b()
{
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(0,6,3,0);
        TTI_SFPCAST(0,1,0);
        TTI_SFPSTORE(1,2,3,0);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
