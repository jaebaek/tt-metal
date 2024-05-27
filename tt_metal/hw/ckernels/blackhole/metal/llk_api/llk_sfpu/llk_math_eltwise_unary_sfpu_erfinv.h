// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_erfinv.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_erfinv_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::erfinv, APPROXIMATE>(sfpu::erfinv_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_erfinv_op(uint dst_index) {
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE>(
        ckernel::sfpu::calculate_erfinv<APPROXIMATE>,
        ckernel::sfpu::calculate_erfinv<APPROXIMATE>,
        dst_index,
        (int)VectorMode::RC);
}

}  // namespace ckernel
