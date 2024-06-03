// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_logical_not_noti.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_logical_not_unary_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::logical_not_unary, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_logical_not_unary_op(uint dst_index) {
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE>(
        ckernel::sfpu::calculate_logical_not_unary<APPROXIMATE>,
        ckernel::sfpu::calculate_logical_not_unary<APPROXIMATE>,
        dst_index,
        (int)VectorMode::RC);
}

}  // namespace ckernel
