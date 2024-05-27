// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_add1.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_add1_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::add1, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_add1(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE>(
        ckernel::sfpu::calculate_add1<APPROXIMATE>, ckernel::sfpu::calculate_add1<APPROXIMATE>, dst_index, vector_mode);
}

}  // namespace ckernel
