// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_mask.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_mask_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::mask, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_mask(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    constexpr int first_iterations = 1;
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE>(
        ckernel::sfpu::calculate_mask<APPROXIMATE, first_iterations>,
        ckernel::sfpu::calculate_mask<APPROXIMATE>,
        dst_index,
        vector_mode);
}

}  // namespace ckernel
