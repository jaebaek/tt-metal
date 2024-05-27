// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_log.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log, APPROXIMATE>(sfpu::log_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE>(
        ckernel::sfpu::calculate_log<APPROXIMATE, false>,
        ckernel::sfpu::calculate_log<APPROXIMATE, false>,
        dst_index,
        vector_mode,
        0);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log_with_base_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log_with_base, APPROXIMATE>(sfpu::log_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log_with_base(
    uint dst_index, uint base_scale, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE>(
        ckernel::sfpu::calculate_log<APPROXIMATE, true>,
        ckernel::sfpu::calculate_log<APPROXIMATE, true>,
        dst_index,
        vector_mode,
        base_scale);
}

}  // namespace ckernel
