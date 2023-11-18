// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_matmul.h"

/*************************************************************************
 * LLK MATMUL
 *************************************************************************/

template <int NUM_FIDELITY_PHASES, DstTileFaceLayout FaceLayout = DstTileFaceLayout::ColMajor>
inline void llk_math_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    const std::uint32_t in0_id = get_operand_id(operandA);
    const std::uint32_t in1_id = get_operand_id(operandB);

    const bool partial_face = get_operand_partial_face(in0_id);

    const auto unpack_tile_dims = get_operand_tile_dims(in0_id);
    const std::uint32_t in0_tile_r_dim = unpack_tile_dims[ckernel::TileDim::R_IDX];
    const std::uint32_t in0_tile_c_dim = unpack_tile_dims[ckernel::TileDim::C_IDX];
    const std::uint32_t in1_tile_r_dim = unpack_tile_dims[ckernel::TileDim::R_IDX];
    const std::uint32_t in1_tile_c_dim = unpack_tile_dims[ckernel::TileDim::C_IDX];

#ifdef ARCH_GRAYSKULL
    _llk_math_matmul_init_<NUM_FIDELITY_PHASES, FaceLayout>(
        in0_tile_r_dim,
        in0_tile_c_dim,
        in1_tile_r_dim,
        in1_tile_c_dim,
        partial_face,
        transpose,
        ct_dim,
        rt_dim,
        kt_dim);
#else
    _llk_math_matmul_init_<NUM_FIDELITY_PHASES>(
        in0_tile_r_dim,
        in0_tile_c_dim,
        in1_tile_r_dim,
        in1_tile_c_dim,
        partial_face,
        transpose,
        ct_dim,
        rt_dim,
        kt_dim);
#endif
}

template <int NUM_FIDELITY_PHASES, DstTileFaceLayout FaceLayout = DstTileFaceLayout::ColMajor>
inline void llk_math_matmul(
    uint dst_index,
    const bool transpose = false,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
#ifdef ARCH_GRAYSKULL
    _llk_math_matmul_<NUM_FIDELITY_PHASES, FaceLayout>(dst_index, transpose, ct_dim, rt_dim, kt_dim);
#else
    _llk_math_matmul_<NUM_FIDELITY_PHASES>(dst_index, transpose, ct_dim, rt_dim, kt_dim);
#endif
}
