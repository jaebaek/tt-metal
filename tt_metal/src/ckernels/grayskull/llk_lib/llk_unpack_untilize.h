/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "llk_io_unpack.h"
#include "llk_param_structs.h"

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::unpacker;

inline void llk_unpack_untilize_mop_config() {
    static constexpr uint unpack_srca =
        TT_OP_UNPACR(SrcA, 0b01000001, 0, 0, 0, 0, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_addcr = TT_OP_ADDRCRZW(0b001, 0, 0, 0, 0, 0b0001);
    static constexpr uint unpack_addr_offset =
        TT_OP_ADDDMAREG(0, p_gpr_unpack::TILE_OFFSET, p_gpr_unpack::TILE_OFFSET, p_gpr_unpack::TILE_SIZE);
    static constexpr uint unpack_wr_addr_offset = TT_OP_REG2FLOP(
        1, 0, 0, 0, THCON_SEC0_REG7_Offset_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TILE_OFFSET);
    ckernel_unpack_template tmp = ckernel_unpack_template(
        true,  // src B
        true,  // halo - just used for 4 unpacks
        unpack_srca,
        unpack_srca,
        unpack_addr_offset,
        unpack_wr_addr_offset,
        0,
        unpack_addcr,
        TT_OP_NOP);

    tmp.program(instrn_buffer);
}

inline void llk_unpack_untilize_hw_configure(const llk_unpack_untilize_params_t *unpack_untilize_params) {
    const uint32_t unpA_operand_id = get_operand_id(unpack_untilize_params->unpA_operand);
    configure_unpack_AB(unpA_operand_id, unpA_operand_id);
}

inline void llk_unpack_untilize_hw_configure_disaggregated(const std::uint32_t unpA_operand) {
    const llk_unpack_untilize_params_t unpack_untilize_params = {
        .unpA_operand = unpA_operand,
    };
    llk_unpack_untilize_hw_configure(&unpack_untilize_params);
}

inline void llk_unpack_untilize_init(const std::uint32_t operand) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t face_r_dim = 1;

    std::uint32_t unpA_ch1_x_stride = (uint) (unpack_dst_format[operand_id]&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (unpack_dst_format[operand_id]&0x3) == (uint) DataFormat::Float16 ? 2 : 1;
    std::uint32_t unpA_ch1_y_stride = FACE_R_DIM*unpA_ch1_x_stride;

    TT_SETADCXX(p_setadc::UNP0, face_r_dim*FACE_C_DIM-1, 0x0);

    unpack_tile_descriptor_u tile_descriptor;
    tile_descriptor.val[0] = 0;
    tile_descriptor.val[1] = 0;

    // Set descriptor 0
    tile_descriptor.f.in_data_format = (uint)unpack_src_format[operand_id];
    tile_descriptor.f.uncompressed = 1;
    tile_descriptor.f.x_dim = FACE_C_DIM;

    // Set descriptor 1
    tile_descriptor.f.y_dim = FACE_R_DIM;
    tile_descriptor.f.z_dim = 4;

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_descriptor.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(tile_descriptor.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG0_TileDescriptor_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_descriptor.val[1]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(tile_descriptor.val[1]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG0_TileDescriptor_ADDR32+1-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);

    std::uint32_t unpA_base_addr = ((((int)unpack_dst_format[operand_id] & 0x3) == 1) ? 0x80 : 0x40)
        << UNP0_ADDR_BASE_REG_1_Base_SHAMT;  // base address skips halo rows in srcA (ch1)
    TT_SETDMAREG(0, LOWER_HALFWORD(unpA_base_addr), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(unpA_base_addr), 0, HI_16(p_gpr_unpack::TMP0));

    std::uint32_t unpA_ch1_xy_stride = (unpA_ch1_x_stride << UNP0_ADDR_CTRL_XY_REG_1_Xstride_SHAMT) |
                                       (unpA_ch1_y_stride << UNP0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT);

    TT_SETDMAREG(0, LOWER_HALFWORD(unpA_ch1_xy_stride), 0, LO_16(p_gpr_unpack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD(unpA_ch1_xy_stride), 0, HI_16(p_gpr_unpack::TMP1));

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, UNP0_ADDR_BASE_REG_0_Base_ADDR32);
    TTI_WRCFG(p_gpr_unpack::TMP1, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32);

    // Clear context state
    TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    unp_cfg_context = 0;

    std::uint32_t tile_size_words = GET_L1_TILE_SIZE((uint)unpack_src_format[operand_id]);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size_words), 0, LO_16(p_gpr_unpack::TILE_SIZE));
    TT_SETDMAREG(0, UPPER_HALFWORD(tile_size_words), 0, HI_16(p_gpr_unpack::TILE_SIZE));
    llk_unpack_untilize_mop_config();
}

inline void llk_unpack_untilize_uninit(uint32_t operand) {
    std::uint32_t operand_id = get_operand_id(operand);
    // Check that unpacker is done (all contexts freed up) before starting hw configuration
    wait_for_idle();

    // Reset address counters
    unpacker_addr_counter_init();

    // Get pointer to registers for current state ID
    volatile uint *cfg = get_cfg_pointer();

    TT_SETADCXX(p_setadc::UNP0, FACE_R_DIM*FACE_C_DIM-1, 0x0);

    unpack_tile_descriptor_u tile_descriptor;
    tile_descriptor.val[0] = 0;
    tile_descriptor.val[1] = 0;

    // Set descriptor 0
    tile_descriptor.f.in_data_format = (uint)unpack_src_format[operand_id];
    tile_descriptor.f.uncompressed = 1;
    tile_descriptor.f.x_dim = 256;

    // Set descriptor 1
    tile_descriptor.f.y_dim = 1;
    tile_descriptor.f.z_dim = 4;

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_descriptor.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(tile_descriptor.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG0_TileDescriptor_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_descriptor.val[1]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(tile_descriptor.val[1]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG0_TileDescriptor_ADDR32+1-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);

    uint unpA_ch1_x_stride = (uint)(unpack_dst_format[operand_id] & 0x3) == (uint)DataFormat::Float32   ? 4
                             : (uint)(unpack_dst_format[operand_id] & 0x3) == (uint)DataFormat::Float16 ? 2
                                                                                                          : 1;
    uint unpA_ch1_y_stride = 16*16*unpA_ch1_x_stride;
    uint reg_val = (unpA_ch1_y_stride << UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT) |
                   (            0 << UNP0_ADDR_CTRL_XY_REG_0_Xstride_SHAMT);
    TT_SETDMAREG(0, LOWER_HALFWORD(reg_val), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(reg_val), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);

    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, UNP0_ADDR_BASE_REG_0_Base_ADDR32); // Clear base address register
    TTI_NOP; TTI_NOP;

}

template <bool first_pass = true>
inline void llk_unpack_untilize_pass(std::uint32_t operand, std::uint32_t block_tile_cols) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = cb_interface[operand_id].fifo_rd_ptr - 1;
    std::uint32_t rem_blocks_in_row = block_tile_cols;

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b0010);  // Clear l1 addr y cnt
    if constexpr (first_pass) {
        // Select bootom faces in the 2nd pass
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Z, 0);
    } else {
        // Select bootom faces in the 2nd pass
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Z, 2);
    }

    // Wait for free context
    wait_for_next_context(1);

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Get tile address
    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = base_address;

    std::uint32_t face_2xr_cnt = 0;
    for (std::uint32_t r = 0; r < FACE_HEIGHT; r++) {
        rem_blocks_in_row = block_tile_cols;  // reset remaining blocks in row

        do {
            if ((face_2xr_cnt + rem_blocks_in_row) >= (FACE_HEIGHT / 2)) {
                // Run MOP
                TT_MOP(0, 8 - face_2xr_cnt - 1, 0);                                              // Run the MOP
                TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);  // set data valid

                TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1000);  // Clear srcA addr y cnt
                rem_blocks_in_row -= (8 - face_2xr_cnt);
                face_2xr_cnt = 0;
            } else {
                TT_MOP(0, rem_blocks_in_row - 1, 0);  // Run the MOP
                face_2xr_cnt += rem_blocks_in_row;
                rem_blocks_in_row = 0;
                // if (face_2xr_cnt==FACE_HEIGHT/2) {
                //   TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); //set data valid
                //   TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1000); // Clear srcA addr y cnt
                //   face_2xr_cnt = 0;
                //}
            }
        } while (rem_blocks_in_row > 0);

        TTI_SETDMAREG(0, 0, 0, LO_16(p_gpr_unpack::TILE_OFFSET));  // Clear offset pointer
        TTI_REG2FLOP(
            1,
            0,
            0,
            0,
            THCON_SEC0_REG7_Offset_address_ADDR32 - THCON_CFGREG_BASE_ADDR32,
            p_gpr::ZERO);                 // Clear offset register
        TTI_INCADCXY(0b001, 0, 0, 1, 0);  // inc l1 addr y cnt
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

}

inline void llk_unpack_untilize(std::uint32_t operand, std::uint32_t block_c_tiles) {
    llk_unpack_untilize_pass<true>(operand, block_c_tiles);
    llk_unpack_untilize_pass<false>(operand, block_c_tiles);
}
