/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include <climits>

#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC)
#include "risc_common.h"
#include "dataflow_api.h"
#else
#include "ckernel.h"
#endif

#ifdef PROFILE_KERNEL
#include "debug_print_buffer.h" // only needed because the address is shared
#endif

#include "hostdevcommon/profiler_common.h"
#include "src/firmware/riscv/common/risc_attribs.h"

namespace kernel_profiler{

    extern uint32_t wIndex;

#if defined(COMPILE_FOR_BRISC)
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_BR;
    uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_BR;
#elif defined(COMPILE_FOR_NCRISC)
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_NC;
    uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_NC;
#elif COMPILE_FOR_TRISC == 0
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_T0;
    uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_T0;
#elif COMPILE_FOR_TRISC == 1
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_T1;
    uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_T1;
#elif COMPILE_FOR_TRISC == 2
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_T2;
    uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_T2;
#endif

    inline __attribute__((always_inline)) void init_profiler()
    {
#if defined(PROFILE_KERNEL)

        volatile uint32_t *profiler_control_buffer = reinterpret_cast<uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
        profiler_control_buffer[deviceBufferEndIndex] = 0;

        wIndex = CUSTOM_MARKERS;
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_time(uint32_t timer_id)
    {
#if defined(PROFILE_KERNEL)
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC)
        uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#else
        uint32_t time_L = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#endif
        volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(profilerBuffer);
        uint32_t index = wIndex;
        buffer[index] = ((time_H & 0x0000FFFF) | (timer_id << 16));
        buffer[index+1] = time_L;
        wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_time_guaranteed_event(uint32_t index)
    {
#if defined(PROFILE_KERNEL)
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC)
        uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#else
        uint32_t time_L = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#endif
        volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(profilerBuffer);
        buffer[index] = ((time_H & 0x0000FFFF) | (index << 15)); // index is devided by 2
        buffer[index+1] = time_L;
#endif //PROFILE_KERNEL
    }


    inline __attribute__((always_inline)) void mark_time_once(uint32_t timer_id, bool * one_time)
    {
#if defined(PROFILE_KERNEL)
        if (*one_time)
        {
            mark_time(timer_id);
        }
        *one_time = false;
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_BR_fw_first_start()
    {
#if defined(PROFILE_KERNEL) & defined(COMPILE_FOR_BRISC)
        uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);

        volatile uint32_t *profiler_control_buffer = reinterpret_cast<uint32_t*>(PROFILER_L1_BUFFER_CONTROL);

        profiler_control_buffer[FW_RESET_L] = time_L;
        profiler_control_buffer[FW_RESET_H] = time_H;
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_fw_start()
    {
#if defined(PROFILE_KERNEL)
        mark_time_guaranteed_event(FW_START);
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_fw_end()
    {
#if defined(PROFILE_KERNEL)
        mark_time_guaranteed_event(FW_END);
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_kernel_start()
    {
#if defined(PROFILE_KERNEL)
        mark_time_guaranteed_event(KERNEL_START);
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_kernel_end()
    {
#if defined(PROFILE_KERNEL)
        mark_time_guaranteed_event(KERNEL_END);
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void finish()
    {
#if defined(PROFILE_KERNEL)
        for (uint32_t i = 0; i < (wIndex % NOC_ALIGNMENT_FACTOR); i++)
        {
            mark_time(PADDING_MARKER);
        }
        volatile uint32_t *profiler_control_buffer = reinterpret_cast<uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
        profiler_control_buffer[kernel_profiler::deviceBufferEndIndex] = wIndex;
#endif //PROFILE_KERNEL
    }
    inline __attribute__((always_inline)) void send_profiler_data_to_host()
    {
#if defined(PROFILE_KERNEL) && defined(COMPILE_FOR_BRISC)
        volatile uint32_t *profiler_control_buffer = reinterpret_cast<uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
        volatile  uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(profilerBuffer);

        const uint32_t NOC_ID_MASK = (1 << NOC_ADDR_NODE_ID_BITS) - 1;
        uint32_t noc_id = noc_local_node_id() & 0xFFF;
        uint32_t dram_noc_x = noc_id & NOC_ID_MASK;
        uint32_t dram_noc_y = (noc_id >> NOC_ADDR_NODE_ID_BITS) & NOC_ID_MASK;

        uint32_t core_flat_id = get_flat_id(dram_noc_x, dram_noc_y);

        finish();


        int hostIndex;
        int deviceIndex;
        for (hostIndex = kernel_profiler::HOST_BUFFER_END_INDEX_BR, deviceIndex = kernel_profiler::DEVICE_BUFFER_END_INDEX_BR;
                (hostIndex <= kernel_profiler::HOST_BUFFER_END_INDEX_T2) && (deviceIndex <= kernel_profiler::DEVICE_BUFFER_END_INDEX_T2);
                hostIndex++, deviceIndex++)
        {
            uint32_t currEndIndex =
                profiler_control_buffer[deviceIndex] +
                profiler_control_buffer[hostIndex];

            uint32_t huge_page_address =
                PROFILER_HUGE_PAGE_ADDRESS +
                (core_flat_id) * PROFILER_RISC_COUNT * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                hostIndex * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                profiler_control_buffer[hostIndex] * sizeof(uint32_t);

            if ( currEndIndex < PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC)
            {
                uint64_t pcie_buffer_dst_noc_addr = get_noc_addr(0, 4, huge_page_address);
                noc_async_write(
                        PROFILER_L1_BUFFER_BR + hostIndex * PROFILER_L1_BUFFER_SIZE,
                        pcie_buffer_dst_noc_addr,
                        profiler_control_buffer[deviceIndex] * sizeof(uint32_t));

                noc_async_write_barrier();
                profiler_control_buffer[hostIndex] = currEndIndex;
            }
            else
            {
                profiler_control_buffer[hostIndex] = PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC+1;
            }
        }
#endif //PROFILE_KERNEL
    }
}
