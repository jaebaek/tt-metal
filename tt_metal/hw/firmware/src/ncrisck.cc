// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "stream_io_map.h"
#ifdef PERF_DUMP
#include "risc_perf.h"
#endif
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "dataflow_api.h"
#include "tensix_functions.h"
#include "c_tensix_core.h"

#include "kernel.cpp"


#include "debug/status.h"
#include "debug/dprint.h"

uint8_t noc_index = NOC_INDEX;

const uint32_t use_multi_noc = true;
const uint32_t noc_index_to_dram_bank_map[NUM_DRAM_BANKS] __attribute__((used)) = {
  NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX,
  1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX
};

// const uint32_t noc_index_to_dram_bank_map[NUM_DRAM_BANKS] __attribute__((used)) = {
//     NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX,
//     NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX };

// const uint32_t noc_index_to_dram_bank_map[NUM_DRAM_BANKS] __attribute__((used)) = {
//     1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX,
//     1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX };

uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];

void kernel_launch() {

  DeviceZoneScopedMainChildN("NCRISC-KERNEL");
#if defined(DEBUG_NULL_KERNELS) && !defined(DISPATCH_KERNEL)
#ifdef KERNEL_RUN_TIME
    uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUN_TIME;
    while (c_tensix_core::read_wall_clock() < KERNEL_RUN_TIME);
#endif
#else
    firmware_kernel_common_init((void tt_l1_ptr *)MEM_NCRISC_INIT_LOCAL_L1_BASE);

    // DPRINT << noc_index_to_dram_bank_map[0] << ENDL();

    noc_local_state_init(0);
    noc_local_state_init(1);

    kernel_main();
#endif
}
