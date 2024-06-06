// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "dev_msgs.h"
#include "stream_io_map.h"
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "risc_attribs.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "circular_buffer.h"

#include "debug/status.h"
// clang-format on

uint32_t halt_stack_ptr_save;

tt_l1_ptr mailboxes_t *const mailboxes = (tt_l1_ptr mailboxes_t *)(MEM_MAILBOX_BASE);

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t atomic_ret_val __attribute__((section("l1_data"))) __attribute__((used));

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
    uint32_t wIndex __attribute__((used));
    uint32_t stackSize __attribute__((used));
    uint32_t sums[SUM_COUNT] __attribute__((used));
    uint32_t sumIDs[SUM_COUNT] __attribute__((used));
    uint16_t core_flat_id __attribute__((used));
    uint32_t nocWriteSize __attribute__((used));
    uint32_t *nocWriteBuffer __attribute__((used));
    uint32_t *nocWriteIndex __attribute__((used));
}
#endif

inline __attribute__((always_inline)) void set_ncrisc_resume_addr() {
#ifdef NCRISC_HAS_IRAM
    extern "C" void ncrisc_resume(void);
    mailboxes->ncrisc_halt.resume_addr = (uint32_t)ncrisc_resume;
#endif
}

inline __attribute__((always_inline)) void halt_ncrisc_with_iram() {
#ifdef NCRISC_HAS_IRAM
    extern "C" void notify_brisc_and_halt(uint32_t status);
    notify_brisc_and_halt(RUN_SYNC_MSG_DONE);
#endif
}

int main(int argc, char *argv[]) {
    DEBUG_STATUS("I");

    disable_lowcache();

    int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
    l1_to_local_mem_copy((uint *)__ldm_data_start, (uint tt_l1_ptr *)MEM_NCRISC_INIT_LOCAL_L1_BASE, num_words);

    risc_init();

    // If NCRISC has IRAM it needs to halt before BRISC copies data from L1 to IRAM
    // Need to save address to jump to after BRISC resumes NCRISC
    set_ncrisc_resume_addr();
    mailboxes->ncrisc_halt.resume_addr = (uint32_t)ncrisc_resume;

    // Cleanup profiler buffer incase we never get the go message
    while (1) {
        DEBUG_STATUS("W");
        halt_ncrisc_with_iram();
        DeviceZoneScopedMainN("NCRISC-FW");

        setup_cb_read_write_interfaces(0, mailboxes->launch.max_cb_index, true, true);

        DEBUG_STATUS("R");
        kernel_init();
        DEBUG_STATUS("D");
    }

    return 0;
}
