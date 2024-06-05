// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "watcher_server.hpp"

#include <unistd.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "dev_mem_map.h"
#include "dev_msgs.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/debug_ring_buffer_common.h"
#include "llrt/llrt.hpp"
#include "llrt/rtoptions.hpp"
#include "noc/noc_overlay_parameters.h"
#include "noc/noc_parameters.h"

namespace tt {
namespace watcher {

constexpr uint64_t DEBUG_SANITIZE_NOC_SENTINEL_OK_64 = 0xbadabadabadabada;
constexpr uint32_t DEBUG_SANITIZE_NOC_SENTINEL_OK_32 = 0xbadabada;
constexpr uint16_t DEBUG_SANITIZE_NOC_SENTINEL_OK_16 = 0xbada;
constexpr uint8_t DEBUG_SANITIZE_NOC_SENTINEL_OK_8 = 0xda;

static std::atomic<bool> enabled = false;
static std::atomic<bool> server_running = false;
static std::atomic<int> dump_count = 0;
static std::mutex watch_mutex;
static std::set<Device *> devices;
static string logfile_path = "generated/watcher/";
static string logfile_name = "watcher.log";
static FILE *logfile = nullptr;
static std::chrono::time_point start_time = std::chrono::system_clock::now();
static std::vector<string> kernel_names;
static FILE *kernel_file = nullptr;
static string kernel_file_name = "kernel_names.txt";

// Flag to signal whether the watcher server has been killed due to a thrown exception.
static std::atomic<bool> watcher_killed_due_to_error = false;

// Description of thrown exception from watcher server, used for testing purposes.
static string watcher_exception_message = "";

static double get_elapsed_secs() {
    std::chrono::time_point now_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_secs = now_time - start_time;

    return elapsed_secs.count();
}

void create_log_file() {
    FILE *f;

    const char *fmode = tt::llrt::OptionsG.get_watcher_append() ? "a" : "w";
    std::filesystem::path output_dir(tt::llrt::OptionsG.get_root_dir() + watcher::logfile_path);
    std::filesystem::create_directories(output_dir);
    string fname = output_dir.string() + watcher::logfile_name;
    if ((f = fopen(fname.c_str(), fmode)) == nullptr) {
        TT_THROW("Watcher failed to create log file\n");
    }
    log_info(LogLLRuntime, "Watcher log file: {}", fname);

    fprintf(f, "At %.3lfs starting\n", watcher::get_elapsed_secs());
    fprintf(f, "Legend:\n");
    fprintf(f, "\tComma separated list specifices waypoint for BRISC,NCRISC,TRISC0,TRISC1,TRISC2\n");
    fprintf(f, "\tI=initialization sequence\n");
    fprintf(f, "\tW=wait (top of spin loop)\n");
    fprintf(f, "\tR=run (entering kernel)\n");
    fprintf(f, "\tD=done (finished spin loop)\n");
    fprintf(f, "\tX=host written value prior to fw launch\n");
    fprintf(f, "\n");
    fprintf(f, "\tA single character status is in the FW, other characters clarify where, eg:\n");
    fprintf(f, "\t\tNRW is \"noc read wait\"\n");
    fprintf(f, "\t\tNWD is \"noc write done\"\n");
    fprintf(f, "\tnoc<n>:<risc>{a, l}=an L1 address used by NOC<n> by <riscv> (eg, local src address)\n");
    fprintf(f, "\tnoc<n>:<riscv>{(x,y), a, l}=NOC<n> unicast address used by <riscv>\n");
    fprintf(f, "\tnoc<n>:<riscv>{(x1,y1)-(x2,y2), a, l}=NOC<n> multicast address used by <riscv>\n");
    fprintf(
        f,
        "\trmsg:<c>=brisc host run message, D/H device/host dispatch; brisc NOC ID; I/G/D init/go/done; | separator; "
        "B/b enable/disable brisc; N/n enable/disable ncrisc; T/t enable/disable TRISC\n");
    fprintf(f, "\tsmsg:<c>=slave run message, I/G/D for NCRISC, TRISC0, TRISC1, TRISC2\n");
    fprintf(f, "\tk_ids:<brisc id>|<ncrisc id>|<trisc id> (ID map to file at end of section)\n");
    fprintf(f, "\n");
    fflush(f);

    watcher::logfile = f;
}

void create_kernel_file() {
    FILE *f;
    const char *fmode = tt::llrt::OptionsG.get_watcher_append() ? "a" : "w";
    std::filesystem::path output_dir(tt::llrt::OptionsG.get_root_dir() + watcher::logfile_path);
    std::filesystem::create_directories(output_dir);
    string fname = output_dir.string() + watcher::kernel_file_name;
    if ((f = fopen(fname.c_str(), fmode)) == nullptr) {
        TT_THROW("Watcher failed to create kernel name file\n");
    }
    watcher::kernel_names.clear();
    watcher::kernel_names.push_back("blank");
    fprintf(f, "0: blank\n");
    fflush(f);

    watcher::kernel_file = f;
}

static void log_running_kernels(const launch_msg_t *launch_msg) {
    log_info("While running kernels:");
    log_info(" brisc : {}", kernel_names[launch_msg->brisc_watcher_kernel_id]);
    log_info(" ncrisc: {}", kernel_names[launch_msg->ncrisc_watcher_kernel_id]);
    log_info(" triscs: {}", kernel_names[launch_msg->triscs_watcher_kernel_id]);
}

static void dump_l1_status(FILE *f, Device *device, CoreCoord core, const launch_msg_t *launch_msg) {
    // Read L1 address 0, looking for memory corruption
    std::vector<uint32_t> data;
    data = tt::llrt::read_hex_vec_from_core(device->id(), core, MEM_L1_BASE, sizeof(uint32_t));
    // XXXX TODO(pgk): get this const from llrt (jump to fw insn)
    if (data[0] != 0x2010006f) {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher found corruption at L1[0] on core {}: read {}", core.str(), data[0]);
    }
}

static const char *get_riscv_name(CoreCoord core, uint32_t type) {
    switch (type) {
        case DebugBrisc: return "brisc";
        case DebugNCrisc: return "ncrisc";
        case DebugErisc: return "erisc";
        case DebugIErisc: return "ierisc";
        case DebugTrisc0: return "trisc0";
        case DebugTrisc1: return "trisc1";
        case DebugTrisc2: return "trisc2";
        default: TT_THROW("Watcher data corrupted, unexpected riscv type on core {}: {}", core.str(), type);
    }
    return nullptr;
}

static string get_kernel_name(CoreCoord core, const launch_msg_t *launch_msg, uint32_t type) {
    switch (type) {
        case DebugBrisc:
        case DebugErisc:
        case DebugIErisc: return kernel_names[launch_msg->brisc_watcher_kernel_id];
        case DebugNCrisc: return kernel_names[launch_msg->ncrisc_watcher_kernel_id];
        case DebugTrisc0:
        case DebugTrisc1:
        case DebugTrisc2: return kernel_names[launch_msg->triscs_watcher_kernel_id];
        default:
            log_running_kernels(launch_msg);
            TT_THROW("Watcher data corrupted, unexpected riscv type on core {}: {}", core.str(), type);
    }
    return "";
}

static string get_debug_status(CoreCoord core, const launch_msg_t *launch_msg, const debug_status_msg_t *debug_status) {
    string out;

    for (int cpu = 0; cpu < num_riscv_per_core; cpu++) {
        string risc_status;
        for (int byte = 0; byte < num_status_bytes_per_riscv; byte++) {
            char v = ((char *)&debug_status[cpu])[byte];
            if (v == 0)
                break;
            if (isprint(v)) {
                risc_status += v;
            } else {
                log_running_kernels(launch_msg);
                TT_THROW(
                    "Watcher data corrupted, unexpected debug status on core {}, unprintable character {}",
                    core.str(),
                    (int)v);
            }
        }
        // Pad risc status to 4 chars for alignment
        string pad(4 - risc_status.length(), ' ');
        out += (pad + risc_status);
        if (cpu != num_riscv_per_core - 1)
            out += ',';
    }

    out += " ";
    return out;
}

static void log_waypoint(CoreCoord core, const launch_msg_t *launch_msg, const debug_status_msg_t *debug_status) {
    string out = get_debug_status(core, launch_msg, debug_status);
    out = string("Last waypoint: ") + out;
    log_info(out.c_str());
}

static string get_ring_buffer(Device *device, CoreCoord phys_core) {
    uint64_t buf_addr = RING_BUFFER_ADDR;
    if (tt::llrt::is_ethernet_core(phys_core, device->id())) {
        // Eth pcores have a different address, but only active ones.
        CoreCoord logical_core = device->logical_core_from_ethernet_core(phys_core);
        if (device->is_active_ethernet_core(logical_core)) {
            buf_addr = eth_l1_mem::address_map::ERISC_RING_BUFFER_ADDR;
        }
    }
    auto from_dev = tt::llrt::read_hex_vec_from_core(device->id(), phys_core, buf_addr, RING_BUFFER_SIZE);
    DebugRingBufMemLayout *ring_buf_data = reinterpret_cast<DebugRingBufMemLayout *>(&(from_dev[0]));
    if (ring_buf_data->current_ptr == DEBUG_RING_BUFFER_STARTING_INDEX)
        return "";

    // Latest written idx is one less than the index read out of L1.
    string out = "\n\tdebug_ring_buffer=\n\t[";
    int curr_idx = ring_buf_data->current_ptr;
    for (int count = 1; count <= RING_BUFFER_ELEMENTS; count++) {
        out += fmt::format("0x{:08x},", ring_buf_data->data[curr_idx]);
        if (count % 8 == 0) {
            out += "\n\t ";
        }
        if (curr_idx == 0) {
            if (ring_buf_data->wrapped == 0)
                break;  // No wrapping, so no extra data available
            else
                curr_idx = RING_BUFFER_ELEMENTS - 1;  // Loop
        } else {
            curr_idx--;
        }
    }
    // Remove the last comma
    out.pop_back();
    out += "]";
    return out;
}

static void log_ring_buffer(Device *device, CoreCoord core) {
    string out = get_ring_buffer(device, core);
    if (!out.empty()) {
        out = string("Last ring buffer status: ") + out;
        log_info(out.c_str());
    }
}

static std::pair<string, string> get_core_and_mem_type(Device *device, CoreCoord &noc_coord, int noc) {
    // Get the physical coord from the noc coord
    const metal_SocDescriptor &soc_d = tt::Cluster::instance().get_soc_desc(device->id());
    CoreCoord phys_core = {NOC_0_X(noc, soc_d.grid_size.x, noc_coord.x), NOC_0_Y(noc, soc_d.grid_size.y, noc_coord.y)};

    CoreType core_type;
    try {
        core_type = device->core_type_from_physical_core(phys_core);
    } catch (std::runtime_error &e) {
        // We may not be able to get a core type if the physical coords are bad.
        return {"Unknown", ""};
    }
    switch (core_type) {
        case CoreType::DRAM: return {"DRAM", "DRAM"};
        case CoreType::ETH: return {"Ethernet", "L1"};
        case CoreType::PCIE: return {"PCIe", "PCIE"};
        case CoreType::WORKER: return {"Tensix", "L1"};
        default: return {"Unknown", ""};
    }
}

static string get_noc_target_str(Device *device, CoreCoord &core, int noc, const debug_sanitize_noc_addr_msg_t *san) {
    string out = fmt::format("{} using noc{} tried to access ", get_riscv_name(core, san->which), noc);
    if (san->multicast) {
        CoreCoord target_phys_noc_core_start = {
            NOC_MCAST_ADDR_START_X(san->noc_addr), NOC_MCAST_ADDR_START_Y(san->noc_addr)};
        CoreCoord target_phys_noc_core_end = {NOC_MCAST_ADDR_END_X(san->noc_addr), NOC_MCAST_ADDR_END_Y(san->noc_addr)};
        auto type_and_mem = get_core_and_mem_type(device, target_phys_noc_core_start, noc);
        out += fmt::format(
            "{} core range w/ physical coords {}-{} {}",
            type_and_mem.first,
            target_phys_noc_core_start.str(),
            target_phys_noc_core_end.str(),
            type_and_mem.second);
    } else {
        CoreCoord target_phys_noc_core = {NOC_UNICAST_ADDR_X(san->noc_addr), NOC_UNICAST_ADDR_Y(san->noc_addr)};
        auto type_and_mem = get_core_and_mem_type(device, target_phys_noc_core, noc);
        out += fmt::format(
            "{} core w/ physical coords {} {}", type_and_mem.first, target_phys_noc_core.str(), type_and_mem.second);
    }

    out += fmt::format("[addr=0x{:08x},len={}]", NOC_LOCAL_ADDR_OFFSET(san->noc_addr), san->len);
    return out;
}

static void dump_noc_sanity_status(
    FILE *f,
    Device *device,
    CoreCoord core,
    const string &core_str,
    const launch_msg_t *launch_msg,
    int noc,
    const debug_sanitize_noc_addr_msg_t *san,
    const debug_status_msg_t *debug_status) {
    string error_msg;
    string error_reason = "Watcher detected NOC error and stopped device: ";

    switch (san->invalid) {
        case DebugSanitizeNocInvalidOK:
            if (san->noc_addr != DEBUG_SANITIZE_NOC_SENTINEL_OK_64 ||
                san->l1_addr != DEBUG_SANITIZE_NOC_SENTINEL_OK_32 || san->len != DEBUG_SANITIZE_NOC_SENTINEL_OK_32 ||
                san->multicast != DEBUG_SANITIZE_NOC_SENTINEL_OK_16 ||
                san->which != DEBUG_SANITIZE_NOC_SENTINEL_OK_16) {
                error_msg = fmt::format(
                    "Watcher unexpected noc debug state on core {}, reported valid got noc{}{{0x{:08x}, {} }}",
                    core.str().c_str(),
                    san->which,
                    san->noc_addr,
                    san->len);
                error_reason += "corrupted noc sanitization state - sanitization memory overwritten.";
            }
            break;
        case DebugSanitizeNocInvalidL1:
            error_msg = fmt::format(
                "{} using noc{} accesses local L1[addr=0x{:08x},len={}]",
                get_riscv_name(core, san->which),
                noc,
                san->l1_addr,
                san->len);
            error_reason += "bad NOC L1/reg address.";
            break;
        case DebugSanitizeNocInvalidUnicast:
            error_msg = get_noc_target_str(device, core, noc, san);
            error_reason += "bad NOC unicast transaction.";
            break;
        case DebugSanitizeNocInvalidMulticast:
            error_msg = get_noc_target_str(device, core, noc, san);
            error_reason += "bad NOC multicast transaction.";
            break;
        case DebugSanitizeNocInvalidAlignment:
            error_msg = get_noc_target_str(device, core, noc, san);
            error_msg += fmt::format(", misaligned with local L1[addr=0x{:08x}]", san->l1_addr);
            error_reason += "bad alignment in NOC transaction.";
            break;
        default:
            error_msg = fmt::format(
                "Watcher unexpected data corruption, noc debug state on core {}, unknown failure code: {}",
                core.str(),
                san->invalid);
            error_reason += "corrupted noc sanitization state - unknown failure code.";
    }

    // If we logged an error, print to stdout and throw.
    if (!error_msg.empty()) {
        log_warning(error_reason.c_str());
        log_warning("{}: {}", core_str, error_msg);
        log_waypoint(core, launch_msg, debug_status);
        log_ring_buffer(device, core);
        log_running_kernels(launch_msg);
        // Save the error string for checking later in unit tests.
        watcher::watcher_exception_message = fmt::format("{}: {}", core_str, error_msg);
        TT_THROW(error_reason);
    }
}

static void dump_assert_status(
    FILE *f,
    Device *device,
    CoreCoord core,
    const string &core_str,
    const launch_msg_t *launch_msg,
    const debug_assert_msg_t *assert_status,
    const debug_status_msg_t *debug_status) {
    switch (assert_status->tripped) {
        case DebugAssertTripped: {
            // TODO: Get rid of this once #6098 is implemented.
            std::string line_num_warning =
                "Note that file name reporting is not yet implemented, and the reported line number for the assert may "
                "be from a different file.";
            string error_msg = fmt::format(
                "{}: {} tripped an assert on line {}. Current kernel: {}. {}",
                core_str,
                get_riscv_name(core, assert_status->which),
                assert_status->line_num,
                get_kernel_name(core, launch_msg, assert_status->which).c_str(),
                line_num_warning.c_str());
            log_warning("Watcher stopped the device due to tripped assert, see watcher log for more details");
            log_warning(error_msg.c_str());
            log_waypoint(core, launch_msg, debug_status);
            log_ring_buffer(device, core);
            log_running_kernels(launch_msg);
            watcher::watcher_exception_message = error_msg;
            TT_THROW("Watcher detected tripped assert and stopped device.");
            break;
        }
        case DebugAssertOK:
            if (assert_status->line_num != DEBUG_SANITIZE_NOC_SENTINEL_OK_16 ||
                assert_status->which != DEBUG_SANITIZE_NOC_SENTINEL_OK_8) {
                TT_THROW(
                    "Watcher unexpected assert state on core {}, reported OK but got risc {}, line {}.",
                    assert_status->which,
                    assert_status->line_num);
            }
            break;
        default:
            log_running_kernels(launch_msg);
            TT_THROW(
                "Watcher data corruption, noc assert state on core {} unknown failure code: {}.\n",
                core.str(),
                assert_status->tripped);
    }
}

static void dump_pause_status(
    CoreCoord core, const debug_pause_msg_t *pause_status, std::set<std::pair<CoreCoord, riscv_id_t>> &paused_cores) {
    // Just record which cores are paused, printing handled at the end.
    for (int risc_id = 0; risc_id < DebugNumUniqueRiscs; risc_id++) {
        auto pause = pause_status->flags[risc_id];
        if (pause == 1) {
            paused_cores.insert({core, static_cast<riscv_id_t>(risc_id)});
        } else if (pause > 1) {
            TT_THROW("Watcher data corruption, pause state on core {} unknown code: {}.\n", core.str(), pause);
        }
    }
}

static void dump_ring_buffer(FILE *f, Device *device, CoreCoord core) {
    string out = get_ring_buffer(device, core);
    fprintf(f, "%s", out.c_str());
}

static void dump_run_state(FILE *f, CoreCoord core, const launch_msg_t *launch_msg, uint32_t state) {
    char code = 'U';
    if (state == RUN_MSG_INIT)
        code = 'I';
    else if (state == RUN_MSG_GO)
        code = 'G';
    else if (state == RUN_MSG_DONE)
        code = 'D';
    if (code == 'U') {
        log_running_kernels(launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected run state on core{}: {} (expected {} or {} or {})",
            core.str(),
            state,
            RUN_MSG_INIT,
            RUN_MSG_GO,
            RUN_MSG_DONE);
    } else {
        fprintf(f, "%c", code);
    }
}

static void dump_run_mailboxes(
    FILE *f, CoreCoord core, const launch_msg_t *launch_msg, const slave_sync_msg_t *slave_sync) {
    fprintf(f, "rmsg:");

    if (launch_msg->mode == DISPATCH_MODE_DEV) {
        fprintf(f, "D");
    } else if (launch_msg->mode == DISPATCH_MODE_HOST) {
        fprintf(f, "H");
    } else {
        log_running_kernels(launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected launch mode on core {}: {} (expected {} or {})",
            core.str(),
            launch_msg->mode,
            DISPATCH_MODE_DEV,
            DISPATCH_MODE_HOST);
    }

    if (launch_msg->brisc_noc_id == 0 || launch_msg->brisc_noc_id == 1) {
        fprintf(f, "%d", launch_msg->brisc_noc_id);
    } else {
        log_running_kernels(launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected brisc noc_id on core {}: {} (expected 0 or 1)",
            core.str(),
            launch_msg->brisc_noc_id);
    }

    dump_run_state(f, core, launch_msg, launch_msg->run);

    fprintf(f, "|");

    if (launch_msg->enable_brisc == 1) {
        fprintf(f, "B");
    } else if (launch_msg->enable_brisc == 0) {
        fprintf(f, "b");
    } else {
        log_running_kernels(launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected brisc enable on core {}: {} (expected 0 or 1)",
            core.str(),
            launch_msg->enable_brisc);
    }

    if (launch_msg->enable_ncrisc == 1) {
        fprintf(f, "N");
    } else if (launch_msg->enable_ncrisc == 0) {
        fprintf(f, "n");
    } else {
        log_running_kernels(launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected ncrisc enable on core {}: {} (expected 0 or 1)",
            core.str(),
            launch_msg->enable_ncrisc);
    }

    if (launch_msg->enable_triscs == 1) {
        fprintf(f, "T");
    } else if (launch_msg->enable_triscs == 0) {
        fprintf(f, "t");
    } else {
        log_running_kernels(launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected trisc enable on core {}: {} (expected 0 or 1)",
            core.str(),
            launch_msg->enable_triscs);
    }

    fprintf(f, " ");

    fprintf(f, "smsg:");
    dump_run_state(f, core, launch_msg, slave_sync->ncrisc);
    dump_run_state(f, core, launch_msg, slave_sync->trisc0);
    dump_run_state(f, core, launch_msg, slave_sync->trisc1);
    dump_run_state(f, core, launch_msg, slave_sync->trisc2);

    fprintf(f, " ");
}

static void dump_debug_status(
    FILE *f, CoreCoord core, const launch_msg_t *launch_msg, const debug_status_msg_t *debug_status) {
    string out = get_debug_status(core, launch_msg, debug_status);
    fprintf(f, "%s ", out.c_str());
}

static void dump_sync_regs(FILE *f, Device *device, CoreCoord core) {
    // Read back all of the stream state, most of it is unused
    std::vector<uint32_t> data;
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
        // XXXX TODO(PGK) get this from device
        const uint32_t OPERAND_START_STREAM = 8;
        uint32_t base = NOC_OVERLAY_START_ADDR + (OPERAND_START_STREAM + operand) * NOC_STREAM_REG_SPACE_SIZE;

        uint32_t rcvd_addr = base + STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX * sizeof(uint32_t);
        data = tt::llrt::read_hex_vec_from_core(device->id(), core, rcvd_addr, sizeof(uint32_t));
        uint32_t rcvd = data[0];

        uint32_t ackd_addr = base + STREAM_REMOTE_DEST_BUF_START_REG_INDEX * sizeof(uint32_t);
        data = tt::llrt::read_hex_vec_from_core(device->id(), core, ackd_addr, sizeof(uint32_t));
        uint32_t ackd = data[0];

        if (rcvd != ackd) {
            fprintf(f, "cb[%d](rcv %d!=ack %d) ", operand, rcvd, ackd);
        }
    }
}

static void validate_kernel_ids(
    FILE *f, std::map<int, bool> &used_kernel_names, chip_id_t device_id, CoreCoord core, const launch_msg_t *launch) {
    if (launch->brisc_watcher_kernel_id >= kernel_names.size()) {
        TT_THROW(
            "Watcher data corruption, unexpected brisc kernel id on Device {} core {}: {} (last valid {})",
            device_id,
            core.str(),
            launch->brisc_watcher_kernel_id,
            kernel_names.size());
    }
    used_kernel_names[launch->brisc_watcher_kernel_id] = true;

    if (launch->ncrisc_watcher_kernel_id >= kernel_names.size()) {
        TT_THROW(
            "Watcher data corruption, unexpected ncrisc kernel id on Device {} core {}: {} (last valid {})",
            device_id,
            core.str(),
            launch->ncrisc_watcher_kernel_id,
            kernel_names.size());
    }
    used_kernel_names[launch->ncrisc_watcher_kernel_id] = true;

    if (launch->triscs_watcher_kernel_id >= kernel_names.size()) {
        TT_THROW(
            "Watcher data corruption, unexpected trisc kernel id on Device {} core {}: {} (last valid {})",
            device_id,
            core.str(),
            launch->triscs_watcher_kernel_id,
            kernel_names.size());
    }
    used_kernel_names[launch->triscs_watcher_kernel_id] = true;
}

static void dump_core(
    FILE *f,
    std::map<int, bool> &used_kernel_names,
    Device *device,
    CoreDescriptor logical_core,
    bool is_active_eth_core,
    std::set<std::pair<CoreCoord, riscv_id_t>> &paused_cores) {
    // Watcher only treats ethernet + worker cores.
    bool is_eth_core = (logical_core.type == CoreType::ETH);
    CoreCoord core = device->physical_core_from_logical_core(logical_core.coord, logical_core.type);

    // Print device id, core coords (logical)
    string core_type = is_eth_core ? "ethnet" : "worker";
    string core_str = fmt::format(
        "Device {} {} core(x={:2},y={:2}) phys(x={:2},y={:2})",
        device->id(),
        core_type,
        logical_core.coord.x,
        logical_core.coord.y,
        core.x,
        core.y);
    fprintf(f, "%s: ", core_str.c_str());

    // Ethernet cores have a different mailbox base addr
    uint64_t mailbox_addr = MEM_MAILBOX_BASE;
    if (is_eth_core) {
        if (is_active_eth_core)
            mailbox_addr = eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE;
        else
            mailbox_addr = MEM_IERISC_MAILBOX_BASE;
    }

    std::vector<uint32_t> data;
    data = tt::llrt::read_hex_vec_from_core(device->id(), core, mailbox_addr, sizeof(mailboxes_t));
    mailboxes_t *mbox_data = (mailboxes_t *)(&data[0]);

    // Validate these first since they are used in diagnostic messages below.
    validate_kernel_ids(f, used_kernel_names, device->id(), core, &mbox_data->launch);

    // Whether or not watcher data is available depends on a flag set on the device.
    bool enabled = (mbox_data->watcher_enable == WatcherEnabled);

    if (enabled) {
        // Dump state only gathered if device is compiled w/ watcher
        if (!tt::llrt::OptionsG.watcher_status_disabled())
            dump_debug_status(f, core, &mbox_data->launch, mbox_data->debug_status);
        // Ethernet cores have firmware that starts at address 0, so no need to check it for a
        // magic value.
        if (!is_eth_core)
            dump_l1_status(f, device, core, &mbox_data->launch);
        if (!tt::llrt::OptionsG.watcher_noc_sanitize_disabled()) {
            for (uint32_t noc = 0; noc < NUM_NOCS; noc++) {
                dump_noc_sanity_status(
                    f,
                    device,
                    core,
                    core_str,
                    &mbox_data->launch,
                    noc,
                    &mbox_data->sanitize_noc[noc],
                    mbox_data->debug_status);
            }
        }
        if (!tt::llrt::OptionsG.watcher_assert_disabled())
            dump_assert_status(
                f, device, core, core_str, &mbox_data->launch, &mbox_data->assert_status, mbox_data->debug_status);
        if (!tt::llrt::OptionsG.watcher_pause_disabled())
            dump_pause_status(core, &mbox_data->pause_status, paused_cores);
    }

    // Ethernet cores don't use the launch message/sync reg
    if (!is_eth_core) {
        // Dump state always available
        dump_run_mailboxes(f, core, &mbox_data->launch, &mbox_data->slave_sync);
        if (tt::llrt::OptionsG.get_watcher_dump_all()) {
            // Reading registers while running can cause hangs, only read if
            // requested explicitly
            dump_sync_regs(f, device, core);
        }
    }

    // Eth core only reports erisc kernel id, uses the brisc field
    if (is_eth_core) {
        fprintf(f, "k_id:%d", mbox_data->launch.brisc_watcher_kernel_id);
    } else {
        fprintf(
            f,
            "k_ids:%d|%d|%d",
            mbox_data->launch.brisc_watcher_kernel_id,
            mbox_data->launch.ncrisc_watcher_kernel_id,
            mbox_data->launch.triscs_watcher_kernel_id);
    }

    // Ring buffer at the end because it can print a bunch of data
    if (enabled) {
        if (!tt::llrt::OptionsG.watcher_ring_buffer_disabled())
            dump_ring_buffer(f, device, core);
    }

    fprintf(f, "\n");

    fflush(f);
}

// noinline so that this fn exists to be called from dgb
static void __attribute__((noinline)) dump(FILE *f) {
    for (Device *device : devices) {
        if (f != stdout && f != stderr) {
            log_info(LogLLRuntime, "Watcher checking device {}", device->id());
        }

        std::set<std::pair<CoreCoord, riscv_id_t>> paused_cores;
        std::map<int, bool> used_kernel_names;
        CoreCoord grid_size = device->logical_grid_size();
        for (uint32_t y = 0; y < grid_size.y; y++) {
            for (uint32_t x = 0; x < grid_size.x; x++) {
                CoreDescriptor logical_core = {{x, y}, CoreType::WORKER};
                if (device->storage_only_cores().find(logical_core.coord) == device->storage_only_cores().end()) {
                    dump_core(f, used_kernel_names, device, logical_core, false, paused_cores);
                }
            }
        }

        for (const CoreCoord &eth_core : device->ethernet_cores()) {
            CoreDescriptor logical_core = {eth_core, CoreType::ETH};
            CoreCoord physical_core = device->ethernet_core_from_logical_core(eth_core);
            if (device->is_active_ethernet_core(eth_core)) {
                dump_core(f, used_kernel_names, device, logical_core, true, paused_cores);
            } else if (device->is_inactive_ethernet_core(eth_core)) {
                dump_core(f, used_kernel_names, device, logical_core, false, paused_cores);
            } else {
                continue;
            }
        }

        for (auto k_id : used_kernel_names) {
            fprintf(f, "k_id[%d]: %s\n", k_id.first, kernel_names[k_id.first].c_str());
        }
        fflush(f);

        // Handle any paused cores, wait for user input.
        if (!paused_cores.empty()) {
            string paused_cores_str = "Paused cores: ";
            for (auto &core_and_risc : paused_cores) {
                paused_cores_str += fmt::format(
                    "{}:{}, ", core_and_risc.first.str(), get_riscv_name(core_and_risc.first, core_and_risc.second));
            }
            paused_cores_str += "\n";
            fprintf(f, "%s", paused_cores_str.c_str());
            log_info(LogLLRuntime, "{}Press ENTER to unpause core(s) and continue...", paused_cores_str);
            if (!tt::llrt::OptionsG.get_watcher_auto_unpause()) {
                while (std::cin.get() != '\n') {
                    ;
                }
            }

            // Clear all pause flags
            for (auto &core_and_risc : paused_cores) {
                const CoreCoord &phys_core = core_and_risc.first;
                riscv_id_t risc_id = core_and_risc.second;

                // Address depends on core type
                uint64_t addr = GET_MAILBOX_ADDRESS_HOST(pause_status);
                if (tt::llrt::is_ethernet_core(phys_core, device->id())) {
                    CoreCoord logical_core = device->logical_core_from_ethernet_core(phys_core);
                    if (device->is_active_ethernet_core(logical_core))
                        addr = GET_ETH_MAILBOX_ADDRESS_HOST(pause_status);
                    else
                        addr = GET_IERISC_MAILBOX_ADDRESS_HOST(pause_status);
                }

                // Clear only the one flag that we saved, in case another one was raised on device
                auto pause_data =
                    tt::llrt::read_hex_vec_from_core(device->id(), phys_core, addr, sizeof(debug_pause_msg_t));
                auto pause_msg = reinterpret_cast<debug_pause_msg_t *>(&(pause_data[0]));
                pause_msg->flags[risc_id] = 0;
                tt::llrt::write_hex_vec_to_core(device->id(), phys_core, pause_data, addr);
            }
        }
    }
}

static void watcher_loop(int sleep_usecs) {
    TT_ASSERT(watcher::server_running == false);
    watcher::server_running = true;
    watcher::dump_count = 1;

    // Print to the user which features are disabled via env vars.
    string disabled_features = "";
    auto &disabled_features_set = tt::llrt::OptionsG.get_watcher_disabled_features();
    if (!disabled_features_set.empty()) {
        for (auto &feature : disabled_features_set) {
            disabled_features += feature + ",";
        }
        disabled_features.pop_back();
    } else {
        disabled_features = "None";
    }
    log_info(LogLLRuntime, "Watcher server initialized, disabled features: {}", disabled_features);

    while (true) {
        // Delay the amount of time specified by the user. Don't include watcher polling time to avoid the case where
        // watcher dominates the communication links due to heavy traffic.
        double last_elapsed_time = watcher::get_elapsed_secs();
        while ((watcher::get_elapsed_secs() - last_elapsed_time) < ((double) sleep_usecs) / 1000000.) {
            // Odds are this thread will be killed during the usleep, the kill signal is
            // watcher::enabled = false from the main thread.
            if (!watcher::enabled)
                break;
            usleep(1);
        }

        {
            const std::lock_guard<std::mutex> lock(watch_mutex);

            // If all devices are detached, we can turn off the server, it will be turned back on
            // when a new device is attached.
            if (!watcher::enabled)
                break;

            fprintf(logfile, "-----\n");
            fprintf(logfile, "Dump #%d at %.3lfs\n", watcher::dump_count.load(), watcher::get_elapsed_secs());

            if (devices.size() == 0) {
                fprintf(logfile, "No active devices\n");
            }

            try {
                dump(logfile);
            } catch (std::runtime_error &e) {
                // Depending on whether test mode is enabled, catch and stop server, or re-throw.
                if (tt::llrt::OptionsG.get_test_mode_enabled()) {
                    watcher::watcher_killed_due_to_error = true;
                    watcher::enabled = false;
                    break;
                } else {
                    throw e;
                }
            }

            fprintf(logfile, "Dump #%d completed at %.3lfs\n", watcher::dump_count.load(), watcher::get_elapsed_secs());
        }
        fflush(logfile);
        watcher::dump_count++;
    }

    log_info(LogLLRuntime, "Watcher thread stopped watching...");
    watcher::server_running = false;
}

}  // namespace watcher

void watcher_init(Device *device) {
    // Initialize debug status values to "unknown"
    std::vector<uint32_t> debug_status_init_val = {'X', 'X', 'X', 'X', 'X'};

    // Initialize debug sanity L1/NOC addresses to sentinel "all ok"
    std::vector<uint32_t> debug_sanity_init_val;
    debug_sanity_init_val.resize(NUM_NOCS * sizeof(debug_sanitize_noc_addr_msg_t) / sizeof(uint32_t));
    static_assert(sizeof(debug_sanitize_noc_addr_msg_t) % sizeof(uint32_t) == 0);
    debug_sanitize_noc_addr_msg_t *data =
        reinterpret_cast<debug_sanitize_noc_addr_msg_t *>(&(debug_sanity_init_val[0]));
    for (int i = 0; i < NUM_NOCS; i++) {
        data[i].noc_addr = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_64;
        data[i].l1_addr = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_32;
        data[i].len = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_32;
        data[i].which = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_16;
        data[i].multicast = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_16;
        data[i].invalid = DebugSanitizeNocInvalidOK;
    }

    // Initialize debug asserts to not tripped.
    std::vector<uint32_t> debug_assert_init_val;
    debug_assert_init_val.resize(sizeof(debug_assert_msg_t) / sizeof(uint32_t));
    static_assert(sizeof(debug_assert_msg_t) % sizeof(uint32_t) == 0);
    debug_assert_msg_t *assert_data = reinterpret_cast<debug_assert_msg_t *>(&(debug_assert_init_val[0]));
    assert_data->line_num = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_16;
    assert_data->tripped = DebugAssertOK;
    assert_data->which = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_8;

    // Initialize pause flags to 0
    std::vector<uint32_t> debug_pause_init_val;
    debug_pause_init_val.resize(sizeof(debug_pause_msg_t) / sizeof(uint32_t));
    static_assert(sizeof(debug_pause_msg_t) % sizeof(uint32_t) == 0);
    debug_pause_msg_t *pause_data = reinterpret_cast<debug_pause_msg_t *>(&(debug_pause_init_val[0]));
    for (int idx = 0; idx < DebugNumUniqueRiscs; idx++) {
        pause_data->flags[idx] = 0;
    }

    // Initialize debug ring buffer to a known init val, we'll check against this to see if any
    // data has been written.
    std::vector<uint32_t> debug_ring_buf_init_val(RING_BUFFER_SIZE / sizeof(uint32_t), 0);
    DebugRingBufMemLayout *ring_buf_data = reinterpret_cast<DebugRingBufMemLayout *>(&(debug_ring_buf_init_val[0]));
    ring_buf_data->current_ptr = DEBUG_RING_BUFFER_STARTING_INDEX;
    ring_buf_data->wrapped = 0;

    // Initialize watcher enable flag according to user setting.
    std::vector<uint32_t> watcher_enable_init_val;
    if (tt::llrt::OptionsG.get_watcher_enabled())
        watcher_enable_init_val.push_back(WatcherEnabled);
    else
        watcher_enable_init_val.push_back(WatcherDisabled);

    CoreCoord grid_size = device->logical_grid_size();

    // Initialize Debug Delay feature
    std::map<CoreCoord, debug_insert_delays_msg_t> debug_delays_val;
    for (tt::llrt::RunTimeDebugFeatures delay_feature = tt::llrt::RunTimeDebugFeatureReadDebugDelay;
         (int)delay_feature <= tt::llrt::RunTimeDebugFeatureAtomicDebugDelay;
         delay_feature = (tt::llrt::RunTimeDebugFeatures)((int)delay_feature + 1)) {
        vector<chip_id_t> chip_ids = tt::llrt::OptionsG.get_feature_chip_ids(delay_feature);
        bool this_chip_enabled = tt::llrt::OptionsG.get_feature_all_chips(delay_feature) ||
                                 std::find(chip_ids.begin(), chip_ids.end(), device->id()) != chip_ids.end();
        if (this_chip_enabled) {
            static_assert(sizeof(debug_sanitize_noc_addr_msg_t) % sizeof(uint32_t) == 0);
            debug_insert_delays_msg_t delay_setup;

            // Create the mask based on the feature
            uint32_t hart_mask = tt::llrt::OptionsG.get_feature_riscv_mask(delay_feature);
            switch (delay_feature) {
                case tt::llrt::RunTimeDebugFeatureReadDebugDelay: delay_setup.read_delay_riscv_mask = hart_mask; break;
                case tt::llrt::RunTimeDebugFeatureWriteDebugDelay:
                    delay_setup.write_delay_riscv_mask = hart_mask;
                    break;
                case tt::llrt::RunTimeDebugFeatureAtomicDebugDelay:
                    delay_setup.atomic_delay_riscv_mask = hart_mask;
                    break;
                default: break;
            }

            for (CoreType core_type : {CoreType::WORKER, CoreType::ETH}) {
                vector<CoreCoord> delayed_cores = tt::llrt::OptionsG.get_feature_cores(delay_feature)[core_type];
                for (tt_xy_pair logical_core : delayed_cores) {
                    CoreCoord phys_core;
                    bool valid_logical_core = true;
                    try {
                        phys_core = device->physical_core_from_logical_core(logical_core, core_type);
                    } catch (std::runtime_error &error) {
                        valid_logical_core = false;
                    }
                    if (valid_logical_core) {
                        // Update the masks for the core
                        if (debug_delays_val.find(phys_core) != debug_delays_val.end()) {
                            debug_delays_val[phys_core].read_delay_riscv_mask |= delay_setup.read_delay_riscv_mask;
                            debug_delays_val[phys_core].write_delay_riscv_mask |= delay_setup.write_delay_riscv_mask;
                            debug_delays_val[phys_core].atomic_delay_riscv_mask |= delay_setup.atomic_delay_riscv_mask;
                        } else {
                            debug_delays_val.insert({phys_core, delay_setup});
                        }
                    } else {
                        log_warning(
                            tt::LogMetal,
                            "TT_METAL_{}_CORES included {} core with logical coordinates {} (physical coordinates {}), which is not a valid core on device {}. This coordinate will be ignored by {} feature.",
                            tt::llrt::RunTimeDebugFeatureNames[delay_feature],
                            tt::llrt::get_core_type_name(core_type),
                            logical_core.str(),
                            valid_logical_core? phys_core.str() : "INVALID",
                            device->id(),
                            tt::llrt::RunTimeDebugFeatureNames[delay_feature]
                        );
                    }
                }
            }
        }
    }

    // Iterate over debug_delays_val and print what got configured where
    for (auto &delay : debug_delays_val) {
        log_info(
            tt::LogMetal,
            "Configured Watcher debug delays for device {}, core {}: read_delay_cores_mask=0x{:x}, "
            "write_delay_cores_mask=0x{:x}, atomic_delay_cores_mask=0x{:x}. Delay cycles: {}",
            device->id(),
            delay.first.str().c_str(),
            delay.second.read_delay_riscv_mask,
            delay.second.write_delay_riscv_mask,
            delay.second.atomic_delay_riscv_mask,
            tt::llrt::OptionsG.get_watcher_debug_delay()
            );
    }

    std::vector<uint32_t>
        debug_insert_delays_init_val_zero;  // Cores that are not involved get this value (0 for both masks)
    debug_insert_delays_init_val_zero.resize(sizeof(debug_insert_delays_msg_t) / sizeof(uint32_t));
    std::vector<uint32_t> debug_insert_delays_init_val;
    debug_insert_delays_init_val.resize(sizeof(debug_insert_delays_msg_t) / sizeof(uint32_t));

    // Initialize worker cores debug values
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = device->worker_core_from_logical_core(logical_core);
            tt::llrt::write_hex_vec_to_core(
                device->id(), worker_core, watcher_enable_init_val, GET_MAILBOX_ADDRESS_HOST(watcher_enable));
            tt::llrt::write_hex_vec_to_core(
                device->id(), worker_core, debug_status_init_val, GET_MAILBOX_ADDRESS_HOST(debug_status));
            tt::llrt::write_hex_vec_to_core(
                device->id(), worker_core, debug_sanity_init_val, GET_MAILBOX_ADDRESS_HOST(sanitize_noc));
            tt::llrt::write_hex_vec_to_core(
                device->id(), worker_core, debug_assert_init_val, GET_MAILBOX_ADDRESS_HOST(assert_status));
            tt::llrt::write_hex_vec_to_core(
                device->id(), worker_core, debug_pause_init_val, GET_MAILBOX_ADDRESS_HOST(pause_status));
            if (debug_delays_val.find(worker_core) != debug_delays_val.end()) {
                debug_insert_delays_init_val[0] = *((uint32_t *)(&debug_delays_val[worker_core]));
                tt::llrt::write_hex_vec_to_core(
                    device->id(),
                    worker_core,
                    debug_insert_delays_init_val,
                    GET_MAILBOX_ADDRESS_HOST(debug_insert_delays));
            } else {
                tt::llrt::write_hex_vec_to_core(
                    device->id(),
                    worker_core,
                    debug_insert_delays_init_val_zero,
                    GET_MAILBOX_ADDRESS_HOST(debug_insert_delays));
            }
            tt::llrt::write_hex_vec_to_core(device->id(), worker_core, debug_ring_buf_init_val, RING_BUFFER_ADDR);
        }
    }

    // Initialize ethernet cores debug values
    for (const CoreCoord &eth_core : device->ethernet_cores()) {
        // Mailbox address is different depending on active vs inactive eth cores.
        bool is_active_eth_core;
        if (device->is_active_ethernet_core(eth_core)) {
            is_active_eth_core = true;
        } else if (device->is_inactive_ethernet_core(eth_core)) {
            is_active_eth_core = false;
        } else {
            continue;
        }
        CoreCoord physical_core = device->ethernet_core_from_logical_core(eth_core);
        tt::llrt::write_hex_vec_to_core(
            device->id(),
            physical_core,
            watcher_enable_init_val,
            is_active_eth_core ? GET_ETH_MAILBOX_ADDRESS_HOST(watcher_enable)
                               : GET_IERISC_MAILBOX_ADDRESS_HOST(watcher_enable));
        tt::llrt::write_hex_vec_to_core(
            device->id(),
            physical_core,
            debug_status_init_val,
            is_active_eth_core ? GET_ETH_MAILBOX_ADDRESS_HOST(debug_status)
                               : GET_IERISC_MAILBOX_ADDRESS_HOST(debug_status));
        tt::llrt::write_hex_vec_to_core(
            device->id(),
            physical_core,
            debug_sanity_init_val,
            is_active_eth_core ? GET_ETH_MAILBOX_ADDRESS_HOST(sanitize_noc)
                               : GET_IERISC_MAILBOX_ADDRESS_HOST(sanitize_noc));
        tt::llrt::write_hex_vec_to_core(
            device->id(),
            physical_core,
            debug_assert_init_val,
            is_active_eth_core ? GET_ETH_MAILBOX_ADDRESS_HOST(assert_status)
                               : GET_IERISC_MAILBOX_ADDRESS_HOST(assert_status));
        tt::llrt::write_hex_vec_to_core(
            device->id(),
            physical_core,
            debug_pause_init_val,
            is_active_eth_core ? GET_ETH_MAILBOX_ADDRESS_HOST(pause_status)
                               : GET_IERISC_MAILBOX_ADDRESS_HOST(pause_status));

        if (debug_delays_val.find(physical_core) != debug_delays_val.end()) {
            debug_insert_delays_init_val[0] = *((uint32_t *)(&debug_delays_val[physical_core]));
            tt::llrt::write_hex_vec_to_core(
                device->id(),
                physical_core,
                debug_insert_delays_init_val,
                GET_MAILBOX_ADDRESS_HOST(debug_insert_delays));
        } else {
            tt::llrt::write_hex_vec_to_core(
                device->id(),
                physical_core,
                debug_insert_delays_init_val_zero,
                GET_MAILBOX_ADDRESS_HOST(debug_insert_delays));
        }

        tt::llrt::write_hex_vec_to_core(
            device->id(),
            physical_core,
            debug_ring_buf_init_val,
            is_active_eth_core ? eth_l1_mem::address_map::ERISC_RING_BUFFER_ADDR : RING_BUFFER_ADDR);
    }

    log_debug(LogLLRuntime, "Watcher initialized device {}", device->id());
}

void watcher_attach(Device *device) {
    const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

    if (!watcher::enabled && tt::llrt::OptionsG.get_watcher_enabled()) {
        watcher::create_log_file();
        if (!watcher::kernel_file) {
            watcher::create_kernel_file();
        }
        watcher::watcher_killed_due_to_error = false;
        watcher::watcher_exception_message = "";

        watcher::enabled = true;

        int sleep_usecs = tt::llrt::OptionsG.get_watcher_interval() * 1000;
        std::thread watcher_thread = std::thread(&watcher::watcher_loop, sleep_usecs);
        watcher_thread.detach();
    }

    if (watcher::logfile != nullptr) {
        fprintf(watcher::logfile, "At %.3lfs attach device %d\n", watcher::get_elapsed_secs(), device->id());
    }

    if (watcher::enabled) {
        log_info(LogLLRuntime, "Watcher attached device {}", device->id());
    }

    // Always register the device w/ watcher, even if disabled
    // This allows dump() to be called from debugger
    watcher::devices.insert(device);
}

void watcher_detach(Device *old) {
    {
        const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

        TT_ASSERT(watcher::devices.find(old) != watcher::devices.end());
        if (watcher::enabled && watcher::logfile != nullptr) {
            log_info(LogLLRuntime, "Watcher detached device {}", old->id());
            fprintf(watcher::logfile, "At %.3lfs detach device %d\n", watcher::get_elapsed_secs(), old->id());
        }
        watcher::devices.erase(old);
        if (watcher::enabled && watcher::devices.empty()) {
            // If no devices remain, shut down the watcher server.
            watcher::enabled = false;
            if (watcher::logfile != nullptr) {
                std::fclose(watcher::logfile);
                watcher::logfile = nullptr;
            }
            if (watcher::kernel_file != nullptr) {
                std::fclose(watcher::kernel_file);
                watcher::kernel_file = nullptr;
            }
        }
    }

    // If we shut down the watcher server, wait until it finishes up. Do this without holding the
    // lock because the watcher server may be waiting on it before it does its exit check.
    if (watcher::devices.empty())
        while (watcher::server_running) {
            ;
        }
}

int watcher_register_kernel(const string &name) {
    const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

    if (!watcher::kernel_file)
        watcher::create_kernel_file();
    int k_id = watcher::kernel_names.size();
    watcher::kernel_names.push_back(name);
    fprintf(watcher::kernel_file, "%d: %s\n", k_id, name.c_str());
    fflush(watcher::kernel_file);

    return k_id;
}

bool watcher_server_killed_due_to_error() { return watcher::watcher_killed_due_to_error; }

void watcher_server_set_error_flag(bool val) { watcher::watcher_killed_due_to_error = val; }

string watcher_server_get_exception_message() { return watcher::watcher_exception_message; }

void watcher_clear_log() { watcher::create_log_file(); }

string watcher_get_log_file_name() {
    return tt::llrt::OptionsG.get_root_dir() + watcher::logfile_path + watcher::logfile_name;
}

int watcher_get_dump_count() { return watcher::dump_count; }

void watcher_dump() {
    if (!watcher::logfile)
        watcher::create_log_file();
    watcher::dump(watcher::logfile);
}

void watcher_read_kernel_ids_from_file() {
    std::filesystem::path output_dir(tt::llrt::OptionsG.get_root_dir() + watcher::logfile_path);
    string fname = output_dir.string() + watcher::kernel_file_name;
    FILE *f;
    if ((f = fopen(fname.c_str(), "r")) == nullptr) {
        TT_THROW("Watcher failed to open kernel name file: {}\n", fname);
    }

    char *line = nullptr;
    size_t len;
    while (getline(&line, &len, f) != -1) {
        string s(line);
        s = s.substr(0, s.length() - 1);            // Strip newline
        int k_id = stoi(s.substr(0, s.find(":")));  // Format is {k_id}: {kernel}
        watcher::kernel_names.push_back(s.substr(s.find(":") + 2));
    }
}

}  // namespace tt
