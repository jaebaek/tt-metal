// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/trace/trace.hpp"
#include "tt_metal/common/core_descriptor.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "impl/debug/dprint_server.hpp"
#include "impl/debug/watcher_server.hpp"
#include "common/env_lib.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include "common/utils.hpp"
#include "llrt/llrt.hpp"
#include "dev_msgs.h"

namespace tt {

namespace tt_metal {

void ::detail::ProgramDeleter::operator()(Program *p) {
    delete p;
}

ActiveDevices Device::active_devices_;

ActiveDevices::ActiveDevices() {
}

ActiveDevices::~ActiveDevices() {
    for (size_t i = 0; i < active_devices_.size(); i++) {
        if (active_devices_[i] == ActiveState::ACTIVE) {
            TT_THROW("Process tear down with device {} still active", i);
        }
    }
}

bool ActiveDevices::activate_device(chip_id_t id) {
    bool already_initialized;
    const std::lock_guard<std::mutex> lock(lock_);
    if (this->active_devices_.size() < id + 1) {
        this->active_devices_.resize(id + 1);
        already_initialized = false;
    } else if (this->active_devices_[id] == ActiveState::ACTIVE) {
        TT_THROW("Cannot re-initialize device {}, must first call close()", id);
    } else {
        already_initialized = (this->active_devices_[id] == ActiveState::INACTIVE) ? true : false;
    }
    this->active_devices_[id] = ActiveState::ACTIVE;

    return already_initialized;
}

void ActiveDevices::deactivate_device(chip_id_t id) {
    const std::lock_guard<std::mutex> lock(lock_);
    this->active_devices_[id] = ActiveState::INACTIVE;
}

bool ActiveDevices::is_device_active(chip_id_t id) {
    if (this->active_devices_.size() < id + 1) {
        return false;
    } else {
        return this->active_devices_[id] == ActiveState::ACTIVE;
    }
}

Device::Device(
    chip_id_t device_id, const uint8_t num_hw_cqs, size_t l1_small_size, const std::vector<uint32_t> &l1_bank_remap, bool minimal, uint32_t worker_core) :
    id_(device_id), num_hw_cqs_(num_hw_cqs), worker_thread_core(worker_core), work_executor(worker_core, device_id) {
    ZoneScoped;
    TT_ASSERT(num_hw_cqs > 0 and num_hw_cqs < 3, "num_hw_cqs can be between 1 and 2");
    this->build_key_ = tt::Cluster::instance().get_harvesting_mask(device_id);
    tunnel_device_dispatch_workers_ = {};
    this->initialize(l1_small_size, l1_bank_remap, minimal);
}

void Device::initialize_cluster() {
    ZoneScoped;
    if (llrt::OptionsG.get_clear_l1()) {
        this->clear_l1_state();
    }
#ifdef TT_METAL_VERSIM_DISABLED
    int ai_clk = tt::Cluster::instance().get_device_aiclk(this->id_);
    log_info(tt::LogMetal, "AI CLK for device {} is:   {} MHz", this->id_, ai_clk);
#endif
}

void Device::initialize_allocator(size_t l1_small_size, const std::vector<uint32_t> &l1_bank_remap) {
    ZoneScoped;
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    // Construct allocator config from soc_desc
    AllocatorConfig config(
        {.num_dram_channels = static_cast<size_t>(soc_desc.get_num_dram_channels()),
         .dram_bank_size = soc_desc.dram_bank_size,
         .dram_bank_offsets = {},
         .worker_grid_size = this->logical_grid_size(),
         .worker_l1_size = static_cast<size_t>(soc_desc.worker_l1_size),
         .l1_bank_size = static_cast<size_t>(get_storage_core_bank_size(this->id_, this->num_hw_cqs_)),
         .l1_small_size = l1_small_size,
         .core_type_from_noc_coord_table = {},  // Populated later
         .worker_log_to_physical_routing_x = soc_desc.worker_log_to_physical_routing_x,
         .worker_log_to_physical_routing_y = soc_desc.worker_log_to_physical_routing_y,
         .l1_bank_remap = l1_bank_remap,
         .compute_grid_size = this->compute_with_storage_grid_size()});
    TT_FATAL(config.l1_small_size < config.l1_bank_size, "Reserved size must be less than bank size");
    TT_FATAL(
        config.l1_small_size % ADDRESS_ALIGNMENT == 0,
        "Reserved size must be aligned to ADDRESS_ALIGNMENT",
        ADDRESS_ALIGNMENT);
    // Initialize dram_offsets from soc_descriptor
    for (auto channel = 0; channel < soc_desc.get_num_dram_channels(); channel++) {
        config.dram_bank_offsets.push_back(soc_desc.get_address_offset(channel));
    }
    // Initialize core_type_from_noc_coord_table table
    for (const auto& core: soc_desc.physical_cores) {
        config.core_type_from_noc_coord_table.insert({core.first, AllocCoreType::Invalid});
    }

    for (const CoreCoord& core : tt::get_logical_compute_cores(id_, num_hw_cqs_)) {
        this->compute_cores_.insert(core);
        const auto noc_coord = this->worker_core_from_logical_core(core);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::ComputeAndStore;
    }
    for (const CoreCoord& core : tt::get_logical_storage_cores(id_, num_hw_cqs_)) {
        this->storage_only_cores_.insert(core);
        const auto noc_coord = this->worker_core_from_logical_core(core);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::StorageOnly;
    }
    for (const CoreCoord& core : tt::get_logical_dispatch_cores(id_, num_hw_cqs_)) {
        CoreType dispatch_core_type = tt::get_dispatch_core_type(id_, num_hw_cqs_);
        const auto noc_coord = this->physical_core_from_logical_core(core, dispatch_core_type);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::Dispatch;
    }
    for (const auto &core : soc_desc.get_logical_ethernet_cores()) {
        this->ethernet_cores_.insert(core);
    }

    // L1_BANKING scheme creates 1 bank per DRAM core and splits up L1 such that there are power 2 num L1 banks
    // This is the only allocator scheme supported because kernel APIs assume num L1 banks are power of 2
    TT_ASSERT(this->allocator_scheme_ == MemoryAllocator::L1_BANKING);
    this->allocator_ = std::make_unique<L1BankingAllocator>(config);
}

void Device::initialize_build() {
    ZoneScoped;

    this->build_env_.init(this->build_key(), this->arch());

    auto init_helper = [this] (bool is_fw) -> JitBuildStateSet {
        std::vector<std::shared_ptr<JitBuildState>> build_states;

        build_states.resize(arch() == tt::ARCH::GRAYSKULL ? 5 : 7);

        build_states[build_processor_type_to_index(JitBuildProcessorType::DATA_MOVEMENT).first + 0] =
            std::make_shared<JitBuildDataMovement>(this->build_env_, 0, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::DATA_MOVEMENT).first + 1] =
            std::make_shared<JitBuildDataMovement>(this->build_env_, 1, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::COMPUTE).first + 0] =
            std::make_shared<JitBuildCompute>(this->build_env_, 0, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::COMPUTE).first + 1] =
            std::make_shared<JitBuildCompute>(this->build_env_, 1, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::COMPUTE).first + 2] =
            std::make_shared<JitBuildCompute>(this->build_env_, 2, is_fw);

        if (arch() != tt::ARCH::GRAYSKULL) {
            build_states[build_processor_type_to_index(JitBuildProcessorType::ETHERNET).first + 0] =
                std::make_shared<JitBuildEthernet>(this->build_env_, 0, is_fw);
            build_states[build_processor_type_to_index(JitBuildProcessorType::ETHERNET).first + 1] =
                std::make_shared<JitBuildEthernet>(this->build_env_, 1, is_fw);
        }

       return build_states;
    };

    this->firmware_build_states_ = init_helper(true);
    this->kernel_build_states_ = init_helper(false);
}

void Device::build_firmware() {
    ZoneScoped;

    detail::GenerateDeviceHeaders(this, this->build_env_.get_out_firmware_root_path());
    jit_build_set(this->firmware_build_states_, nullptr, "");
}

void Device::initialize_firmware(CoreCoord phys_core, launch_msg_t *launch_msg) {
    ZoneScoped;

    if (llrt::is_ethernet_core(phys_core, this->id())) {
        //ethernet core.
        //Determine if its a connected or unconnected ethernet core.
        //Unconnected ethernet cores will get idle_erisc fw.
        auto active_eth_cores = this->get_active_ethernet_cores();

        if (active_eth_cores.find(logical_core_from_ethernet_core(phys_core)) != active_eth_cores.end()) {
            int eriscv_id = build_processor_type_to_index(JitBuildProcessorType::ETHERNET).first + 0;
            ll_api::memory binary_mem = llrt::get_risc_binary(firmware_build_states_[eriscv_id]->get_target_out_path(""));
            uint32_t kernel_size16 = llrt::get_binary_code_size16(binary_mem, eriscv_id);
            log_debug(LogDevice, "ERISC fw binary size: {} in bytes", kernel_size16 * 16);
            llrt::test_load_write_read_risc_binary(binary_mem, this->id(), phys_core, eriscv_id);
            llrt::launch_erisc_app_fw_on_core(this->id(), phys_core);
        } else {
            tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->id(), phys_core));
            int eriscv_id = build_processor_type_to_index(JitBuildProcessorType::ETHERNET).first + 1;
            ll_api::memory binary_mem = llrt::get_risc_binary(firmware_build_states_[eriscv_id]->get_target_out_path(""));
            uint32_t kernel_size16 = llrt::get_binary_code_size16(binary_mem, eriscv_id);
            log_debug(LogDevice, "ERISC fw binary size: {} in bytes", kernel_size16 * 16);
            llrt::test_load_write_read_risc_binary(binary_mem, this->id(), phys_core, eriscv_id);
            llrt::program_risc_startup_addr(this->id(), phys_core);
        }
    } else {
        llrt::program_risc_startup_addr(this->id(), phys_core);
        for (int riscv_id = 0; riscv_id < 5; riscv_id++) {
            ll_api::memory binary_mem =
                llrt::get_risc_binary(firmware_build_states_[riscv_id]->get_target_out_path(""));
            uint32_t kernel_size16 = llrt::get_binary_code_size16(binary_mem, riscv_id);
            if (riscv_id == 1) {
                launch_msg->ncrisc_kernel_size16 = kernel_size16;
            }
            log_debug(LogDevice, "RISC {} fw binary size: {} in bytes", riscv_id, kernel_size16 * 16);
            llrt::test_load_write_read_risc_binary(binary_mem, this->id(), phys_core, riscv_id);
        }
    }
    //This is an initialization launch message.
    //Clears launch message fields to 0 in target core L1.
    //Sets launch.run to RUN_MSG_INIT.
    llrt::write_launch_msg_to_core(this->id(), phys_core, launch_msg);
}

void Device::initialize_and_launch_firmware() {
    ZoneScoped;

    launch_msg_t launch_msg = {
        .brisc_watcher_kernel_id = 0,
        .ncrisc_watcher_kernel_id = 0,
        .triscs_watcher_kernel_id = 0,
        .ncrisc_kernel_size16 = 0,
        .mode = DISPATCH_MODE_HOST,
        .brisc_noc_id = 0,
        .enable_brisc = 0,
        .enable_ncrisc = 0,
        .enable_triscs = 0,
        .enable_erisc = 0,
        .run = RUN_MSG_INIT,
    };

    // Download to worker cores
    log_debug("Initializing firmware");
    CoreCoord grid_size = this->logical_grid_size();
    std::unordered_set<CoreCoord> not_done_cores;


    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            if (!this->storage_only_cores_.count(logical_core)) {
                CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);
                this->initialize_firmware(worker_core, &launch_msg);
                not_done_cores.insert(worker_core);
            }
        }
    }

    // Load erisc app base FW to eth cores
    for (const auto &eth_core : this->get_active_ethernet_cores()) {
        CoreCoord phys_eth_core = this->ethernet_core_from_logical_core(eth_core);
        this->initialize_firmware(phys_eth_core, &launch_msg);
    }

    for (const auto &eth_core : this->get_inactive_ethernet_cores()) {
        CoreCoord phys_eth_core = this->ethernet_core_from_logical_core(eth_core);
        this->initialize_firmware(phys_eth_core, &launch_msg);
        not_done_cores.insert(phys_eth_core);
    }

    // Barrier between L1 writes above and deassert below
    tt::Cluster::instance().l1_barrier(this->id());

    // Deassert worker cores
    for(const auto& worker_core : not_done_cores)
        tt::Cluster::instance().deassert_risc_reset_at_core(tt_cxy_pair(this->id(), worker_core));

    // Wait until fw init is done, ensures the next launch msg doesn't get
    // written while fw is still in init
    log_debug("Waiting for firmware init complete");
    llrt::internal_::wait_until_cores_done(this->id(), RUN_MSG_INIT, not_done_cores);
    log_debug("Firmware init complete");
}

void Device::clear_l1_state() {
    CoreCoord logical_grid_size = this->logical_grid_size();
    TT_ASSERT(this->l1_size_per_core() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> zero_vec(this->l1_size_per_core() / sizeof(uint32_t), 0);
    constexpr uint32_t start_address = 0;
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            detail::WriteToDeviceL1(this, logical_core, start_address, zero_vec);
        }
    }

    for (const auto &eth_core : this->get_inactive_ethernet_cores()) {
        CoreCoord physical_core = this->ethernet_core_from_logical_core(eth_core);
        std::vector<uint32_t> zero_vec_mailbox(128 / sizeof(uint32_t), 0);
        llrt::write_hex_vec_to_core(this->id(), physical_core, zero_vec_mailbox, MEM_IERISC_MAILBOX_BASE);
    }

    // Clear erisc sync info
    for (const auto &eth_core : this->get_active_ethernet_cores()) {
        CoreCoord physical_core = this->ethernet_core_from_logical_core(eth_core);
        // These L1 ranges are restricted becase UMD base routing FW uses L1 below FIRMWARE_BASE and
        // between TILE_HEADER_BUFFER_BASE to COMMAND_Q_BASE
        std::vector<uint32_t> zero_vec_above_tile_header_buffer(
            (eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::TILE_HEADER_BUFFER_BASE) /
                sizeof(uint32_t),
            0);

        llrt::write_hex_vec_to_core(
            this->id(),
            physical_core,
            zero_vec_above_tile_header_buffer,
            eth_l1_mem::address_map::TILE_HEADER_BUFFER_BASE);

        /* TODO: removing this section of code fixes the n300 hangs, what's the proper fix?
        std::vector<uint32_t> zero_vec_below_command_q_base(
            (eth_l1_mem::address_map::COMMAND_Q_BASE - eth_l1_mem::address_map::FIRMWARE_BASE) / sizeof(uint32_t), 0);

        llrt::write_hex_vec_to_core(
            this->id(), physical_core, zero_vec_below_command_q_base, eth_l1_mem::address_map::FIRMWARE_BASE);
        */
    }
    // TODO: clear idle eriscs as well
}

void Device::configure_kernel_variant(
    Program& program,
    string path,
    std::vector<uint32_t> compile_args,
    CoreCoord kernel_core,
    CoreCoord kernel_physical_core,
    CoreType dispatch_core_type,
    CoreCoord upstream_physical_core,
    CoreCoord downstream_physical_core,
    std::map<string, string> defines_in,
    bool is_active_eth_core) {

    std::map<string, string> defines = {
        {"DISPATCH_KERNEL", "1"},
        {"MY_NOC_X", std::to_string(kernel_physical_core.x)},
        {"MY_NOC_Y", std::to_string(kernel_physical_core.y)},
        {"UPSTREAM_NOC_X", std::to_string(upstream_physical_core.x)},
        {"UPSTREAM_NOC_Y", std::to_string(upstream_physical_core.y)},
        {"DOWNSTREAM_NOC_X", std::to_string(downstream_physical_core.x)},
        {"DOWNSTREAM_NOC_Y", std::to_string(downstream_physical_core.y)},
    };
    defines.insert(defines_in.begin(), defines_in.end());

    if (dispatch_core_type == CoreType::WORKER) {
        tt::tt_metal::CreateKernel(
            program,
            path,
            kernel_core,
            tt::tt_metal::DataMovementConfig {
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = NOC::NOC_0,
                .compile_args = compile_args,
                .defines = defines
            }
        );
    } else {
        tt::tt_metal::CreateKernel(
            program,
            path,
            kernel_core,
            tt::tt_metal::EthernetConfig{
                .eth_mode = is_active_eth_core ? Eth::SENDER : Eth::IDLE,
                .noc = NOC::NOC_0,
                .compile_args = compile_args,
                .defines = defines
            }
        );
    }
}

void Device::update_workers_build_settings(std::vector<std::vector<std::tuple<tt_cxy_pair, worker_build_settings_t>>> &device_worker_variants) {
    for (uint32_t dwv = 0; dwv < device_worker_variants.size(); dwv++)
    {
        if (device_worker_variants[dwv].size() == 0) {
            continue;
        }
        log_debug(tt::LogMetal, "Setting up {} Arguments", magic_enum::enum_name((tt::tt_metal::DispatchWorkerType)dwv));
        switch(dwv) {
            case PREFETCH:
            {
                uint32_t num_prefetchers = device_worker_variants[PREFETCH].size();
                TT_ASSERT(device_worker_variants[MUX].size() == 1, "Cannot have more than one Mux.");
                auto mux_settings = std::get<1>(device_worker_variants[MUX][0]);
                TT_ASSERT(num_prefetchers == mux_settings.semaphores.size(), "Mux does not have required number of semaphores for Prefetchers. Exptected = {}. Fount = {}", num_prefetchers, mux_settings.semaphores.size());
                uint32_t mux_sem = mux_settings.consumer_semaphore_id;
                for (auto&[core, settings] : device_worker_variants[PREFETCH]) {
                    auto dispatch_core_type = settings.dispatch_core_type;
                    uint32_t downstream_cb_base = mux_settings.cb_start_address + mux_settings.cb_size_bytes * mux_sem;
                    settings.upstream_cores.push_back(tt_cxy_pair(0, 0, 0));
                    settings.downstream_cores.push_back(mux_settings.worker_physical_core);
                    settings.compile_args.resize(23);
                    auto& compile_args = settings.compile_args;
                    compile_args[0]  = downstream_cb_base;
                    compile_args[1]  = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
                    compile_args[2]  = dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_pages();
                    compile_args[3]  = settings.producer_semaphore_id;
                    compile_args[4]  = mux_sem++;
                    compile_args[5]  = settings.issue_queue_start_addr;
                    compile_args[6]  = settings.issue_queue_size;
                    compile_args[7]  = dispatch_constants::PREFETCH_Q_BASE;
                    compile_args[8]  = dispatch_constants::get(dispatch_core_type).prefetch_q_size();
                    compile_args[9]  = CQ_PREFETCH_Q_RD_PTR;
                    compile_args[10] = dispatch_constants::get(dispatch_core_type).cmddat_q_base();
                    compile_args[11] = dispatch_constants::get(dispatch_core_type).cmddat_q_size();
                    compile_args[12] = dispatch_constants::get(dispatch_core_type).scratch_db_base(); // unused for prefetch_h
                    compile_args[13] = dispatch_constants::get(dispatch_core_type).scratch_db_size(); // unused for prefetch_h
                    compile_args[14] = 0; //prefetch_sync_sem unused for prefetch_h
                    compile_args[15] = dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_pages(); // prefetch_d only
                    compile_args[16] = 0; // prefetch_d only
                    compile_args[17] = 0; //prefetch_downstream_cb_sem, // prefetch_d only
                    compile_args[18] = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
                    compile_args[19] = dispatch_constants::PREFETCH_D_BUFFER_BLOCKS; // prefetch_d only
                    compile_args[20] = 2; //prefetch_h_exec_buf_sem,
                    compile_args[21] = false;  // is_dram_variant
                    compile_args[22] = true;    // is_host_variant
                }
                break;
            }
            case MUX:
            {
                uint32_t num_prefetchers = device_worker_variants[PREFETCH].size();
                TT_ASSERT(device_worker_variants[MUX].size() == 1, "Cannot have more than one Mux.");
                auto &mux_settings = std::get<1>(device_worker_variants[MUX][0]);
                TT_ASSERT(num_prefetchers == mux_settings.semaphores.size(), "Mux does not have required number of semaphores for Prefetchers. Exptected = {}. Fount = {}", num_prefetchers, mux_settings.semaphores.size());
                uint32_t mux_sem = mux_settings.consumer_semaphore_id;

                auto& compile_args = mux_settings.compile_args;
                compile_args.resize(25);
                compile_args[0] = 0; // 0: reserved
                compile_args[1] = mux_settings.cb_start_address >> 4; // 1: rx_queue_start_addr_words
                compile_args[2] = mux_settings.cb_size_bytes >> 4; // 2: rx_queue_size_words
                compile_args[3] = num_prefetchers; // 3: mux_fan_in
                uint32_t arg_index = 4;
                for (auto&[core, settings] : device_worker_variants[PREFETCH]) {
                    compile_args[arg_index++] = packet_switch_4B_pack((uint32_t)settings.worker_physical_core.x,
                                                                    (uint32_t)settings.worker_physical_core.y,
                                                                    1,
                                                                    (uint32_t)DispatchRemoteNetworkType::NOC0); // 4,5,6,7: src x info
                }
                TT_ASSERT(device_worker_variants[US_TUNNELER_REMOTE].size() == 1, "Unexpected number of ethernet tunnelers.");
                auto &tunneler_settings = std::get<1>(device_worker_variants[US_TUNNELER_REMOTE][0]);

                compile_args[8] = tunneler_settings.cb_start_address >> 4; // 8: remote_tx_queue_start_addr_words
                compile_args[9] = tunneler_settings.cb_size_bytes >> 4; // 9: remote_tx_queue_size_words
                compile_args[10] = tunneler_settings.worker_physical_core.x; // 10: remote_tx_x
                compile_args[11] = tunneler_settings.worker_physical_core.y; // 11: remote_tx_y
                compile_args[12] = 0; // 12: remote_tx_queue_id
                compile_args[13] = (uint32_t)DispatchRemoteNetworkType::NOC0; // 13: tx_network_type
                compile_args[14] = BRISC_L1_RESULT_BASE; // 14: test_results_addr
                compile_args[15] = 1024; // 15: test_results_size
                compile_args[16] = 0; // 16: timeout_cycles
                compile_args[17] = 0x0; // 17: output_depacketize
                compile_args[18] = 0x0; // 18: output_depacketize info
                arg_index = 19; // 19, 20, 21, 22: input x packetize info:
                for (auto&[core, settings] : device_worker_variants[PREFETCH]) {
                    compile_args[arg_index++] = packet_switch_4B_pack(0x1,
                                dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                                settings.producer_semaphore_id,  // upstream sem
                                mux_sem++); // local sem
                }
                compile_args[23] = packet_switch_4B_pack(0xA1, 0xA2, 0xA3, 0xA4); // 23: packetized input src id
                compile_args[24] = packet_switch_4B_pack(0xB1, 0xB2, 0xB3, 0xB4); // 24: packetized input dest id
                break;
            }
            case US_TUNNELER_REMOTE:
            {
                TT_ASSERT(device_worker_variants[US_TUNNELER_REMOTE].size() == 1, "Unexpected number of ethernet tunnelers.");
                auto &tunneler_settings = std::get<1>(device_worker_variants[US_TUNNELER_REMOTE][0]);
                bool is_tunnel_start = tunneler_settings.tunnel_stop == 0;
                auto &compile_args = tunneler_settings.compile_args;
                compile_args.resize(16);
                compile_args[0] = 0xDACADACA; // 0: endpoint_id_start_index
                compile_args[1] = 2; // tunnel_lanes. 1 => Unidirectional. 2 => Bidirectional.
                compile_args[2] = tunneler_settings.cb_start_address >> 4; // 2: rx_queue_start_addr_words
                compile_args[3] = tunneler_settings.cb_size_bytes >> 4; // 3: rx_queue_size_words

                compile_args[4] = packet_switch_4B_pack(tunneler_settings.eth_partner_physical_core.x,
                                    tunneler_settings.eth_partner_physical_core.y,
                                    0,
                                    (uint32_t)DispatchRemoteNetworkType::ETH); // 4: remote_receiver_0_info
                compile_args[6] = tunneler_settings.cb_start_address >> 4; // 6: remote_receiver_queue_start_addr_words 0
                compile_args[7] = tunneler_settings.cb_size_bytes >> 4; // 7: remote_receiver_queue_size_words 0

                if (is_tunnel_start) {
                    auto &demux_settings = std::get<1>(device_worker_variants[DEMUX][0]);
                    auto &mux_settings = std::get<1>(device_worker_variants[MUX][0]);

                    compile_args[5] = packet_switch_4B_pack(demux_settings.worker_physical_core.x,
                                        demux_settings.worker_physical_core.y,
                                        device_worker_variants[DISPATCH].size(),//num_dest_endpoints,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0); // 5: remote_receiver_1_info
                    compile_args[8] = demux_settings.cb_start_address >> 4; // 8: remote_receiver_queue_start_addr_words 1
                    compile_args[9] = demux_settings.cb_size_bytes >> 4; // 9: remote_receiver_queue_size_words 1
                    compile_args[10] = packet_switch_4B_pack(mux_settings.worker_physical_core.x,
                                        mux_settings.worker_physical_core.y,
                                        device_worker_variants[PREFETCH].size(), // mux output queue id
                                        (uint32_t)DispatchRemoteNetworkType::NOC0); // 10: remote_sender_0_info
                } else {
                    auto &mux_d_settings = std::get<1>(device_worker_variants[MUX_D][0]);
                    auto &demux_d_settings = std::get<1>(device_worker_variants[DEMUX_D][0]);

                    compile_args[5] = packet_switch_4B_pack(mux_d_settings.worker_physical_core.x,
                                        mux_d_settings.worker_physical_core.y,
                                        1,//num_dest_endpoints,
                                        (uint32_t)DispatchRemoteNetworkType::NOC0); // 5: remote_receiver_1_info
                    compile_args[8] = (mux_d_settings.cb_start_address + mux_d_settings.cb_size_bytes) >> 4; // 8: remote_receiver_queue_start_addr_words 1
                    compile_args[9] = mux_d_settings.cb_size_bytes >> 4; // 9: remote_receiver_queue_size_words 1
                    compile_args[10] = packet_switch_4B_pack(demux_d_settings.worker_physical_core.x,
                                        demux_d_settings.worker_physical_core.y,
                                        1, // demux output queue id
                                        (uint32_t)DispatchRemoteNetworkType::NOC0); // 10: remote_sender_0_info
                }

                compile_args[11] = packet_switch_4B_pack(tunneler_settings.eth_partner_physical_core.x,
                                    tunneler_settings.eth_partner_physical_core.y,
                                    3, // r tunneler output queue id
                                    (uint32_t)DispatchRemoteNetworkType::ETH); // 11: remote_sender_1_info

                compile_args[12] = 0x39000; // 12: test_results_addr
                compile_args[13] = 0x7000; // 13: test_results_size
                compile_args[14] = 0; // 14: timeout_cycles

                break;
            }
            case DEMUX:
            {
                TT_ASSERT(device_worker_variants[DEMUX].size() == 1, "Unexpected number of ethernet tunnelers.");
                auto &tunneler_settings = std::get<1>(device_worker_variants[US_TUNNELER_REMOTE][0]);
                auto &demux_settings = std::get<1>(device_worker_variants[DEMUX][0]);
                auto &dispatch_settings = std::get<1>(device_worker_variants[DISPATCH][0]);

                auto &compile_args = demux_settings.compile_args;
                compile_args.resize(30);

                compile_args[0] = 0xD1; // 0: endpoint_id_start_index
                compile_args[1] = demux_settings.cb_start_address >> 4; // 1: rx_queue_start_addr_words
                compile_args[2] = demux_settings.cb_size_bytes >> 4; // 2: rx_queue_size_words
                compile_args[3] = device_worker_variants[DISPATCH].size(); // 3: demux_fan_out

                uint32_t arg_index = 4;
                for (auto&[core, settings] : device_worker_variants[DISPATCH]) {
                    compile_args[arg_index++] = packet_switch_4B_pack((uint32_t)settings.worker_physical_core.x,
                                                                    (uint32_t)settings.worker_physical_core.y,
                                                                    0,
                                                                    (uint32_t)DispatchRemoteNetworkType::NOC0); // 4,5,6,7: remote_tx_x_info
                }
                arg_index = 8;
                for (auto&[core, settings] : device_worker_variants[DISPATCH]) {
                    compile_args[arg_index++] = settings.cb_start_address >> 4; // 8, 10, 12, 14: remote_tx_queue_start_addr_words x
                    compile_args[arg_index++] = settings.cb_size_bytes >> 4; // 9, 11, 13, 15: remote_tx_queue_size_words x
                }
                compile_args[16] = tunneler_settings.worker_physical_core.x; // 16: remote_rx_x
                compile_args[17] = tunneler_settings.worker_physical_core.y; // 17: remote_rx_y
                compile_args[18] = 3; // 18: remote_rx_queue_id
                compile_args[19] = (uint32_t)DispatchRemoteNetworkType::NOC0; // 19: tx_network_type
                uint32_t dest_map_array[4] = {0, 1, 2, 3};
                uint64_t dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
                compile_args[20] = (uint32_t)(dest_endpoint_output_map >> 32); // 20: dest_endpoint_output_map_hi
                compile_args[21] = (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF); // 21: dest_endpoint_output_map_lo
                compile_args[22] = BRISC_L1_RESULT_BASE; // 22: test_results_addr
                compile_args[23] = 1024; // 23: test_results_size
                compile_args[24] = 0; // 24: timeout_cycles
                compile_args[25] = 0xF; // 25: output_depacketize_mask
                arg_index = 26;
                uint32_t demux_sem = demux_settings.producer_semaphore_id;
                for (auto&[core, settings] : device_worker_variants[DISPATCH]) {
                     // 26, 27, 28, 29: output x depacketize info:
                    compile_args[arg_index++] = packet_switch_4B_pack(settings.cb_log_page_size,
                                                                        settings.consumer_semaphore_id, // downstream sem
                                                                        demux_sem++,    // local sem
                                                                        1); // remove header
                }
                break;
            }
            case DISPATCH:
            {
                uint32_t num_dispatchers = device_worker_variants[DISPATCH].size();
                TT_ASSERT(device_worker_variants[DEMUX].size() == 1, "Cannot have more than one Demux.");
                auto demux_settings = std::get<1>(device_worker_variants[DEMUX][0]);
                TT_ASSERT(num_dispatchers == demux_settings.semaphores.size(), "Demux does not have required number of semaphores for Dispatchers. Exptected = {}. Fount = {}", num_dispatchers, demux_settings.semaphores.size());
                uint32_t demux_sem = demux_settings.producer_semaphore_id;
                for (auto&[core, settings] : device_worker_variants[DISPATCH]) {
                    auto dispatch_core_type = settings.dispatch_core_type;
                    settings.upstream_cores.push_back(demux_settings.worker_physical_core);
                    settings.downstream_cores.push_back(tt_cxy_pair(0, 0, 0));
                    settings.compile_args.resize(17);
                    auto& compile_args = settings.compile_args;
                    compile_args[0] = settings.cb_start_address;
                    compile_args[1] = settings.cb_log_page_size;
                    compile_args[2] = settings.cb_pages;
                    compile_args[3] = settings.consumer_semaphore_id;
                    compile_args[4] = demux_sem++;
                    compile_args[5] = dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS;
                    compile_args[6] = 0; //unused prefetch_sync_sem
                    compile_args[7] = settings.command_queue_start_addr;
                    compile_args[8] = settings.completion_queue_start_addr;
                    compile_args[9] = settings.completion_queue_size;
                    compile_args[10] = dispatch_constants::DISPATCH_BUFFER_BASE; // unused
                    compile_args[11] = (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) * dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages(); // unused
                    compile_args[12] = 0; // unused: local ds semaphore
                    compile_args[13] = 0; // unused: remote ds semaphore
                    compile_args[14] = 0; // preamble size
                    compile_args[15] = false; // is_dram_variant
                    compile_args[16] = true; // is_host_variant
                }
                break;
            }
            case US_TUNNELER_LOCAL:
            {
                bool is_tunnel_end = device_worker_variants[US_TUNNELER_REMOTE].size() == 0;
                TT_ASSERT(device_worker_variants[US_TUNNELER_LOCAL].size() == 1, "Unexpected number of ethernet tunnelers.");
                auto &tunneler_settings = std::get<1>(device_worker_variants[US_TUNNELER_LOCAL][0]);
                auto &demux_d_settings = std::get<1>(device_worker_variants[DEMUX_D][0]);
                auto &mux_d_settings = std::get<1>(device_worker_variants[MUX_D][0]);

                auto &compile_args = tunneler_settings.compile_args;
                compile_args.resize(16);
                compile_args[0] = 0xDACADACA; // 0: endpoint_id_start_index
                compile_args[1] = 2; // tunnel_lanes. 1 => Unidirectional. 2 => Bidirectional.
                compile_args[2] = tunneler_settings.cb_start_address >> 4; // 2: rx_queue_start_addr_words
                compile_args[3] = tunneler_settings.cb_size_bytes >> 4; // 3: rx_queue_size_words

                compile_args[4] = packet_switch_4B_pack(demux_d_settings.worker_physical_core.x,
                                    demux_d_settings.worker_physical_core.y,
                                    is_tunnel_end ? 1 : 2,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0); // 4: remote_receiver_0_info

                compile_args[5] = packet_switch_4B_pack(tunneler_settings.eth_partner_physical_core.x,
                                    tunneler_settings.eth_partner_physical_core.y,
                                    1, // input q id of remote ethernet tunneler
                                    (uint32_t)DispatchRemoteNetworkType::ETH); // 5: remote_receiver_1_info

                compile_args[6] = demux_d_settings.cb_start_address >> 4; // 6: remote_receiver_queue_start_addr_words 0
                compile_args[7] = demux_d_settings.cb_size_bytes >> 4; // 7: remote_receiver_queue_size_words 0
                compile_args[8] = (tunneler_settings.cb_start_address + tunneler_settings.cb_size_bytes) >> 4; // 8: remote_receiver_queue_start_addr_words 1
                compile_args[9] = tunneler_settings.cb_size_bytes >> 4; // 9: remote_receiver_queue_size_words 1

                compile_args[10] = packet_switch_4B_pack(tunneler_settings.eth_partner_physical_core.x,
                                    tunneler_settings.eth_partner_physical_core.y,
                                    2, // queue id of remote eth tunneler sender
                                    (uint32_t)DispatchRemoteNetworkType::ETH); // 10: remote_sender_0_info
                compile_args[11] = packet_switch_4B_pack(mux_d_settings.worker_physical_core.x,
                                    mux_d_settings.worker_physical_core.y,
                                    is_tunnel_end ? 1 : 2, // mux_d output queue id
                                    (uint32_t)DispatchRemoteNetworkType::NOC0); // 11: remote_sender_1_info
                compile_args[12] = 0x39000; // 12: test_results_addr
                compile_args[13] = 0x7000; // 13: test_results_size
                compile_args[14] = 0; // 14: timeout_cycles
                if (!is_tunnel_end && tunneler_settings.tunnel_stop > 1) {
                    auto &us_tunneler_remote_settings = std::get<1>(device_worker_variants[US_TUNNELER_REMOTE][0]);
                    auto mux_d_sender = us_tunneler_remote_settings.worker_physical_core;
                    compile_args[15] = (0x3 << 16) | (mux_d_sender.y << 8) | (mux_d_sender.x);
                    log_debug(tt::LogMetal, "Tunner Inner Device {} will send done to {}", tunneler_settings.worker_physical_core.str(), mux_d_sender.str());
                }

                break;
            }
            case DEMUX_D:
            {
                bool is_tunnel_end = device_worker_variants[US_TUNNELER_REMOTE].size() == 0;
                TT_ASSERT(device_worker_variants[DEMUX_D].size() == 1, "Unexpected number of device demux.");

                auto &tunneler_settings = std::get<1>(device_worker_variants[US_TUNNELER_LOCAL][0]);
                auto &demux_d_settings = std::get<1>(device_worker_variants[DEMUX_D][0]);
                auto &prefetch_d_settings = std::get<1>(device_worker_variants[PREFETCH_D][0]);

                TT_ASSERT(demux_d_settings.tunnel_stop > 0 && demux_d_settings.tunnel_stop <= 4, "Invalid Demux D tunnel stop.");

                auto &compile_args = demux_d_settings.compile_args;
                compile_args.resize(30);

                compile_args[0] = 0xB1; // 0: endpoint_id_start_index
                compile_args[1] = demux_d_settings.cb_start_address >> 4; // 1: rx_queue_start_addr_words
                compile_args[2] = demux_d_settings.cb_size_bytes >> 4; // 2: rx_queue_size_words
                compile_args[3] = is_tunnel_end ? 1 : 2; // 3: demux_fan_out

                compile_args[4] = packet_switch_4B_pack(prefetch_d_settings.worker_physical_core.x,
                                                        prefetch_d_settings.worker_physical_core.y,
                                                        0,
                                                        (uint32_t)DispatchRemoteNetworkType::NOC0); // 4: remote_tx_0_info

                compile_args[8] = prefetch_d_settings.cb_start_address >> 4; // 8: remote_tx_queue_start_addr_words 0
                compile_args[9] = prefetch_d_settings.cb_size_bytes >> 4; // 9: remote_tx_queue_size_words 0

                if(!is_tunnel_end) {
                    auto &us_tunneler_remote_settings = std::get<1>(device_worker_variants[US_TUNNELER_REMOTE][0]);
                    compile_args[5] = packet_switch_4B_pack((uint32_t)us_tunneler_remote_settings.worker_physical_core.x,
                                                                    (uint32_t)us_tunneler_remote_settings.worker_physical_core.y,
                                                                    0,
                                                                    (uint32_t)DispatchRemoteNetworkType::NOC0); // 5: remote_tx_1_info

                    compile_args[10] = us_tunneler_remote_settings.cb_start_address >> 4;    // 10: remote_tx_queue_start_addr_words 1
                    compile_args[11] = us_tunneler_remote_settings.cb_size_bytes >> 4;   // 11: remote_tx_queue_size_words 1
                }

                compile_args[16] = tunneler_settings.worker_physical_core.x; // 16: remote_rx_x
                compile_args[17] = tunneler_settings.worker_physical_core.y; // 17: remote_rx_y
                compile_args[18] = 2; // 18: remote_rx_queue_id
                compile_args[19] = (uint32_t)DispatchRemoteNetworkType::NOC0; // 19: tx_network_type
                uint32_t dest_map_array[4] = {1, 1, 1, 1}; // needs to be based on tunnel stop.
                dest_map_array[demux_d_settings.tunnel_stop-1] = 0;
                uint64_t dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
                compile_args[20] = (uint32_t)(dest_endpoint_output_map >> 32); // 20: dest_endpoint_output_map_hi
                compile_args[21] = (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF); // 21: dest_endpoint_output_map_lo
                compile_args[22] = BRISC_L1_RESULT_BASE; // 22: test_results_addr
                compile_args[23] = 1024; // 23: test_results_size
                compile_args[24] = 0; // 24: timeout_cycles
                compile_args[25] = 0x1; // 25: output_depacketize_mask
                compile_args[26] = packet_switch_4B_pack(prefetch_d_settings.cb_log_page_size,
                                                                        prefetch_d_settings.consumer_semaphore_id, // downstream sem
                                                                        demux_d_settings.producer_semaphore_id,    // local sem
                                                                        0); // remove header
                break;
            }
            case PREFETCH_D:
            {

                uint32_t num_prefetchers = device_worker_variants[PREFETCH_D].size();
                TT_ASSERT(device_worker_variants[DEMUX_D].size() == 1, "Cannot have more than one Demux D.");
                auto &prefetch_d_settings = std::get<1>(device_worker_variants[PREFETCH_D][0]);
                auto demux_d_settings = std::get<1>(device_worker_variants[DEMUX_D][0]);
                auto dispatch_d_settings = std::get<1>(device_worker_variants[DISPATCH_D][0]);

                TT_ASSERT(num_prefetchers == demux_d_settings.semaphores.size(), "Demux D does not have required number of semaphores for Prefetcher D. Exptected = {}. Fount = {}", num_prefetchers, demux_d_settings.semaphores.size());

                auto dispatch_core_type = prefetch_d_settings.dispatch_core_type;
                prefetch_d_settings.upstream_cores.push_back(demux_d_settings.worker_physical_core);
                prefetch_d_settings.downstream_cores.push_back(dispatch_d_settings.worker_physical_core);

                uint32_t scratch_db_base = (prefetch_d_settings.cb_start_address + prefetch_d_settings.cb_size_bytes + PCIE_ALIGNMENT - 1) & (~(PCIE_ALIGNMENT - 1));
                uint32_t scratch_db_size = dispatch_constants::get(dispatch_core_type).scratch_db_size();
                const uint32_t l1_size = dispatch_core_type == CoreType::WORKER ? MEM_L1_SIZE : MEM_ETH_SIZE;
                TT_ASSERT(scratch_db_base + scratch_db_size <= l1_size);

                auto& compile_args = prefetch_d_settings.compile_args;
                compile_args.resize(23);
                compile_args[0]  = dispatch_d_settings.cb_start_address;
                compile_args[1]  = dispatch_d_settings.cb_log_page_size;
                compile_args[2]  = dispatch_d_settings.cb_pages;
                compile_args[3]  = prefetch_d_settings.producer_semaphore_id;
                compile_args[4]  = dispatch_d_settings.consumer_semaphore_id;
                compile_args[5]  = 0;
                compile_args[6]  = 0;
                compile_args[7]  = 0;
                compile_args[8]  = dispatch_constants::get(dispatch_core_type).prefetch_q_size();
                compile_args[9]  = CQ_PREFETCH_Q_RD_PTR;
                compile_args[10] = prefetch_d_settings.cb_start_address;
                compile_args[11] = prefetch_d_settings.cb_size_bytes;
                compile_args[12] = scratch_db_base;
                compile_args[13] = scratch_db_size;
                compile_args[14] = 0; //prefetch_sync_sem
                compile_args[15] = prefetch_d_settings.cb_pages; // prefetch_d only
                compile_args[16] = prefetch_d_settings.consumer_semaphore_id; // prefetch_d only
                compile_args[17] = demux_d_settings.producer_semaphore_id; //prefetch_downstream_cb_sem, // prefetch_d only
                compile_args[18] = prefetch_d_settings.cb_log_page_size;;
                compile_args[19] = dispatch_constants::PREFETCH_D_BUFFER_BLOCKS; // prefetch_d only
                compile_args[20] = 2; //prefetch_h_exec_buf_sem,
                compile_args[21] = true;  // is_dram_variant
                compile_args[22] = false; // is_host_variant
                break;
            }
            case DISPATCH_D:
            {
                uint32_t num_dispatchers = device_worker_variants[DISPATCH_D].size();
                TT_ASSERT(device_worker_variants[MUX_D].size() == 1, "Cannot have more than one Mux D.");
                auto mux_d_settings = std::get<1>(device_worker_variants[MUX_D][0]);
                TT_ASSERT(num_dispatchers == mux_d_settings.semaphores.size(), "Mux D does not have required number of semaphores for Dispatchers. Exptected = {}. Fount = {}", num_dispatchers, mux_d_settings.semaphores.size());
                uint32_t sem = 0;
                auto &dispatch_d_settings = std::get<1>(device_worker_variants[DISPATCH_D][0]);
                auto prefetch_d_settings = std::get<1>(device_worker_variants[PREFETCH_D][0]);

                auto dispatch_core_type = dispatch_d_settings.dispatch_core_type;
                dispatch_d_settings.upstream_cores.push_back(prefetch_d_settings.worker_physical_core);
                dispatch_d_settings.downstream_cores.push_back(mux_d_settings.worker_physical_core);
                dispatch_d_settings.compile_args.resize(17);
                auto& compile_args = dispatch_d_settings.compile_args;
                compile_args[0] = dispatch_d_settings.cb_start_address;
                compile_args[1] = dispatch_d_settings.cb_log_page_size;
                compile_args[2] = dispatch_d_settings.cb_pages;
                compile_args[3] = dispatch_d_settings.consumer_semaphore_id;
                compile_args[4] = prefetch_d_settings.producer_semaphore_id;
                compile_args[5] = dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS;
                compile_args[6] = 0;
                compile_args[7] = dispatch_d_settings.command_queue_start_addr;
                compile_args[8] = dispatch_d_settings.completion_queue_start_addr;
                compile_args[9] = dispatch_d_settings.completion_queue_size;
                compile_args[10] = mux_d_settings.cb_start_address;
                compile_args[11] = mux_d_settings.cb_size_bytes;
                compile_args[12] = dispatch_d_settings.producer_semaphore_id; // unused: local ds semaphore
                compile_args[13] = mux_d_settings.consumer_semaphore_id; // unused: remote ds semaphore
                compile_args[14] = sizeof(dispatch_packet_header_t); // preamble size
                compile_args[15] = true; // is_dram_variant
                compile_args[16] = false; // is_host_variant
                break;
            }
            case MUX_D:
            {
                uint32_t num_dispatchers = device_worker_variants[DISPATCH_D].size();
                TT_ASSERT(device_worker_variants[MUX_D].size() == 1, "Cannot have more than one Mux D.");
                auto &mux_d_settings = std::get<1>(device_worker_variants[MUX_D][0]);
                auto dispatch_d_settings = std::get<1>(device_worker_variants[DISPATCH_D][0]);

                TT_ASSERT(num_dispatchers == mux_d_settings.semaphores.size(), "Mux D does not have required number of semaphores for Dispatchers. Exptected = {}. Fount = {}", num_dispatchers, mux_d_settings.semaphores.size());
                uint32_t sem = 0;
                bool is_tunnel_end = device_worker_variants[US_TUNNELER_REMOTE].size() == 0;

                auto& compile_args = mux_d_settings.compile_args;
                compile_args.resize(25);
                compile_args[0] = 0; // 0: reserved
                compile_args[1] = mux_d_settings.cb_start_address >> 4; // 1: rx_queue_start_addr_words
                compile_args[2] = mux_d_settings.cb_size_bytes >> 4; // 2: rx_queue_size_words
                compile_args[3] = is_tunnel_end ? 1 : 2; // 3: mux_fan_in
                uint32_t arg_index = 4;
                compile_args[4] = packet_switch_4B_pack(dispatch_d_settings.worker_physical_core.x,
                                                        dispatch_d_settings.worker_physical_core.y,
                                                        1,
                                                        DispatchRemoteNetworkType::NOC0); // 4,5,6,7: src x info

                if (!is_tunnel_end) {
                    TT_ASSERT(device_worker_variants[US_TUNNELER_REMOTE].size() == 1, "Unexpected number of ethernet tunnelers.");
                    auto &us_tunneler_remote_settings = std::get<1>(device_worker_variants[US_TUNNELER_REMOTE][0]);
                    compile_args[5] = packet_switch_4B_pack(us_tunneler_remote_settings.worker_physical_core.x,
                                        us_tunneler_remote_settings.worker_physical_core.y,
                                        3,
                                        DispatchRemoteNetworkType::NOC0); // 4,5,6,7: src x info

                }

                TT_ASSERT(device_worker_variants[US_TUNNELER_LOCAL].size() == 1, "Unexpected number of ethernet tunnelers.");
                auto &tunneler_settings = std::get<1>(device_worker_variants[US_TUNNELER_LOCAL][0]);

                compile_args[8] = (tunneler_settings.cb_start_address + tunneler_settings.cb_size_bytes) >> 4; // 8: remote_tx_queue_start_addr_words
                compile_args[9] = tunneler_settings.cb_size_bytes >> 4; // 9: remote_tx_queue_size_words
                compile_args[10] = tunneler_settings.worker_physical_core.x; // 10: remote_tx_x
                compile_args[11] = tunneler_settings.worker_physical_core.y; // 11: remote_tx_y
                compile_args[12] = 1; // 12: remote_tx_queue_id
                compile_args[13] = (uint32_t)DispatchRemoteNetworkType::NOC0; // 13: tx_network_type
                compile_args[14] = BRISC_L1_RESULT_BASE; // 14: test_results_addr
                compile_args[15] = 1024; // 15: test_results_size
                compile_args[16] = 0; // 16: timeout_cycles
                compile_args[17] = 0x0; // 17: output_depacketize
                compile_args[18] = 0x0; // 18: output_depacketize info

                compile_args[19] = packet_switch_4B_pack(0x1,
                            dispatch_d_settings.cb_log_page_size,
                            dispatch_d_settings.producer_semaphore_id,  // upstream sem
                            mux_d_settings.consumer_semaphore_id); // local sem
                uint32_t src_id = 0xC1 + mux_d_settings.tunnel_stop - 1;
                uint32_t dest_id = 0xD1 + mux_d_settings.tunnel_stop - 1;
                compile_args[23] = packet_switch_4B_pack(src_id, src_id, src_id, src_id); // 23: packetized input src id
                compile_args[24] = packet_switch_4B_pack(dest_id, dest_id, dest_id, dest_id); // 24: packetized input dest id
                break;
            }
        }
    }
}

void Device::setup_tunnel_for_remote_devices() {
    chip_id_t mmio_device_id = this->id_;
    uint32_t num_tunnels = tt::Cluster::instance().get_mmio_device_tunnel_count(mmio_device_id);
    if (num_tunnels == 0) {
        //no remote device conected to this mmio device.
        return;
    }


    tunnels_from_mmio_ = tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_device_id);
    uint32_t index = 0;
    for (auto tunnel : tunnels_from_mmio_) {
        for (auto remote_dev : tunnel) {
            log_info(tt::LogMetal, "MMIO Device {} : Tunnel {} : Device {}", mmio_device_id, index, remote_dev);
        }
        index++;
    }

    std::map<uint32_t, std::vector<std::vector<std::tuple<tt_cxy_pair, worker_build_settings_t>>>> tunnel_dispatch_core_allocations = {};

    uint32_t tunnel_id = 0;
    for (auto &tunnel: tunnels_from_mmio_) {
        std::vector<std::vector<std::tuple<tt_cxy_pair, worker_build_settings_t>>> tunnel_core_allocations = {};
        tunnel_core_allocations.resize(tt::tt_metal::DispatchWorkerType::COUNT);

        for (uint32_t tunnel_stop = 1; tunnel_stop < tunnel.size(); tunnel_stop++) {
            //uint32_t tunnel_stop = tt::Cluster::instance().get_device_tunnel_depth(device_id);
            chip_id_t device_id = tunnel[tunnel_stop];
            // a remote device.
            // tunnel_stop hops away.
            uint8_t num_hw_cqs = 1;
            uint32_t cq_id = 0;
            uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
            CoreType dispatch_core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(mmio_device_id);

            worker_build_settings_t settings = {};
            //allocations below are on mmio chip.
            settings.tunnel_stop = 0;
            uint32_t cq_size = this->sysmem_manager().get_cq_size();
            settings.command_queue_start_addr = get_absolute_cq_offset(channel, cq_id, cq_size);
            settings.issue_queue_start_addr = settings.command_queue_start_addr + CQ_START;
            settings.issue_queue_size = this->sysmem_manager_->get_issue_queue_size(cq_id);
            settings.completion_queue_start_addr = settings.issue_queue_start_addr + settings.issue_queue_size;
            settings.completion_queue_size = this->sysmem_manager_->get_completion_queue_size(cq_id);
            settings.dispatch_core_type = dispatch_core_type;

            tt_cxy_pair prefetch_location = dispatch_core_manager::get(num_hw_cqs).prefetcher_core(device_id, channel, cq_id);
            settings.worker_physical_core = tt_cxy_pair(prefetch_location.chip, get_physical_core_coordinate(prefetch_location, dispatch_core_type));
            settings.kernel_file = "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp";
            //prefetch needs three semaphores.
            settings.semaphores.push_back(0);
            settings.semaphores.push_back(dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_pages());
            settings.semaphores.push_back(0);
            settings.producer_semaphore_id = 1;
            tunnel_core_allocations[PREFETCH].push_back(std::make_tuple(prefetch_location, settings));

            settings.semaphores.clear();
            tt_cxy_pair dispatch_location = dispatch_core_manager::get(num_hw_cqs).dispatcher_core(device_id, channel, cq_id);
            settings.worker_physical_core = tt_cxy_pair(dispatch_location.chip, get_physical_core_coordinate(dispatch_location, dispatch_core_type));
            settings.kernel_file = "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp";
            //dispatch needs one semaphore.
            settings.semaphores.push_back(0);
            settings.producer_semaphore_id = 0;
            settings.consumer_semaphore_id = 0;
            settings.cb_start_address = dispatch_constants::DISPATCH_BUFFER_BASE;
            settings.cb_log_page_size = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
            settings.cb_pages = dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages();
            settings.cb_size_bytes = (1 << settings.cb_log_page_size) * settings.cb_pages;
            tunnel_core_allocations[DISPATCH].push_back(std::make_tuple(dispatch_location, settings));
            log_debug(LogMetal, "Device {} Channel {} : Dispatch: Issue Q Start Addr: {} - Completion Q Start Addr: {}",  device_id, channel, settings.issue_queue_start_addr, settings.completion_queue_start_addr);

            if (tunnel_stop == 1) {
                //need to allocate mux/demux on mmio chip only once.
                //all tunnel stops, share the same mux/demux on mmio chip.
                settings.semaphores.clear();
                //mux/demux need a semaphore per remote device in the tunnel.
                //Tunnel includes the mmio device as well, so tunnel.size() - 1 is the number of remote devices.
                settings.semaphores.resize(tunnel.size()-1);
                settings.producer_semaphore_id = 0;
                settings.consumer_semaphore_id = 0;
                tt_cxy_pair mux_location = dispatch_core_manager::get(num_hw_cqs).mux_core(device_id, channel, cq_id);
                settings.worker_physical_core = tt_cxy_pair(mux_location.chip, get_physical_core_coordinate(mux_location, dispatch_core_type));
                settings.kernel_file = "tt_metal/impl/dispatch/kernels/packet_mux.cpp";
                settings.cb_start_address = dispatch_constants::DISPATCH_BUFFER_BASE;
                settings.cb_size_bytes = dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_size();

                tunnel_core_allocations[MUX].push_back(std::make_tuple(mux_location, settings));

                tt_cxy_pair demux_location = dispatch_core_manager::get(num_hw_cqs).demux_core(device_id, channel, cq_id);
                settings.worker_physical_core = tt_cxy_pair(demux_location.chip, get_physical_core_coordinate(demux_location, dispatch_core_type));
                settings.kernel_file = "tt_metal/impl/dispatch/kernels/packet_demux.cpp";
                settings.cb_start_address = L1_UNRESERVED_BASE;
                settings.cb_size_bytes = 0x10000;
                tunnel_core_allocations[DEMUX].push_back(std::make_tuple(demux_location, settings));
            }

            settings.tunnel_stop = tunnel_stop - 1;
            settings.semaphores.clear();
            chip_id_t us_device = tunnel[tunnel_stop - 1];
            tt_cxy_pair us_location = dispatch_core_manager::get(num_hw_cqs).tunneler_core(us_device, device_id, channel, cq_id);
            tt_cxy_pair local_location = dispatch_core_manager::get(num_hw_cqs).us_tunneler_core_local(device_id, channel, cq_id);

            settings.worker_physical_core = tt_cxy_pair(us_location.chip, get_physical_core_coordinate(us_location, CoreType::ETH));
            settings.eth_partner_physical_core = tt_cxy_pair(local_location.chip, get_physical_core_coordinate(local_location, CoreType::ETH));
            settings.kernel_file = "tt_metal/impl/dispatch/kernels/eth_tunneler.cpp";
            settings.cb_start_address = 0x19000;
            settings.cb_size_bytes = 0x10000;
            tunnel_core_allocations[US_TUNNELER_REMOTE].push_back(std::make_tuple(us_location, settings));

            //all allocation below this are on a remote chip.
            settings.tunnel_stop = tunnel_stop;

            //swap the two etnernet link pair cores for downstream chip on the link pair.
            tt_cxy_pair temp = settings.worker_physical_core;
            settings.worker_physical_core = settings.eth_partner_physical_core;
            settings.eth_partner_physical_core = temp;
            settings.kernel_file = "tt_metal/impl/dispatch/kernels/eth_tunneler.cpp";
            tunnel_core_allocations[US_TUNNELER_LOCAL].push_back(std::make_tuple(local_location, settings));

            TT_ASSERT(us_location.chip == us_device,
                "Upstream Tunneler is on device {} but it is expected to be on device {}", us_location.chip, us_device);
            TT_ASSERT(local_location.chip == device_id,
                "Upstream Local Tunneler is on device {} but it is expected to be on device {}", local_location.chip, device_id);

            dispatch_core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(device_id);
            settings.dispatch_core_type = dispatch_core_type;

            tt_cxy_pair mux_d_location = dispatch_core_manager::get(num_hw_cqs).mux_d_core(device_id, channel, cq_id);
            settings.worker_physical_core = tt_cxy_pair(mux_d_location.chip, get_physical_core_coordinate(mux_d_location, dispatch_core_type));
            settings.kernel_file = "tt_metal/impl/dispatch/kernels/packet_mux.cpp";
            settings.semaphores.push_back(0);
            settings.consumer_semaphore_id = 0;
            settings.cb_start_address = dispatch_constants::DISPATCH_BUFFER_BASE;
            settings.cb_log_page_size = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
            settings.cb_pages = dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages();
            settings.cb_size_bytes = (1 << settings.cb_log_page_size) * settings.cb_pages;
            tunnel_core_allocations[MUX_D].push_back(std::make_tuple(mux_d_location, settings));

            tt_cxy_pair demux_d_location = dispatch_core_manager::get(num_hw_cqs).demux_d_core(device_id, channel, cq_id);
            settings.worker_physical_core = tt_cxy_pair(demux_d_location.chip, get_physical_core_coordinate(demux_d_location, dispatch_core_type));
            settings.kernel_file = "tt_metal/impl/dispatch/kernels/packet_demux.cpp";
            settings.producer_semaphore_id = 0;
            settings.cb_start_address = L1_UNRESERVED_BASE;
            settings.cb_size_bytes = 0x10000;
            tunnel_core_allocations[DEMUX_D].push_back(std::make_tuple(demux_d_location, settings));

            settings.semaphores.clear();
            uint32_t dispatch_buffer_pages = dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages();
            settings.semaphores.push_back(0);// prefetch_d_sync_sem
            settings.semaphores.push_back(0);// prefetch_d_upstream_cb_sem
            settings.semaphores.push_back(dispatch_buffer_pages);// prefetch_d_downstream_cb_sem
            settings.consumer_semaphore_id = 1;
            settings.producer_semaphore_id = 2;

            tt_cxy_pair prefetch_d_location = dispatch_core_manager::get(num_hw_cqs).prefetcher_d_core(device_id, channel, cq_id);
            settings.worker_physical_core = tt_cxy_pair(prefetch_d_location.chip, get_physical_core_coordinate(prefetch_d_location, dispatch_core_type));
            settings.kernel_file = "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp";
            settings.cb_start_address = dispatch_constants::DISPATCH_BUFFER_BASE;
            settings.cb_size_bytes = dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_size();
            settings.cb_pages = dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_pages();
            settings.cb_log_page_size = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
            tunnel_core_allocations[PREFETCH_D].push_back(std::make_tuple(prefetch_d_location, settings));

            settings.semaphores.clear();
            settings.semaphores.push_back(0);// dispatch_sem
            settings.semaphores.push_back(dispatch_buffer_pages);// dispatch_downstream_cb_sem
            settings.consumer_semaphore_id = 0;
            settings.producer_semaphore_id = 1;
            settings.cb_start_address = dispatch_constants::DISPATCH_BUFFER_BASE;
            settings.cb_log_page_size = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
            settings.cb_pages = dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages();
            settings.cb_size_bytes = (1 << settings.cb_log_page_size) * settings.cb_pages;
            tt_cxy_pair dispatch_d_location = dispatch_core_manager::get(num_hw_cqs).dispatcher_d_core(device_id, channel, cq_id);
            settings.worker_physical_core = tt_cxy_pair(dispatch_d_location.chip, get_physical_core_coordinate(dispatch_d_location, dispatch_core_type));
            settings.kernel_file = "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp";
            tunnel_core_allocations[DISPATCH_D].push_back(std::make_tuple(dispatch_d_location, settings));
        }
        tunnel_dispatch_core_allocations.insert(std::make_pair(tunnel_id, tunnel_core_allocations));
        tunnel_id++;
    }

    //separate out all the dispatch workers on the tunnel into individual devices.
    for (const auto& pair : tunnel_dispatch_core_allocations) {
        std::map<chip_id_t, std::vector<std::vector<std::tuple<tt_cxy_pair, worker_build_settings_t>>>> device_dispatch_workers = {};
        for (uint32_t i = 0; i < pair.second.size(); i++) {
            if (pair.second[i].size()) {
                //some workers of allocated.
                auto tunnel_workers = pair.second[i];
                for (auto &[worker, settings] : tunnel_workers) {
                    if (device_dispatch_workers.find(worker.chip) == device_dispatch_workers.end()) {
                        std::vector<std::vector<std::tuple<tt_cxy_pair, worker_build_settings_t>>> temp = {};
                        temp.resize(tt::tt_metal::DispatchWorkerType::COUNT);
                        temp[i].push_back(std::make_tuple(worker, settings));
                        device_dispatch_workers.insert(std::make_pair(worker.chip, temp));
                    } else {
                        device_dispatch_workers[worker.chip][i].push_back(std::make_tuple(worker, settings));
                    }
                }
            }
        }
        tunnel_device_dispatch_workers_.insert(std::make_pair(pair.first, device_dispatch_workers));
    }

    log_debug(LogMetal, "{} tunnels found.",  tunnel_device_dispatch_workers_.size());

    for (const auto& tunnel : tunnel_device_dispatch_workers_) {
        for (const auto& pair : tunnel.second) {
            for (uint32_t i = 0; i < pair.second.size(); i++) {
                for (auto [core, settings] : pair.second[i]) {
                    log_debug(LogMetal, "Tunnel {} Device {} has {} on core {}.", tunnel.first, pair.first, magic_enum::enum_name((tt::tt_metal::DispatchWorkerType)i), core.str());
                }
            }
        }
    }

    for (uint32_t t = 0; t < tunnels_from_mmio_.size(); t++) {
        auto tunnel = tunnels_from_mmio_[t];
        TT_ASSERT(tunnel_device_dispatch_workers_.find(t) != tunnel_device_dispatch_workers_.end(),
                "Tunnel {} not found on MMIO Device {}", t, mmio_device_id);
        auto &tunnel_devices = tunnel_device_dispatch_workers_[t];
        for (uint32_t tunnel_stop = 0; tunnel_stop < tunnel.size(); tunnel_stop++) {
            //last iteration is used to loop in tunnel workers that run on mmio device.
            auto tunnel_device = tunnel[tunnel_stop];
            TT_ASSERT(tunnel_devices.find(tunnel_device) != tunnel_devices.end(),
                "Device {} not found in Tunnel {} on MMIO Device {}", tunnel_device, t, mmio_device_id);
            auto &device_worker_variants = tunnel_devices[tunnel_device];
            update_workers_build_settings(device_worker_variants);

            for (uint32_t dwv = 0; dwv < device_worker_variants.size(); dwv++)
            {
                if (device_worker_variants[dwv].size()) {
                    for (auto &[core, settings] : device_worker_variants[dwv]) {
                        log_debug(LogMetal, "Tunnel {} Stop {} is Device {}. Core {} - Physical {} will run {}.", t, tunnel_stop, tunnel_device, core.str(), settings.worker_physical_core.str(), magic_enum::enum_name((tt::tt_metal::DispatchWorkerType)dwv));
                        for (uint32_t arg = 0; arg < settings.compile_args.size(); arg++) {
                            log_debug(LogMetal, "CompileArgs[{}] = {}", arg, settings.compile_args[arg]);
                        }

                    }
                }
            }
        }
    }
}

void Device::compile_command_queue_programs() {
    ZoneScoped;
    unique_ptr<Program, detail::ProgramDeleter> command_queue_program_ptr(new Program);
    unique_ptr<Program, detail::ProgramDeleter> mmio_command_queue_program_ptr(new Program);

    std::string prefetch_kernel_path = "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp";
    std::string dispatch_kernel_path = "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp";

    // TODO: These are semaphore IDs, remove these when CreateSemaphore returns ID rather than address
    constexpr uint32_t prefetch_sync_sem = 0;
    constexpr uint32_t prefetch_downstream_cb_sem = 1;
    constexpr uint32_t prefetch_sem = 1;
    constexpr uint32_t dispatch_sem = 0;
    constexpr uint32_t mux_sem = 0;
    constexpr uint32_t demux_sem = 0;

    constexpr uint32_t prefetch_d_sync_sem = 0;
    constexpr uint32_t prefetch_d_upstream_cb_sem = 1;
    constexpr uint32_t prefetch_d_downstream_cb_sem = 2;
    constexpr uint32_t prefetch_h_exec_buf_sem = 2;
    constexpr uint32_t dispatch_downstream_cb_sem = 1;

    if (this->is_mmio_capable()) {
        auto device_id = this->id();
        uint8_t num_hw_cqs = this->num_hw_cqs();
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        uint32_t cq_size = this->sysmem_manager().get_cq_size();

        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            CoreType dispatch_core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(device_id);
            tt_cxy_pair prefetch_core = dispatch_core_manager::get(num_hw_cqs).prefetcher_core(device_id, channel, cq_id);
            tt_cxy_pair dispatch_core = dispatch_core_manager::get(num_hw_cqs).dispatcher_core(device_id, channel, cq_id);

            CoreCoord prefetch_physical_core = get_physical_core_coordinate(prefetch_core, dispatch_core_type);
            CoreCoord dispatch_physical_core = get_physical_core_coordinate(dispatch_core, dispatch_core_type);

            log_debug(LogDevice, "Dispatching out of {} cores",  magic_enum::enum_name(dispatch_core_type));
            log_debug(LogDevice, "Prefetch HD logical location: {} physical core: {}", prefetch_core.str(), prefetch_physical_core.str());
            log_debug(LogDevice, "Dispatch HD logical location: {} physical core {}", dispatch_core.str(), dispatch_physical_core.str());

            uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id, cq_size);
            uint32_t issue_queue_start_addr = command_queue_start_addr + CQ_START;
            uint32_t issue_queue_size = this->sysmem_manager_->get_issue_queue_size(cq_id);
            uint32_t completion_queue_start_addr = issue_queue_start_addr + issue_queue_size;
            uint32_t completion_queue_size = this->sysmem_manager_->get_completion_queue_size(cq_id);

            std::vector<uint32_t> prefetch_compile_args = {
                dispatch_constants::DISPATCH_BUFFER_BASE,
                dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages(),
                prefetch_sem,
                dispatch_sem,
                issue_queue_start_addr,
                issue_queue_size,
                dispatch_constants::PREFETCH_Q_BASE,
                dispatch_constants::get(dispatch_core_type).prefetch_q_size(),
                CQ_PREFETCH_Q_RD_PTR,
                dispatch_constants::get(dispatch_core_type).cmddat_q_base(),
                dispatch_constants::get(dispatch_core_type).cmddat_q_size(),
                dispatch_constants::get(dispatch_core_type).scratch_db_base(),
                dispatch_constants::get(dispatch_core_type).scratch_db_size(),
                prefetch_sync_sem,
                dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_pages(), // prefetch_d only
                0, //prefetch_d_upstream_cb_sem, // prefetch_d only
                0, //prefetch_downstream_cb_sem, // prefetch_d only
                dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE,
                dispatch_constants::PREFETCH_D_BUFFER_BLOCKS, // prefetch_d only
                prefetch_h_exec_buf_sem,
                true,   // is_dram_variant
                true    // is_host_variant
            };

            configure_kernel_variant(
                *command_queue_program_ptr,
                "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",
                prefetch_compile_args,
                prefetch_core,
                prefetch_physical_core,
                dispatch_core_type,
                CoreCoord{0, 0},
                dispatch_physical_core,
                std::map<string, string> {}
            );

            tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, prefetch_core, 0, dispatch_core_type); // prefetch_sync_sem
            tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, prefetch_core, dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages(), dispatch_core_type); // prefetch_sem
            tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, prefetch_core, 0, dispatch_core_type); // prefetch_h_exec_buf_sem

            std::vector<uint32_t> dispatch_compile_args = {
                dispatch_constants::DISPATCH_BUFFER_BASE,
                dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages(),
                dispatch_sem,
                prefetch_sem,
                dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS,
                prefetch_sync_sem,
                command_queue_start_addr,
                completion_queue_start_addr,
                completion_queue_size,
                dispatch_constants::DISPATCH_BUFFER_BASE,
                (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) * dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages(),
                0, // unused
                0, // unused
                0, // unused
                true,   // is_dram_variant
                true    // is_host_variant
            };

            configure_kernel_variant(
                *command_queue_program_ptr,
                "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
                dispatch_compile_args,
                dispatch_core,
                dispatch_physical_core,
                dispatch_core_type,
                prefetch_physical_core,
                CoreCoord{0, 0},
                std::map<string, string> {}
            );

            tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, dispatch_core, 0, dispatch_core_type); // dispatch_sem
        }
        detail::CompileProgram(this, *command_queue_program_ptr);
        this->command_queue_programs.push_back(std::move(command_queue_program_ptr));
        this->setup_tunnel_for_remote_devices();
    } else {
        chip_id_t device_id = this->id();
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        Device *mmio_device = tt::tt_metal::detail::GetDeviceHandle(mmio_device_id);

        auto &tunnel_device_dispatch_workers = mmio_device->tunnel_device_dispatch_workers_;
        auto &tunnels_from_mmio = mmio_device->tunnels_from_mmio_;

        std::vector<std::vector<std::tuple<tt_cxy_pair, worker_build_settings_t>>> device_worker_variants;
        std::vector<std::vector<std::tuple<tt_cxy_pair, worker_build_settings_t>>> mmio_device_worker_variants;

        uint32_t tunnel_id = 0;
        for (auto tunnel : tunnel_device_dispatch_workers) {
            TT_ASSERT(tunnel.second.find(mmio_device_id) != tunnel.second.end(), "MMIO Device {} not found in tunnel map.", mmio_device_id);
            if (tunnel.second.find(device_id) != tunnel.second.end()) {
                tunnel_id = tunnel.first;
                device_worker_variants = tunnel.second[device_id];
                mmio_device_worker_variants = tunnel.second[mmio_device_id];
                break;
            }
        }
        TT_ASSERT(device_worker_variants.size() != 0, "No worker variants found for Device {}.", device_id);

        //determine if its first tunnel stop.
        //FD2 kernels running on mmio device are launched with first tunnel stop.
        bool first_tunnel_stop = true;
        auto tunnel = tunnels_from_mmio[tunnel_id];
        for (uint32_t ts = 1; ts < tunnel.size(); ts++) {
            if (tunnel[ts] == device_id) {
                first_tunnel_stop = ts == 1;
                break;
            }
            TT_ASSERT(ts < (tunnel.size() - 1) , "Device {} tunnel stop cannot be determined on tunnel {}.", device_id, tunnel_id);
        }

        if (first_tunnel_stop) {
            /////////////////Following section is for mmio device serving Remote Device
            for (auto [prefetch_core, prefetch_settings] : mmio_device_worker_variants[PREFETCH]) {
                //auto [prefetch_core, prefetch_settings] = mmio_device_worker_variants[PREFETCH][0];
                for (auto sem : prefetch_settings.semaphores) {
                    //size of semaphores vector is number of needed semaphores on the core.
                    //Value of each vector entry is the initialization value for the semaphore.
                    tt::tt_metal::CreateSemaphore(*mmio_command_queue_program_ptr, prefetch_core, sem, prefetch_settings.dispatch_core_type);
                }
                configure_kernel_variant(
                    *mmio_command_queue_program_ptr,
                    prefetch_settings.kernel_file,//"tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",
                    prefetch_settings.compile_args,
                    prefetch_core,
                    prefetch_settings.worker_physical_core,
                    prefetch_settings.dispatch_core_type,
                    prefetch_settings.upstream_cores[0],
                    prefetch_settings.downstream_cores[0],
                    std::map<string, string> {}
                );
            }

            auto [mux_core, mux_settings] = mmio_device_worker_variants[MUX][0];
            for (auto sem : mux_settings.semaphores) {
                //size of semaphores vector is number of needed semaphores on the core.
                //Value of each vector entry is the initialization value for the semaphore.
                tt::tt_metal::CreateSemaphore(*mmio_command_queue_program_ptr, mux_core, sem, mux_settings.dispatch_core_type);
            }
            configure_kernel_variant(
                *mmio_command_queue_program_ptr,
                mux_settings.kernel_file,//"tt_metal/impl/dispatch/kernels/packet_mux.cpp",
                mux_settings.compile_args,
                mux_core,
                CoreCoord{0, 0},
                mux_settings.dispatch_core_type,
                CoreCoord{0, 0},
                CoreCoord{0, 0},
                std::map<string, string> {{"SKIP_NOC_LOGGING", "1"}}
            );

            auto [tunneler_core, tunneler_settings] = mmio_device_worker_variants[US_TUNNELER_REMOTE][0];
            configure_kernel_variant(
                *mmio_command_queue_program_ptr,
                tunneler_settings.kernel_file,//"tt_metal/impl/dispatch/kernels/eth_tunneler.cpp",
                tunneler_settings.compile_args,
                tunneler_core,
                CoreCoord{0, 0},
                CoreType::ETH,
                CoreCoord{0, 0},
                CoreCoord{0, 0},
                std::map<string, string> {{"SKIP_NOC_LOGGING", "1"}},
                true
            );

            auto [demux_core, demux_settings] = mmio_device_worker_variants[DEMUX][0];
            for (auto sem : demux_settings.semaphores) {
                //size of semaphores vector is number of needed semaphores on the core.
                //Value of each vector entry is the initialization value for the semaphore.
                tt::tt_metal::CreateSemaphore(*mmio_command_queue_program_ptr, demux_core, sem, demux_settings.dispatch_core_type);
            }
            configure_kernel_variant(
                *mmio_command_queue_program_ptr,
                demux_settings.kernel_file,//"tt_metal/impl/dispatch/kernels/packet_demux.cpp",
                demux_settings.compile_args,
                demux_core,
                CoreCoord{0, 0},
                demux_settings.dispatch_core_type,
                CoreCoord{0, 0},
                CoreCoord{0, 0},
                std::map<string, string> {{"SKIP_NOC_LOGGING", "1"}}
            );

            for (auto [dispatch_core, dispatch_settings] : mmio_device_worker_variants[DISPATCH]) {
                //auto [dispatch_core, dispatch_settings] = mmio_device_worker_variants[DISPATCH][0];
                for (auto sem : dispatch_settings.semaphores) {
                    //size of semaphores vector is number of needed semaphores on the core.
                    //Value of each vector entry is the initialization value for the semaphore.
                    tt::tt_metal::CreateSemaphore(*mmio_command_queue_program_ptr, dispatch_core, sem, dispatch_settings.dispatch_core_type);
                }
                configure_kernel_variant(
                    *mmio_command_queue_program_ptr,
                    dispatch_settings.kernel_file,//"tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
                    dispatch_settings.compile_args,
                    dispatch_core,
                    dispatch_settings.worker_physical_core,
                    dispatch_settings.dispatch_core_type,
                    dispatch_settings.upstream_cores[0],
                    CoreCoord{0xffffffff, 0xffffffff},
                    std::map<string, string> {}
                );
            }
        }
        /////////////////Following section is for Remote Device

        //Upstream device tunneler. Goes towards MMIO Device.
        auto [us_tunneler_core, us_tunneler_settings] = device_worker_variants[US_TUNNELER_LOCAL][0];
        configure_kernel_variant(
            *command_queue_program_ptr,
            us_tunneler_settings.kernel_file,//"tt_metal/impl/dispatch/kernels/eth_tunneler.cpp",
            us_tunneler_settings.compile_args,
            us_tunneler_core,
            CoreCoord{0, 0},
            CoreType::ETH,
            CoreCoord{0, 0},
            CoreCoord{0, 0},
            std::map<string, string> {{"SKIP_NOC_LOGGING", "1"}},
            true
        );

        //Downstream device tunneler. Goes towards tunnel end.
        if (device_worker_variants[US_TUNNELER_REMOTE].size()) {
            auto [ds_tunneler_core, ds_tunneler_settings] = device_worker_variants[US_TUNNELER_REMOTE][0];
            configure_kernel_variant(
                *command_queue_program_ptr,
                ds_tunneler_settings.kernel_file,//"tt_metal/impl/dispatch/kernels/eth_tunneler.cpp",
                ds_tunneler_settings.compile_args,
                ds_tunneler_core,
                CoreCoord{0, 0},
                CoreType::ETH,
                CoreCoord{0, 0},
                CoreCoord{0, 0},
                std::map<string, string> {{"SKIP_NOC_LOGGING", "1"}},
                true
            );
        }

        auto [demux_d_core, demux_d_settings] = device_worker_variants[DEMUX_D][0];
        for (auto sem : demux_d_settings.semaphores) {
            //size of semaphores vector is number of needed semaphores on the core.
            //Value of each vector entry is the initialization value for the semaphore.
            tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, demux_d_core, sem, demux_d_settings.dispatch_core_type);
        }
        configure_kernel_variant(
            *command_queue_program_ptr,
            demux_d_settings.kernel_file,//"tt_metal/impl/dispatch/kernels/packet_demux.cpp",
            demux_d_settings.compile_args,
            demux_d_core,
            CoreCoord{0, 0},
            demux_d_settings.dispatch_core_type,
            CoreCoord{0, 0},
            CoreCoord{0, 0},
            std::map<string, string> {{"SKIP_NOC_LOGGING", "1"}}
        );

        auto [prefetch_d_core, prefetch_d_settings] = device_worker_variants[PREFETCH_D][0];
        for (auto sem : prefetch_d_settings.semaphores) {
            //size of semaphores vector is number of needed semaphores on the core.
            //Value of each vector entry is the initialization value for the semaphore.
            tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, prefetch_d_core, sem, prefetch_d_settings.dispatch_core_type);
        }
        configure_kernel_variant(
            *command_queue_program_ptr,
            prefetch_d_settings.kernel_file,//"tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",
            prefetch_d_settings.compile_args,
            prefetch_d_core,
            prefetch_d_settings.worker_physical_core,
            prefetch_d_settings.dispatch_core_type,
            prefetch_d_settings.upstream_cores[0],
            prefetch_d_settings.downstream_cores[0],
            std::map<string, string> {}
        );

        auto [dispatch_d_core, dispatch_d_settings] = device_worker_variants[DISPATCH_D][0];
        for (auto sem : dispatch_d_settings.semaphores) {
            //size of semaphores vector is number of needed semaphores on the core.
            //Value of each vector entry is the initialization value for the semaphore.
            tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, dispatch_d_core, sem, dispatch_d_settings.dispatch_core_type);
        }
        configure_kernel_variant(
            *command_queue_program_ptr,
            dispatch_d_settings.kernel_file,//"tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
            dispatch_d_settings.compile_args,
            dispatch_d_core,
            dispatch_d_settings.worker_physical_core,
            dispatch_d_settings.dispatch_core_type,
            dispatch_d_settings.upstream_cores[0],
            dispatch_d_settings.downstream_cores[0],
            std::map<string, string> {}
        );

        auto [mux_d_core, mux_d_settings] = device_worker_variants[MUX_D][0];
        for (auto sem : mux_d_settings.semaphores) {
            //size of semaphores vector is number of needed semaphores on the core.
            //Value of each vector entry is the initialization value for the semaphore.
            tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, mux_d_core, sem, mux_d_settings.dispatch_core_type);
        }
        configure_kernel_variant(
            *command_queue_program_ptr,
            mux_d_settings.kernel_file,//"tt_metal/impl/dispatch/kernels/packet_mux.cpp",
            mux_d_settings.compile_args,
            mux_d_core,
            CoreCoord{0, 0},
            mux_d_settings.dispatch_core_type,
            CoreCoord{0, 0},
            CoreCoord{0, 0},
            std::map<string, string> {{"SKIP_NOC_LOGGING", "1"}}
        );

        detail::CompileProgram(this, *command_queue_program_ptr);
        this->command_queue_programs.push_back(std::move(command_queue_program_ptr));
        if (first_tunnel_stop) {
            detail::CompileProgram(mmio_device, *mmio_command_queue_program_ptr);
            this->command_queue_programs.push_back(std::move(mmio_command_queue_program_ptr));
        }
    }
}

// Writes issue and completion queue pointers to device and in sysmem and loads fast dispatch program onto dispatch cores
void Device::configure_command_queue_programs() {
    chip_id_t device_id = this->id();
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    Device *mmio_device = tt::tt_metal::detail::GetDeviceHandle(mmio_device_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
    log_debug(tt::LogMetal, "Device {} - Channel {}", this->id_, channel);

    std::vector<uint32_t> zero = {0x0}; // Reset state in case L1 Clear is disabled.
    std::vector<uint32_t> pointers(CQ_START / sizeof(uint32_t), 0);
    uint32_t cq_size = this->sysmem_manager().get_cq_size();

    if (this->is_mmio_capable()) {
        TT_ASSERT(this->command_queue_programs.size() == 1);
    } else {
        uint32_t program_size = tt::Cluster::instance().get_device_tunnel_depth(device_id) == 1 ? 2 : 1;
        TT_ASSERT(this->command_queue_programs.size() == program_size);
    }

    Program& command_queue_program = *this->command_queue_programs[0];

    for (uint8_t cq_id = 0; cq_id < this->num_hw_cqs(); cq_id++) {
        // Reset the host manager's pointer for this command queue
        this->sysmem_manager_->reset(cq_id);

        pointers[HOST_CQ_ISSUE_READ_PTR / sizeof(uint32_t)] = (CQ_START + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
        pointers[HOST_CQ_COMPLETION_WRITE_PTR / sizeof(uint32_t)] = (CQ_START + this->sysmem_manager_->get_issue_queue_size(cq_id) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;

        tt::Cluster::instance().write_sysmem(pointers.data(), pointers.size() * sizeof(uint32_t), get_absolute_cq_offset(channel, cq_id, cq_size), mmio_device_id, get_umd_channel(channel));
    }

    uint8_t num_hw_cqs = device_id == mmio_device_id ? this->num_hw_cqs() : 1;
    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        tt_cxy_pair prefetch_location = dispatch_core_manager::get(num_hw_cqs).prefetcher_core(device_id, channel, cq_id);
        tt_cxy_pair completion_q_writer_location = dispatch_core_manager::get(num_hw_cqs).completion_queue_writer_core(device_id, channel, cq_id);
        tt_cxy_pair dispatch_location = dispatch_core_manager::get(num_hw_cqs).dispatcher_core(device_id, channel, cq_id);
        CoreType dispatch_core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(mmio_device_id);

        TT_ASSERT(prefetch_location.chip == mmio_device_id and completion_q_writer_location.chip == mmio_device_id,
            "Issue queue interface is on device {} and completion queue interface is on device {} but they are expected to be on device {}", prefetch_location.chip, completion_q_writer_location.chip, mmio_device_id);

        // Initialize the FetchQ
        std::vector<uint32_t> prefetch_q(dispatch_constants::get(dispatch_core_type).prefetch_q_entries(), 0);
        std::vector<uint32_t> prefetch_q_rd_ptr_addr_data = {
            (uint32_t)(dispatch_constants::PREFETCH_Q_BASE + dispatch_constants::get(dispatch_core_type).prefetch_q_size())
        };
        detail::WriteToDeviceL1(mmio_device, prefetch_location, CQ_PREFETCH_Q_RD_PTR, prefetch_q_rd_ptr_addr_data, dispatch_core_type);
        detail::WriteToDeviceL1(mmio_device, prefetch_location, dispatch_constants::PREFETCH_Q_BASE, prefetch_q, dispatch_core_type);

        // Initialize completion queue write pointer and read pointer copy
        uint32_t issue_queue_size = this->sysmem_manager_->get_issue_queue_size(cq_id);
        uint32_t completion_queue_start_addr = CQ_START + issue_queue_size + get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t completion_queue_start_addr_16B = completion_queue_start_addr >> 4;
        vector<uint32_t> completion_queue_wr_ptr = {completion_queue_start_addr_16B};
        detail::WriteToDeviceL1(mmio_device, completion_q_writer_location, CQ_COMPLETION_READ_PTR, completion_queue_wr_ptr, dispatch_core_type);
        detail::WriteToDeviceL1(mmio_device, completion_q_writer_location, CQ_COMPLETION_WRITE_PTR, completion_queue_wr_ptr, dispatch_core_type);
        detail::WriteToDeviceL1(mmio_device, completion_q_writer_location, CQ0_COMPLETION_LAST_EVENT, zero, dispatch_core_type);
        detail::WriteToDeviceL1(mmio_device, completion_q_writer_location, CQ1_COMPLETION_LAST_EVENT, zero, dispatch_core_type);

        // Initialize address where workers signal to completion to dispatch core
        // This value is always increasing
        detail::WriteToDeviceL1(mmio_device, dispatch_location, DISPATCH_MESSAGE_ADDR, zero, dispatch_core_type);
        if (device_id != mmio_device_id) {
            tt_cxy_pair dispatch_d_location = dispatch_core_manager::get(num_hw_cqs).dispatcher_d_core(device_id, channel, cq_id);
            dispatch_core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(device_id);
            detail::WriteToDeviceL1(this, dispatch_d_location, DISPATCH_MESSAGE_ADDR, zero, dispatch_core_type);
        }
    }

    detail::ConfigureDeviceWithProgram(this, command_queue_program, true);
    tt::Cluster::instance().l1_barrier(this->id());
    if (device_id != mmio_device_id) {
        if (tt::Cluster::instance().get_device_tunnel_depth(device_id) == 1) {
            //first or only remote device on the tunnel, launch fd2 kernels on mmio device for all remote devices.
            Program& mmio_command_queue_program = *this->command_queue_programs[1];
            detail::ConfigureDeviceWithProgram(mmio_device, mmio_command_queue_program, true);
            tt::Cluster::instance().l1_barrier(mmio_device_id);
        }
    }
}

void Device::initialize_command_queue() {
    TT_ASSERT(this->is_mmio_capable() or (not this->is_mmio_capable() and this->num_hw_cqs() == 1), "Only support one hardware command queue for fast dispatch on remote device");
    using_fast_dispatch = true;
    this->sysmem_manager_ = std::make_unique<SystemMemoryManager>(this->id_, this->num_hw_cqs());
    hw_command_queues_.resize(num_hw_cqs());
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        hw_command_queues_[cq_id] = std::make_unique<HWCommandQueue>(this, cq_id);
        // Need to do this since CommandQueue constructor is private
        sw_command_queues_.push_back(std::unique_ptr<CommandQueue>(new CommandQueue(this, cq_id)));
    }

    this->compile_command_queue_programs();
    if (this->is_mmio_capable()) {
        TT_ASSERT(this->command_queue_programs.size() == 1);
    } else {
        uint32_t program_size = tt::Cluster::instance().get_device_tunnel_depth(this->id()) == 1 ? 2 : 1;
        TT_ASSERT(this->command_queue_programs.size() == program_size);
    }
    this->configure_command_queue_programs();
    Program& command_queue_program = *this->command_queue_programs[0];

    for (uint8_t cq_id = 0; cq_id < this->num_hw_cqs(); cq_id++) {
        for (const auto &[core_type, logical_dispatch_cores] : command_queue_program.logical_cores()) {
            for (const CoreCoord &logical_dispatch_core : logical_dispatch_cores) {
                launch_msg_t msg = command_queue_program.kernels_on_core(logical_dispatch_core, core_type)->launch_msg;
                tt::llrt::write_launch_msg_to_core(this->id(), this->physical_core_from_logical_core(logical_dispatch_core, core_type), &msg);
            }
        }
    }

    if (!this->is_mmio_capable()) {
        if (tt::Cluster::instance().get_device_tunnel_depth(this->id()) == 1) {
            chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->id());
            Device *mmio_device = tt::tt_metal::detail::GetDeviceHandle(mmio_device_id);
            Program& mmio_command_queue_program = *this->command_queue_programs[1];
            for (const auto &[core_type, logical_dispatch_cores] : mmio_command_queue_program.logical_cores()) {
                for (const CoreCoord &logical_dispatch_core : logical_dispatch_cores) {
                    launch_msg_t msg = mmio_command_queue_program.kernels_on_core(logical_dispatch_core, core_type)->launch_msg;
                    tt::llrt::write_launch_msg_to_core(mmio_device_id, mmio_device->physical_core_from_logical_core(logical_dispatch_core, core_type), &msg);
                }
            }
        }
    }
    // Added this for safety while debugging hangs with FD v1.3 tunnel to R, should experiment with removing it
    // tt::Cluster::instance().l1_barrier(this->id());
}

void Device::initialize_synchronous_sw_cmd_queue() {
    // Initialize a single Software Command Queue for SD, using passthrough mode.
    // This queue is used for all host bound functions using the Software CQ in SD mode.
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        // Need to do this since CommandQueue constructor is private
        sw_command_queues_.push_back(std::unique_ptr<CommandQueue>(new CommandQueue(this, cq_id)));
        sw_command_queues_[cq_id]->set_mode(CommandQueue::CommandQueueMode::PASSTHROUGH);
    }
}

bool Device::initialize(size_t l1_small_size, const std::vector<uint32_t> &l1_bank_remap, bool minimal) {
    ZoneScoped;
    log_info(tt::LogMetal, "Initializing device {}. Program cache is {}enabled", this->id_, this->program_cache.is_enabled() ? "": "NOT ");
    this->initialize_cluster();
    this->initialize_allocator(l1_small_size, l1_bank_remap);
    this->initialize_build();
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    tt::tt_metal::device_pool::devices.resize(num_devices, nullptr);
    TT_ASSERT(id_ < num_devices);
    tt::tt_metal::device_pool::devices[id_] = this;
    // For minimal setup, don't initialize FW, watcher, dprint. They won't work if we're attaching to a hung chip.
    if (minimal)
        return true;

    bool already_initialized = this->active_devices_.activate_device(this->id_);
    if (!already_initialized) {
        this->build_firmware();
    }

    DprintServerAttach(this);
    watcher_init(this);

    this->initialize_and_launch_firmware();

    watcher_attach(this);

    // Mark initialized before compiling and sending dispatch kernels to device because compilation expects device to be initialized
    this->initialized_ = true;

    // Create system memory writer for this device to have an associated interface to hardware command queue (i.e. hugepage)
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        detail::DispatchStateCheck(true);
        this->initialize_command_queue();
    } else {
        detail::DispatchStateCheck(false);
        this->initialize_synchronous_sw_cmd_queue();
        TT_ASSERT(this->num_hw_cqs() == 1, "num_hw_cqs must be 1 in slow dispatch");
    }

    return true;
}

bool Device::close() {
    log_info(tt::LogMetal, "Closing device {}", this->id_);
    if (not this->initialized_) {
        TT_THROW("Cannot close device {} that has not been initialized!", this->id_);
    }
    this->deallocate_buffers();
    watcher_detach(this);

    for (const std::unique_ptr<HWCommandQueue> &hw_command_queue : hw_command_queues_) {
        if (hw_command_queue->manager.get_bypass_mode()) {
            hw_command_queue->record_end();
        }
        hw_command_queue->terminate();
    }
    this->trace_buffer_pool_.clear();
    detail::EnableAllocs(this);

    std::unordered_set<CoreCoord> not_done_dispatch_cores;
    std::unordered_set<CoreCoord> cores_to_skip;


    if (this->is_mmio_capable()) {
        for (const chip_id_t &device_id : tt::Cluster::instance().get_devices_controlled_by_mmio_device(this->id_)) {
            uint8_t curr_num_hw_cqs = device_id == this->id_ ? this->num_hw_cqs() : 1;
            uint16_t curr_channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
            CoreType dispatch_core_type = dispatch_core_manager::get(curr_num_hw_cqs).get_dispatch_core_type(device_id);
            for (uint8_t cq_id = 0; cq_id < curr_num_hw_cqs; cq_id++) {
                if (device_id == this->id_) {
                    //mmio device.
                    if (dispatch_core_manager::get(curr_num_hw_cqs).is_dispatcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair dispatch_location = dispatch_core_manager::get(curr_num_hw_cqs).dispatcher_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);
                        not_done_dispatch_cores.insert(phys_core);
                        log_debug(tt::LogMetal, "MMIO Device Dispatch core: Logical: {} - Physical: {}", dispatch_location.str(), phys_core.str());
                    }
                    if (dispatch_core_manager::get(curr_num_hw_cqs).is_prefetcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair prefetch_location = dispatch_core_manager::get(curr_num_hw_cqs).prefetcher_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(prefetch_location, dispatch_core_type);
                        not_done_dispatch_cores.insert(phys_core);
                        log_debug(tt::LogMetal, "MMIO Device Prefetch core: Logical: {} - Physical: {}", prefetch_location.str(), phys_core.str());
                    }
                } else if (this->active_devices_.is_device_active(device_id)) {
                    //non mmio devices serviced by this mmio capable device.
                    //skip remote dispatch cores only if respective remote device is active.
                    if (dispatch_core_manager::get(curr_num_hw_cqs).is_dispatcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair dispatch_location = dispatch_core_manager::get(curr_num_hw_cqs).dispatcher_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);
                        cores_to_skip.insert(phys_core);
                        log_debug(tt::LogMetal, "Remote Device Dispatch core: Logical: {} - Physical: {} will keep running on MMIO Device.", dispatch_location.str(), phys_core.str());
                    }
                    if (dispatch_core_manager::get(curr_num_hw_cqs).is_prefetcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair prefetch_location = dispatch_core_manager::get(curr_num_hw_cqs).prefetcher_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(prefetch_location, dispatch_core_type);
                        cores_to_skip.insert(phys_core);
                        log_debug(tt::LogMetal, "Remote Device Prefetch core: Logical: {} - Physical: {} will keep running on MMIO Device.", prefetch_location.str(), phys_core.str());
                    }
                    if (dispatch_core_manager::get(curr_num_hw_cqs).is_mux_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair mux_location = dispatch_core_manager::get(curr_num_hw_cqs).mux_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(mux_location, dispatch_core_type);
                        cores_to_skip.insert(phys_core);
                        log_debug(tt::LogMetal, "Remote Device Mux core: Logical: {} - Physical: {} will keep running on MMIO Device.", mux_location.str(), phys_core.str());
                    }
                    if (dispatch_core_manager::get(curr_num_hw_cqs).is_demux_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair demux_location = dispatch_core_manager::get(curr_num_hw_cqs).demux_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(demux_location, dispatch_core_type);
                        cores_to_skip.insert(phys_core);
                        log_debug(tt::LogMetal, "Remote Device Demux core: Logical: {} - Physical: {} will keep running on MMIO Device.", demux_location.str(), phys_core.str());
                    }
                }
            }
        }
    } else {
        //remote device that is active
        uint8_t curr_num_hw_cqs = 1;
        auto device_id = this->id_;
        uint16_t curr_channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        CoreType dispatch_core_type = dispatch_core_manager::get(curr_num_hw_cqs).get_dispatch_core_type(device_id);
        for (uint8_t cq_id = 0; cq_id < curr_num_hw_cqs; cq_id++) {
            if (dispatch_core_manager::get(curr_num_hw_cqs).is_dispatcher_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair dispatch_location = dispatch_core_manager::get(curr_num_hw_cqs).dispatcher_core(device_id, curr_channel, cq_id);
                CoreCoord phys_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);
                not_done_dispatch_cores.insert(phys_core);
                log_debug(tt::LogMetal, "Remote Device Dispatch core: Logical: {} - Physical: {} will be reset on MMIO Device.", dispatch_location.str(), phys_core.str());
            }
            if (dispatch_core_manager::get(curr_num_hw_cqs).is_prefetcher_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair prefetch_location = dispatch_core_manager::get(curr_num_hw_cqs).prefetcher_core(device_id, curr_channel, cq_id);
                CoreCoord phys_core = get_physical_core_coordinate(prefetch_location, dispatch_core_type);
                not_done_dispatch_cores.insert(phys_core);
                log_debug(tt::LogMetal, "Remote Device Prefetch core: Logical: {} - Physical: {} will be reset on MMIO Device.", prefetch_location.str(), phys_core.str());
            }
            if (dispatch_core_manager::get(curr_num_hw_cqs).is_mux_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair mux_location = dispatch_core_manager::get(curr_num_hw_cqs).mux_core(device_id, curr_channel, cq_id);
                CoreCoord phys_core = get_physical_core_coordinate(mux_location, dispatch_core_type);
                not_done_dispatch_cores.insert(phys_core);
                log_debug(tt::LogMetal, "Remote Device Mux core: Logical: {} - Physical: {} will be reset on MMIO Device.", mux_location.str(), phys_core.str());
            }
            if (dispatch_core_manager::get(curr_num_hw_cqs).is_demux_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair demux_location = dispatch_core_manager::get(curr_num_hw_cqs).demux_core(device_id, curr_channel, cq_id);
                CoreCoord phys_core = get_physical_core_coordinate(demux_location, dispatch_core_type);
                not_done_dispatch_cores.insert(phys_core);
                log_debug(tt::LogMetal, "Remote Device Demux core: Logical: {} - Physical: {} will be reset on MMIO Device.", demux_location.str(), phys_core.str());
            }
        }
    }

    auto mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->id_);
    std::unordered_set<CoreCoord> wait_for_cores = not_done_dispatch_cores;

    llrt::internal_::wait_until_cores_done(mmio_device_id, RUN_MSG_GO, wait_for_cores);

    DprintServerDetach(this);

    // Assert worker cores
    CoreCoord grid_size = this->logical_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);

            if (cores_to_skip.find(worker_core) == cores_to_skip.end()) {
                if (this->storage_only_cores_.find(logical_core) == this->storage_only_cores_.end()) {
                    tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->id(), worker_core));
                }
            } else {
                log_debug(tt::LogMetal, "{} will not be Reset when closing Device {}", worker_core.str(), this->id());
            }
        }
    }

    if (this->id_ != mmio_device_id) {
        for (auto it = not_done_dispatch_cores.begin(); it != not_done_dispatch_cores.end(); it++) {
            const auto &phys_core = *it;
            if(llrt::is_ethernet_core(phys_core, this->id_)) {
                log_debug(tt::LogMetal, "Ethernet dispatch core {} on Device {} is idle. Closing Device {}", phys_core.str(), mmio_device_id, this->id());
            } else {
                log_debug(tt::LogMetal, "Resetting core {} on Device {} when closing Device {}", phys_core.str(), mmio_device_id, this->id());
                tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(mmio_device_id, phys_core));
            }
        }
    }

    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);

    tt::Cluster::instance().l1_barrier(id_);
    allocator::clear(*this->allocator_);

    this->active_devices_.deactivate_device(this->id_);
    this->disable_and_clear_program_cache();
    this->command_queue_programs.clear();
    this->sw_command_queues_.clear();
    this->hw_command_queues_.clear();

    this->initialized_ = false;

    return true;
}

Device::~Device() {
    if (this->initialized_) {
        this->close();
    }
}

tt::ARCH Device::arch() const {
    return tt::Cluster::instance().arch();
}

int Device::num_dram_channels() const {
    return tt::Cluster::instance().get_soc_desc(id_).get_num_dram_channels();
}

uint32_t Device::l1_size_per_core() const {
    return tt::Cluster::instance().get_soc_desc(id_).worker_l1_size;
}
uint32_t Device::dram_size_per_channel() const {
    return tt::Cluster::instance().get_soc_desc(id_).dram_bank_size;
}

CoreCoord Device::logical_grid_size() const {
    return tt::Cluster::instance().get_soc_desc(id_).worker_grid_size;
}

CoreCoord Device::compute_with_storage_grid_size() const {
    return tt::get_compute_grid_size(id_, num_hw_cqs_);
}

CoreCoord Device::dram_grid_size() const {
    return tt::Cluster::instance().get_soc_desc(id_).get_dram_grid_size();
}

CoreCoord Device::physical_core_from_logical_core(const CoreCoord &logical_coord, const CoreType &core_type) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return soc_desc.get_physical_core_from_logical_core(logical_coord, core_type);
}

CoreType Device::core_type_from_physical_core(const CoreCoord &physical_coord) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    if (soc_desc.physical_cores.find(physical_coord) == soc_desc.physical_cores.end())
        TT_THROW("Physical core {} doesn't exist in metal_SocDescriptor.", physical_coord);

    return soc_desc.physical_cores.at(physical_coord).type;
}

CoreCoord Device::worker_core_from_logical_core(const CoreCoord &logical_core) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return soc_desc.get_physical_tensix_core_from_logical(logical_core);
}

std::vector<CoreCoord> Device::worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> worker_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++)
        worker_cores[idx] = worker_core_from_logical_core(logical_cores[idx]);

    return worker_cores;
}

CoreCoord Device::dram_core_from_logical_core(const CoreCoord &logical_core) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return soc_desc.get_physical_dram_core_from_logical(logical_core);
}

std::vector<CoreCoord> Device::dram_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> dram_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++)
        dram_cores[idx] = dram_core_from_logical_core(logical_cores[idx]);

    return dram_cores;
}

CoreCoord Device::ethernet_core_from_logical_core(const CoreCoord &logical_core) const {
    return tt::Cluster::instance().ethernet_core_from_logical_core(id_, logical_core);
}

CoreCoord Device::logical_core_from_ethernet_core(const CoreCoord &physical_core) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return soc_desc.get_logical_ethernet_core_from_physical(physical_core);
}

std::vector<CoreCoord> Device::ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> ethernet_cores(logical_cores.size());

    for (std::size_t idx = 0; idx < logical_cores.size(); idx++)
        ethernet_cores[idx] = ethernet_core_from_logical_core(logical_cores[idx]);
    return ethernet_cores;
}

void Device::check_allocator_is_initialized() const {
    if (this->allocator_ == nullptr) {
        TT_THROW("No memory allocator! Device has not been initialized, did you forget to call InitializeDevice?");
    }
}

uint32_t Device::num_banks(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::num_banks(*this->allocator_, buffer_type);
}

uint32_t Device::bank_size(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::bank_size(*this->allocator_, buffer_type);
}

uint32_t Device::dram_channel_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::dram_channel_from_bank_id(*this->allocator_, bank_id);
}

CoreCoord Device::dram_core_from_dram_channel(uint32_t dram_channel) const {
    return tt::Cluster::instance().get_soc_desc(id_).get_preferred_worker_core_for_dram_channel(dram_channel);
}

CoreCoord Device::logical_core_from_dram_channel(uint32_t dram_channel) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return tt::Cluster::instance().get_soc_desc(id_).get_logical_core_for_dram_channel(dram_channel);
}

uint32_t Device::dram_channel_from_logical_core(const CoreCoord& logical_core) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return tt::Cluster::instance().get_soc_desc(id_).get_dram_channel_from_logical_core(logical_core);
}

int32_t Device::bank_offset(BufferType buffer_type, uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::bank_offset(*this->allocator_, buffer_type, bank_id);
}

CoreCoord Device::logical_core_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::logical_core_from_bank_id(*this->allocator_, bank_id);
}

const std::vector<uint32_t> &Device::bank_ids_from_dram_channel(uint32_t dram_channel) const {
    this->check_allocator_is_initialized();
    return allocator::bank_ids_from_dram_channel(*this->allocator_, dram_channel);
}

const std::vector<uint32_t> &Device::bank_ids_from_logical_core(
    BufferType buffer_type, const CoreCoord &logical_core) const {
    this->check_allocator_is_initialized();
    return allocator::bank_ids_from_logical_core(*this->allocator_, buffer_type, logical_core);
}

allocator::Statistics Device::get_memory_allocation_statistics(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::get_statistics(*this->allocator_, buffer_type);
}

size_t Device::get_l1_small_size() const {
    this->check_allocator_is_initialized();
    return this->allocator_->config.l1_small_size;
}

void Device::dump_memory_blocks(const BufferType &buffer_type, std::ofstream &out) const {
    this->check_allocator_is_initialized();
    return allocator::dump_memory_blocks(*this->allocator_, buffer_type, out);
}

void Device::deallocate_buffers(){
    allocator::deallocate_buffers(*allocator_);
}

float Device::sfpu_eps() const {

  float value = std::numeric_limits<float>::epsilon();
  if( arch() == tt::ARCH::GRAYSKULL  ) {
    value = tt::tt_metal::EPS_GS;
  } else if( arch() == tt::ARCH::WORMHOLE_B0 ) {
    value = tt::tt_metal::EPS_WHB0;
  }

  return value;
}

pair<int, int> Device::build_processor_type_to_index(JitBuildProcessorType t) const {
    constexpr int DataMovementBuildCount = 2;
    constexpr int ComputeBuildCount = 3;
    constexpr int EthernetBuildCount = 2;

    switch (t) {
    case JitBuildProcessorType::DATA_MOVEMENT: return pair<int, int>(0, DataMovementBuildCount);
    case JitBuildProcessorType::COMPUTE: return pair<int, int>(DataMovementBuildCount, ComputeBuildCount);
    case JitBuildProcessorType::ETHERNET: return pair<int, int>(DataMovementBuildCount + ComputeBuildCount, EthernetBuildCount);
    default: TT_ASSERT("Bad processor type: {}", static_cast<std::underlying_type<JitBuildProcessorType>::type>(t));
    }

    // shh the warnings
    return pair<int, int>(0, 0);
}

// Ideally the firmware getter would be private to the device, however, tests look for this
const JitBuildState& Device::build_firmware_state(JitBuildProcessorType t, int i) const {
    return *(this->firmware_build_states_[build_processor_type_to_index(t).first + i]);
}

const JitBuildState& Device::build_kernel_state(JitBuildProcessorType t, int i) const {
    return *(this->kernel_build_states_[build_processor_type_to_index(t).first + i]);
}

const JitBuildStateSubset Device::build_kernel_states(JitBuildProcessorType t) const {
    pair<int, int> bptti = build_processor_type_to_index(t);
    JitBuildStateSubset subset = {
        &this->kernel_build_states_[bptti.first],
        bptti.second
    };
    return subset;
}

const string Device::build_firmware_target_path(JitBuildProcessorType t, int i) const {
    const JitBuildState& bs = build_firmware_state(t, i);
    return bs.get_target_out_path("");
}

const string Device::build_kernel_target_path(JitBuildProcessorType t, int i, const string& kernel_name) const {
    const JitBuildState& bs = build_kernel_state(t, i);
    return bs.get_target_out_path(kernel_name);
}

HWCommandQueue& Device::hw_command_queue(size_t cq_id) {
    detail::DispatchStateCheck(true);
    TT_FATAL( cq_id < hw_command_queues_.size(), "cq_id {} is out of range", cq_id );
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *hw_command_queues_[cq_id];
}

CommandQueue &Device::command_queue(size_t cq_id) {
    detail::DispatchStateCheck(using_fast_dispatch);
    TT_FATAL( cq_id < sw_command_queues_.size(), "cq_id {} is out of range", cq_id );
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *sw_command_queues_[cq_id];
}

void Device::push_work(std::function<void()>&& work, bool blocking) {
    this->work_executor.push_work(work, blocking);
}

void Device::push_work(std::shared_ptr<std::function<void()>> work, bool blocking) {
    this->work_executor.push_work(work, blocking);
}

void Device::synchronize() {
    this->work_executor.synchronize();
}

void Device::set_worker_mode(const WorkExecutorMode& mode) {
    this->work_executor.set_worker_mode(mode);
}

void Device::enable_async(bool enable) {
    auto mode = enable ? WorkExecutorMode::ASYNCHRONOUS : WorkExecutorMode::SYNCHRONOUS;
    this->set_worker_mode(mode);
}

bool Device::using_slow_dispatch() const {
    return not (this->using_fast_dispatch);
}

void Device::begin_trace(const uint8_t cq_id, const uint32_t tid, const uint32_t trace_buff_size) {
    TT_FATAL(this->trace_buffer_pool_.count(tid) == 0, "Trace already exists for tid {} on device", tid);
    TT_FATAL(!this->hw_command_queues_[cq_id]->tid.has_value(), "CQ {} is already being used for tracing tid {}", (uint32_t)cq_id, tid);
    auto desc = std::make_shared<detail::TraceDescriptor>();
    detail::EnableAllocs(this);
    this->trace_buffer_pool_.insert({tid, Trace::create_trace_buffer(this->command_queue(cq_id), desc, trace_buff_size)});
    this->hw_command_queues_[cq_id]->record_begin(tid, desc);
}

void Device::end_trace(const uint8_t cq_id, const uint32_t tid) {
    TT_FATAL(this->hw_command_queues_[cq_id]->tid == tid, "CQ {} is not being used for tracing tid {}", (uint32_t)cq_id, tid);
    TT_FATAL(this->trace_buffer_pool_.count(tid) > 0, "Trace instance " + std::to_string(tid) + " must exist on device");
    this->hw_command_queues_[cq_id]->record_end();
    auto &data = this->trace_buffer_pool_[tid]->desc->data;
    data = std::move(this->sysmem_manager().get_bypass_data());
    // Add command to terminate the trace buffer
    DeviceCommand command_sequence(CQ_PREFETCH_CMD_BARE_MIN_SIZE);
    command_sequence.add_prefetch_exec_buf_end();
    for (int i = 0; i < command_sequence.size_bytes() / sizeof(uint32_t); i++) {
        data.push_back(((uint32_t*)command_sequence.data())[i]);
    }
    Trace::initialize_buffer(this->command_queue(cq_id), this->trace_buffer_pool_[tid]);
    detail::DisableAllocs(this);
}

void Device::replay_trace(const uint8_t cq_id, const uint32_t tid, const bool blocking) {
    constexpr bool check = false;
    TT_FATAL(this->trace_buffer_pool_.count(tid) > 0, "Trace instance " + std::to_string(tid) + " must exist on device");
    if constexpr (check) {
        Trace::validate_instance(*this->trace_buffer_pool_[tid]);
    }
    this->command_queue(cq_id).run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_TRACE,
        .blocking = blocking,
        .trace_id = tid
    });
}

void Device::release_trace(const uint32_t tid) {
    uint32_t erased = this->trace_buffer_pool_.erase(tid);
    // Only enable allocations once all captured traces are released
    if (this->trace_buffer_pool_.empty()) {
        detail::EnableAllocs(this);
    }
}

std::shared_ptr<TraceBuffer> Device::get_trace(const uint32_t tid) {
    if (auto trace = this->trace_buffer_pool_.find(tid); trace != this->trace_buffer_pool_.end()) {
        return trace->second;
    } else {
        return nullptr;
    }
}

}  // namespace tt_metal

}  // namespace tt
