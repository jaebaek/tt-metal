#include "tt_metal/impl/kernels/kernel_args.hpp"

#include "common/utils.hpp"
#include "build_kernels_for_riscv/hlk_desc.hpp"

namespace tt {

namespace tt_metal {

KernelArgs::KernelArgs(const CoreCoord &logical_core, const std::vector<uint32_t> &compile_time_args) {
    core_to_compile_time_args_.insert({logical_core, compile_time_args});
}

void KernelArgs::set_kernel_args_map(const CoreRangeSet &core_ranges, const std::vector<std::vector<uint32_t>> &args_spec, bool set_compile_time_args) {
    TT_ASSERT(core_ranges.size() == args_spec.size());
    int core_range_index = 0;
    // TODO: validate core_ranges
    for (auto core_range : core_ranges) {
        auto start_core = core_range.start;
        auto end_core = core_range.end;
        auto args = args_spec.at(core_range_index);
        for (auto x = start_core.x; x <= end_core.x; x++) {
            for (auto y = start_core.y; y <= end_core.y; y++) {
                auto core_in_range = CoreCoord{.x=x, .y=y};
                if (set_compile_time_args) {
                    this->core_to_compile_time_args_.insert({core_in_range, args});
                } else {
                    this->core_to_runtime_args_.insert({core_in_range, args});
                }
            }
        }
        core_range_index++;
    }
}

KernelArgs::KernelArgs(const CoreRange &core_range, const std::vector<uint32_t> &compile_time_args) {
    this->set_kernel_args_map({core_range}, {compile_time_args}, /*set_compile_time_args=*/true);
}

KernelArgs::KernelArgs(const CoreRangeSet &core_ranges, const std::vector<std::vector<uint32_t>> &compile_time_args) {
    this->set_kernel_args_map(core_ranges, compile_time_args, /*set_compile_time_args=*/true);
}

KernelArgs::KernelArgs(const KernelArgs &other) : core_to_compile_time_args_(other.core_to_compile_time_args_), core_to_runtime_args_(other.core_to_runtime_args_) {}

KernelArgs &KernelArgs::operator=(const KernelArgs &other) {
    if (this != &other) {
        this->core_to_compile_time_args_ = other.core_to_compile_time_args_;
        this->core_to_runtime_args_ = other.core_to_runtime_args_;
    }
    return *this;
}

KernelArgs::KernelArgs(KernelArgs &&other)
    : core_to_compile_time_args_(other.core_to_compile_time_args_), core_to_runtime_args_(other.core_to_runtime_args_) {
    other.core_to_compile_time_args_.clear();
    other.core_to_runtime_args_.clear();
}

KernelArgs &KernelArgs::operator=(KernelArgs &&other) {
    if (this != &other) {
        this->core_to_compile_time_args_ = other.core_to_compile_time_args_;
        this->core_to_runtime_args_ = other.core_to_runtime_args_;
        other.core_to_compile_time_args_.clear();
        other.core_to_runtime_args_.clear();
    }
    return *this;
}

std::vector<uint32_t> KernelArgs::compile_time_args(const CoreCoord &logical_core) const {
    if (core_to_compile_time_args_.find(logical_core) != core_to_compile_time_args_.end()) {
        return core_to_compile_time_args_.at(logical_core);
    }
    return {};
}

std::vector<uint32_t> KernelArgs::runtime_args(const CoreCoord &logical_core) const {
    if (core_to_runtime_args_.find(logical_core) != core_to_runtime_args_.end()) {
        return core_to_runtime_args_.at(logical_core);
    }
    return {};
}

void KernelArgs::set_runtime_args(const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args) {
    core_to_runtime_args_.insert_or_assign(logical_core, runtime_args);
}

size_t KernelArgsHash::operator()(const KernelArgs& args) const {
    return tt::utils::vector_hash<uint32_t>{}(args.compile_time_args(logical_core));
}

size_t KernelDefinesHash::operator()(const std::map<std::string, std::string> &c_defines) const {
    size_t hash_value = 0;
    for (auto it = c_defines.begin(); it != c_defines.end(); ++it)
        boost::hash_combine(hash_value, std::hash<std::string>{}(it->first + it->second));
    return hash_value;
}


}  // namespace tt_metal

}  // namespace tt
