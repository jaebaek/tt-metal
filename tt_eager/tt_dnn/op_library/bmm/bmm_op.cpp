// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/bmm/bmm_op.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/types.hpp"

using namespace tt::constants;

vector<uint32_t> _get_prime_factors(uint32_t n) {
    uint32_t i = 2;

    vector<uint32_t> prime_factors;
    while (i * i <= n) {
        if (n % i != 0)
            i++;
        else {
            n /= i;
            prime_factors.push_back(i);
        }
    }
    if (n > 1)
        prime_factors.push_back(n);

    return prime_factors;
}

vector<uint32_t> _get_possible_products(vector<uint32_t> factors) {
    if (factors.size() == 0)
        return {1};

    vector<uint32_t> products;
    for (uint32_t& fac : factors) {
        vector<uint32_t> new_products;
        if (not std::count(products.begin(), products.end(), fac))
            new_products.push_back(fac);
        for (uint32_t& prod : products) {
            if (not std::count(products.begin(), products.end(), fac * prod))
                new_products.push_back(fac * prod);
        }

        // Insert all new products to product
        products.reserve(products.size() + distance(new_products.begin(), new_products.end()));
        products.insert(products.end(), new_products.begin(), new_products.end());
    }

    // Sort products
    std::sort(products.begin(), products.end());

    return products;
}

uint32_t _get_maximum_block_dim(int32_t block_dim, int32_t in0_block_w) {
    int32_t other_dim = (400 - 2 * in0_block_w * block_dim) / (2 * in0_block_w + block_dim);
    if (other_dim > 0)
        return other_dim;
    return 0;
}

uint32_t get_batch_size(const ttnn::types::Shape& shape) {
    uint32_t result = 1;
    for (auto i = 0; i < shape.rank() - 2; i++) {
        result *= shape[i];
    }
    return result;
}

namespace {
using namespace tt;
using namespace tt::tt_metal;
operation::OpPerformanceModel create_op_performance_model_for_matmul(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& in_a_shape = input_tensors.at(0).get_shape();
    const auto& in_b_shape = input_tensors.at(1).get_shape();
    const auto& out_shape = output_tensors.at(0).get_shape();

    const auto& t = output_tensors.at(0);
    if (t.storage_type() != StorageType::DEVICE) {
        tt::log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }

    auto arch = t.storage_type() == StorageType::DEVICE ? t.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    const int num_cores = (arch == ARCH::WORMHOLE_B0) ? 8 * 8 : 9 * 12;
    const int tensix_mul_adds_per_cycle_lofi = (arch == ARCH::WORMHOLE_B0) ? 4096 : 2048;

    // Calculate number of mul/add operations
    // TODO: add bias modeling
    int64_t num_mul_adds_per_elem = in_a_shape[3] * 2;  // 1 multiply and 1 add per element
    int64_t num_mul_adds = num_mul_adds_per_elem * out_shape[2] * out_shape[3] * out_shape[1] * out_shape[0];

    MathFidelity math_fidelity = MathFidelity::Invalid;

    std::visit(
        [&](auto&& compute_kernel_config) {
            using T = std::decay_t<decltype(compute_kernel_config)>;
            if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
                math_fidelity = compute_kernel_config.math_fidelity;
            } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                math_fidelity = compute_kernel_config.math_fidelity;
            } else {
                TT_FATAL("arch not supported");
            }
        },
        compute_kernel_config);

    int ideal_dev_clock_cycles = std::ceil(
        ((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi)) *
        (float)operation::OpPerformanceModel::fidelity_multiplier(math_fidelity));

    operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_dev_clock_cycles);
#if 0
    tt::log_info(tt::LogOp, "Matmul PerfModel:");
    tt::log_info(tt::LogOp, "\t Batch: ({}, {})", out_shape[0], out_shape[1]);
    tt::log_info(tt::LogOp, "\t In A (H, W): ({}, {})", in_a_shape[2], in_a_shape[3]);
    tt::log_info(tt::LogOp, "\t In B (H, W): ({}, {})", in_b_shape[2], in_b_shape[3]);
    tt::log_info(tt::LogOp, "\t Out (H, W): ({}, {})", out_shape[2], out_shape[3]);
    tt::log_info(tt::LogOp, "\t ideal_dev_clock_cycles: {}", ideal_dev_clock_cycles);
#endif
    return result;
}
}  // namespace
namespace bmm_op_utils {
using namespace tt;
using namespace tt::tt_metal;

tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_large_matmul_params(
    uint32_t Mt, uint32_t Nt, uint32_t num_cores_y, uint32_t num_cores_x, uint32_t in0_block_w) {
    auto Nt_fac = _get_prime_factors(Nt);
    auto Mt_fac = _get_prime_factors(Mt);
    uint32_t Npc_min = 1;
    uint32_t Mpc_min = 1;

    for (auto it = Nt_fac.begin(); it != Nt_fac.end(); ++it) {
        auto ele = *it;
        if (ele > num_cores_x) {
            Npc_min *= ele;
            Nt_fac.erase(it);
            --it;
        }
    }
    for (auto it = Mt_fac.begin(); it != Mt_fac.end(); ++it) {
        auto ele = *it;
        if (ele > num_cores_y) {
            Mpc_min *= ele;
            Mt_fac.erase(it);
            --it;
        }
    }

    if (Npc_min > _get_maximum_block_dim(Mpc_min, in0_block_w))
        return {0, 0, 0, 0};

    uint32_t Mpc = Mpc_min;
    uint32_t Npc = Npc_min;
    if (Mpc_min > 1) {
        auto Npc_choices = _get_possible_products(Nt_fac);
        auto Npc_max = _get_maximum_block_dim(Mpc_min, in0_block_w);
        for (auto& ele : Npc_choices) {
            if (ele * Npc_min <= Npc_max)
                Npc = ele * Npc_min;
            else
                break;
        }

        if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x)
            return {0, 0, 0, 0};

        for (auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
            auto subblock_h = std::get<0>(subblock_hw);
            auto subblock_w = std::get<1>(subblock_hw);
            if (Mpc % subblock_h == 0 and Npc % subblock_w == 0)
                return {Mpc, Npc, subblock_h, subblock_w};
        }
    }

    else if (Npc_min > 1) {
        auto Mpc_choices = _get_possible_products(Mt_fac);
        auto Mpc_max = _get_maximum_block_dim(Npc_min, in0_block_w);
        for (auto& ele : Mpc_choices) {
            if (ele * Mpc_min <= Mpc_max)
                Mpc = ele * Mpc_min;
            else
                break;
        }

        if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x) {
            return {0, 0, 0, 0};
        }

        for (auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
            auto subblock_h = std::get<0>(subblock_hw);
            auto subblock_w = std::get<1>(subblock_hw);
            if (Mpc % subblock_h == 0 and Npc % subblock_w == 0)
                return {Mpc, Npc, subblock_h, subblock_w};
        }
    }

    else {
        auto Mpc_choices = _get_possible_products(Mt_fac);
        auto Npc_choices = _get_possible_products(Nt_fac);
        for (auto& Npc : Npc_choices) {
            auto Mpc_max = _get_maximum_block_dim(Npc, in0_block_w);
            for (auto& ele : Mpc_choices) {
                if (ele <= Mpc_max)
                    Mpc = ele;
            }

            if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x)
                continue;

            for (auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
                auto subblock_h = std::get<0>(subblock_hw);
                auto subblock_w = std::get<1>(subblock_hw);
                if (Mpc % subblock_h == 0 and Npc % subblock_w == 0)
                    return {Mpc, Npc, subblock_h, subblock_w};
            }
        }
    }

    return {0, 0, 0, 0};
}

CoreCoord get_core_range(
    uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols) {
    CoreCoord core_range(0, 0);
    if (!(num_blocks_rows == 1 && num_blocks_cols == 1) && num_blocks_rows <= max_num_rows &&
        num_blocks_cols <= max_num_cols) {
        core_range.x = num_blocks_cols;
        core_range.y = num_blocks_rows;
    }
    return core_range;
}

tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig get_mcast_1d_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool fuse_batch,
    std::optional<UnaryWithParam> fused_activation,
    bool mcast_in0,
    bool out_sharded,
    std::optional<CoreCoord> compute_with_storage_grid_size,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto device = input_tensor_a.device();
    auto grid_size = compute_with_storage_grid_size.value_or(device->compute_with_storage_grid_size());
    uint32_t M = fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1]
                            : input_tensor_a.get_legacy_shape()[-2];
    uint32_t K = input_tensor_a.get_legacy_shape()[-1];
    uint32_t N = input_tensor_b.get_legacy_shape()[-1];
    uint32_t per_core_M, per_core_N;
    if (mcast_in0) {
        per_core_M = M / TILE_HEIGHT;
        per_core_N = div_up(div_up(N, grid_size.x * grid_size.y), TILE_WIDTH);
    } else {
        per_core_M = div_up(div_up(M, grid_size.x * grid_size.y), TILE_HEIGHT);
        per_core_N = N / TILE_WIDTH;
    }
    uint32_t in0_block_w = K / TILE_WIDTH % 2 == 0 ? 2 : 1;
    bool per_core_N_equals_subblock_w_constraint = out_sharded && !mcast_in0;
    bool per_core_M_equals_subblock_h_constraint = out_sharded && mcast_in0;
    bool fp32_dest_acc_en = bmm_op_utils::get_fp32_dest_acc_en(compute_kernel_config);
    auto subblock_hw = get_matmul_subblock_params(
        per_core_M,
        per_core_N,
        per_core_M_equals_subblock_h_constraint,
        per_core_N_equals_subblock_w_constraint,
        fp32_dest_acc_en);
    auto out_subblock_h = std::get<0>(subblock_hw);
    auto out_subblock_w = std::get<1>(subblock_hw);

    return tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig{
        .compute_with_storage_grid_size = grid_size,
        .in0_block_w = in0_block_w,
        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
        .per_core_M = per_core_M,
        .per_core_N = per_core_N,
        .fuse_batch = fuse_batch,
        .fused_activation = fused_activation,
        .mcast_in0 = mcast_in0};
}

tuple<uint32_t, uint32_t> get_matmul_subblock_params(
    const uint32_t per_core_M,
    const uint32_t per_core_N,
    const bool per_core_M_equals_subblock_h_constraint,
    bool per_core_N_equals_subblock_w_constraint,
    bool fp32_dest_acc_en) {
    TT_FATAL(
        !(per_core_M_equals_subblock_h_constraint and per_core_N_equals_subblock_w_constraint),
        "Only one constraint may be true for h or w!");

    uint32_t out_subblock_h, out_subblock_w;
    for (auto& subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
        out_subblock_h = std::get<0>(subblock_hw);
        out_subblock_w = std::get<1>(subblock_hw);
        if (fp32_dest_acc_en) {
            if ((out_subblock_h * out_subblock_w) > 4) {
                continue;  // Total number of tiles in a subblock must be less than 4 when in fp32_dest_acc mode
            }
        }
        if (per_core_N_equals_subblock_w_constraint) {
            if (out_subblock_w != per_core_N || out_subblock_h != 1) {
                continue;
            }
        }
        if (per_core_M_equals_subblock_h_constraint) {
            if (out_subblock_h != per_core_M || out_subblock_w != 1) {
                continue;
            }
        }
        if (per_core_M % out_subblock_h == 0 and per_core_N % out_subblock_w == 0) {
            return {out_subblock_h, out_subblock_w};
        }
    }

    TT_FATAL(false, "Matmul subblock parameters could not be determined for given input shapes");
}

// TODO: Only supports sharded matmul for now; infer most matmul params from shard spec
tt::operations::primary::MatmulProgramConfig get_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const MemoryConfig& output_mem_config,
    std::optional<UnaryWithParam> fused_activation,
    const bool matmul,
    const std::optional<const CoreCoord> user_core_coord,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(input_tensor_a.is_sharded());
    bool fp32_dest_acc_en = bmm_op_utils::get_fp32_dest_acc_en(compute_kernel_config);
    // TODO: allow overwriting of grid size by user_core_coord after allowing support of arbitrary compute grid and more
    // generic sharded output tensor creation
    auto grid_size = input_tensor_a.shard_spec().value().grid.bounding_box().grid_size();

    // MCAST matmuls only support input_b in INTERLEAVED
    if (matmul) {
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if ((input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED or
             input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) and
            (grid_size.x > 1 or grid_size.y > 1)) {
            TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);

            bool per_core_N_equals_subblock_w_constraint = false;
            if (output_mem_config.is_sharded()) {
                TT_FATAL(input_tensor_a.memory_config().buffer_type == output_mem_config.buffer_type);
                TT_FATAL(input_tensor_a.memory_config().memory_layout == output_mem_config.memory_layout);
                per_core_N_equals_subblock_w_constraint = true;
            }

            uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
            uint32_t K = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;
            uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
            auto shard_shape = input_tensor_a.shard_spec().value().shape;

            bool mcast_in0;
            uint32_t per_core_M;
            uint32_t per_core_N;
            uint32_t in0_block_w;
            if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                mcast_in0 = true;
                per_core_M = M;
                per_core_N = div_up(N, input_tensor_a.shard_spec().value().grid.num_cores());
                in0_block_w = shard_shape[1] / TILE_WIDTH;
            } else if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                mcast_in0 = false;
                per_core_M = shard_shape[0] / TILE_HEIGHT;
                per_core_N = N;  // Only necessary if output is sharded; otherwise, can set this to be < N
                in0_block_w = K;
            } else {
                TT_FATAL(false, "Input tensor must be WIDTH or HEIGHT sharded for 1D mcast matmul!");
            }

            auto subblock_hw = get_matmul_subblock_params(
                per_core_M, per_core_N, false, per_core_N_equals_subblock_w_constraint, fp32_dest_acc_en);
            auto out_subblock_h = std::get<0>(subblock_hw);
            auto out_subblock_w = std::get<1>(subblock_hw);

            return tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig{
                .compute_with_storage_grid_size = grid_size,
                .in0_block_w = in0_block_w,
                .out_subblock_h = out_subblock_h,
                .out_subblock_w = out_subblock_w,
                .per_core_M = per_core_M,
                .per_core_N = per_core_N,
                .fuse_batch = true,
                .fused_activation = fused_activation,
                .mcast_in0 = mcast_in0,
            };
        } else if (
            input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED and
            (grid_size.x > 1 and grid_size.y > 1)) {
            bool transpose_mcast = input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR;

            bool per_core_N_equals_subblock_w_constraint = false;
            if (output_mem_config.is_sharded()) {
                TT_FATAL(input_tensor_a.memory_config().buffer_type == output_mem_config.buffer_type);
                TT_FATAL(input_tensor_a.memory_config().memory_layout == output_mem_config.memory_layout);
                per_core_N_equals_subblock_w_constraint = true;
            }

            uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
            uint32_t K = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;
            uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;

            auto shard_shape = input_tensor_a.shard_spec().value().shape;
            uint32_t virtual_x = transpose_mcast ? grid_size.y : grid_size.x;
            uint32_t virtual_y = transpose_mcast ? grid_size.x : grid_size.y;
            TT_FATAL(
                virtual_y == (M / (shard_shape[0] / TILE_HEIGHT)), "Num cores along y must match provided grid size!");
            TT_FATAL(
                virtual_x == (K / (shard_shape[1] / TILE_WIDTH)), "Num cores along x must match provided grid size!");

            uint32_t per_core_M = M / virtual_y;
            uint32_t per_core_N = N / virtual_x;
            uint32_t in0_block_w = shard_shape[1] / TILE_WIDTH;

            auto subblock_hw = get_matmul_subblock_params(
                per_core_M, per_core_N, false, per_core_N_equals_subblock_w_constraint, fp32_dest_acc_en);
            auto out_subblock_h = std::get<0>(subblock_hw);
            auto out_subblock_w = std::get<1>(subblock_hw);

            return tt::operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = grid_size,
                .in0_block_w = in0_block_w,
                .out_subblock_h = out_subblock_h,
                .out_subblock_w = out_subblock_w,
                .per_core_M = per_core_M,
                .per_core_N = per_core_N,
                .transpose_mcast = transpose_mcast,
                .fused_activation = fused_activation,
            };
        }
    } else {
        // TODO: Need a better criteria for BMMs and MatmulMultiCoreReuseProgramConfig
        TT_FATAL(input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED);

        bool per_core_N_equals_subblock_w_constraint = false;
        if (output_mem_config.is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().buffer_type == output_mem_config.buffer_type);
            TT_FATAL(input_tensor_a.memory_config().memory_layout == output_mem_config.memory_layout);
            per_core_N_equals_subblock_w_constraint = true;
        }

        uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
        uint32_t K = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;
        uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;

        auto in0_shard_shape = input_tensor_a.shard_spec().value().shape;
        uint32_t per_core_M = in0_shard_shape[0] / TILE_HEIGHT;
        uint32_t per_core_N = N;
        uint32_t in0_block_w = in0_shard_shape[1] / TILE_WIDTH;

        auto subblock_hw = get_matmul_subblock_params(
            per_core_M, per_core_N, false, per_core_N_equals_subblock_w_constraint, fp32_dest_acc_en);
        auto out_subblock_h = std::get<0>(subblock_hw);
        auto out_subblock_w = std::get<1>(subblock_hw);

        // TODO: Temporarily allow for single core; should support bcast_batch in general
        bool broadcast_batch =
            (input_tensor_a.get_legacy_shape()[0] * input_tensor_a.get_legacy_shape()[1] > 1 and
             input_tensor_b.get_legacy_shape()[0] * input_tensor_b.get_legacy_shape()[1] == 1);
        TT_FATAL(!broadcast_batch);

        if (input_tensor_b.is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().buffer_type == input_tensor_b.memory_config().buffer_type);
            TT_FATAL(input_tensor_a.memory_config().memory_layout == input_tensor_b.memory_config().memory_layout);
            TT_FATAL(input_tensor_a.shard_spec().value().grid == input_tensor_b.shard_spec().value().grid);
            TT_FATAL(
                input_tensor_a.shard_spec().value().orientation == input_tensor_b.shard_spec().value().orientation);
        }

        return tt::operations::primary::MatmulMultiCoreReuseProgramConfig{
            .compute_with_storage_grid_size = grid_size,
            .in0_block_w = in0_block_w,
            .out_subblock_h = out_subblock_h,
            .out_subblock_w = out_subblock_w,
            .per_core_M = per_core_M,
            .per_core_N = per_core_N,
        };
    }
    TT_FATAL(false, "Matmul program config could not be determined for given input shapes!");
}

tuple<uint32_t, uint32_t> get_subblock_sizes(
    uint32_t m_tiles_per_core, uint32_t n_tiles_per_core, bool fp32_dest_acc_en) {
    uint32_t out_subblock_h, out_subblock_w;
    for (auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
        out_subblock_w = std::get<0>(subblock_hw);
        out_subblock_h = std::get<1>(subblock_hw);
        if ((out_subblock_h * out_subblock_w) <= 4 || !fp32_dest_acc_en) {
            if (m_tiles_per_core % out_subblock_h == 0 && n_tiles_per_core % out_subblock_w == 0) {
                return {out_subblock_h, out_subblock_w};
            }
        }
    }
    TT_FATAL(false, "Unable to find subblock sizes");
}

}  // namespace bmm_op_utils

namespace tt {
namespace tt_metal {

CoreCoord get_falcon_matmul_grid_size(Device* device) {
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    // TODO: Remove this once there are no more hangs on 8x8 (Issue #6795)
    if (device->arch() == ARCH::WORMHOLE_B0 and grid_size.y >= 8) {
        grid_size.y = 7;
    }
    return grid_size;
}

/**
 * Falcon matmuls using operations::primary::matmul + program_config
 */
Tensor falcon_fused_qkv_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    std::optional<const Tensor> bias,
    const MemoryConfig& mem_config,
    std::optional<const DataType> output_dtype) {
    CoreCoord grid_size = get_falcon_matmul_grid_size(input_tensor_a.device());
    auto program_config = bmm_op_utils::get_mcast_1d_config(
        input_tensor_a, input_tensor_b, true, std::nullopt, true, mem_config.is_sharded(), grid_size);
    return operations::primary::matmul_1d(
        input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor falcon_selfout_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    std::optional<const Tensor> bias,
    const MemoryConfig& mem_config,
    std::optional<const DataType> output_dtype) {
    CoreCoord grid_size = get_falcon_matmul_grid_size(input_tensor_a.device());
    auto program_config = bmm_op_utils::get_mcast_1d_config(
        input_tensor_a, input_tensor_b, true, std::nullopt, true, mem_config.is_sharded(), grid_size);
    return operations::primary::matmul_1d(
        input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor falcon_dense_4h_to_h_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    std::optional<const Tensor> bias,
    const MemoryConfig& mem_config,
    std::optional<const DataType> output_dtype,
    std::optional<bool> packer_l1_acc) {
    CoreCoord grid_size = get_falcon_matmul_grid_size(input_tensor_a.device());
    std::optional<const DeviceComputeKernelConfig> config = std::nullopt;
    auto compute_kernel_config = init_device_compute_kernel_config(
        input_tensor_a.device()->arch(), config, MathFidelity::LoFi, true, false, packer_l1_acc.value_or(false));
    auto program_config = bmm_op_utils::get_mcast_1d_config(
        input_tensor_a,
        input_tensor_b,
        true,
        std::nullopt,
        true,
        mem_config.is_sharded(),
        grid_size,
        compute_kernel_config);
    return operations::primary::matmul_1d(
        input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype, compute_kernel_config);
}

Tensor falcon_dense_h_to_4h_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    std::optional<const Tensor> bias,
    std::optional<UnaryWithParam> fused_activation,
    const MemoryConfig& mem_config,
    std::optional<const DataType> output_dtype) {
    auto seq_len = input_tensor_a.get_legacy_shape()[2];
    if (seq_len > 1024) {
        TT_FATAL(not fused_activation.has_value());
        // TODO: Check support for seq_len == 128, 256, 512, ..., 2048
        TT_FATAL(seq_len % TILE_HEIGHT == 0, "Falcon mm's seq_len must be a multiple of 32!");
        TT_FATAL(seq_len >= 128, "Falcon mm's seq_len must be greater than 128!");
        TT_FATAL((input_tensor_a.get_legacy_shape() == Shape({1, 1, seq_len, 4544})), "Unsupported input shape");
        TT_FATAL((input_tensor_b.get_legacy_shape() == Shape({1, 1, 4544, 18176})), "Unsupported input shape");
        TT_FATAL(!fused_activation.has_value());
        return operation::run_with_autoformat(
                   tt::operations::primary::Matmul{
                       .program_config = std::nullopt,
                       .bcast_batch = true,
                       .output_mem_config = mem_config,
                       .output_dtype = output_dtype.value_or(input_tensor_a.get_dtype())},
                   {input_tensor_a, input_tensor_b})
            .at(0);
    } else {
        CoreCoord grid_size = get_falcon_matmul_grid_size(input_tensor_a.device());
        std::optional<const DeviceComputeKernelConfig> config = std::nullopt;
        auto compute_kernel_config = init_device_compute_kernel_config(
            input_tensor_a.device()->arch(),
            config,
            MathFidelity::LoFi,
            true /* math_approx_mode */,
            false /* fp32_dest_acc_en */,
            true /* packer_l1_acc */);
        auto program_config = bmm_op_utils::get_mcast_1d_config(
            input_tensor_a,
            input_tensor_b,
            true,
            fused_activation,
            true,
            mem_config.is_sharded(),
            grid_size,
            compute_kernel_config);
        return operations::primary::matmul_1d(
            input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype, compute_kernel_config);
    }
}

Tensor falcon_lm_head_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    std::optional<const Tensor> bias,
    const MemoryConfig& mem_config,
    std::optional<const DataType> output_dtype) {
    auto seq_len = input_tensor_a.get_legacy_shape()[2];
    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({input_tensor_a, input_tensor_b}, {bias}))};
    if (seq_len > 512) {
        // TODO: Check support for seq_len == 128, 256, 512, ..., 2048
        operation::launch_with_autoformat(
            [seq_len, mem_config, output_dtype](
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                auto& input_tensor_a = input_tensors.at(0);
                auto& input_tensor_b = input_tensors.at(1);
                auto& bias = optional_input_tensors.at(0);
                TT_FATAL(seq_len % TILE_HEIGHT == 0, "Falcon mm's seq_len must be a multiple of 32!");
                TT_FATAL(seq_len >= 128, "Falcon mm's seq_len must be greater than 128!");
                TT_FATAL(
                    (input_tensor_a.get_legacy_shape() == Shape({1, 1, seq_len, 4544})), "Unsupported input shape");
                TT_FATAL((input_tensor_b.get_legacy_shape() == Shape({1, 1, 4544, 65024})), "Unsupported input shape");
                return operation::run_with_autoformat(
                    tt::operations::primary::Matmul{
                        .program_config = std::nullopt,
                        .bcast_batch = true,
                        .output_mem_config = mem_config,
                        .output_dtype = output_dtype.value_or(input_tensor_a.get_dtype())},
                    {input_tensor_a, input_tensor_b},
                    {bias});
            },
            {input_tensor_a, input_tensor_b},
            output_tensors,
            {bias});

    } else {
        operation::launch_op(
            [mem_config, output_dtype](
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                auto& input_tensor_a = input_tensors.at(0);
                auto& input_tensor_b = input_tensors.at(1);
                auto& bias = optional_input_tensors.at(0);
                CoreCoord grid_size = get_falcon_matmul_grid_size(input_tensor_a.device());
                std::optional<const DeviceComputeKernelConfig> config = std::nullopt;
                auto compute_kernel_config = init_device_compute_kernel_config(
                    input_tensor_a.device()->arch(),
                    config,
                    MathFidelity::LoFi,
                    true /* math_approx_mode */,
                    false /* fp32_dest_acc_en */,
                    true /* packer_l1_acc */);
                auto program_config = bmm_op_utils::get_mcast_1d_config(
                    input_tensor_a,
                    input_tensor_b,
                    true,
                    std::nullopt,
                    true,
                    mem_config.is_sharded(),
                    grid_size,
                    compute_kernel_config);
                return {operations::primary::matmul_1d(
                    input_tensor_a,
                    input_tensor_b,
                    bias,
                    program_config,
                    mem_config,
                    output_dtype,
                    compute_kernel_config)};
            },
            {input_tensor_a, input_tensor_b},
            output_tensors,
            {bias});
    }
    return output_tensors.at(0);
}

/**
 * Resnet50 matmul with fused batch
 */
Tensor resnet_matmul(
    const Tensor& input_a,
    const Tensor& input_b,
    std::optional<const Tensor> bias,
    const MemoryConfig& mem_config,
    std::optional<const DataType> output_dtype,
    const MathFidelity math_fidelity) {
    std::optional<const DeviceComputeKernelConfig> config = std::nullopt;
    auto compute_kernel_config =
        init_device_compute_kernel_config(input_a.device()->arch(), config, math_fidelity, true, false, false);
    auto program_config = bmm_op_utils::get_mcast_1d_config(
        input_a,
        input_b,
        true /* fuse_batch */,
        std::nullopt /* fused_activation */,
        true /* mcast_in0 */,
        false /* out_sharded */,
        std::nullopt /* compute_with_storage_grid_size */,
        compute_kernel_config);
    return operations::primary::matmul_1d(
        input_a, input_b, bias, program_config, mem_config, output_dtype, compute_kernel_config);
}

}  // namespace tt_metal

namespace operations {

namespace primary {
inline MatmulProgramConfig create_simple_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    const auto &ashape = input_tensor_a.get_legacy_shape(), bshape = input_tensor_b.get_legacy_shape();
    uint32_t num_output_tiles = ashape[0] * ashape[1] * ashape[2] * bshape[3] / TILE_HW;  // Output M x N

    // Parameters for large matmul with reuse
    uint32_t B = ashape[0] * ashape[1];
    uint32_t Mt = ashape[2] / TILE_HEIGHT;
    uint32_t Kt = ashape[3] / TILE_WIDTH;
    uint32_t Nt = bshape[3] / TILE_WIDTH;
    uint32_t in0_block_w = 2;

    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "input tensor needs to be on device");
    tt::tt_metal::Device* device = input_tensor_a.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t per_core_M, per_core_N, out_subblock_h, out_subblock_w;
    uint32_t num_blocks_x, num_blocks_y;

    // out_subblock h/w doesn't matter
    per_core_M = 16;
    per_core_N = 16;

    // Calculate number of blocks along x and y; tensor dims are padded up to 512
    num_blocks_y = (Mt - 1) / per_core_M + 1;
    num_blocks_x = (Nt - 1) / per_core_N + 1;

    if (num_blocks_x * num_blocks_y <= num_cores_x * num_cores_y and Kt % in0_block_w == 0) {
        CoreCoord core_range = bmm_op_utils::get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);
        if (core_range.y == 1) {
            return bmm_op_utils::get_mcast_1d_config(
                input_tensor_a,
                input_tensor_b,
                false /* fuse_batch */,
                std::nullopt /* fused_activation */,
                true /* mcast_in0 */,
                false /* out_sharded */,
                std::nullopt /* compute_with_storage_grid_size */,
                compute_kernel_config);
        } else if (core_range.x == 1) {
            return bmm_op_utils::get_mcast_1d_config(
                input_tensor_a,
                input_tensor_b,
                false /* fuse_batch */,
                std::nullopt /* fused_activation */,
                false /* mcast_in0 */,
                false /* out_sharded */,
                std::nullopt /* compute_with_storage_grid_size */,
                compute_kernel_config);
        } else if (core_range.y > 0) {
            uint32_t in0_block_w = Kt % 2 == 0 ? 2 : 1;
            return MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = {num_cores_x, num_cores_y},
                .in0_block_w = in0_block_w,
                .out_subblock_h = 4,
                .out_subblock_w = 2,
                .per_core_M = 16,
                .per_core_N = 16,
                .transpose_mcast = false,
                .fused_activation = std::nullopt,
                .fuse_batch = false,
            };
        }
        // If we don't need padding, use the default multi_core reuse/reuse_mcast
        else if (Mt % per_core_M == 0 and Nt % per_core_N == 0) {
            return MatmulMultiCoreNonOptimizedReuseProgramConfig{};
        } else {
            return MatmulMultiCoreProgramConfig{};
        }
    } else {
        return MatmulMultiCoreProgramConfig{};
    }
}

inline MatmulProgramConfig generate_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const MemoryConfig& mem_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreCoord> user_core_coord,
    const std::optional<const UnaryWithParam> user_fused_activation,
    const std::optional<const bool> user_run_batched) {
    const bool has_user_grid = user_core_coord.has_value();
    if (has_user_grid || !input_tensor_a.is_sharded()) {
        CoreCoord core_coord;
        if (has_user_grid) {
            core_coord = user_core_coord.value();
            return create_matmul_program_config(
                input_tensor_a, input_tensor_b, user_core_coord, user_fused_activation, compute_kernel_config);
        } else {
            return create_simple_matmul_program_config(input_tensor_a, input_tensor_b, compute_kernel_config);
        }
    } else {
        bool bmm = user_run_batched.value_or(false);
        return bmm_op_utils::get_matmul_program_config(
            input_tensor_a, input_tensor_b, mem_config, std::nullopt, !bmm, user_core_coord, compute_kernel_config);
    }
}

inline MatmulProgramConfig get_program_config(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, const struct Matmul* matmul) {
    if (matmul->program_config.has_value()) {
        return matmul->program_config.value();
    }
    auto config = generate_matmul_program_config(
        input_tensor_a,
        input_tensor_b,
        matmul->output_mem_config,
        matmul->compute_kernel_config,
        matmul->user_core_coord,
        matmul->user_fused_activation,
        matmul->user_run_batched);
    tt::log_debug(tt::LogOp, "Auto generated program config: {}", config);

    // Sanity checks for matmul program configs
    std::visit(
        [input_tensor_a](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                not std::is_same_v<ProgramConfigType, MatmulMultiCoreProgramConfig> and
                not std::is_same_v<ProgramConfigType, MatmulMultiCoreNonOptimizedReuseProgramConfig>) {
                TT_FATAL(
                    program_config.compute_with_storage_grid_size.x <=
                        input_tensor_a.device()->compute_with_storage_grid_size().x,
                    "Number of columns in matmul compute grid exceeds maximum device compute grid size!");
                TT_FATAL(
                    program_config.compute_with_storage_grid_size.y <=
                        input_tensor_a.device()->compute_with_storage_grid_size().y,
                    "Number of rows in matmul compute grid exceeds maximum device compute grid size!");
                TT_FATAL(
                    program_config.compute_with_storage_grid_size.x > 0,
                    "Number of columns in matmul compute grid must be greater than 0!");
                TT_FATAL(
                    program_config.compute_with_storage_grid_size.y > 0,
                    "Number of rows in matmul compute grid must be greater than 0!");
                TT_FATAL(program_config.in0_block_w > 0, "in0_block_w must be greater than 0!");
                TT_FATAL(program_config.out_subblock_h > 0, "out_subblock_h must be greater than 0!");
                TT_FATAL(program_config.out_subblock_w > 0, "out_subblock_w must be greater than 0!");
                TT_FATAL(program_config.per_core_M > 0, "per_core_M must be greater than 0!");
                TT_FATAL(program_config.per_core_N > 0, "per_core_N must be greater than 0!");
                TT_FATAL(
                    program_config.per_core_M % program_config.out_subblock_h == 0,
                    "per_core_M must be divisible by out_subblock_h!");
                TT_FATAL(
                    program_config.per_core_N % program_config.out_subblock_w == 0,
                    "per_core_N must be divisible by out_subblock_w!");
            }
        },
        config);
    return config;
}

void Matmul::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 2);
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_FATAL(
        (input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE),
        "Inputs to matmul must be tilized");
    TT_FATAL(
        input_tensor_a.get_legacy_shape()[3] == input_tensor_b.get_legacy_shape()[2] &&
        "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op");  // A.K == B.K

    TT_FATAL(is_floating_point(input_tensor_a.get_dtype()), "Unsupported data format");
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE,
        "Operands to matmul need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
        "Operands to matmul need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");

    MatmulProgramConfig chosen_program_config = get_program_config(input_tensor_a, input_tensor_b, this);

    TT_FATAL(optional_input_tensors.size() == 1);
    const auto& optional_bias = optional_input_tensors.at(0);
    if (optional_bias.has_value()) {
        const auto& bias = optional_bias.value();
        TT_FATAL(bias.get_layout() == Layout::TILE, "Unsupported input layout");
        TT_FATAL(
            bias.get_legacy_shape() == Shape({1, 1, TILE_HEIGHT, input_tensor_b.get_legacy_shape()[3]}),
            "Unsupported bias shape");
    }

    if (this->untilize_out) {
        TT_FATAL((this->output_dtype == DataType::BFLOAT16) || (this->output_dtype == DataType::FLOAT32));
    }

    std::visit(
        [input_tensor_a, input_tensor_b, this](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            // TODO: For 1D and 2D mcasts, we don't check if tensor is single core or single row/col
            // We can uplift these variants to skip mcasting to support single core (1D) or single row/col (2D)
            if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                if (program_config.mcast_in0) {
                    if (input_tensor_a.is_sharded()) {
                        TT_FATAL(program_config.fuse_batch);
                        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED);
                        if (this->output_mem_config.is_sharded()) {
                            TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type);
                            TT_FATAL(
                                input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout);
                        }
                        TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);
                        uint32_t M =
                            (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1]
                                                       : input_tensor_a.get_legacy_shape()[-2]) /
                            TILE_HEIGHT;
                        uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                        uint32_t K = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;
                        auto shard_shape = input_tensor_a.shard_spec().value().shape;

                        // No padding
                        TT_FATAL(M == per_core_M);
                        TT_FATAL(per_core_M == (shard_shape[0] / TILE_HEIGHT));
                        TT_FATAL(K % program_config.in0_block_w == 0);
                        TT_FATAL((shard_shape[1] / TILE_WIDTH) % program_config.in0_block_w == 0);
                    }
                    if (this->output_mem_config.is_sharded()) {
                        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED);
                        uint32_t M =
                            (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1]
                                                       : input_tensor_a.get_legacy_shape()[-2]) /
                            TILE_HEIGHT;
                        uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        // No padding
                        TT_FATAL(M == per_core_M);

                        TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                    }
                } else {
                    if (input_tensor_a.memory_config().is_sharded()) {
                        TT_FATAL(program_config.fuse_batch);
                        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
                        if (this->output_mem_config.is_sharded()) {
                            TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type);
                            TT_FATAL(
                                input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout);
                        }
                        TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);
                        uint32_t M =
                            (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1]
                                                       : input_tensor_a.get_legacy_shape()[-2]) /
                            TILE_HEIGHT;
                        uint32_t K = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        auto shard_shape = input_tensor_a.shard_spec().value().shape;

                        TT_FATAL(div_up(M, per_core_M) == input_tensor_a.shard_spec().value().grid.num_cores());
                        TT_FATAL(per_core_M == (shard_shape[0] / TILE_HEIGHT));
                        TT_FATAL(K == program_config.in0_block_w);
                        TT_FATAL(program_config.in0_block_w == (shard_shape[1] / TILE_WIDTH));
                    }
                    if (this->output_mem_config.is_sharded()) {
                        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
                        uint32_t M =
                            (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1]
                                                       : input_tensor_a.get_legacy_shape()[-2]) /
                            TILE_HEIGHT;
                        uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        TT_FATAL(N == per_core_N);
                        TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                    }
                }
                TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                if (input_tensor_a.memory_config().is_sharded()) {
                    auto tensor_a_memory_layout = input_tensor_a.memory_config().memory_layout;
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
                    uint32_t K = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    auto shard_shape = input_tensor_a.shard_spec().value().shape;

                    TT_FATAL(
                        tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
                        tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);

                    if (tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                        if (program_config.transpose_mcast) {
                            TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR);
                        } else {
                            TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);
                        }
                        if (this->output_mem_config.is_sharded()) {
                            TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type);
                            TT_FATAL(
                                input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout);
                        }

                        TT_FATAL(K / (shard_shape[1] / TILE_WIDTH) == div_up(N, program_config.per_core_N));
                    } else if (tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                        TT_FATAL(K == program_config.in0_block_w);
                        TT_FATAL(program_config.in0_block_w == (shard_shape[1] / TILE_WIDTH));
                        TT_FATAL(
                            input_tensor_a.shard_spec()->grid.bounding_box().start.x ==
                            input_tensor_a.shard_spec()->grid.bounding_box().end.x);
                    }

                    TT_FATAL(per_core_M == (shard_shape[0] / TILE_HEIGHT));
                    TT_FATAL((shard_shape[1] / TILE_WIDTH) % program_config.in0_block_w == 0);
                }

                if (input_tensor_b.memory_config().is_sharded()) {
                    auto tensor_b_memory_layout = input_tensor_b.memory_config().memory_layout;
                    TT_FATAL(tensor_b_memory_layout == TensorMemoryLayout::WIDTH_SHARDED);
                    TT_FATAL(program_config.per_core_N == (input_tensor_b.shard_spec().value().shape[0] / TILE_WIDTH));
                    TT_FATAL(
                        input_tensor_b.shard_spec()->grid.bounding_box().start.y ==
                        input_tensor_b.shard_spec()->grid.bounding_box().end.y);
                }

                if (this->output_mem_config.is_sharded()) {
                    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                }
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                uint32_t M = input_tensor_a.get_legacy_shape()[-2] / TILE_HEIGHT;
                uint32_t total_M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
                uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                uint32_t K = input_tensor_a.get_legacy_shape()[-1];
                uint32_t per_core_M = program_config.per_core_M;
                uint32_t per_core_N = program_config.per_core_N;
                if (per_core_M > M) {
                    TT_FATAL(per_core_M % M == 0, "per_core_M must be a multiple of M if per_core_M > M!");
                    TT_FATAL(total_M % per_core_M == 0, "input a total height must be divisible by per_core_M!");
                } else {
                    TT_FATAL(M % per_core_M == 0, "per_core_M must divide M if per_core_M < M!");
                }
                TT_FATAL(N == per_core_N);
                if (input_tensor_a.is_sharded()) {
                    TT_FATAL(input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
                    auto in0_shard_shape = input_tensor_a.shard_spec().value().shape;

                    TT_FATAL(K == in0_shard_shape[1]);
                    TT_FATAL(in0_shard_shape[1] == program_config.in0_block_w * TILE_WIDTH);
                    TT_FATAL(per_core_M * TILE_HEIGHT == in0_shard_shape[0]);

                    if (input_tensor_b.is_sharded()) {
                        TT_FATAL(
                            input_tensor_a.memory_config().buffer_type == input_tensor_b.memory_config().buffer_type);
                        TT_FATAL(
                            input_tensor_a.memory_config().memory_layout ==
                            input_tensor_b.memory_config().memory_layout);
                        TT_FATAL(input_tensor_a.shard_spec().value().grid == input_tensor_b.shard_spec().value().grid);
                        TT_FATAL(
                            input_tensor_a.shard_spec().value().orientation ==
                            input_tensor_b.shard_spec().value().orientation);
                    }
                    if (this->output_mem_config.is_sharded()) {
                        TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type);
                        TT_FATAL(input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout);
                    }
                }

                bool broadcast_batch =
                    (input_tensor_a.get_legacy_shape()[0] * input_tensor_a.get_legacy_shape()[1] > 1 and
                     input_tensor_b.get_legacy_shape()[0] * input_tensor_b.get_legacy_shape()[1] == 1);
                TT_FATAL(!broadcast_batch);

                if (input_tensor_b.is_sharded()) {
                    TT_FATAL(per_core_M % M == 0, "per_core_M must be a multiple of M if input b is sharded!");
                    TT_FATAL(input_tensor_b.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
                    auto in1_shard_shape = input_tensor_b.shard_spec().value().shape;
                    TT_FATAL(in1_shard_shape[1] == input_tensor_b.get_legacy_shape()[-1]);
                    TT_FATAL(per_core_N * TILE_HEIGHT == in1_shard_shape[1]);
                    TT_FATAL(in1_shard_shape[0] % K == 0);
                }
                if (this->output_mem_config.is_sharded()) {
                    TT_FATAL(this->output_mem_config.memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
                    TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                }
            } else {
                TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
                TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
                TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
            }
            if constexpr (
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig> ||
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig> ||
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                TT_FATAL(
                    (input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH) % program_config.in0_block_w == 0,
                    "Kt must be divisible by in0_block_w");
                TT_FATAL(
                    program_config.per_core_M % program_config.out_subblock_h == 0,
                    "per_core_M must be divisible by out_subblock_h");
                TT_FATAL(
                    program_config.per_core_N % program_config.out_subblock_w == 0,
                    "per_core_N must be divisible by out_subblock_w");
            }
        },
        chosen_program_config);
}

std::vector<Shape> Matmul::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto input_shape_a = input_tensors.at(0).get_legacy_shape();
    const auto input_shape_b = input_tensors.at(1).get_legacy_shape();

    auto output_shape = input_shape_a;
    output_shape[-1] = input_shape_b[-1];
    auto dimensions_pads = std::vector<Padding::PadDimension>();
    for (auto index = 0; index < input_shape_a.rank() - 1; index++) {
        dimensions_pads.push_back(input_shape_a.padding()[index]);
    }
    dimensions_pads.push_back(input_shape_b.padding()[input_shape_b.rank() - 1]);
    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    return {Shape(output_shape, padding)};
}

std::vector<Tensor> Matmul::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto output_layout = this->untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
    if (this->output_mem_config.is_sharded()) {
        MatmulProgramConfig chosen_program_config = get_program_config(input_tensor_a, input_tensor_b, this);
        return std::visit(
            [&](const auto& program_config) -> std::vector<Tensor> {
                using ProgramConfigType = std::decay_t<decltype(program_config)>;
                if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                    uint32_t M =
                        (program_config.fuse_batch ? input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1]
                                                   : input_tensor_a.get_legacy_shape()[-2]) /
                        TILE_HEIGHT;
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
                    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
                    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    CoreRangeSet all_cores =
                        num_cores_to_corerange_set(num_cores, program_config.compute_with_storage_grid_size, true);
                    ShardSpec shard_spec = ShardSpec{
                        all_cores, {per_core_M * TILE_HEIGHT, per_core_N * TILE_WIDTH}, ShardOrientation::ROW_MAJOR};
                    auto mem_config = this->output_mem_config;
                    mem_config.shard_spec = shard_spec;
                    return {create_device_tensor(
                        this->compute_output_shapes(input_tensors).at(0),
                        this->output_dtype,
                        output_layout,
                        input_tensor_a.device(),
                        mem_config)};
                } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
                    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
                    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    CoreRangeSet all_cores({});
                    ShardOrientation shard_orientation;
                    if (program_config.transpose_mcast) {
                        all_cores = CoreRangeSet({CoreRange({0, 0}, {num_blocks_y - 1, num_blocks_x - 1})});
                        shard_orientation = ShardOrientation::COL_MAJOR;
                    } else {
                        all_cores = CoreRangeSet({CoreRange({0, 0}, {num_blocks_x - 1, num_blocks_y - 1})});
                        shard_orientation = ShardOrientation::ROW_MAJOR;
                    }
                    ShardSpec shard_spec =
                        ShardSpec{all_cores, {per_core_M * TILE_HEIGHT, per_core_N * TILE_WIDTH}, shard_orientation};
                    auto mem_config = this->output_mem_config;
                    mem_config.shard_spec = shard_spec;
                    return {create_device_tensor(
                        this->compute_output_shapes(input_tensors).at(0),
                        this->output_dtype,
                        output_layout,
                        input_tensor_a.device(),
                        mem_config)};
                } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                    uint32_t M = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
                    uint32_t N = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
                    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
                    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    ShardOrientation shard_orientation = ShardOrientation::COL_MAJOR;
                    if (input_tensor_a.is_sharded()) {
                        shard_orientation = input_tensor_a.shard_spec().value().orientation;
                    } else if (input_tensor_b.is_sharded()) {
                        shard_orientation = input_tensor_b.shard_spec().value().orientation;
                    }

                    CoreRangeSet all_cores = num_cores_to_corerange_set(
                        num_cores,
                        program_config.compute_with_storage_grid_size,
                        shard_orientation == ShardOrientation::ROW_MAJOR);
                    ShardSpec shard_spec =
                        ShardSpec{all_cores, {per_core_M * TILE_HEIGHT, per_core_N * TILE_WIDTH}, shard_orientation};
                    auto mem_config = this->output_mem_config;
                    mem_config.shard_spec = shard_spec;
                    return {create_device_tensor(
                        this->compute_output_shapes(input_tensors).at(0),
                        this->output_dtype,
                        output_layout,
                        input_tensor_a.device(),
                        mem_config)};
                } else {
                    TT_FATAL(false, "Unsupported op for output sharding");
                    return {};
                }
            },
            chosen_program_config);
    }
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

uint32_t get_per_core_for_multiple_blocks(uint32_t per_core, uint32_t tiles) {
    static std::vector<uint32_t> divisors = {2, 3, 5, 7, 11, 13};
    uint32_t num_blocks = (tiles - 1) / per_core + 1;
    while (per_core > 1 && num_blocks == 1) {
        bool divided = false;
        for (uint32_t divisor : divisors) {
            if (per_core % divisor == 0) {
                per_core /= divisor;
                divided = true;
                break;
            }
        }
        if (!divided) {
            per_core = 1;
        }
        num_blocks = (tiles - 1) / per_core + 1;
    }
    return per_core;
}

operation::ProgramWithCallbacks Matmul::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& bias = optional_input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    tt::tt_metal::DataType output_dtype = this->output_dtype;

    bool fuse_batch = true;
    // TODO: If input_tensor_a.get_legacy_shape()[0] * input_tensor_a.get_legacy_shape()[1] == 1, does matmuls work if
    // we treat it as bmm
    // TODO: Only for MatmulMultiCoreReuseProgramConfig we allow this as single core matmul/bmm
    bool broadcast_batch = this->bcast_batch;

    MatmulProgramConfig chosen_program_config = get_program_config(input_tensor_a, input_tensor_b, this);

    return std::visit(
        [&](const auto& program_config) -> operation::ProgramWithCallbacks {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                TT_FATAL(!bias.has_value(), "Bias is not supported for MatmulMultiCoreReuseProgramConfig!");
                // TODO: fuse_batch doesn't do anything for this variant! Code is doing fuse_batch=false
                return bmm_multi_core_reuse_optimized(
                    input_tensor_a,
                    input_tensor_b,
                    output_tensor,
                    broadcast_batch,
                    program_config.compute_with_storage_grid_size,
                    output_dtype,
                    this->compute_kernel_config,
                    program_config.in0_block_w,
                    program_config.out_subblock_h,
                    program_config.out_subblock_w,
                    program_config.per_core_M,
                    program_config.per_core_N,
                    /*fuse_batch=*/false,
                    this->untilize_out);
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                return matmul_multi_core_reuse_mcast_2d_optimized(
                    input_tensor_a,
                    input_tensor_b,
                    bias,
                    output_tensor,
                    broadcast_batch,
                    program_config.compute_with_storage_grid_size,
                    this->compute_kernel_config,
                    program_config.in0_block_w,
                    program_config.out_subblock_h,
                    program_config.out_subblock_w,
                    program_config.per_core_M,
                    program_config.per_core_N,
                    program_config.fuse_batch,
                    program_config.transpose_mcast,
                    program_config.fused_activation,
                    this->untilize_out);
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                return matmul_multi_core_reuse_mcast_1d_optimized(
                    input_tensor_a,
                    input_tensor_b,
                    bias,
                    output_tensor,
                    broadcast_batch,
                    program_config.compute_with_storage_grid_size,
                    this->compute_kernel_config,
                    program_config.in0_block_w,
                    program_config.out_subblock_h,
                    program_config.out_subblock_w,
                    program_config.per_core_M,
                    program_config.per_core_N,
                    program_config.fuse_batch,
                    program_config.fused_activation,
                    program_config.mcast_in0,
                    this->untilize_out);
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreNonOptimizedReuseProgramConfig>) {
                TT_FATAL(!bias.has_value(), "Bias is not supported for matmul multi core non-optimized reuse");
                return matmul_multi_core_reuse(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreProgramConfig>) {
                TT_FATAL(!bias.has_value(), "Bias is not supported for matmul multi core");
                return matmul_multi_core(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
            } else {
                TT_THROW("Unrecognized Config");
            }
        },
        chosen_program_config);
}

Tensor matmul_1d(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    std::optional<const Tensor> bias,
    std::optional<MatmulMultiCoreReuseMultiCast1DProgramConfig> program_config,
    const MemoryConfig& mem_config,
    std::optional<const DataType> output_dtype,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool untilize_out) {
    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({input_tensor_a, input_tensor_b}, {bias}))};
    operation::launch_op(
        [program_config, mem_config, output_dtype, compute_kernel_config, untilize_out](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor_a = input_tensors.at(0);
            const auto& input_tensor_b = input_tensors.at(1);
            if (!program_config.has_value()) {
                program_config = bmm_op_utils::get_mcast_1d_config(
                    input_tensor_a,
                    input_tensor_b,
                    false /* fuse_batch */,
                    std::nullopt /* fused_activation */,
                    true /* mcast_in0 */,
                    false /* out_sharded */,
                    std::nullopt /* compute_with_storage_grid_size */,
                    compute_kernel_config);
            }
            auto kernel_config_val =
                init_device_compute_kernel_config(input_tensor_a.device()->arch(), compute_kernel_config);
            return {operations::primary::matmul(
                input_tensor_a,
                input_tensor_b,
                optional_input_tensors.at(0),
                program_config.value(),
                mem_config,
                output_dtype,
                kernel_config_val,
                untilize_out)};
        },
        {input_tensor_a, input_tensor_b},
        output_tensors,
        {bias});
    return output_tensors.at(0);
}

operation::OpPerformanceModel Matmul::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ::create_op_performance_model_for_matmul(
        input_tensors, optional_input_tensors, output_tensors, this->compute_kernel_config);
}

MatmulProgramConfig create_matmul_1d_systolic_array_program_config(
    const ttnn::types::Shape& input_shape_a,
    const ttnn::types::Shape& input_shape_b,
    const CoreCoord& core_coord,
    const std::optional<const UnaryWithParam> fused_activation,
    const bool fp32_dest_acc_en) {
    auto a_padded_shape = input_shape_a.with_tile_padding();
    auto b_padded_shape = input_shape_b.with_tile_padding();
    auto k_size = a_padded_shape[-1];
    auto m_size = a_padded_shape[-2];
    auto n_size = b_padded_shape[-1];
    uint32_t batch_size_a = get_batch_size(a_padded_shape);
    uint32_t batch_size_b = get_batch_size(b_padded_shape);
    bool input_b_is_batched = batch_size_b > 1;
    TT_FATAL(batch_size_b == 1, "Second input cannot be currently batched when running matmul using 1d systolic array");
    TT_FATAL(
        (batch_size_a * m_size) % ttnn::TILE_SIZE == 0 && k_size % ttnn::TILE_SIZE == 0 &&
            n_size % ttnn::TILE_SIZE == 0,
        "The last two dimensions of the first tensor and the last dimension of the second tensor must be a multiple of "
        "tile size");
    uint32_t batch_and_m_tiles = (batch_size_a * m_size) / ttnn::TILE_SIZE;
    uint32_t k_tiles = k_size / ttnn::TILE_SIZE;
    uint32_t n_tiles = n_size / ttnn::TILE_SIZE;
    uint32_t num_cores = core_coord.x * core_coord.y;
    bool is_tall = batch_and_m_tiles > n_tiles;
    bool is_wide = !is_tall;
    uint32_t batch_and_m_tiles_per_core;
    uint32_t k_tiles_per_core;
    uint32_t n_tiles_per_core;
    if (is_tall) {
        batch_and_m_tiles_per_core = div_up(batch_and_m_tiles, num_cores);
        k_tiles_per_core = div_up(k_tiles, num_cores);
        n_tiles_per_core = n_tiles;
    } else {
        batch_and_m_tiles_per_core = batch_and_m_tiles;
        k_tiles_per_core = div_up(k_tiles, num_cores);
        n_tiles_per_core = div_up(n_tiles, num_cores);
    }
    while (k_tiles % k_tiles_per_core != 0) {
        k_tiles_per_core -= 1;
    }
    auto matmul_params =
        bmm_op_utils::get_subblock_sizes(batch_and_m_tiles_per_core, n_tiles_per_core, fp32_dest_acc_en);
    uint32_t out_subblock_h = std::get<0>(matmul_params);
    uint32_t out_subblock_w = std::get<1>(matmul_params);
    return MatmulMultiCoreReuseMultiCast1DProgramConfig{
        .compute_with_storage_grid_size = {core_coord.x, core_coord.y},
        .in0_block_w = k_tiles_per_core,
        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
        .per_core_M = batch_and_m_tiles_per_core,
        .per_core_N = n_tiles_per_core,
        .fuse_batch = true,
        .fused_activation = fused_activation,
        .mcast_in0 = is_wide,
    };
}

MatmulProgramConfig create_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const CoreCoord> user_core_coord,
    std::optional<UnaryWithParam> fused_activation,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto a_shape = input_tensor_a.get_shape();
    auto b_shape = input_tensor_b.get_shape();
    auto a_padded_shape = a_shape.with_tile_padding();
    auto b_padded_shape = b_shape.with_tile_padding();
    auto inteneded_k_size_of_a = a_shape[-1];
    auto inteneded_k_size_of_b = b_shape[-2];
    auto k_size = a_padded_shape[-1];
    auto m_size = a_padded_shape[-2];
    auto n_size = b_padded_shape[-1];
    uint32_t batch_size_a = get_batch_size(a_padded_shape);
    uint32_t batch_size_b = get_batch_size(b_padded_shape);
    bool input_b_is_batched = batch_size_b > 1;
    bool any_size_within_tile = k_size <= ttnn::TILE_SIZE || m_size <= ttnn::TILE_SIZE || n_size <= ttnn::TILE_SIZE;
    auto input_tensor_a_memory_config = input_tensor_a.memory_config();
    auto input_tensor_b_memory_config = input_tensor_b.memory_config();
    bool fp32_dest_acc_en = bmm_op_utils::get_fp32_dest_acc_en(compute_kernel_config);
    bool a_is_sharded = input_tensor_a.is_sharded();
    TT_FATAL(inteneded_k_size_of_a == inteneded_k_size_of_b, "The k dimension does not match between tensors");
    TT_FATAL(
        (batch_size_a * m_size) % ttnn::TILE_SIZE == 0 && k_size % ttnn::TILE_SIZE == 0 &&
            n_size % ttnn::TILE_SIZE == 0,
        "The last two dimensions of the first tensor and the last dimension of the second tensor must be a multiple of "
        "tile size");
    auto core_coord = input_tensor_a.device()->compute_with_storage_grid_size();
    bool has_user_core_coord = user_core_coord.has_value();
    if (has_user_core_coord) {
        auto x = user_core_coord.value().x;
        auto y = user_core_coord.value().y;
        if (x <= core_coord.x && y <= core_coord.y) {
            core_coord = user_core_coord.value();
        }
    }

    uint32_t m_tiles_per_core;
    uint32_t n_tiles_per_core;
    uint32_t k_tiles_per_core;
    if (input_b_is_batched) {
        TT_FATAL(!fused_activation.has_value(), "Cannot use activation with batched input b");
        if (!a_is_sharded && !input_tensor_b.is_sharded()) {
            m_tiles_per_core = div_up(m_size, ttnn::TILE_SIZE);
            n_tiles_per_core = div_up(n_size, ttnn::TILE_SIZE);
            k_tiles_per_core = 1;  // TODO(arakhmati): Can it be more than 1 without running out of memory?
        } else if (a_is_sharded) {
            TT_FATAL(
                input_tensor_a_memory_config.memory_layout != TensorMemoryLayout::WIDTH_SHARDED,
                "MatmulMultiCoreReuseProgramConfig: Cannot be width sharded");
            auto shard_shape = input_tensor_a_memory_config.shard_spec.value().shape;
            uint32_t n = b_shape[-1] / ttnn::TILE_SIZE;
            m_tiles_per_core = shard_shape[0] / ttnn::TILE_SIZE;
            n_tiles_per_core = n;
            k_tiles_per_core = shard_shape[1] / ttnn::TILE_SIZE;
        } else {
            TT_FATAL(
                input_tensor_b_memory_config.memory_layout != TensorMemoryLayout::WIDTH_SHARDED,
                "MatmulMultiCoreReuseProgramConfig: Cannot be width sharded");
            auto shard_shape = input_tensor_b_memory_config.shard_spec.value().shape;
            m_tiles_per_core = div_up(m_size, ttnn::TILE_SIZE);
            n_tiles_per_core = shard_shape[1] / ttnn::TILE_SIZE;
            k_tiles_per_core = 1;
        }

        auto matmul_params = bmm_op_utils::get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, fp32_dest_acc_en);
        uint32_t out_subblock_h = std::get<0>(matmul_params);
        uint32_t out_subblock_w = std::get<1>(matmul_params);

        return MatmulMultiCoreReuseProgramConfig{
            .compute_with_storage_grid_size = {core_coord.x, core_coord.y},
            .in0_block_w = k_tiles_per_core,
            .out_subblock_h = out_subblock_h,
            .out_subblock_w = out_subblock_w,
            .per_core_M = m_tiles_per_core,
            .per_core_N = n_tiles_per_core,
        };
    }

    auto height = batch_size_a * m_size;
    auto width = n_size;
    auto height_width_ratio = (height > width) ? height / width : width / height;
    if (height_width_ratio > 8 || any_size_within_tile) {
        return create_matmul_1d_systolic_array_program_config(
            a_shape, b_shape, core_coord, fused_activation, fp32_dest_acc_en);
    }
    if (!a_is_sharded) {
        m_tiles_per_core = (uint32_t)std::ceil((((double)batch_size_a * m_size) / ttnn::TILE_SIZE) / core_coord.y);
        n_tiles_per_core = (uint32_t)std::ceil((double)n_size / ttnn::TILE_SIZE / core_coord.x);
        k_tiles_per_core = 4;  // TODO(arakhmati): What is a good starting point?
        while ((k_size / ttnn::TILE_SIZE) % k_tiles_per_core != 0) {
            k_tiles_per_core -= 1;
        }
    } else {
        TT_FATAL(
            input_tensor_a_memory_config.memory_layout == TensorMemoryLayout::BLOCK_SHARDED,
            "MatmulMultiCoreReuseMultiCastProgramConfig: Must be block sharded");
        uint32_t k = a_shape[-1] / ttnn::TILE_SIZE;
        uint32_t n = b_shape[-1] / ttnn::TILE_SIZE;
        auto shard_shape = input_tensor_a_memory_config.shard_spec.value().shape;
        m_tiles_per_core = shard_shape[0] / ttnn::TILE_SIZE;
        n_tiles_per_core = (n * shard_shape[1]) / (k * ttnn::TILE_SIZE);
        k_tiles_per_core = shard_shape[1] / ttnn::TILE_SIZE;
    }

    auto matmul_params = bmm_op_utils::get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, fp32_dest_acc_en);
    uint32_t out_subblock_h = std::get<0>(matmul_params);
    uint32_t out_subblock_w = std::get<1>(matmul_params);

    return MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {core_coord.x, core_coord.y},
        .in0_block_w = k_tiles_per_core,
        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
        .per_core_M = m_tiles_per_core,
        .per_core_N = n_tiles_per_core,
        .transpose_mcast = false,
        .fused_activation = fused_activation,
    };
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
