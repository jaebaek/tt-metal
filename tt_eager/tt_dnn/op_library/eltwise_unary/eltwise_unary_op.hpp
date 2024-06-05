// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/types.hpp"

namespace tt {

namespace tt_metal {

// These operations have a corresponding LLK available
enum class UnaryOpType {
    EXP,
    RECIP,
    GELU,
    RELU,
    SQRT,
    SIGMOID,
    LOG,
    TANH,
    LOG2,
    LOG10,
    SIN,
    COS,
    ABS,
    SIGN,
    SQUARE,
    EQZ,
    NEZ,
    GTZ,
    LTZ,
    GEZ,
    LEZ,
    RELU_MAX,
    RELU_MIN,
    POWER,
    LEAKY_RELU,
    ELU,
    EXP2,
    HEAVISIDE,
    EXPM1,
    SIGNBIT,
    ASIN,
    ACOS,
    RSQRT,
    RELU6,
    ATAN,
    ERF,
    ERFC,
    ISINF,
    ISPOSINF,
    ISNEGINF,
    ISNAN,
    LOGICAL_NOT_UNARY,
    ISFINITE,
    ERFINV,
    I0,
    TAN,
    RSUB,
    RDIV,
    SILU,
    SOFTPLUS,
    IDENTITY,
    NEG,
    ADD_UNARY_SFPU,
    SUB_UNARY_SFPU,
    MUL_UNARY_SFPU,
    DIV_UNARY_SFPU,
    IDENTITY_UINT32,
    UNARY_NE,
    UNARY_GT,
    UNARY_LT,
    TILED_PROD,
    TYPECAST
};

template <typename T>
bool is_parametrized_type(T val) {
    switch (val) {
        case UnaryOpType::RELU_MAX:
        case UnaryOpType::RELU_MIN:
        case UnaryOpType::POWER:
        case UnaryOpType::LEAKY_RELU:
        case UnaryOpType::ELU:
        case UnaryOpType::GELU:
        case UnaryOpType::RSQRT:
        case UnaryOpType::HEAVISIDE:
        case UnaryOpType::ERF:
        case UnaryOpType::ERFC:
        case UnaryOpType::RSUB:
        case UnaryOpType::RDIV:
        case UnaryOpType::EXP:
        case UnaryOpType::SOFTPLUS:
        case UnaryOpType::ADD_UNARY_SFPU:
        case UnaryOpType::SUB_UNARY_SFPU:
        case UnaryOpType::MUL_UNARY_SFPU:
        case UnaryOpType::DIV_UNARY_SFPU:
        case UnaryOpType::UNARY_NE:
        case UnaryOpType::UNARY_GT:
        case UnaryOpType::UNARY_LT: return true;
        default: return false;
    }
    return false;
}

struct UnaryWithParam {
    UnaryOpType op_type;
    std::vector<float> params;

    UnaryWithParam(UnaryOpType op_type, const std::vector<float>& params) : op_type{op_type}, params{params} {}
    UnaryWithParam(UnaryOpType op_type, float param) : op_type{op_type}, params{param} {}
    UnaryWithParam(UnaryOpType op_type) : op_type{op_type} {}

    bool has_parameter() const { return params.size() > 0; }

    static constexpr auto attribute_names = std::make_tuple("op_type", "param");
    const auto attribute_values() const { return std::make_tuple(std::cref(this->op_type), std::cref(this->params)); }
};

inline UnaryWithParam string_to_unary_with_param(const std::string& name) {
    if (name == "relu")
        return UnaryWithParam(UnaryOpType::RELU);
    else if (name == "gelu")
        return UnaryWithParam(UnaryOpType::GELU, static_cast<float>(true));
    else if (name == "silu")
        return UnaryWithParam(UnaryOpType::SILU);
    else if (name == "sigmoid")
        return UnaryWithParam(UnaryOpType::SIGMOID);
    else if (name == "sqrt")
        return UnaryWithParam(UnaryOpType::SQRT);
    else if (name == "exp")
        return UnaryWithParam(UnaryOpType::EXP, static_cast<float>(true));
    else if (name == "recip")
        return UnaryWithParam(UnaryOpType::RECIP);
    else if (name == "log")
        return UnaryWithParam(UnaryOpType::LOG);
    else if (name == "tanh")
        return UnaryWithParam(UnaryOpType::TANH);
    else if (name == "log2")
        return UnaryWithParam(UnaryOpType::LOG2);
    else if (name == "log10")
        return UnaryWithParam(UnaryOpType::LOG10);
    else if (name == "sin")
        return UnaryWithParam(UnaryOpType::SIN);
    else if (name == "cos")
        return UnaryWithParam(UnaryOpType::COS);
    else if (name == "abs")
        return UnaryWithParam(UnaryOpType::ABS);
    else if (name == "sign")
        return UnaryWithParam(UnaryOpType::SIGN);
    else if (name == "square")
        return UnaryWithParam(UnaryOpType::SQUARE);
    else if (name == "softplus")
        return UnaryWithParam(UnaryOpType::SOFTPLUS);
    TT_THROW("Unknown unary op: " + name);
}

enum class UnaryOpParallelizationStrategy { MULTI_CORE, SHARDED_MULTI_CORE };

struct EltwiseUnary {
    const std::vector<UnaryWithParam> op_chain;
    const MemoryConfig output_mem_config;
    bool fp32_dest_acc_en;
    DataType output_dtype;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    UnaryOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("op_chain", "output_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->op_chain), std::cref(this->output_mem_config));
    }

    const operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

operation::ProgramWithCallbacks eltwise_unary_sharded(
    const Tensor& a, Tensor& output, const std::vector<UnaryWithParam> op_chain, bool fp32_dest_acc_en);
operation::ProgramWithCallbacks eltwise_unary_multi_core(
    const Tensor& a, Tensor& output, const std::vector<UnaryWithParam> op_chain, bool fp32_dest_acc_en);

inline Tensor run_eltwise_unary(
    const Tensor& input_tensor,
    const std::vector<UnaryWithParam>& ops_chain,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_FATAL(ops_chain.size() > 0, "At least 1 unary op must be specified");
    DataType output_dtype = (ops_chain[0].op_type == UnaryOpType::TYPECAST) ? DataType::UINT32 : input_tensor.get_dtype();
    bool fp32_dest_acc_en =
        output_dtype == DataType::UINT32 or
        input_tensor.get_dtype() == DataType::UINT32 or
        input_tensor.get_dtype() ==
            DataType::INT32;  // MT: Currently only uint32/int32 is moved to DST directly, fp32 is converted to fp16b
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    if (output_mem_config.is_sharded()) {
        operation::launch_op(
            [ops_chain, output_mem_config, fp32_dest_acc_en, output_dtype](
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                return operation::run_without_autoformat(
                    EltwiseUnary{ops_chain, output_mem_config, fp32_dest_acc_en, output_dtype}, input_tensors);
            },
            {input_tensor},
            output_tensors);
    } else {
        operation::launch_with_autoformat(
            [ops_chain, output_mem_config, fp32_dest_acc_en, output_dtype](
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                Tensor input_tensor = input_tensors.at(0);
                Shape pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
                FormatParams input_format_params = {
                    .pad_shape = pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
                return operation::run_with_autoformat(
                    EltwiseUnary{ops_chain, output_mem_config, fp32_dest_acc_en, output_dtype},
                    {input_tensor},
                    {input_format_params},
                    {Layout::TILE});
            },
            {input_tensor},
            output_tensors);
    }
    return output_tensors.at(0);
}

inline Tensor run_eltwise_unary(
    uint8_t cq_id,
    const Tensor& input_tensor,
    const std::vector<UnaryWithParam>& ops_chain,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_FATAL(ops_chain.size() > 0, "At least 1 unary op must be specified");
    DataType output_dtype = (ops_chain[0].op_type == UnaryOpType::TYPECAST) ? DataType::UINT32 : input_tensor.get_dtype();
    bool fp32_dest_acc_en =
        output_dtype == DataType::UINT32 or
        input_tensor.get_dtype() == DataType::UINT32 or
        input_tensor.get_dtype() ==
            DataType::INT32;  // MT: Currently only uint32/int32 is moved to DST directly, fp32 is converted to fp16b
    return operation::run(
               EltwiseUnary{ops_chain, output_mem_config, fp32_dest_acc_en, output_dtype},
               {input_tensor}, {}, {}, cq_id)
        .at(0);
}

template <UnaryOpType unary_op_type, typename T = float>
struct make_eltwise_unary_with_param {
    Tensor operator()(
        const Tensor& input_tensor,
        T param,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        return run_eltwise_unary(
            input_tensor, {UnaryWithParam(unary_op_type, static_cast<float>(param))}, output_mem_config);
    }
};

template <UnaryOpType unary_op_type>
struct make_eltwise_unary {
    Tensor operator()(
        const Tensor& input_tensor,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        return run_eltwise_unary(input_tensor, {UnaryWithParam(unary_op_type)}, output_mem_config);
    }
};

template <UnaryOpType unary_op_type, typename T = float>
struct make_eltwise_symmetric_binop_unary_with_param {
    Tensor operator()(
        const Tensor& input_tensor,
        T param,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        return run_eltwise_unary(
            input_tensor, {UnaryWithParam(unary_op_type, static_cast<float>(param))}, output_mem_config);
    }
    Tensor operator()(
        T param,
        const Tensor& input_tensor,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        return run_eltwise_unary(
            input_tensor, {UnaryWithParam(unary_op_type, static_cast<float>(param))}, output_mem_config);
    }
};

template <UnaryOpType unary_op_type, UnaryOpType unary_op_rev_type, typename T = float>
struct make_eltwise_asymmetric_binop_unary_with_param {
    Tensor operator()(
        const Tensor& input_tensor,
        T param,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        return run_eltwise_unary(
            input_tensor, {UnaryWithParam(unary_op_type, static_cast<float>(param))}, output_mem_config);
    }
    Tensor operator()(
        T param,
        const Tensor& input_tensor,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        return run_eltwise_unary(
            input_tensor, {UnaryWithParam(unary_op_rev_type, static_cast<float>(param))}, output_mem_config);
    }
};

inline Tensor sqrt(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return run_eltwise_unary(input_tensor, {UnaryWithParam(UnaryOpType::SQRT)}, output_mem_config);
}

inline Tensor sqrt(
    uint8_t cq_id,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return run_eltwise_unary(cq_id, input_tensor, {UnaryWithParam(UnaryOpType::SQRT)}, output_mem_config);
}

constexpr auto recip = make_eltwise_unary<UnaryOpType::RECIP>{};
constexpr auto relu = make_eltwise_unary<UnaryOpType::RELU>{};
constexpr auto relu6 = make_eltwise_unary<UnaryOpType::RELU6>{};
constexpr auto sigmoid = make_eltwise_unary<UnaryOpType::SIGMOID>{};
constexpr auto tiled_prod = make_eltwise_unary<UnaryOpType::TILED_PROD>{};
constexpr auto log = make_eltwise_unary<UnaryOpType::LOG>{};
constexpr auto tanh = make_eltwise_unary<UnaryOpType::TANH>{};
constexpr auto log2 = make_eltwise_unary<UnaryOpType::LOG2>{};
constexpr auto log10 = make_eltwise_unary<UnaryOpType::LOG10>{};
constexpr auto exp2 = make_eltwise_unary<UnaryOpType::EXP2>{};
constexpr auto expm1 = make_eltwise_unary<UnaryOpType::EXPM1>{};
constexpr auto sin = make_eltwise_unary<UnaryOpType::SIN>{};
constexpr auto cos = make_eltwise_unary<UnaryOpType::COS>{};
constexpr auto asin = make_eltwise_unary<UnaryOpType::ASIN>{};
constexpr auto acos = make_eltwise_unary<UnaryOpType::ACOS>{};
constexpr auto abs = make_eltwise_unary<UnaryOpType::ABS>{};
constexpr auto isfinite = make_eltwise_unary<UnaryOpType::ISFINITE>{};
constexpr auto isinf = make_eltwise_unary<UnaryOpType::ISINF>{};
constexpr auto isposinf = make_eltwise_unary<UnaryOpType::ISPOSINF>{};
constexpr auto isneginf = make_eltwise_unary<UnaryOpType::ISNEGINF>{};
constexpr auto isnan = make_eltwise_unary<UnaryOpType::ISNAN>{};
constexpr auto sign = make_eltwise_unary<UnaryOpType::SIGN>{};
constexpr auto signbit = make_eltwise_unary<UnaryOpType::SIGNBIT>{};
constexpr auto square = make_eltwise_unary<UnaryOpType::SQUARE>{};
constexpr auto atan = make_eltwise_unary<UnaryOpType::ATAN>{};
constexpr auto eqz = make_eltwise_unary<UnaryOpType::EQZ>{};
constexpr auto nez = make_eltwise_unary<UnaryOpType::NEZ>{};
constexpr auto gez = make_eltwise_unary<UnaryOpType::GEZ>{};
constexpr auto lez = make_eltwise_unary<UnaryOpType::LEZ>{};
constexpr auto gtz = make_eltwise_unary<UnaryOpType::GTZ>{};
constexpr auto ltz = make_eltwise_unary<UnaryOpType::LTZ>{};
constexpr auto logical_not_unary = make_eltwise_unary<UnaryOpType::LOGICAL_NOT_UNARY>{};
constexpr auto i0 = make_eltwise_unary<UnaryOpType::I0>{};
constexpr auto erfinv = make_eltwise_unary<UnaryOpType::ERFINV>{};
constexpr auto tan = make_eltwise_unary<UnaryOpType::TAN>{};
constexpr auto neg = make_eltwise_unary<UnaryOpType::NEG>{};
constexpr auto relu_max = make_eltwise_unary_with_param<UnaryOpType::RELU_MAX>{};
constexpr auto relu_min = make_eltwise_unary_with_param<UnaryOpType::RELU_MIN>{};
constexpr auto power = make_eltwise_unary_with_param<UnaryOpType::POWER, uint32_t>{};
constexpr auto leaky_relu = make_eltwise_unary_with_param<UnaryOpType::LEAKY_RELU>{};
constexpr auto prelu = leaky_relu;
constexpr auto elu = make_eltwise_unary_with_param<UnaryOpType::ELU>{};
constexpr auto heaviside = make_eltwise_unary_with_param<UnaryOpType::HEAVISIDE>{};
constexpr auto unary_ne = make_eltwise_unary_with_param<UnaryOpType::UNARY_NE>{};
constexpr auto rsub = make_eltwise_unary_with_param<UnaryOpType::RSUB>{};
constexpr auto silu = make_eltwise_unary<UnaryOpType::SILU>{};
constexpr auto identity = make_eltwise_unary<UnaryOpType::IDENTITY>{};
constexpr auto identity_uint32 = make_eltwise_unary<UnaryOpType::IDENTITY_UINT32>{};
constexpr auto eltwise_typecast = make_eltwise_unary<UnaryOpType::TYPECAST>{};
constexpr auto add_unary_sfpu = make_eltwise_symmetric_binop_unary_with_param<UnaryOpType::ADD_UNARY_SFPU>{};
constexpr auto mul_unary_sfpu = make_eltwise_symmetric_binop_unary_with_param<UnaryOpType::MUL_UNARY_SFPU>{};
constexpr auto unary_gt = make_eltwise_unary_with_param<UnaryOpType::UNARY_GT>{};
constexpr auto unary_lt = make_eltwise_unary_with_param<UnaryOpType::UNARY_LT>{};
constexpr auto sub_unary_sfpu =
    make_eltwise_asymmetric_binop_unary_with_param<UnaryOpType::SUB_UNARY_SFPU, UnaryOpType::RSUB>{};
constexpr auto div_unary_sfpu =
    make_eltwise_asymmetric_binop_unary_with_param<UnaryOpType::DIV_UNARY_SFPU, UnaryOpType::RDIV>{};

inline Tensor exp(
    const Tensor& input_tensor,
    bool fast_and_approx,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return make_eltwise_unary_with_param<UnaryOpType::EXP>{}(input_tensor, fast_and_approx, output_mem_config);
}
inline Tensor exp(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return exp(input_tensor, false, output_mem_config);
}

inline Tensor erf(
    const Tensor& input_tensor,
    bool fast_and_approx = true,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return make_eltwise_unary_with_param<UnaryOpType::ERF>{}(input_tensor, fast_and_approx, output_mem_config);
}
inline Tensor erfc(
    const Tensor& input_tensor,
    bool fast_and_approx = true,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return make_eltwise_unary_with_param<UnaryOpType::ERFC>{}(input_tensor, fast_and_approx, output_mem_config);
}

inline Tensor gelu(
    const Tensor& input_tensor,
    bool fast_and_approx = true,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return make_eltwise_unary_with_param<UnaryOpType::GELU>{}(input_tensor, fast_and_approx, output_mem_config);
}
inline Tensor rsqrt(
    const Tensor& input_tensor,
    bool fast_and_approx = true,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return make_eltwise_unary_with_param<UnaryOpType::RSQRT>{}(input_tensor, fast_and_approx, output_mem_config);
}

inline Tensor log_sigmoid(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return run_eltwise_unary(
        input_tensor, {UnaryWithParam(UnaryOpType::SIGMOID), UnaryWithParam(UnaryOpType::LOG)}, output_mem_config);
}

inline Tensor sigmoid_accurate(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return run_eltwise_unary(
        input_tensor,
        {UnaryWithParam(UnaryOpType::NEG),
         UnaryWithParam(UnaryOpType::EXP, 1.0f),
         UnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, 1.0f),
         UnaryWithParam(UnaryOpType::RECIP)},
        output_mem_config);
}

inline Tensor softplus(
    const Tensor& input_tensor,
    float beta,
    float threshold,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_ASSERT(input_tensor.device()->arch() != tt::ARCH::GRAYSKULL, "Softplus is not currently supported on Grayskull");
    return run_eltwise_unary(
        input_tensor, {UnaryWithParam(UnaryOpType::SOFTPLUS, {beta, threshold})}, output_mem_config);
}

inline Tensor unary_chain(
    const Tensor& input_tensor,
    std::vector<UnaryWithParam> ops_chain,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return run_eltwise_unary(input_tensor, ops_chain, output_mem_config);
}

Tensor sub_unary(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor sub_unary(
    float value,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor add_unary(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor add_unary(
    float value,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor mul_unary(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor mul_unary(
    float value,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor div_unary(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor div_unary(
    float value,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
// relops with unary argument
Tensor lte_unary(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor lte_unary(
    float value,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor gte_unary(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor gte_unary(
    float value,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor eq_unary(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor eq_unary(
    float value,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
// same as div_unary(value,tensor)
Tensor rdiv(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// deg2rad(a) using scale pi/180.
inline Tensor deg2rad(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return mul_unary(input_tensor, (float)(M_PI / 180.0), output_mem_config);
}

// rad2deg(a) using scale 180/pi.
inline Tensor rad2deg(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return mul_unary(input_tensor, (float)(180.0 / M_PI), output_mem_config);
}

// add 1
// use transformation y = 1.0 + x by broadcast
inline Tensor add1(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return add_unary(input_tensor, 1.0f, output_mem_config);
}

}  // namespace tt_metal

namespace operations {

namespace primary {

inline Tensor relu(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [output_mem_config](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor = input_tensors.at(0);
            bool fp32_dest_acc_en =
                input_tensor.get_dtype() == DataType::UINT32 or
                input_tensor.get_dtype() == DataType::INT32;  // MT: Currently only uint32/int32 is moved to DST
                                                              // directly, fp32 is converted to fp16b
            return operation::run(EltwiseUnary{{UnaryWithParam(UnaryOpType::RELU)}, output_mem_config}, {input_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;

bool get_op_approx_mode(UnaryOpType op_type);
std::pair<string, string> get_op_init_and_func(UnaryOpType op_type, std::vector<float> params = {}, string idst = "0");
std::map<string, string> get_defines(
    UnaryOpType op_type, std::optional<std::vector<float>> params = std::nullopt, string id = "0", string idst = "0");
std::map<string, string> get_block_defines(
    const std::vector<UnaryWithParam>& op_chain, string block_id = "0", string idst = "0");
}  // namespace eltwise_unary_op_utils
