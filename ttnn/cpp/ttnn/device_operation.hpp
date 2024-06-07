// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <optional>
#include <tt_eager/tensor/tensor.hpp>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/operation_history.hpp"
#include "tt_stl/concepts.hpp"
#include "tt_stl/reflection.hpp"
#include "tt_stl/unique_any.hpp"

namespace ttnn {

namespace device_operation {

template <typename... attributes_t>
struct CachedProgram {
    tt::tt_metal::Program program;
    // Cached program needs to share attributes between create and override_runtime_arguments functions
    std::tuple<attributes_t...> attributes;

    CachedProgram(tt::tt_metal::Program&& program, attributes_t... attributes) :
        program{std::move(program)}, attributes{std::tuple{attributes...}} {}
};

template <typename program_manager_t>
concept ProgramManagerConcept = requires { [](auto&&... args) { program_manager_t::create(args...); }; };

template <typename program_manager_t>
concept CacheableProgramManagerConcept = ProgramManagerConcept<program_manager_t> and requires {
    [](auto&&... args) { program_manager_t::override_runtime_arguments(args...); };
};

template <typename operation_t>
concept DeviceOperationConcept = requires {
    [](const typename operation_t::operation_attributes_t& attributes,
       const typename operation_t::tensor_args_t& tensor_args) {
        const auto program_manager = operation_t::select_program_manager(attributes, tensor_args);

        operation_t::validate(program_manager, attributes, tensor_args);

        using shape_return_t = typename operation_t::shape_return_t;
        static_assert(std::same_as<
                      decltype(operation_t::compute_output_shapes(program_manager, attributes, tensor_args)),
                      shape_return_t>);

        using tensor_return_value_t = typename operation_t::tensor_return_value_t;
        static_assert(std::same_as<
                      decltype(operation_t::create_output_tensors(program_manager, attributes, tensor_args)),
                      tensor_return_value_t>);
    };
};

template <typename operation_t>
concept DeviceOperationWithCustomProgramCacheConcept = DeviceOperationConcept<operation_t> and requires {
    [](auto&& program_manager,
       const typename operation_t::operation_attributes_t& attributes,
       const typename operation_t::tensor_args_t& tensor_args) {
        operation_t::compute_program_hash(program_manager, attributes, tensor_args);
    };
};

template <typename program_manager_t, typename operation_t>
    requires ProgramManagerConcept<program_manager_t>
constexpr auto create_or_get_program_from_cache(
    auto& program_cache, const typename operation_t::operation_attributes_t& attributes, auto&&... args) {
    if constexpr (CacheableProgramManagerConcept<program_manager_t>) {
        auto program_hash = [&]() {
            const auto& tensor_args = std::get<0>(std::forward_as_tuple(args...));
            if constexpr (DeviceOperationWithCustomProgramCacheConcept<operation_t>) {
                ZoneScopedN("Compute Custom Program Hash");
                return operation_t::compute_program_hash(program_manager_t{}, attributes, tensor_args);
            } else {
                ZoneScopedN("Compute Default Program Hash");
                return tt::stl::hash::hash_objects_with_default_seed(
                    typeid(operation_t).hash_code(), attributes, tensor_args);
            }
        }();

        using cached_program_t = decltype(program_manager_t::create(attributes, std::forward<decltype(args)>(args)...));

        auto cache_hit = program_cache.contains(program_hash);
        if (not cache_hit) {
            program_cache.insert(
                program_hash, program_manager_t::create(attributes, std::forward<decltype(args)>(args)...));
            auto& cached_program = program_cache.template get<cached_program_t>(program_hash);
            return std::reference_wrapper{cached_program.program};
        } else {
            auto& cached_program = program_cache.template get<cached_program_t>(program_hash);
            program_manager_t::override_runtime_arguments(
                cached_program, attributes, std::forward<decltype(args)>(args)...);
            return std::reference_wrapper{cached_program.program};
        }

    } else {
        return program_manager_t::create(attributes, std::forward<decltype(args)>(args)...);
    }
}

struct void_t {};

template <typename operation_t>
    requires DeviceOperationConcept<operation_t>
constexpr typename operation_t::tensor_return_value_t run(
    const typename operation_t::operation_attributes_t& attributes,
    const typename operation_t::tensor_args_t& tensor_args) {
    auto program_manager = operation_t::select_program_manager(attributes, tensor_args);

    operation_t::validate(program_manager, attributes, tensor_args);

    using tensor_return_value_t = typename operation_t::tensor_return_value_t;
    auto tensor_return_value = [&program_manager, &attributes, &tensor_args]() {
        ZoneScopedN("Create Output Tensors");
        if constexpr (std::is_same_v<tensor_return_value_t, void>) {
            operation_t::create_output_tensors(program_manager, attributes, tensor_args);
            return void_t{};
        } else {
            return operation_t::create_output_tensors(program_manager, attributes, tensor_args);
        }
    }();

    auto cq_id = 0;
    auto device = tensor_args.input_tensor_a.device();
    auto& queue = device->command_queue(cq_id);

    auto program = std::visit(
        [&device, &attributes, &tensor_args, &tensor_return_value](auto&& program_manager)
            -> std::variant<tt::tt_metal::Program, std::reference_wrapper<tt::tt_metal::Program>> {
            ZoneScopedN("Create Or Get Program From the Cache");
            using program_manager_t = std::decay_t<decltype(program_manager)>;
            if constexpr (std::is_same_v<tensor_return_value_t, void>) {
                return create_or_get_program_from_cache<program_manager_t, operation_t>(
                    device->program_cache, attributes, tensor_args);
            } else {
                return create_or_get_program_from_cache<program_manager_t, operation_t>(
                    device->program_cache, attributes, tensor_args, tensor_return_value);
            }
        },
        program_manager);

    std::visit(
        [&queue](auto&& program) {
            ZoneScopedN("Enqueue Program");
            tt::tt_metal::EnqueueProgram(queue, program, false);
        },
        program);

    if constexpr (not std::is_same_v<tensor_return_value_t, void>) {
        return tensor_return_value;
    }
}

}  // namespace device_operation

}  // namespace ttnn
