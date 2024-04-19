// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/binary.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace binary {

void py_module(py::module& module) {
    module.def(
        "add",
        [](const ttnn::Tensor& input_tensor_a,
           const float scalar,
           const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
           const std::optional<const DataType> dtype = std::nullopt) -> ttnn::Tensor {
            return ttnn::operations::binary::add(input_tensor_a, scalar, memory_config, dtype, std::nullopt);
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG,  // TODO(arakhmati): set to std::nullopt
        py::arg("dtype") = std::nullopt);

    module.def(
        "add",
        [](const ttnn::Tensor& input_tensor_a,
           const ttnn::Tensor& input_tensor_b,
           const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
           const std::optional<const DataType> dtype = std::nullopt) -> ttnn::Tensor {
            return ttnn::operations::binary::add(input_tensor_a, input_tensor_b, memory_config, dtype, std::nullopt);
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG,  // TODO(arakhmati): set to std::nullopt
        py::arg("dtype") = std::nullopt);

    module.def(
        "subtract",
        [](const ttnn::Tensor& input_tensor_a,
           const float scalar,
           const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
           const std::optional<const DataType> dtype = std::nullopt) -> ttnn::Tensor {
            return ttnn::operations::binary::subtract(input_tensor_a, scalar, memory_config, dtype, std::nullopt);
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG,  // TODO(arakhmati): set to std::nullopt
        py::arg("dtype") = std::nullopt);

    module.def(
        "subtract",
        [](const ttnn::Tensor& input_tensor_a,
           const ttnn::Tensor& input_tensor_b,
           const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
           const std::optional<const DataType> dtype = std::nullopt) -> ttnn::Tensor {
            return ttnn::operations::binary::subtract(
                input_tensor_a, input_tensor_b, memory_config, dtype, std::nullopt);
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG,  // TODO(arakhmati): set to std::nullopt
        py::arg("dtype") = std::nullopt);

    module.def(
        "multiply",
        [](const ttnn::Tensor& input_tensor_a,
           const float scalar,
           const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
           const std::optional<const DataType> dtype = std::nullopt) -> ttnn::Tensor {
            return ttnn::operations::binary::multiply(input_tensor_a, scalar, memory_config, dtype, std::nullopt);
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG,  // TODO(arakhmati): set to std::nullopt
        py::arg("dtype") = std::nullopt);

    module.def(
        "multiply",
        [](const ttnn::Tensor& input_tensor_a,
           const ttnn::Tensor& input_tensor_b,
           const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
           const std::optional<const DataType> dtype = std::nullopt) -> ttnn::Tensor {
            return ttnn::operations::binary::multiply(
                input_tensor_a, input_tensor_b, memory_config, dtype, std::nullopt);
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG,  // TODO(arakhmati): set to std::nullopt
        py::arg("dtype") = std::nullopt);
}

}  // namespace binary
}  // namespace operations
}  // namespace ttnn
