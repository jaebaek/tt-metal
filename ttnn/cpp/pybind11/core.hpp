// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/core.hpp"

namespace py = pybind11;

namespace ttnn {
namespace core {
void py_module(py::module& module) {
    module.def("get_memory_config", &ttnn::get_memory_config);
    module.def("set_printoptions", &ttnn::set_printoptions, py::kw_only(), py::arg("profile"));
    module.def("dump_stack_trace_on_segfault", &ttnn::core::dump_stack_trace_on_segfault);
}

}  // namespace core
}  // namespace ttnn
