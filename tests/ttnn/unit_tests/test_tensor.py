# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from ttnn import Tensor

from tests.ttnn.utils_for_testing import assert_with_pcc

EXPECTED_TENSOR_METHODS = [
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "binary_function",
    "cbrt",
    "clone",
    "cos",
    "cosh",
    "deg2rad",
    "device",
    "digamma",
    "dtype",
    "elu",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "exp2",
    "expm1",
    "full_like",
    "geglu",
    "gelu",
    "glu",
    "hardshrink",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "heaviside",
    "i0",
    "identity",
    "is_contiguous",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "layout",
    "leaky_relu",
    "lerp",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "log_sigmoid",
    "logical_andi",
    "logical_not_unary",
    "logical_noti",
    "logical_ori",
    "logical_xori",
    "logit",
    "mish",
    "move",
    "multigammaln",
    "neg",
    "ones_like",
    "polygamma",
    "pow",
    "prelu",
    "rad2deg",
    "rdiv",
    "recip",
    "reglu",
    "relu",
    "relu6",
    "relu_max",
    "relu_min",
    "rpow",
    "rsqrt",
    "rsub",
    "shape",
    "sigmoid",
    "sign",
    "silu",
    "sin",
    "sinh",
    "softplus",
    "softshrink",
    "softsign",
    "sqrt",
    "square",
    "swiglu",
    "swish",
    "tan",
    "tanh",
    "tanhshrink",
    "ternary_function",
    "threshold",
    "tril",
    "triu",
    "unary_function",
    "value",
    "zeros_like",
]


def test_check_symbols(device):
    for symbol in EXPECTED_TENSOR_METHODS:
        assert getattr(Tensor, symbol)
