# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import json
import os
import pathlib
import pprint
from typing import Optional

from loguru import logger


@dataclasses.dataclass
class Config:
    cache_path: pathlib.Path = pathlib.Path().home() / ".cache" / "ttnn"
    model_cache_path: pathlib.Path = cache_path / "models"
    tmp_dir: pathlib.Path = pathlib.Path("/") / "tmp" / "ttnn"
    enable_model_cache: bool = False
    enable_fast_runtime_mode: bool = False
    throw_exception_on_fallback: bool = False
    enable_logging: bool = False
    enable_graph_report: bool = False
    enable_detailed_buffer_report: bool = False
    enable_detailed_tensor_report: bool = False
    enable_comparison_mode: bool = False
    comparison_mode_pcc: float = 0.9999
    root_report_path: pathlib.Path = pathlib.Path("generated") / "ttnn" / "reports"
    report_name: Optional[str] = None

    @property
    def report_path(self):
        if self.report_name is None:
            return None
        return self.root_report_path / f"{hash(self.report_name)}"


CONFIG = Config()
CONFIG_PATH = pathlib.Path.home() / ".config" / "ttnn" / "config.json"
if "TTNN_CONFIG_PATH" in os.environ:
    CONFIG_PATH = pathlib.Path(os.environ["TTNN_CONFIG_PATH"])

CONFIG_OVERRIDES = os.environ.get("TTNN_CONFIG_OVERRIDES", "{}")


def load_config_from_dictionary(config, from_file=False):
    global CONFIG
    for key, value in config.items():
        if hasattr(CONFIG, key):
            if getattr(CONFIG, key) is not None:
                value = type(getattr(CONFIG, key))(value)
            setattr(CONFIG, key, value)
        elif from_file:
            logger.warning(
                f"Unknown configuration key: {key}. Please update your configuration file: {CONFIG_PATH}. Or delete it to get the new default config"
            )
        else:
            raise ValueError(f"Unknown configuration key: {key}")


def load_config_from_json_file(json_path):
    global CONFIG
    try:
        with open(json_path, "r") as f:
            config = json.load(f)
        load_config_from_dictionary(config, from_file=True)
    except Exception as e:
        logger.warning(f"Failed to load ttnn configuration from {json_path}: {e}")


def save_config_to_json_file(json_path):
    with open(json_path, "w") as f:
        normalized_config = dataclasses.asdict(CONFIG)
        for key, value in normalized_config.items():
            if isinstance(value, pathlib.Path):
                value = str(value)
            normalized_config[key] = value
        json.dump(normalized_config, f, indent=4)


if CONFIG_PATH.exists():
    logger.debug(f"Loading ttnn configuration from {CONFIG_PATH}")
    load_config_from_json_file(CONFIG_PATH)
else:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_config_to_json_file(CONFIG_PATH)


if CONFIG_OVERRIDES:
    logger.debug(f"Loading ttnn configuration overrides from environment variable TTNN_CONFIG_OVERRIDES")
    load_config_from_dictionary(json.loads(CONFIG_OVERRIDES))


logger.debug(f"Initial ttnn.CONFIG:\n{pprint.pformat(dataclasses.asdict(CONFIG))}")


@contextlib.contextmanager
def manage_config_attribute(name, value):
    global CONFIG
    original_value = getattr(CONFIG, name)
    setattr(CONFIG, name, value)
    logger.debug(f"Set ttnn.CONFIG.{name} to {value}")
    yield
    setattr(CONFIG, name, original_value)
    logger.debug(f"Restored ttnn.CONFIG.{name} to {original_value}")


import tt_lib as _tt_lib
import ttnn._ttnn

from ttnn._ttnn.multi_device import get_device_tensors, aggregate_as_tensor

from ttnn.types import (
    TILE_SIZE,
    DataType,
    uint16,
    int32,
    uint32,
    bfloat8_b,
    bfloat4_b,
    bfloat16,
    float32,
    MathFidelity,
    MemoryConfig,
    BufferType,
    TensorMemoryLayout,
    DRAM_MEMORY_CONFIG,
    L1_MEMORY_CONFIG,
    L1_BLOCK_SHARDED_MEMORY_CONFIG,
    L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    L1_WIDTH_SHARDED_MEMORY_CONFIG,
    ShardStrategy,
    ShardOrientation,
    Layout,
    ROW_MAJOR_LAYOUT,
    TILE_LAYOUT,
    StorageType,
    DEVICE_STORAGE_TYPE,
    MULTI_DEVICE_STORAGE_TYPE,
    CoreGrid,
    CoreRange,
    Shape,
    Tensor,
    DeviceComputeKernelConfig,
    WormholeComputeKernelConfig,
    GrayskullComputeKernelConfig,
    DeviceGrid,
)

from ttnn.device import (
    Device,
    open_device,
    close_device,
    enable_program_cache,
    disable_and_clear_program_cache,
    manage_device,
    synchronize_device,
    dump_device_memory_state,
)

from ttnn.multi_device import (
    DeviceMesh,
    open_device_mesh,
    close_device_mesh,
    get_num_pcie_devices,
    get_pcie_device_ids,
    get_device_ids,
    create_device_mesh,
    TensorToMesh,
    ShardTensorToMesh,
    ReplicateTensorToMesh,
    MeshToTensor,
    ConcatMeshToTensor,
    ListMeshToTensor,
    default_mesh_mapper,
    default_mesh_composer,
)

from ttnn.core import (
    set_printoptions,
    has_storage_type_of,
    is_tensor_storage_on_device,
    has_tile_padding,
    is_sharded,
    get_memory_config,
    create_sharded_memory_config,
    dump_memory_config,
    load_memory_config,
)

import ttnn.reflection
from ttnn.validation import validate_input_tensor
import ttnn.tracer
import ttnn.database

from ttnn.decorators import (
    register_operation,
    query_operations,
    register_pre_operation_hook,
    register_post_operation_hook,
)

import ttnn.experimental
import ttnn.experimental.golden_functions

from ttnn.operations.core import (
    from_torch,
    to_torch,
    to_device,
    from_device,
    to_layout,
    reshape,
    to_memory_config,
    deallocate,
    reallocate,
    load_tensor,
    dump_tensor,
    unsqueeze_to_4D,
    squeeze,
    clone,
    as_tensor,
)

from ttnn.operations.matmul import (
    matmul,
    linear,
    create_matmul_program_config,
    create_matmul_1d_systolic_array_program_config,
)

from ttnn.operations.others import (
    embedding,
    # fused operations
    softmax,
    # reduction operations
    mean,
    upsample,
    pearson_correlation_coefficient,
)

from ttnn.operations.creation import (
    arange,
    empty,
    full,
    full_like,
    ones,
    ones_like,
    zeros,
    zeros_like,
)

from ttnn.operations.reduction import (
    std,
    var,
    max,
    min,
    sum,
)

from ttnn.operations.losses import (
    l1_loss,
    mse_loss,
)

from ttnn.operations.data_movement import (
    concat,
    pad,
    permute,
    split,
    repeat_interleave,
    repeat,
)

from ttnn.operations.unary import (
    exp,
    tanh,
    gelu,
    rsqrt,
    relu,
    silu,
    log,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    asinh,
    acosh,
    atanh,
    logical_not,
    logit,
    signbit,
)

from ttnn.operations.binary import (
    pow,
    add,
    sub,
    subtract,
    mul,
    multiply,
    ldexp,
    logical_and,
    logical_or,
    logical_xor,
    logaddexp,
    logaddexp2,
    xlogy,
    add_and_apply_activation,
    add_and_apply_activation_,
    nextafter,
    polyval,
    maximum,
    minimum,
)

from ttnn.operations.ternary import (
    addcdiv,
    addcmul,
    mac,
    where,
)

from ttnn.operations.relational import (
    gtz,
    ltz,
    gez,
    lez,
    nez,
    eqz,
    gt,
    gte,
    lt,
    lte,
    eq,
    ne,
    isclose,
)

from ttnn.operations.activation import (
    clip,
    elu,
    hardshrink,
    hardsigmoid,
    hardswish,
    hardtanh,
    heaviside,
    leaky_relu,
    log_sigmoid,
    mish,
    prelu,
    relu6,
    sigmoid,
    sign,
    softshrink,
    softsign,
    swish,
    softplus,
    tanhshrink,
    threshold,
    glu,
    geglu,
    reglu,
    swiglu,
)

from ttnn.operations.math import (
    i0,
    isfinite,
    isinf,
    isnan,
    isneginf,
    isposinf,
    lgamma,
    log10,
    log1p,
    log2,
    multigammaln,
    neg,
    abs,
    cbrt,
    deg2rad,
    digamma,
    erf,
    erfc,
    erfinv,
    exp2,
    expm1,
    atan2,
    hypot,
    squared_difference,
    lerp,
    polygamma,
    rad2deg,
    reciprocal,
    sqrt,
    square,
    tril,
    triu,
)

from ttnn.operations.normalization import (
    layer_norm,
    rms_norm,
    group_norm,
    create_group_norm_weight_bias_rm,
    create_group_norm_input_mask,
    determine_expected_group_norm_sharded_config_and_grid_size,
    get_group_norm_cores_accross_channel,
)

from ttnn.operations.ccl import all_gather

from ttnn.operations import transformer
from ttnn.operations import kv_cache
from ttnn.operations.conv2d import Conv2d
from ttnn.operations.maxpool2d import (
    MaxPool2d,
    global_avg_pool2d,
)
