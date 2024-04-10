# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib
from models.utility_functions import pad_by_zero, torch2tt_tensor


def get_weights_cached(
    devices,
    model_config,
    tt_cache_path,
    weight_cache_str,
    weight_config_str,
    weights_to_cache,
    overwrite=False,
    padzero=False,
    weights_dict=None,
):
    """Load weights from weights_dict or cache and duplicate per device. Store if not cached."""

    path = tt_cache_path / f"{weight_cache_str}_{model_config[f'{weight_config_str}_DTYPE'].name}.bin"
    if weights_dict and str(path) in weights_dict.keys():
        weights = weights_dict[str(path)]
    elif not overwrite and path.exists():
        # Load cached weights
        weights_host = tt_lib.tensor.load_tensor(str(path))
        # Duplicate weights on all devices
        weights = [weights_host.to(device, model_config[f"{weight_config_str}_MEMCFG"]) for device in devices]
        # Add to weights_dict
        if weights_dict is not None:
            weights_dict[str(path)] = weights
    else:
        # Duplicate weights on all devices
        if padzero:
            weights = [
                pad_by_zero(
                    weights_to_cache,
                    device,
                    tt_memory_config=model_config[f"{weight_config_str}_MEMCFG"],
                    tt_dtype=model_config[f"{weight_config_str}_DTYPE"],
                )[0]
                for device in devices
            ]
        else:
            weights = [
                torch2tt_tensor(
                    weights_to_cache,
                    device,
                    tt_memory_config=model_config[f"{weight_config_str}_MEMCFG"],
                    tt_dtype=model_config[f"{weight_config_str}_DTYPE"],
                )
                for device in devices
            ]
        # Save weights for reuse between prefill/decode
        if weights_dict is not None:
            weights_dict[str(path)] = weights[0]
        # Store weights (from first device)
        tt_lib.tensor.dump_tensor(
            str(path),
            weights[0].cpu(),
        )
    return weights
