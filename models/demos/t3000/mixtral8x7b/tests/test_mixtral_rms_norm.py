# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger

# Set Mixtral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["MIXTRAL_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor

from models.demos.t3000.mixtral8x7b.tt.mixtral_rms_norm import TtRMSNormSharded
from models.demos.t3000.mixtral8x7b.reference.model import RMSNorm
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


def test_mistral_rms_norm_inference(t3k_device_mesh, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(t3k_device_mesh.get_device(0))
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[24:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention_norm."))}
    reference_model = RMSNorm(dim=model_args.dim)
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtRMSNormSharded(
        device_mesh=t3k_device_mesh,
        state_dict=state_dict,
        args=model_args,
        dtype=dtype,
        layer_num=0,
        weight_key="attention_norm",
    )
    input = torch.rand(1, 1, 32, 4096)
    reference_output = reference_model(input)[0]

    tt_input = ttnn.from_torch(
        input,
        device=t3k_device_mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(t3k_device_mesh),
    )

    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0))[0]
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mixtral_rms_norm Passed!")
    else:
        logger.warning("Mixtral_rms_norm Failed!")

    assert passing, f"Mixtral_rms_norm output does not meet PCC requirement {0.99}."
