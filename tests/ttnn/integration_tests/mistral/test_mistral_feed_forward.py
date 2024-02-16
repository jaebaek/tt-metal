# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from time import time
from operator import mul
from functools import reduce

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

import json
from pathlib import Path
from models.experimental.functional_mistral.tt.mistral_configuration import TtModelArgs
from models.experimental.functional_mistral.reference.model import Transformer
from models.experimental.functional_mistral.reference.model import FeedForward
from models.experimental.functional_mistral.tt.ttnn_functional_feed_forward import MistralMLP, feed_forward

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_mistral_feed_forward_inference(model_location_generator, device, reset_seeds):
    model_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    transformer = Transformer.from_folder(Path(model_path), n_layers=1, max_batch_size=1, is_whole_model=False)

    state_dict = torch.load(model_path / "consolidated.00.pth")
    with open(model_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))
    state_dict = {k[22:]: v for k, v in state_dict.items() if (k.startswith("layers.0.feed_forward"))}

    ref_model = transformer.layers[0].feed_forward
    ref_model.eval()

    model_args.max_batch_size = 1
    model_args.n_layers = 32
    dim = 4096

    reference_model = FeedForward(args=model_args)
    reference_model.load_state_dict(state_dict)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        device=device,
    )

    # Not sure why preprocess_model_parameters doesn't do this
    for v in parameters.values():
        v.weight = ttnn.unsqueeze_to_4D(v.weight)

    input = torch.rand(1, 1, 11, dim)
    reference_output = reference_model(input)

    ttnn.enable_program_cache()

    ttnn_input = ttnn.from_torch(
        input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    def show_duration(duration_sec):
        bytes_per_element = {"BFLOAT16": 2, "BFLOAT8_B": 1, "FLOAT32": 4, "UINT16": 2, "UINT32": 4}
        tensor_bytes = lambda x: reduce(mul, x.shape, 1) * bytes_per_element[x.dtype.name]
        total_dram_bytes = sum(tensor_bytes(x.weight) for x in parameters.values())
        dram_gb_per_sec = total_dram_bytes / 1024**3 / duration_sec
        logger.info(
            f"End-to-end duration: {duration_sec * 1000:.1f} ms = {dram_gb_per_sec:.2f} GB/s DRAM weight utilization"
        )

    logger.info("Kernel compilation pass...")
    mlp = MistralMLP(input_shape=ttnn_input.shape, parameters=parameters, grid=(8, 8))
    output = mlp(ttnn_input)

    logger.info("Performance timing pass (MistralMLP)...")
    start = time()
    output = mlp(ttnn_input)
    duration = time() - start
    show_duration(duration)

    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(reference_output, output.to(reference_output.dtype), 0.99)

    logger.info("Performance timing pass (feed_forward)...")
    start = time()
    output = feed_forward(model_args, ttnn_input, parameters)
    duration = time() - start
    show_duration(duration)

    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(reference_output, output.to(reference_output.dtype), 0.99)
