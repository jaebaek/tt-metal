import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_pcc, tt2torch_tensor
import ttnn
import torch


def run_test(device):
    M = 2048
    K = 4544
    K_padded = 4608

    activations_shape = [1, 1, M, K]
    activations_padding = [1, 1, M, K_padded - K]

    weights1_shape = [K_padded, 4 * K_padded]
    weights2_shape = [4 * K_padded, K_padded]

    # Create activations tensor
    activations = torch.ones(activations_shape)
    torch_activations = torch.ones((1, 1, M, K_padded))
    tt_activations = ttnn.from_torch(
        activations,
        ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Create padding tensor
    tt_activations_padding = ttnn.from_torch(
        torch.ones(activations_padding),
        ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Concat padding tensor to activations tensor
    tt_activations = ttnn.concat([tt_activations, tt_activations_padding], dim=3)

    # Create weights tensor
    weights1 = torch.randn(weights1_shape)
    tt_weights1 = ttnn.from_torch(
        weights1,
        ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weights2 = torch.randn(weights2_shape)
    tt_weights2 = ttnn.from_torch(
        weights2,
        ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Perform matmul
    tt_out1 = ttnn.matmul(tt_activations, tt_weights1)
    tt_out2 = ttnn.matmul(tt_out1, tt_weights2)

    # Unpad output
    tt_out2 = tt_out2[:, :, :, :K]

    # Convert to torch tensor
    out = ttnn.to_torch(tt_out2)
    # Compare output to torch
    torch_out = torch.matmul(torch_activations, weights1)
    torch_out = torch.matmul(torch_out, weights2)
    torch_out = torch_out[:, :, :, :K]
    # Check that the output is close to the expected output
    assert comp_pcc(out, torch_out)[1] > 0.99


def test_concat_and_unpad(device):
    run_test(device)
