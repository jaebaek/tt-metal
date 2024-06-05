# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
import tempfile
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor, ListMeshToTensor


@pytest.mark.parametrize("shape", [(1, 1, 512, 512), (1, 1, 32, 32), (1, 3, 512, 512), (1, 3, 32, 32)])
@pytest.mark.parametrize("use_all_gather", [True, False])
@pytest.mark.parametrize("enable_async", [True, False])
def test_multi_device_single_trace(pcie_device_mesh, shape, use_all_gather, enable_async):
    if pcie_device_mesh.get_num_devices() <= 1:
        pytest.skip("This test requires multiple devices")

    # Trace requires program cache to be enabled
    for device_id in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device_id).enable_async(enable_async)
        pcie_device_mesh.get_device(device_id).enable_program_cache()

    # Preallocate activation tensors. These will be used when capturing and executing the trace
    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, pcie_device_mesh)
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, pcie_device_mesh)

    # Op chains to be traced
    def run_op_chain(input_0, input_1):
        single_dev_output = ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1)))
        if use_all_gather:
            return ttnn.all_gather(single_dev_output, dim=0, num_links=1)
        return single_dev_output

    # Compile program binaries
    run_op_chain(input_0_dev, input_1_dev)

    # Capture Trace
    logger.info("Capture Trace")
    tid = ttnn.begin_trace_capture(pcie_device_mesh, trace_buffer_size=106496, cq_id=0)
    output_tensor = run_op_chain(input_0_dev, input_1_dev)
    ttnn.end_trace_capture(pcie_device_mesh, tid, cq_id=0)

    for i in range(50):
        # Create torch inputs
        torch_input_tensor_0 = torch.rand(
            (pcie_device_mesh.get_num_devices(), shape[1], shape[2], shape[3]), dtype=torch.bfloat16
        )
        torch_input_tensor_1 = torch.rand(
            (pcie_device_mesh.get_num_devices(), shape[1], shape[2], shape[3]), dtype=torch.bfloat16
        )
        # Compute PT Golden
        torch_output_golden = torch.neg(
            torch.add(
                torch.mul(torch_input_tensor_1, torch.neg(torch.nn.functional.gelu(torch_input_tensor_0))),
                torch.relu(torch_input_tensor_1),
            )
        )
        # Convert torch tensors to TTNN Multi-Device Host Tensors
        ttnn_input_tensor_0 = ttnn.from_torch(
            torch_input_tensor_0, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0)
        )
        ttnn_input_tensor_1 = ttnn.from_torch(
            torch_input_tensor_1, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0)
        )

        # Copy TTNN host tensors into preallocated Mult-Device tensors
        logger.info("Send Inputs to Device")
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_0, input_0_dev)
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_1, input_1_dev)
        logger.info("Execute Trace")
        # Execute trace
        ttnn.execute_trace(pcie_device_mesh, tid, cq_id=0, blocking=False)

        if use_all_gather:
            # Device All-Gather: Iterate through tensors on all devices. Ensure they match the full tensor
            logger.info("Read Back Trace Outputs")
            device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(output_tensor)
            for device_tensor in device_tensors:
                device_tensor_torch = ttnn.to_torch(device_tensor)
                assert_with_pcc(device_tensor_torch, torch_output_golden, pcc=0.99)

        else:
            # Perform host All-Gather
            ttnn_torch_output_tensor = ttnn.to_torch(
                output_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=0)
            )
            assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.99)

    # Release trace buffer once workload is complete
    ttnn.release_trace(pcie_device_mesh, tid)

    for device_id in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device_id).enable_async(False)


@pytest.mark.parametrize("shape", [(1, 1, 512, 512), (1, 1, 32, 32), (1, 3, 512, 512), (1, 3, 32, 32)])
@pytest.mark.parametrize("use_all_gather", [True, False])
@pytest.mark.parametrize("enable_async", [True, False])
def test_multi_device_multi_trace(pcie_device_mesh, shape, use_all_gather, enable_async):
    if use_all_gather:
        # Currently all-gather tests pass only if blocking == False
        if shape == (1, 1, 32, 32) or shape == (1, 3, 512, 512) or shape == (1, 3, 32, 32):
            pytest.skip("This configuration is not working with all-gather")

    if pcie_device_mesh.get_num_devices() <= 1:
        pytest.skip("This test requires multiple devices")

    # Trace requires program cache to be enabled
    for device_id in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device_id).enable_async(enable_async)
        pcie_device_mesh.get_device(device_id).enable_program_cache()

    # Preallocate activation tensors. These will be used when capturing and executing the trace
    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, pcie_device_mesh)
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, pcie_device_mesh)
    weight_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, pcie_device_mesh)

    # Op chains to be traced
    def run_op_chain(input_0, input_1, weight):
        single_dev_output = ttnn.neg(
            ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1))
        ) @ ttnn.silu(weight)
        if use_all_gather:
            return ttnn.all_gather(single_dev_output, dim=0, num_links=1)
        return single_dev_output

    def run_op_chain_1(input_0, input_1, weight):
        single_dev_output = ttnn.tanh(ttnn.mul(ttnn.sub(input_0, input_1), weight))
        if use_all_gather:
            return ttnn.all_gather(single_dev_output, dim=0, num_links=1)
        return single_dev_output

    # Compile program binaries
    run_op_chain(input_0_dev, input_1_dev, weight_dev)
    run_op_chain_1(input_0_dev, input_1_dev, weight_dev)

    # Capture Trace 0
    logger.info("Capture Trace 0")
    tid = ttnn.begin_trace_capture(pcie_device_mesh, trace_buffer_size=106496, cq_id=0)
    output_tensor = run_op_chain(input_0_dev, input_1_dev, weight_dev)
    ttnn.end_trace_capture(pcie_device_mesh, tid, cq_id=0)

    # Capture Trace 1
    logger.info("Capture Trace 1")
    tid_1 = ttnn.begin_trace_capture(pcie_device_mesh, trace_buffer_size=26624, cq_id=0)
    output_tensor_1 = run_op_chain_1(input_0_dev, input_1_dev, weight_dev)
    ttnn.end_trace_capture(pcie_device_mesh, tid_1, cq_id=0)

    # Execute and verify trace against pytorch
    torch_silu = torch.nn.SiLU()
    for i in range(50):
        # Create torch inputs
        torch_input_tensor_0 = torch.rand(
            (pcie_device_mesh.get_num_devices(), shape[1], shape[2], shape[3]), dtype=torch.bfloat16
        )
        torch_input_tensor_1 = torch.rand(
            (pcie_device_mesh.get_num_devices(), shape[1], shape[2], shape[3]), dtype=torch.bfloat16
        )
        torch_weight = torch.rand(shape, dtype=torch.bfloat16)
        # Compute PT Golden
        torch_output_golden = torch.neg(
            torch.add(
                torch.mul(torch_input_tensor_1, torch.neg(torch.nn.functional.gelu(torch_input_tensor_0))),
                torch.relu(torch_input_tensor_1),
            )
        ) @ torch_silu(torch_weight)

        torch_output_golden_1 = torch.tanh(
            torch.mul(torch.sub(torch_input_tensor_0, torch_input_tensor_1), torch_weight)
        )

        # Convert torch tensors to TTNN Multi-Device Host Tensors
        ttnn_input_tensor_0 = ttnn.from_torch(
            torch_input_tensor_0, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0)
        )
        ttnn_input_tensor_1 = ttnn.from_torch(
            torch_input_tensor_1, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0)
        )
        ttnn_weight = ttnn.from_torch(
            torch_weight, layout=ttnn.TILE_LAYOUT, mesh_mapper=ReplicateTensorToMesh(pcie_device_mesh)
        )

        # Copy TTNN host tensors into preallocated Mult-Device tensors
        logger.info("Send Inputs to Device")
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_0, input_0_dev)
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_1, input_1_dev)
        ttnn.copy_host_to_device_tensor(ttnn_weight, weight_dev)

        logger.info("Execute Trace 0")
        # Execute trace
        ttnn.execute_trace(pcie_device_mesh, tid, cq_id=0, blocking=False)
        logger.info("Execute Trace 1")
        ttnn.execute_trace(pcie_device_mesh, tid_1, cq_id=0, blocking=False)
        if use_all_gather:
            # Device All-Gather: Iterate through tensors on all devices. Ensure they match the full tensor
            logger.info("Read Back Trace 0 Outputs")
            device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(output_tensor)
            for device_tensor in device_tensors:
                device_tensor_torch = ttnn.to_torch(device_tensor)
                assert_with_pcc(device_tensor_torch, torch_output_golden, pcc=0.99)

            logger.info("Read Back Trace 1 Outputs")
            device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(output_tensor_1)
            for device_tensor in device_tensors:
                device_tensor_torch = ttnn.to_torch(device_tensor)
                assert_with_pcc(device_tensor_torch, torch_output_golden_1, pcc=0.99)
        else:
            # Perform host All-Gather
            ttnn_torch_output_tensor = ttnn.to_torch(
                output_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=0)
            )
            assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.99)

            ttnn_torch_output_tensor = ttnn.to_torch(
                output_tensor_1, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=0)
            )
            assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden_1, pcc=0.99)

    # Release trace buffer once workload is complete
    ttnn.release_trace(pcie_device_mesh, tid)
    ttnn.release_trace(pcie_device_mesh, tid_1)

    for device_id in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device_id).enable_async(False)
