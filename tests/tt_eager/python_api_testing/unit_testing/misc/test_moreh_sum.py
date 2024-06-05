# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_allclose_and_pcc
from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)

TILE_HEIGHT = 32
TILE_WIDTH = 32


# For keepdim in torch
def filter_indices(output_shape, dims):
    def not_in_dims(index_value_pair):
        index, value = index_value_pair
        return index not in dims

    filtered_elements = list(filter(not_in_dims, enumerate(output_shape)))
    filtered_values = [value for index, value in filtered_elements]

    return filtered_values


# For keep_batch_dim in tt
def filter_indices_with_last_two(output_shape, dims):
    last_two_elements = output_shape[-2:]
    remaining_elements = output_shape[:-2]

    def not_in_dims(index_value_pair):
        index, _ = index_value_pair
        return index not in dims

    filtered_remaining_elements = list(filter(not_in_dims, enumerate(remaining_elements)))
    filtered_remaining_values = [value for index, value in filtered_remaining_elements]
    final_output_shape = filtered_remaining_values + last_two_elements

    return final_output_shape


def get_tensors(input_shape, dim, device, *, with_padding=True, use_randint=True, keep_batch_dim=False):
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.bfloat16
    npu_layout = ttl.tensor.Layout.TILE

    output_shape = input_shape.copy()
    if dim is None or dim == []:
        dim = list(range(len(input_shape)))

    if isinstance(dim, int):
        dim = [dim]

    for d in dim:
        output_shape[d] = 1

    if keep_batch_dim:
        torch_output_shape = output_shape.copy()
        tt_output_shape = output_shape.copy()
    else:
        torch_output_shape = filter_indices(output_shape, dim)
        tt_output_shape = filter_indices_with_last_two(output_shape, dim)

    if use_randint:
        torch_input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype, requires_grad=True)
        torch_output = torch.randint(-2, 3, tt_output_shape, dtype=cpu_dtype)
    else:
        torch_input = torch.rand(input_shape, dtype=cpu_dtype, requires_grad=True)
        torch_output = torch.rand(tt_output_shape, dtype=cpu_dtype)

    if with_padding:
        tt_input = ttl.tensor.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
        tt_output = ttl.tensor.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    else:
        tt_input = ttl.tensor.Tensor(torch_input, npu_dtype).to(npu_layout).to(device)
        tt_output = ttl.tensor.Tensor(torch_output, npu_dtype).to(npu_layout).to(device)

    return tt_input, tt_output, tt_output_shape, torch_output_shape, torch_input


def get_backward_tensors(
    tt_output_grad_shape,
    torch_output_grad_shape,
    torch_input_grad_shape,
    device,
    *,
    with_padding=True,
    use_randint=True,
):
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.bfloat16
    npu_layout = ttl.tensor.Layout.TILE

    if use_randint:
        torch_output_grad = torch.randint(-2, 3, torch_output_grad_shape, dtype=cpu_dtype, requires_grad=True)
        torch_input_grad = torch.randint(-2, 3, torch_input_grad_shape, dtype=cpu_dtype)
    else:
        torch_output_grad = torch.rand(torch_output_grad_shape, dtype=cpu_dtype, requires_grad=True)
        torch_input_grad = torch.rand(torch_input_grad_shape, dtype=cpu_dtype)

    if with_padding:
        tt_output_grad = (
            ttl.tensor.Tensor(torch_output_grad.reshape(tt_output_grad_shape), npu_dtype)
            .pad_to_tile(float("nan"))
            .to(npu_layout)
            .to(device)
        )
        tt_input_grad = (
            ttl.tensor.Tensor(torch_input_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
        )
    else:
        tt_output_grad = (
            ttl.tensor.Tensor(torch_output_grad.reshape(tt_output_grad_shape), npu_dtype).to(npu_layout).to(device)
        )
        tt_input_grad = ttl.tensor.Tensor(torch_input_grad, npu_dtype).to(npu_layout).to(device)

    return tt_output_grad, tt_input_grad, torch_output_grad


def moreh_sum(input_shape, dim, keep_batch_dim, use_provide_output, compute_kernel_options, device):
    (tt_input, tt_output, output_shape, _, torch_input) = get_tensors(
        input_shape, dim, device, keep_batch_dim=keep_batch_dim
    )
    torch_output = torch.sum(torch_input, dim, keep_batch_dim)

    if not use_provide_output:
        tt_output = None

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output_cpu = (
        ttl.operations.primary.moreh_sum(
            tt_input,
            dim=dim,
            keep_batch_dim=keep_batch_dim,
            output=tt_output,
            compute_kernel_config=compute_kernel_config,
        )
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    # test for equivalance
    # TODO(Dongjin) : check while changing rtol after enabling fp32_dest_acc_en
    rtol = atol = 0.12
    passing, output_pcc = comp_allclose_and_pcc(
        torch_output if keep_batch_dim else torch_output.reshape(-1),
        tt_output_cpu if keep_batch_dim else tt_output_cpu.reshape(-1),
        pcc=0.999,
        rtol=rtol,
        atol=atol,
    )

    logger.debug(f"input_shape={input_shape}, dim={dim}, tt_output_shape={tt_output_cpu.shape}")
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    return passing


@pytest.mark.parametrize(
    "input_shape",
    (([3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1]),),
    ids=[
        "3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (
        None,
        0,
        1,
        2,
        3,
        [],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2],
        [1, 2, 3],
        [1, 3],
        [2, 3],
    ),
    ids=["None", "0", "1", "2", "3", "[]", "0,1", "0,1,2", "0,1,2,3", "0,1,3", "0,2,3", "1,2", "1,2,3", "1,3", "2,3"],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize("keep_batch_dim", (True, False), ids=["keep_batch_dim-true", "keep_batch_dim-flase"])
def test_moreh_sum(input_shape, dim, keep_batch_dim, compute_kernel_options, device):
    torch.manual_seed(2023)
    passing = moreh_sum(input_shape, dim, keep_batch_dim, True, compute_kernel_options, device)
    assert passing


@pytest.mark.parametrize(
    "input_shape",
    (
        ([TILE_HEIGHT, TILE_WIDTH]),
        ([TILE_HEIGHT - 1, TILE_WIDTH - 1]),
        ([2, 3, 2, 4, TILE_HEIGHT * 4, TILE_WIDTH * 4]),
        ([3, 2, 4, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1]),
    ),
    ids=[
        "TILE_HEIGHT, TILE_WIDTH",
        "TILE_HEIGHT - 1, TILE_WIDTH - 1",
        "2, 3, 2, 4, TILE_HEIGHT * 4, TILE_WIDTH * 4",
        "3, 2, 4, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (0, 1, 2, 3, 4, 5),
    ids=["0", "1", "2", "3", "4", "5"],
)
@pytest.mark.parametrize("use_provide_output", (True, False), ids=["True", "False"])
@pytest.mark.parametrize("keep_batch_dim", (True, False), ids=["keep_batch_dim-true", "keep_batch_dim-flase"])
def test_moreh_sum_non_4d(input_shape, dim, keep_batch_dim, use_provide_output, device):
    torch.manual_seed(2023)
    input_rank = len(input_shape)
    if dim >= input_rank:
        pytest.skip(f"input dim {dim} exceeds the dims of input tensor {len(input_shape)}.")

    passing = moreh_sum(input_shape, dim, keep_batch_dim, use_provide_output, False, device)
    assert passing


@pytest.mark.parametrize(
    "input_shape",
    [
        [4, TILE_HEIGHT * 4, TILE_WIDTH * 4],
    ],
    ids=[
        "4, TILE_HEIGHT * 4, TILE_WIDTH * 4",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (0, 1, 2),
    ids=["0", "1", "2"],
)
def test_moreh_sum_enable_cache(input_shape, dim, device, use_program_cache):
    torch.manual_seed(3072)
    keep_batch_dim = [True, False]
    use_provide_output = [True, False]
    for i in range(2):
        passing = moreh_sum(input_shape, dim, keep_batch_dim[i], use_provide_output[i], False, device)
        assert passing
    assert device.num_program_cache_entries() == 1


@pytest.mark.parametrize(
    "input_shape",
    (
        [10, TILE_HEIGHT * 12, TILE_WIDTH * 12],
        [10, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 12 - 1],
    ),
    ids=[
        "10, TILE_HEIGHT * 12, TILE_WIDTH * 12",
        "10, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 12 - 1",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (0, 1, 2),
    ids=["dim-n", "dim-h", "dim-w"],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_sum_fp32_dest_acc(input_shape, dim, compute_kernel_options, device):
    torch.manual_seed(2023)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    (tt_input, tt_output, output_shape, torch_output_shape, torch_input) = get_tensors(
        input_shape, dim, device, use_randint=False, keep_batch_dim=True
    )
    torch_input = torch_input.float()
    torch_output = torch.sum(torch_input, dim, True)

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output_cpu = (
        ttl.operations.primary.moreh_sum(
            tt_input, dim=dim, keep_batch_dim=True, output=tt_output, compute_kernel_config=compute_kernel_config
        )
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")
    logger.debug(f"std={torch.std(torch.abs(torch_output - tt_output_cpu))}")
    logger.debug(f"mean={torch.abs(torch_output - tt_output_cpu).mean()}")

    # TODO: Need to check the accuracy for fp32 mode
    # assert passing


def moreh_sum_backward(input_shape, dim, keep_batch_dim, use_provide_output, compute_kernel_options, device):
    torch.manual_seed(2023)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    (tt_input, _, tt_output_shape, torch_output_shape, torch_input) = get_tensors(
        input_shape, dim, device, keep_batch_dim=keep_batch_dim
    )
    (tt_output_grad, tt_input_grad, torch_output_grad) = get_backward_tensors(
        tt_output_shape, torch_output_shape, input_shape, device
    )

    if not use_provide_output:
        tt_input_grad = None

    torch_output = torch.sum(torch_input, dim, keep_batch_dim)
    torch_output.backward(torch_output_grad)

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_input_grad_cpu = (
        ttl.operations.primary.moreh_sum_backward(
            tt_output_grad,
            tt_input,
            dim=dim,
            keep_batch_dim=keep_batch_dim,
            input_grad=tt_input_grad,
            compute_kernel_config=compute_kernel_config,
        )
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(input_shape)
        .to_torch()
    )

    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, tt_input_grad_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    return passing


@pytest.mark.parametrize(
    "input_shape",
    (([3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1]),),
    ids=[
        "3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (
        None,
        0,
        1,
        2,
        3,
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2],
        [1, 2, 3],
        [1, 3],
        [2, 3],
    ),
    ids=["None", "0", "1", "2", "3", "0,1", "0,1,2", "0,1,2,3", "0,1,3", "0,2,3", "1,2", "1,2,3", "1,3", "2,3"],
)
@pytest.mark.parametrize("keep_batch_dim", (True, False), ids=["keep_batch_dim-true", "keep_batch_dim-flase"])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_sum_backward(input_shape, dim, keep_batch_dim, compute_kernel_options, device):
    torch.manual_seed(2023)
    passing = moreh_sum_backward(input_shape, dim, keep_batch_dim, True, compute_kernel_options, device)
    assert passing


@pytest.mark.parametrize(
    "input_shape",
    (([3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1]),),
    ids=[
        "3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (
        None,
        1,
        3,
        [0, 1],
        [0, 2, 3],
        [1, 3],
    ),
    ids=["None", "1", "3", "0,1", "0,2,3", "1,3"],
)
def test_moreh_sum_backward_wo_input_grad(input_shape, dim, device):
    torch.manual_seed(2023)
    passing = moreh_sum_backward(input_shape, dim, True, False, False, device)
    assert passing


@pytest.mark.parametrize(
    "input_shape",
    [
        [4, TILE_HEIGHT * 4, TILE_WIDTH * 4],
    ],
    ids=[
        "4, TILE_HEIGHT * 4, TILE_WIDTH * 4",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (0, 1, 2),
    ids=["0", "1", "2"],
)
def test_moreh_sum_backward_enable_cache(input_shape, dim, device, use_program_cache):
    torch.manual_seed(3072)
    keep_batch_dim = [True, False]
    use_provide_output = [True, False]
    num_cache_entires = [2, 1, 1]
    for i in range(2):
        passing = moreh_sum_backward(input_shape, dim, keep_batch_dim[i], use_provide_output[i], False, device)
        assert passing
    assert device.num_program_cache_entries() == num_cache_entires[dim]


@pytest.mark.parametrize(
    "input_shape",
    ([2, 3, 2, 4, TILE_HEIGHT * 6 - 1, TILE_WIDTH * 6 - 1],),
    ids=[
        "2, 3, 2, 4, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1",
    ],
)
@pytest.mark.parametrize(
    "dim",
    (0, 4, 5, [4, 5], [1, 4, 5]),
    ids=["dim-n", "dim-h", "dim-w", "dim-hw", "dim-nhw"],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_sum_backward_fp32_dest_acc(input_shape, dim, compute_kernel_options, device):
    torch.manual_seed(2023)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    (tt_input, _, tt_output_shape, torch_output_shape, torch_input) = get_tensors(
        input_shape, dim, device, use_randint=False
    )
    (tt_output_grad, tt_input_grad, torch_output_grad) = get_backward_tensors(
        tt_output_shape, torch_output_shape, input_shape, device, use_randint=False
    )

    # convert torch_input to float32 dtype
    torch_input = torch_input.detach().clone().to(dtype=torch.float32).requires_grad_(True)
    torch_output_grad = torch_output_grad.float()
    torch_output = torch.sum(torch_input, dim)
    torch_output.backward(torch_output_grad)

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_input_grad_cpu = (
        ttl.operations.primary.moreh_sum_backward(
            tt_output_grad,
            tt_input,
            dim=dim,
            input_grad=tt_input_grad,
            compute_kernel_config=compute_kernel_config,
        )
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(input_shape)
        .to_torch()
    )

    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, tt_input_grad_cpu, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")
    logger.debug(f"std={torch.std(torch.abs(torch_input.grad- tt_input_grad_cpu))}")
    logger.debug(f"mean={torch.abs(torch_input.grad - tt_input_grad_cpu).mean()}")

    assert passing
