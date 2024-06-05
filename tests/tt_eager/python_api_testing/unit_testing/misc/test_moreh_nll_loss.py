# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib as ttl
import pytest
from models.utility_functions import comp_allclose_and_pcc, is_wormhole_b0
from loguru import logger


fp32_dest_acc_en = [
    False,  # for grayskull
]
fp32_dest_acc_en_ids = ["fp32_dest_acc_en=False"]
if is_wormhole_b0:
    fp32_dest_acc_en.append(True)
    fp32_dest_acc_en_ids.append("fp32_dest_acc_en=True")


def get_compute_kernel_options(fp32_dest_acc_en):
    if fp32_dest_acc_en is None:
        return None

    if is_wormhole_b0():
        packer_l1_acc = False
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )
    else:
        # Grayskull doesn't support fp32 but test passing a GS config is ok
        compute_kernel_config = ttl.tensor.GrayskullComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,
            math_approx_mode=True,
        )
    return compute_kernel_config


def get_torch_tensors(shape):
    C = shape[1]
    target_shape = shape[:1] + shape[2:]

    cpu_dtype = torch.float32
    cpu_index_dtype = torch.long

    torch_input = torch.rand(shape, dtype=cpu_dtype).requires_grad_()
    torch_target = torch.randint(0, C, target_shape, dtype=cpu_index_dtype)
    torch_weight = torch.rand(C, dtype=cpu_dtype)
    torch_divisor = torch.tensor([0], dtype=cpu_dtype)
    torch_output = torch.tensor([0], dtype=cpu_dtype)

    return torch_input, torch_target, torch_weight, torch_divisor, torch_output


def get_tt_tensors(torch_input, torch_target, torch_weight, torch_divisor, torch_output, device):
    C = torch_input.shape[1]

    npu_dtype = ttl.tensor.DataType.BFLOAT16
    npu_index_dtype = ttl.tensor.DataType.INT32
    npu_layout = ttl.tensor.Layout.TILE
    npu_weight_layout = ttl.tensor.Layout.TILE

    tt_input = ttl.tensor.Tensor(torch_input, npu_dtype).pad_to_tile(0).to(npu_layout).to(device)

    if len(torch_target.shape) == 1:
        N = torch_input.shape[0]
        tt_target = ttl.tensor.Tensor(torch_target.reshape(1, N), npu_index_dtype)
    else:
        tt_target = ttl.tensor.Tensor(torch_target, npu_index_dtype)

    tt_target = tt_target.pad_to_tile(0).to(npu_layout).to(device)

    if torch_weight is not None:
        tt_weight = (
            ttl.tensor.Tensor(torch_weight.reshape(1, C), npu_dtype).pad_to_tile(0).to(npu_weight_layout).to(device)
        )
    else:
        tt_weight = None

    if torch_divisor is not None:
        tt_divisor = (
            ttl.tensor.Tensor(torch_divisor.reshape(1, 1), npu_dtype)
            .pad_to_tile(float("nan"))
            .to(npu_layout)
            .to(device)
        )
    else:
        tt_divisor = None

    if torch_output is not None:
        if len(torch_output.shape) == 1:
            tt_output = ttl.tensor.Tensor(torch_output.reshape(1, torch_output.numel()), npu_dtype)
        else:
            tt_output = ttl.tensor.Tensor(torch_output, npu_dtype)

        tt_output = tt_output.pad_to_tile(float("nan")).to(npu_layout).to(device)
    else:
        tt_output = None

    return tt_input, tt_target, tt_weight, tt_divisor, tt_output


@pytest.mark.parametrize(
    "shape",
    [
        (5, 10),
        (500, 100),
        (45, 100, 90),
        (45, 100, 50, 60),
        (5, 100, 2, 7, 50, 70),
    ],
)
@pytest.mark.parametrize("ignore_index", [1])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("none_weight", [True, False])
@pytest.mark.parametrize("fp32_dest_acc_en", fp32_dest_acc_en, ids=fp32_dest_acc_en_ids)
def test_moreh_nll_loss(shape, ignore_index, reduction, none_weight, fp32_dest_acc_en, device, use_program_cache):
    torch.manual_seed(0)

    compute_kernel_config = get_compute_kernel_options(fp32_dest_acc_en)

    (torch_input, torch_target, torch_weight, torch_divisor, torch_output) = get_torch_tensors(shape)

    if none_weight:
        torch_weight = None

    nll_loss = torch.nn.NLLLoss(weight=torch_weight, ignore_index=ignore_index, reduction=reduction)
    torch_loss = torch.tensor([nll_loss(torch_input, torch_target)])

    (tt_input, tt_target, tt_weight, tt_divisor, tt_output) = get_tt_tensors(
        torch_input, torch_target, torch_weight, torch_divisor, torch_output, device
    )

    reduction_mean = reduction == "mean"
    tt_loss = ttl.operations.primary.moreh_nll_loss(
        tt_input,
        tt_target,
        tt_weight,
        tt_divisor,
        tt_output,
        ignore_index,
        reduction_mean,
        compute_kernel_config=compute_kernel_config,
    )

    tt_loss_to_cpu = tt_loss.cpu().to(ttl.tensor.Layout.ROW_MAJOR).unpad_from_tile([1, 1]).to_torch().reshape([1])
    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_loss, tt_loss_to_cpu, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing


@pytest.mark.parametrize(
    "shape",
    [
        (5, 10),
        (5, 10, 10),
        (5, 10, 10, 20),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("none_weight", [True, False])
def test_moreh_nll_loss_callback(shape, reduction, none_weight, device, use_program_cache):
    torch.manual_seed(0)

    ignore_index = 1
    reduction_mean = reduction == "mean"

    # run TT
    for _ in range(2):
        (torch_input, torch_target, torch_weight, torch_divisor, torch_output) = get_torch_tensors(shape)
        if none_weight:
            torch_weight = None

        (tt_input, tt_target, tt_weight, tt_divisor, tt_output) = get_tt_tensors(
            torch_input, torch_target, torch_weight, torch_divisor, torch_output, device
        )

        tt_loss = ttl.operations.primary.moreh_nll_loss(
            tt_input,
            tt_target,
            tt_weight,
            tt_divisor,
            tt_output,
            ignore_index,
            reduction_mean,
        )

    tt_loss_to_cpu = tt_loss.cpu().to(ttl.tensor.Layout.ROW_MAJOR).unpad_from_tile([1, 1]).to_torch().reshape([1])

    # run torch
    nll_loss = torch.nn.NLLLoss(weight=torch_weight, ignore_index=ignore_index, reduction=reduction)
    torch_loss = torch.tensor([nll_loss(torch_input, torch_target)])

    # compare result
    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_loss, tt_loss_to_cpu, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing


@pytest.mark.skip(reason="https://github.com/tenstorrent/tt-metal/issues/9076")
@pytest.mark.parametrize(
    "shape",
    [
        (400, 300),
        (20, 300, 320),
        (3, 4, 32 * 5, 32 * 6),
        (5, 2, 5, 40, 70),
    ],
)
@pytest.mark.parametrize("ignore_index", [1])
@pytest.mark.parametrize("reduction_mean", [True, False])
@pytest.mark.parametrize("none_weight", [True, False])
@pytest.mark.parametrize("fp32_dest_acc_en", fp32_dest_acc_en, ids=fp32_dest_acc_en_ids)
def test_moreh_nll_loss_backward(
    shape, ignore_index, reduction_mean, none_weight, fp32_dest_acc_en, device, use_program_cache
):
    torch.manual_seed(0)

    compute_kernel_config = get_compute_kernel_options(fp32_dest_acc_en)

    (torch_input, torch_target, torch_weight, torch_divisor, torch_output) = get_torch_tensors(shape)
    if none_weight:
        torch_weight = None

    nll_loss = torch.nn.NLLLoss(
        weight=torch_weight, ignore_index=ignore_index, reduction="mean" if reduction_mean else "sum"
    )
    torch_loss = nll_loss(torch_input, torch_target)

    (tt_input, tt_target, tt_weight, tt_divisor, tt_output) = get_tt_tensors(
        torch_input, torch_target, torch_weight, torch_divisor, torch_output, device
    )
    if reduction_mean == False:
        tt_divisor = None
    tt_loss = ttl.operations.primary.moreh_nll_loss(
        tt_input,
        tt_target,
        tt_weight,
        tt_divisor,
        tt_output,
        ignore_index,
        reduction_mean,
        compute_kernel_config=compute_kernel_config,
    )

    # run backward
    output_grad = torch.randn_like(torch_loss)
    torch_loss.backward(output_grad)

    tt_output_grad = (
        ttl.tensor.Tensor(output_grad.reshape(1, 1), ttl.tensor.DataType.BFLOAT16)
        .pad_to_tile(float("nan"))
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    tt_input_grad = (
        ttl.tensor.Tensor(torch_input, ttl.tensor.DataType.BFLOAT16)
        .pad_to_tile(float("nan"))
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    tt_input_grad = ttl.operations.primary.moreh_nll_loss_backward(
        tt_target,
        tt_weight,
        tt_divisor,
        tt_output_grad,
        tt_input_grad,
        ignore_index,
        reduction_mean,
        compute_kernel_config=compute_kernel_config,
    )
    tt_input_grad_to_cpu = tt_input_grad.cpu().to(ttl.tensor.Layout.ROW_MAJOR).unpad_from_tile(shape).to_torch()

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_input.grad, tt_input_grad_to_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing


@pytest.mark.skip(reason="https://github.com/tenstorrent/tt-metal/issues/9076")
@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (2, 3, 4),
        (2, 3, 5, 4),
    ],
)
@pytest.mark.parametrize("reduction_mean", [True, False])
@pytest.mark.parametrize("none_weight", [True, False])
def test_moreh_nll_loss_backward_test_callback(shape, reduction_mean, none_weight, device, use_program_cache):
    torch.manual_seed(0)

    ignore_index = 0

    # run TT
    for _ in range(2):
        (torch_input, torch_target, torch_weight, torch_divisor, torch_output) = get_torch_tensors(shape)
        if none_weight:
            torch_weight = None

        (tt_input, tt_target, tt_weight, tt_divisor, tt_output) = get_tt_tensors(
            torch_input, torch_target, torch_weight, torch_divisor, torch_output, device
        )
        if reduction_mean == False:
            tt_divisor = None
        tt_loss = ttl.operations.primary.moreh_nll_loss(
            tt_input, tt_target, tt_weight, tt_divisor, tt_output, ignore_index, reduction_mean
        )

        output_grad = torch.rand([])

        tt_output_grad = (
            ttl.tensor.Tensor(output_grad.reshape(1, 1), ttl.tensor.DataType.BFLOAT16)
            .pad_to_tile(float("nan"))
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )

        tt_input_grad = (
            ttl.tensor.Tensor(torch_input, ttl.tensor.DataType.BFLOAT16)
            .pad_to_tile(float("nan"))
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )

        tt_input_grad = ttl.operations.primary.moreh_nll_loss_backward(
            tt_target,
            tt_weight,
            tt_divisor,
            tt_output_grad,
            tt_input_grad,
            ignore_index,
            reduction_mean,
        )

    tt_input_grad_to_cpu = tt_input_grad.cpu().to(ttl.tensor.Layout.ROW_MAJOR).unpad_from_tile(shape).to_torch()

    # run torch
    nll_loss = torch.nn.NLLLoss(
        weight=torch_weight, ignore_index=ignore_index, reduction="mean" if reduction_mean else "sum"
    )
    torch_loss = nll_loss(torch_input, torch_target)

    torch_loss.backward(output_grad)

    # compare result
    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_input.grad, tt_input_grad_to_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing
