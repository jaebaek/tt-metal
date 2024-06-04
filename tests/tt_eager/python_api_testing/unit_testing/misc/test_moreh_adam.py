# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.optim as optim

import tt_lib as ttl
import pytest
from models.utility_functions import skip_for_wormhole_b0, comp_allclose_and_pcc, comp_pcc, is_wormhole_b0
from loguru import logger

fp32_dest_acc_en = [
    False,  # for grayskull
]
fp32_dest_acc_en_ids = ["fp32_dest_acc_en=False"]
# if is_wormhole_b0:
#     fp32_dest_acc_en.append(True)
#     fp32_dest_acc_en_ids.append("fp32_dest_acc_en=True")


def create_tt_tensor(tensor, device):
    ret = (
        ttl.tensor.Tensor(
            tensor,
            ttl.tensor.DataType.BFLOAT16,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    return ret


def get_compute_kernel_options(fp32_dest_acc_en):
    if fp32_dest_acc_en is None:
        return None

    if is_wormhole_b0:
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=False,
        )
    else:
        # Grayskull doesn't support fp32 but test passing a GS config is ok
        compute_kernel_config = ttl.tensor.GrayskullComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,
            math_approx_mode=True,
        )
    return compute_kernel_config


@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],
        [3, 32, 32],
        [12, 6, 64, 64],
        [3, 4, 6, 64, 64],
    ],
)
@pytest.mark.parametrize("lr", [0.0, 1e-1])
@pytest.mark.parametrize("betas", [[0.9, 0.999], [0.5, 0.555]])
@pytest.mark.parametrize("eps", [1e-06, 1e-08])
@pytest.mark.parametrize("weight_decay", [0.0, 0.3])
@pytest.mark.parametrize("amsgrad", [True, False])
@pytest.mark.parametrize("fp32_dest_acc_en", fp32_dest_acc_en, ids=fp32_dest_acc_en_ids)
def test_moreh_adam(shape, lr, betas, eps, weight_decay, amsgrad, fp32_dest_acc_en, device):
    torch.manual_seed(0)

    x_data = torch.rand(shape).to(torch.bfloat16)
    y_data = torch.rand(shape).to(torch.bfloat16)

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.weight = nn.Parameter(torch.randn(shape).to(torch.bfloat16)).to(torch.bfloat16)

        def forward(self, x):
            return torch.mul(x, self.weight)

    model = SimpleModel()
    cpu_exp_avg = torch.zeros_like(model.weight)
    cpu_exp_avg_sq = torch.zeros_like(model.weight)
    cpu_max_exp_avg_sq = torch.zeros_like(model.weight)

    dev_param = create_tt_tensor(model.weight, device)
    dev_exp_avg = create_tt_tensor(cpu_exp_avg, device)
    dev_exp_avg_sq = create_tt_tensor(cpu_exp_avg_sq, device)
    dev_max_exp_avg_sq = create_tt_tensor(cpu_max_exp_avg_sq, device)

    dev_param_out = create_tt_tensor(model.weight, device)
    dev_exp_avg_out = create_tt_tensor(cpu_exp_avg, device)
    dev_exp_avg_sq_out = create_tt_tensor(cpu_exp_avg_sq, device)
    dev_max_exp_avg_sq_out = create_tt_tensor(cpu_max_exp_avg_sq, device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam({model.weight}, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    optimizer.zero_grad()
    optimizer_state_dict = optimizer.state_dict()
    outputs = model(x_data)
    loss = criterion(outputs, y_data)
    loss.backward()

    cpu_grad = model.weight.grad.clone()
    dev_grad = (
        ttl.tensor.Tensor(
            cpu_grad,
            ttl.tensor.DataType.BFLOAT16,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    optimizer.step()
    optimizer_state_dict = optimizer.state_dict()

    cpu_exp_avg_result = optimizer_state_dict["state"][0]["exp_avg"]
    cpu_exp_avg_sq_result = optimizer_state_dict["state"][0]["exp_avg_sq"]
    if "max_exp_avg_sq" in optimizer_state_dict["state"][0]:
        cpu_max_exp_avg_sq_result = optimizer_state_dict["state"][0]["max_exp_avg_sq"]
    else:
        cpu_max_exp_avg_sq_result = None

    compute_kernel_config = get_compute_kernel_options(fp32_dest_acc_en)

    step = 1

    dev_param_out, dev_exp_avg_out, dev_exp_avg_sq_out, dev_max_exp_avg_sq_out = ttl.operations.primary.moreh_adam(
        dev_param,
        dev_grad,
        dev_exp_avg,
        dev_exp_avg_sq,
        lr,
        betas[0],
        betas[1],
        eps,
        weight_decay,
        step,
        amsgrad,
        dev_max_exp_avg_sq,
        dev_param_out,
        dev_exp_avg_out,
        dev_exp_avg_sq_out,
        dev_max_exp_avg_sq_out,
        compute_kernel_config=compute_kernel_config,
    )

    assert dev_param.get_legacy_shape() == list(model.weight.shape)

    param_result = dev_param_out.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)
    exp_avg_result = dev_exp_avg_out.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)
    exp_avg_sq_result = dev_exp_avg_sq_out.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)
    if "max_exp_avg_sq" in optimizer_state_dict["state"][0]:
        max_exp_avg_sq_result = (
            dev_max_exp_avg_sq_out.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)
        )
    else:
        max_exp_avg_sq_result = None

    rtol = atol = 0.01
    passing, out = comp_allclose_and_pcc(model.weight, param_result, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    passing, out = comp_allclose_and_pcc(cpu_exp_avg_result, exp_avg_result, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (exp_avg)={passing}")
    logger.debug(f"Output pcc={out}")

    passing, out = comp_allclose_and_pcc(cpu_exp_avg_sq_result, exp_avg_sq_result, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (exp_avg_sq)={passing}")
    logger.debug(f"Output pcc={out}")

    if "max_exp_avg_sq" in optimizer_state_dict["state"][0]:
        passing, out = comp_allclose_and_pcc(
            cpu_max_exp_avg_sq_result, max_exp_avg_sq_result, pcc=0.999, rtol=rtol, atol=atol
        )
        logger.debug(f"Out passing (max_exp_avg_sq)={passing}")
        logger.debug(f"Output pcc={out}")
    assert passing
