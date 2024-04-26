# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
    data_gen_with_val,
)
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        # (torch.Size([1, 1, 320, 384])),
        # (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_remainder(input_shapes, device):
    # in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    # grad_data, grad_tensor = data_gen_with_range(input_shapes, -80, 80, device)
    # in_data, input_tensor = data_gen_with_val(input_shapes, device, True, val=10.6875)
    # grad_data, grad_tensor = data_gen_with_val(input_shapes, device, val=-79)#d1 = -68.0 , d2 = -68.5 if diff=0.5, ret -68.5
    # in_data, input_tensor = data_gen_with_val(input_shapes, device, True, val=14.875)  # 71.5 = 14.8750
    # grad_data, grad_tensor = data_gen_with_val(
    #    input_shapes, device, val=-71.5
    # )  # d1 = -56.5 , d2 = -56.75 if diff=0.25 ret -56.5 both same
    in_data, input_tensor = data_gen_with_val(input_shapes, device, True, val=-12.125)
    grad_data, grad_tensor = data_gen_with_val(
        input_shapes, device, val=-79.0
    )  # d1 = -55.75 , d2 = -56.0 if diff=0.25 ret -56.0
    print(in_data)
    print(grad_data)

    golden_tensor = torch.remainder(in_data, grad_data)
    golden_tensor = torch.where(torch.isnan(golden_tensor), torch.tensor(float("inf")), golden_tensor)

    tt_output_tensor_on_device = tt_lib.tensor.atan2(input_tensor, grad_tensor)
    tt_out_tensor = tt_output_tensor_on_device.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    comp_pass, comp_out = comparison_funcs.comp_pcc(golden_tensor, tt_out_tensor)
    comp_all, _ = comparison_funcs.comp_allclose(golden_tensor, tt_out_tensor, atol=4, rtol=1e-1)
    print(comp_pass)
    print(comp_all)
    print(comp_out)
    print(tt_out_tensor)
    print(golden_tensor)

    differing_elements = torch.ne(golden_tensor, tt_out_tensor)
    total_differences = differing_elements.sum().item()
    print("Total differing elements --> ", total_differences)
    diff = torch.abs(golden_tensor - tt_out_tensor)
    max_diff = torch.max(diff)

    if True:
        print("Inputs for which the outputs differ by more than 0:")
        indices = torch.nonzero(diff)
        iter = 0
        for idx in indices:
            if iter < 100:
                input1_val = in_data[idx[0], idx[1], idx[2], idx[3]]
                input2_val = grad_data[idx[0], idx[1], idx[2], idx[3]]
                expected_output_val = golden_tensor[idx[0], idx[1], idx[2], idx[3]]
                actual_output_val = tt_out_tensor[idx[0], idx[1], idx[2], idx[3]]
                print(
                    f"Input 1 value: {input1_val}, Input 2 value: {input2_val}, Expected output: {expected_output_val}, Actual output: {actual_output_val}"
                )
                print("diff ", torch.abs(expected_output_val) - torch.abs(actual_output_val))
                iter += 1

    assert comp_pass
