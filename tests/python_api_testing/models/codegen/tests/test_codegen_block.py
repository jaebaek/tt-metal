from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib
import pytest

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger
import python_api_testing.models.codegen.tt.codegen_block as codegen_block
from transformers import CodeGenConfig, CodeGenModel


from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

def run_codegen_block_test(device, pcc):

    model_hf = CodeGenModel.from_pretrained('Salesforce/codegen-350M-mono')
    sd = model_hf.state_dict()
    model_hf.eval()
    block = 0
    base_address = f"h.{block}"

    torch.manual_seed(0)

    test_in = torch.rand(1,1,1024)

    tt_test_in = torch2tt_tensor(test_in, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)


    config = CodeGenConfig('Salesforce/codegen-350M-mono')


    #pt_block = model_hf.h[block]
    #pt_out = pt_block.forward(test_in)

    tt_block = codegen_block.TtCodeGenBlock(base_address, config, sd, device)

    tt_out = tt_block.forward(
        device,
        tt_test_in
    )

    """
    tt_out_converted = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("codegen_block: Passed!")
    else:
        logger.warning("codegen_block: Failed!")

    assert does_pass
    """
    print('DONE')
@pytest.mark.parametrize(
    "pcc",
    (
        (
            0.99,
        ),
    ),
)
def test_codegen_block(pcc):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    run_codegen_block_test(device, pcc)
    tt_lib.device.CloseDevice(device)
