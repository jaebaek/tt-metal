# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from loguru import logger
from transformers import VisionEncoderDecoderModel, TrOCRConfig

# import tt_lib
# tt-metal/models/experimental/trocr/reference/torch_trocr.py
from models.experimental.trocr.reference.torch_trocr import TrOCRDecoderLayer, TrOCRDecoder, TrOCRForCausalLM

# from models.experimental.trocr.reference.trocr_configuration import Torch_TrOCRConfig
# from models.experimental.trocr.tt.trocr_decoder_layer import TtTrOCRDecoderLayer
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.66),),
)
def test_trocr_causallm(device, pcc, reset_seeds):
    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        config = model.decoder.config
        # input shapes for causallm tensor([[2]]) tensor([[1]]) torch.Size([1, 577, 768])

        torch_model = model.decoder.model

        input1 = torch.rand(1, 1).long()
        input2 = torch.rand(1, 1)
        input3 = torch.rand(1, 577, 768)

        model_output = torch_model(input1, input2, input3, None, None, None, None, None, False, False, False, True)

        print("\nTorch run is completed!\n")
        ref_model = TrOCRForCausalLM(config)

        ref_output = ref_model(input1, input2, input3, None, None, None, None, None, False, False, False, True)

        # run tt model
        # tt_input = torch_to_tt_tensor_rm(input, device=device, put_on_device=False)
        # tt_output = tt_model(tt_input)
        # tt_output_torch = tt_to_torch_tensor(tt_output.last_hidden_state).squeeze(0)

        # # compare output
        # passing, pcc_message = comp_pcc(model_output, ref_output, pcc)

        # logger.info(comp_allclose(model_output, ref_output))
        # logger.info(pcc_message)

        # if passing:
        #     logger.info("TrOCRDecoder Passed!")
        # else:
        #     logger.warning("TrOCRDecoder Failed!")

        # assert passing, f"TrOCRDecoder output does not meet PCC requirement {pcc}."


# @pytest.mark.parametrize(
#     "pcc",
#     ((0.66),),
# )
# def test_trocr_decoder_inference(device, pcc, reset_seeds):
#     with torch.no_grad():
#         model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

#         config = model.decoder.config
#         print("\n\nßAt start",config)

#         # print(model.decoder.config)

#         base_address = f"decoder"

#         torch_model = model.decoder.model.decoder

#         # run torch model
#         input = torch.rand(1, 5).long()

#         model_output = torch_model(input.long()).last_hidden_state

#         ref_model = TrOCRDecoder(config)

#         ref_output = ref_model(input)

#         # run tt model
#         # tt_input = torch_to_tt_tensor_rm(input, device=device, put_on_device=False)
#         # tt_output = tt_model(tt_input)
#         # tt_output_torch = tt_to_torch_tensor(tt_output.last_hidden_state).squeeze(0)

#         # # compare output
#         # passing, pcc_message = comp_pcc(model_output, ref_output, pcc)

#         # logger.info(comp_allclose(model_output, ref_output))
#         # logger.info(pcc_message)

#         # if passing:
#         #     logger.info("TrOCRDecoder Passed!")
#         # else:
#         #     logger.warning("TrOCRDecoder Failed!")

#         # assert passing, f"TrOCRDecoder output does not meet PCC requirement {pcc}."


# @pytest.mark.parametrize(
#     "pcc",
#     ((0.99),),
# )
# def test_trocr_decoder_layer_inference(device, pcc, reset_seeds):
#     with torch.no_grad():
#         model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

#         config = model.decoder.config
#         print(config)

#         # config2 = Torch_TrOCRConfig()
#         # print(config2)

#         # exit(0)

#         base_address = f"decoder.model.decoder.layers.0"

#         torch_model = model.decoder.model.decoder.layers[0]

#         ref_model = TrOCRDecoderLayer(
#             config=config
#             # base_address=base_address,
#             # state_dict=model.state_dict(),
#             # device=device,
#         )

#         # run torch model
#         input = torch.rand(1, 3, 1024)

#         model_output = torch_model(input)[0]
#         ref_output = ref_model(input)[0]

#         # # run tt model
#         # tt_input = torch_to_tt_tensor_rm(input, device)
#         # tt_output = tt_model(tt_input)
#         # tt_output_torch = tt_to_torch_tensor(tt_output[0])
#         # tt_output_torch = tt_output_torch.squeeze(0)

#         # # compare output
#         passing, pcc_message = comp_pcc(model_output, ref_output, pcc)

#         # logger.info(comp_allclose(model_output, tt_output_torch))
#         logger.info(pcc_message)

#         if passing:
#             logger.info("TrOCRDecoderLayer Passed!")
#         else:
#             logger.warning("TrOCRDecoderLayer Failed!")

#         assert passing, f"TrOCRDecoderLayer output does not meet PCC requirement {pcc}."
