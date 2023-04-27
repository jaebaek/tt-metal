
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
from torchvision import models

from loguru import logger
import pytest

from libs import tt_lib as ttl
from utility_functions import comp_allclose_and_pcc, comp_pcc
from torch_vgg.vgg import vgg16

_batch_size = 1

@pytest.mark.parametrize("fuse_ops", [False, True], ids=['Not Fused', "Ops Fused"])
def test_vgg16_inference(fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input
    batch_size = _batch_size
    with torch.no_grad():

        torch_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        torch_vgg.eval()

        state_dict = torch_vgg.state_dict()

        tt_vgg = vgg16(state_dict)
        tt_vgg.eval()

        if fuse_ops:
            indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
            modules_to_fuse = [[f"features.{ind}", f"features.{ind+1}",] for ind in indices]
            tt_vgg = torch.ao.quantization.fuse_modules(tt_vgg, modules_to_fuse)

        torch_output = torch_vgg(image).unsqueeze(1).unsqueeze(1)
        tt_output = tt_vgg(image)

        passing = comp_pcc(torch_output, tt_output)
        assert passing[0], passing[1:]

    logger.info(f"vgg16 PASSED {passing[1]}")
