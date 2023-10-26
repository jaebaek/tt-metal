# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from torchvision import models
from transformers import AutoImageProcessor
import pytest
import numpy as np
from loguru import logger
import tt_lib

from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    profiler,
    prep_report,
    is_e75,
)
from tests.models.resnet.tests.demo_utils import get_data
from tests.models.resnet.metalResnetBlock50 import ResNet, Bottleneck


def run_perf_resnet(
    model_location_generator,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    iterations,
    device,
):
    disable_persistent_kernel_cache()
    batch_size = 1
    first_key = f"first_iter"
    second_key = f"second_iter"
    third_key = f"accuracy_loop"
    cpu_key = f"ref_key"
    model_name = "microsoft/resnet-50"

    image = hf_cat_image_sample_input
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    inputs = image_processor(image, return_tensors="pt")

    inputs = inputs["pixel_values"]
    comments = f"{list(inputs.shape)[-2]}x{list(inputs.shape)[-1]}_batchsize{batch_size}"

    torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    torch_resnet50.eval()

    state_dict = torch_resnet50.state_dict()
    sharded = False
    tt_resnet50 = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        device=device,
        state_dict=state_dict,
        base_address="",
        fold_batchnorm=True,
        storage_in_dram=False,
        batch_size=batch_size,
        sharded=sharded,
    )

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = torch_resnet50(inputs)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_resnet50(inputs)
        tt_lib.device.Synchronize()
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_resnet50(inputs)
        tt_lib.device.Synchronize()
        profiler.end(second_key)
        del tt_output

        logger.info("ImageNet-1k validation Dataset")
        if iterations <= 50:
            input_loc = str(model_location_generator("sample_data"))
        else:
            input_loc = str(model_location_generator("ImageNet_data"))
        image_examples = get_data(input_loc)
        reference_labels = []
        predicted_labels = []
        profiler.start(third_key)
        for i in range(iterations):
            input_image = image_examples[i].image
            if input_image.mode == "L":
                input_image = input_image.convert(mode="RGB")
            input = image_processor(input_image, return_tensors="pt")
            input = input["pixel_values"]
            tt_output = tt_resnet50(input)
            prediction = tt_output[0][0][0].argmax()
            prediction = prediction.item()
            predicted_labels.append(prediction)
            reference_labels.append(image_examples[i].label)
        predicted_labels = np.array(predicted_labels)
        reference_labels = np.array(reference_labels)
        accuracy = np.mean(predicted_labels == reference_labels)
        logger.info("Accuracy")
        logger.info(accuracy)
        tt_lib.device.Synchronize()
        profiler.end(third_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time
    prep_report(
        model_name=f"resnet50",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"resnet50 inference time: {second_iter_time}")
    logger.info(f"resnet50 compile time: {compile_time}")
    logger.info(f"resnet50 inference for {iterations} Samples: {third_iter_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time,iterations",
    ((0.225, 33, 50),),
)
def test_perf_bare_metal(
    use_program_cache,
    model_location_generator,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    iterations,
    device,
):
    if is_e75(device):
        pytest.skip("Resnet is not supported on E75")

    run_perf_resnet(
        model_location_generator,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        iterations,
        device,
    )


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time,iterations",
    ((0.3, 36, 50),),
)
def test_perf_virtual_machine(
    use_program_cache,
    model_location_generator,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    iterations,
    device,
):
    if is_e75(device):
        pytest.skip("Resnet is not supported on E75")

    run_perf_resnet(
        model_location_generator,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        iterations,
        device,
    )
