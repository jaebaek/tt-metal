# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from torchvision import models
from transformers import AutoImageProcessor
import pytest
import tt_lib

from models.utility_functions import is_e75, profiler, divup, disable_persistent_kernel_cache, skip_for_wormhole_b0
from models.perf.perf_utils import prep_perf_report

from loguru import logger
from models.demos.resnet.tt.metalResnetBlock50 import ResNet, Bottleneck

model_config = {
    "MATH_FIDELITY": tt_lib.tensor.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": tt_lib.tensor.DataType.BFLOAT8_B,
    "ACTIVATIONS_DTYPE": tt_lib.tensor.DataType.BFLOAT8_B,
}


def run_perf_resnet(
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    device,
):
    disable_persistent_kernel_cache()
    if batch_size <= 2:
        pytest.skip("Batch size 1 and 2 are not supported with sharded data")
    first_key = f"first_iter_batchsize{batch_size}"
    second_key = f"second_iter_batchsize{batch_size}"
    cpu_key = f"ref_key_batchsize{batch_size}"
    model_name = "microsoft/resnet-50"

    image = hf_cat_image_sample_input
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    inputs = image_processor(image, return_tensors="pt")

    inputs = inputs["pixel_values"]
    comments = f"{list(inputs.shape)[-2]}x{list(inputs.shape)[-1]}_batchsize{batch_size}"

    inputs1 = inputs
    for i in range(batch_size - 1):
        inputs = torch.cat((inputs, inputs1), dim=0)

    torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    torch_resnet50.eval()

    state_dict = torch_resnet50.state_dict()
    sharded = False
    if batch_size >= 8:
        sharded = True
    tt_resnet50 = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        device=device,
        state_dict=state_dict,
        base_address="",
        fold_batchnorm=True,
        storage_in_dram=False,
        batch_size=batch_size,
        model_config=model_config,
        sharded=sharded,
    )

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = torch_resnet50(inputs)
        profiler.end(cpu_key)

        tt_inputs = tt_resnet50.preprocessing(inputs)
        input_shape = tt_inputs.get_legacy_shape()
        shard_spec = tt_lib.tensor.ShardSpec(
            tt_lib.tensor.CoreRangeSet(
                {
                    tt_lib.tensor.CoreRange(
                        tt_lib.tensor.CoreCoord(0, 0),
                        tt_lib.tensor.CoreCoord(7, 0),
                    )
                }
            ),
            [
                divup(tt_inputs.volume() // input_shape[3], 8),
                input_shape[3],
            ],
            tt_lib.tensor.ShardOrientation.ROW_MAJOR,
            False,
        )
        sharded_mem_config_DRAM = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.DRAM, shard_spec
        )
        tt_image_res = tt_lib.tensor.allocate_tensor_on_device(
            tt_inputs.shape, tt_inputs.dtype, tt_inputs.layout, device, sharded_mem_config_DRAM
        )
        op_event = tt_lib.device.CreateEvent()
        write_event = tt_lib.device.CreateEvent()
        # Initialize the op event so we can write
        tt_lib.device.RecordEvent(device, 0, op_event)
        warmup_end = 5
        for iter in range(0, warmup_end):
            profiler.start(f"{iter}_key")
            tt_lib.device.WaitForEvent(device, 1, op_event)
            tt_lib.tensor.write_tensor(tt_inputs, tt_image_res, 1)
            tt_lib.device.RecordEvent(device, 1, write_event)
            _ = tt_resnet50(tt_image_res, write_event, op_event).cpu(blocking=True)
            profiler.end(f"{iter}_key")
            tt_lib.device.DumpDeviceProfiler(device)

        num_warm_iterations = 10
        warm_start = warmup_end
        warm_end = warm_start + num_warm_iterations

        outputs = []
        profiler.start(f"run")
        for iter in range(warm_start, warm_end):
            tt_lib.device.WaitForEvent(device, 1, op_event)
            tt_lib.tensor.write_tensor(tt_inputs, tt_image_res, 1)
            tt_lib.device.RecordEvent(device, 1, write_event)
            outputs.append(tt_resnet50(tt_image_res, write_event, op_event).cpu(blocking=False))
        tt_lib.device.Synchronize(device)
        profiler.end(f"run")
        tt_lib.device.DumpDeviceProfiler(device)

        # enable_persistent_kernel_cache()

    first_iter_time = profiler.get(f"{0}_key")

    # ensuring inference time fluctuations is not noise
    inference_time_avg = profiler.get("run") / num_warm_iterations

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - inference_time_avg
    prep_perf_report(
        model_name=f"resnet50_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"resnet50 {comments} inference time (avg): {inference_time_avg}")
    logger.info(f"resnet50 compile time: {compile_time}")


@skip_for_wormhole_b0(reason_str="Not tested on single WH")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "num_hw_cqs": 2}], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    (
        # (1, 0.001, 1),
        # (2, 0.001, 1),
        # (16, 0.007, 7),
        (20, 0.007, 7),
    ),
)
def test_perf_bare_metal(
    device,
    use_program_cache,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
):
    if is_e75(device):
        pytest.skip("Resnet is not supported on E75")

    run_perf_resnet(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
    )


def run_perf_resnet_trace(
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    device,
):
    disable_persistent_kernel_cache()
    if batch_size <= 2:
        pytest.skip("Batch size 1 and 2 are not supported with sharded data")
    first_key = f"first_iter_batchsize{batch_size}"
    second_key = f"second_iter_batchsize{batch_size}"
    cpu_key = f"ref_key_batchsize{batch_size}"
    model_name = "microsoft/resnet-50"

    image = hf_cat_image_sample_input
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    inputs = image_processor(image, return_tensors="pt")

    inputs = inputs["pixel_values"]
    comments = f"{list(inputs.shape)[-2]}x{list(inputs.shape)[-1]}_batchsize{batch_size}"

    inputs1 = inputs
    for i in range(batch_size - 1):
        inputs = torch.cat((inputs, inputs1), dim=0)

    torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    torch_resnet50.eval()

    state_dict = torch_resnet50.state_dict()
    sharded = False
    if batch_size >= 8:
        sharded = True
    tt_resnet50 = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        device=device,
        state_dict=state_dict,
        base_address="",
        fold_batchnorm=True,
        storage_in_dram=False,
        batch_size=batch_size,
        model_config=model_config,
        sharded=sharded,
    )

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = torch_resnet50(inputs)
        profiler.end(cpu_key)

        tt_inputs = tt_resnet50.preprocessing(inputs)
        interleaved_mem_config_DRAM = tt_lib.tensor.MemoryConfig(
            memory_layout=tt_lib.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=tt_lib.tensor.BufferType.DRAM,
        )
        tt_image_res = tt_inputs.to(device, interleaved_mem_config_DRAM)
        # Compile
        profiler.start(f"{0}_key")
        tt_lib.tensor.write_tensor(tt_inputs, tt_image_res)
        tt_resnet50(tt_image_res).cpu(blocking=True)
        profiler.end(f"{0}_key")
        tt_lib.device.DumpDeviceProfiler(device)

        # Capture
        tid = tt_lib.device.BeginTraceCapture(device, 0, 1334880)
        tt_output_res = tt_resnet50(tt_image_res)
        tt_lib.device.EndTraceCapture(device, 0, tid)
        tt_lib.device.DumpDeviceProfiler(device)

        warmup_end = 6
        for iter in range(1, warmup_end):
            profiler.start(f"{iter}_key")
            tt_lib.tensor.write_tensor(tt_inputs, tt_image_res)
            tt_lib.device.ReplayTrace(device, 0, tid, False)
            _ = tt_output_res.cpu(blocking=True)
            profiler.end(f"{iter}_key")
            tt_lib.device.DumpDeviceProfiler(device)

        num_warm_iterations = 15
        warm_start = warmup_end
        warm_end = warm_start + num_warm_iterations

        outputs = []
        profiler.start(f"run")
        for iter in range(warm_start, warm_end):
            tt_lib.tensor.write_tensor(tt_inputs, tt_image_res)
            tt_lib.device.ReplayTrace(device, 0, tid, False)
            outputs.append(tt_output_res.cpu(blocking=False))
        tt_lib.device.Synchronize(device)
        profiler.end(f"run")
        tt_lib.device.DumpDeviceProfiler(device)

        # enable_persistent_kernel_cache()

    first_iter_time = profiler.get(f"{0}_key")

    # ensuring inference time fluctuations is not noise
    inference_time_avg = profiler.get("run") / num_warm_iterations

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - inference_time_avg
    prep_perf_report(
        model_name=f"resnet50_trace_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"resnet50 {comments} inference time (avg): {inference_time_avg}")
    logger.info(f"resnet50 compile time: {compile_time}")

    tt_lib.device.ReleaseTrace(device, tid)

    assert inference_time_avg < expected_inference_time, f"resnet50 {comments} inference is too slow"
    assert compile_time < expected_compile_time, f"resnet50 {comments} compilation is too slow"


@skip_for_wormhole_b0(reason_str="Not tested on single WH")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    (
        (16, 0.04, 25),
        (20, 0.04, 25),
    ),
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_perf_trace_bare_metal(
    device,
    use_program_cache,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    enable_async,
):
    if is_e75(device):
        pytest.skip("Resnet is not supported on E75")
    device.enable_async(enable_async)
    run_perf_resnet_trace(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
    )
    device.enable_async(False)
