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


def run_model(device, tt_inputs, tt_resnet50, num_warmup_iterations, num_measurement_iterations):
    profiler.start("compile")
    _ = tt_resnet50(tt_inputs).cpu(blocking=True)
    profiler.end("compile")
    tt_lib.device.DumpDeviceProfiler(device)

    for iter in range(0, num_warmup_iterations):
        _ = tt_resnet50(tt_inputs).cpu(blocking=True)
        tt_lib.device.DumpDeviceProfiler(device)

    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        outputs.append(tt_resnet50(tt_inputs).cpu(blocking=False))
    tt_lib.device.Synchronize(device)
    profiler.end(f"run")
    tt_lib.device.DumpDeviceProfiler(device)


def run_2cq_model(device, tt_inputs, tt_resnet50, num_warmup_iterations, num_measurement_iterations):
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

    profiler.start("compile")
    tt_lib.device.WaitForEvent(device, 1, op_event)
    tt_lib.tensor.write_tensor(tt_inputs, tt_image_res, 1)
    tt_lib.device.RecordEvent(device, 1, write_event)
    _ = tt_resnet50(tt_image_res, write_event, op_event).cpu(blocking=True)
    profiler.end("compile")
    tt_lib.device.DumpDeviceProfiler(device)

    for iter in range(0, num_warmup_iterations):
        tt_lib.device.WaitForEvent(device, 1, op_event)
        tt_lib.tensor.write_tensor(tt_inputs, tt_image_res, 1)
        tt_lib.device.RecordEvent(device, 1, write_event)
        _ = tt_resnet50(tt_image_res, write_event, op_event).cpu(blocking=True)
        tt_lib.device.DumpDeviceProfiler(device)

    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        tt_lib.device.WaitForEvent(device, 1, op_event)
        tt_lib.tensor.write_tensor(tt_inputs, tt_image_res, 1)
        tt_lib.device.RecordEvent(device, 1, write_event)
        outputs.append(tt_resnet50(tt_image_res, write_event, op_event).cpu(blocking=False))
    tt_lib.device.Synchronize(device)
    profiler.end(f"run")
    tt_lib.device.DumpDeviceProfiler(device)


def run_trace_model(device, tt_inputs, tt_resnet50, num_warmup_iterations, num_measurement_iterations):
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
    # Compile
    profiler.start("compile")
    tt_lib.tensor.write_tensor(tt_inputs, tt_image_res)
    tt_resnet50(tt_image_res).cpu(blocking=True)
    profiler.end("compile")
    tt_lib.device.DumpDeviceProfiler(device)

    # Capture
    tid = tt_lib.device.BeginTraceCapture(device, 0, 1500000)
    tt_output_res = tt_resnet50(tt_image_res)
    tt_lib.device.EndTraceCapture(device, 0, tid)
    tt_lib.device.DumpDeviceProfiler(device)

    for iter in range(0, num_warmup_iterations):
        tt_lib.tensor.write_tensor(tt_inputs, tt_image_res)
        tt_lib.device.ReplayTrace(device, 0, tid, False)
        _ = tt_output_res.cpu(blocking=True)
        tt_lib.device.DumpDeviceProfiler(device)

    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        tt_lib.tensor.write_tensor(tt_inputs, tt_image_res)
        tt_lib.device.ReplayTrace(device, 0, tid, False)
        outputs.append(tt_output_res.cpu(blocking=False))
    tt_lib.device.Synchronize(device)
    profiler.end(f"run")
    tt_lib.device.DumpDeviceProfiler(device)


def run_perf_resnet(
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    device,
    model_version,
):
    if is_e75(device):
        pytest.skip("Resnet is not supported on E75")
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
    tt_lib.device.Synchronize(device)

    num_warmup_iterations = 5
    num_measurement_iterations = 15

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = torch_resnet50(inputs)
        profiler.end(cpu_key)

        tt_inputs = tt_resnet50.preprocessing(inputs)
        if "resnet50_2cqs" in model_version:
            run_2cq_model(device, tt_inputs, tt_resnet50, num_warmup_iterations, num_measurement_iterations)
        elif "resnet50_trace" in model_version:
            run_trace_model(device, tt_inputs, tt_resnet50, num_warmup_iterations, num_measurement_iterations)
        elif "resnet50" in model_version:
            run_model(device, tt_inputs, tt_resnet50, num_warmup_iterations, num_measurement_iterations)
        else:
            assert False, f"Model version to run {model_version} not found"

    first_iter_time = profiler.get(f"compile")

    # ensuring inference time fluctuations is not noise
    inference_time_avg = profiler.get("run") / num_measurement_iterations

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - inference_time_avg
    prep_perf_report(
        model_name=f"{model_version}_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"{model_name} {comments} inference time (avg): {inference_time_avg}")
    logger.info(f"{model_name} compile time: {compile_time}")


@skip_for_wormhole_b0(reason_str="Not tested on single WH")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    (
        (16, 0.007, 16),
        (20, 0.007, 16),
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
        batch_size, expected_inference_time, expected_compile_time, hf_cat_image_sample_input, device, "resnet50"
    )


@skip_for_wormhole_b0(reason_str="Not tested on single WH")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    ((20, 0.008, 16),),
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
    device.enable_async(enable_async)
    mode = "async" if enable_async else "sync"
    run_perf_resnet(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
        f"resnet50_trace_{mode}",
    )
    device.enable_async(False)
