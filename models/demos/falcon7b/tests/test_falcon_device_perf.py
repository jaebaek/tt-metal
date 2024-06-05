# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import numpy as np
from sklearn.metrics import top_k_accuracy_score

import tt_lib
from models.demos.falcon7b.tt.falcon_causallm import TtFalconCausalLM

# TODO: Remove this?
from models.demos.falcon7b.tt.falcon_common import (
    PytorchFalconCausalLM,
)

from models.demos.falcon7b.tt.model_config import (
    get_model_config,
)
from models.demos.falcon7b.tests.test_utils import (
    get_rand_falcon_inputs,
    concat_device_out_layer_present,
    load_hf_model,
)
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    get_atol_rtol_pcc,
)

from models.utility_functions import (
    tt_tensors_to_torch_tensors,
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    disable_compilation_reports,
    skip_for_grayskull,
)
from models.perf.perf_utils import prep_perf_report


def get_inputs_on_device(llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len):
    if llm_mode == "prefill":
        tt_input_ids, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(
                    llm_mode, model_input[i::batch], kv_cache_len, num_input_tokens=seq_len
                )
                for i in range(batch)
            ]
        )
    elif llm_mode == "decode":
        tt_input_ids, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )
    return tt_input_ids, tt_attention_mask


def run_test_FalconCausalLM_end_to_end(
    devices,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    expected_pccs,
    model_config,
    model_config_str,
    tt_cache_path,
    model_location_generator,
    expected_inference_time,
    async_mode=False,
):
    # Clear global profiler state before starting measurements
    profiler.clear()

    disable_persistent_kernel_cache()

    num_devices = len(devices)
    global_batch = batch * num_devices

    profiler.start("hugging_face_model_setup")
    hugging_face_reference_model, state_dict = load_hf_model(model_location_generator, model_version)
    configuration = hugging_face_reference_model.config
    pytorch_FalconCausalLM = PytorchFalconCausalLM(hugging_face_reference_model, num_layers)
    profiler.end("hugging_face_model_setup")

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    base_url = ""
    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    use_cache = True

    if True:
        model_input = torch.arange(seq_len * global_batch).reshape(global_batch, seq_len)
    else:
        # batch identical sequences for debugging
        model_input = torch.stack([torch.arange(seq_len)] * global_batch).reshape(global_batch, seq_len)

    # Generate dummy kv_cache --------------------------------------------------------------
    (
        past_key_values,
        tt_layer_past,
        kv_len,
    ) = get_rand_falcon_inputs(
        llm_mode,
        seq_len,
        batch,
        kv_cache_len,
        devices,
        global_batch,
        head_dim,
        max_position_embeddings,
        configuration,
        model_config,
        num_layers=num_layers,
        generate_attention_inputs=False,
    )

    profiler.start("TtFalcon_model_setup")
    tt_FalconCausalLM = TtFalconCausalLM(
        devices,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_position_embeddings,
        model_config,
        tt_cache_path,
        seq_len,
    )
    profiler.end("TtFalcon_model_setup")

    profiler.start("processing_of_input")
    # TODO: Generate attention_mask on device
    tt_input_ids, tt_attention_mask = get_inputs_on_device(
        llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len
    )
    profiler.end("processing_of_input")

    # First run to fill compile cache ----------------------------------------------------
    logger.info(f"Running Falcon model once to fill caches -> disable profiler")
    profiler.disable()

    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run_with_compile", force_enable=True)
    if llm_mode == "prefill":
        tt_outs = []
        # Device transfer time is included in model run time for prefill
        tt_input_ids, tt_attention_mask = get_inputs_on_device(
            llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len
        )
        for user_id in range(batch):
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_ids=tt_input_ids[user_id],
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask[user_id],
                user_id=user_id,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=use_cache,
                device_perf_run=True,
            )
            tt_outs.append(tt_out)
        tt_out = tt_outs

    elif llm_mode == "decode":
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_ids=tt_input_ids,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
            device_perf_run=True,
        )
    for device in devices:
        tt_lib.device.Synchronize(device)
    profiler.end("first_model_run_with_compile", force_enable=True)

    # Dump device profiler data before second run to avoid exceeding profiler memory limits when using tracy
    for device in devices:
        tt_lib.device.DumpDeviceProfiler(device)

    # Prepare reference output -----------------------------------------------------------

    profiler.start("hugging_face_reference_model")
    pytorch_out, pytorch_layer_present = pytorch_FalconCausalLM(
        input_ids=model_input, past_key_values=past_key_values, use_cache=use_cache
    )
    profiler.end("hugging_face_reference_model")

    # Second run for perf ----------------------------------------------------------------

    logger.info(f"Enable profiler and enable binary and compile cache")
    profiler.enable()

    # Regenerate input ids and attention_mask on device
    tt_input_ids, tt_attention_mask = get_inputs_on_device(
        llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len
    )

    if llm_mode == "prefill":
        tt_out_tmp = torch.zeros(global_batch, seq_len, configuration.vocab_size)  # Output tensor to overwrite
        for user_id, tt_out in enumerate(tt_outs):
            # Get outputs from all devices
            tt_out_tmp[user_id::batch] = torch.concat(
                [tt_out_torch.squeeze(1) for tt_out_torch in tt_tensors_to_torch_tensors(tt_out)]
            )
        tt_out = tt_out_tmp
    elif llm_mode == "decode":
        tt_out = [tt_out_torch.squeeze(1).transpose(0, 1) for tt_out_torch in tt_tensors_to_torch_tensors(tt_out)]
        tt_out = torch.concat(tt_out)

    # check outputs ----------------------------------------------------------------------
    does_pass = True
    tt_out_tmp = tt_out.type(pytorch_out.dtype)
    _, _, device_pcc, pcc_str = get_atol_rtol_pcc(pytorch_out, tt_out_tmp)
    logger.info(f"Output: {pcc_str}")
    if device_pcc < expected_pccs[0]:
        does_pass = False
        logger.warning(f"Output PCC {device_pcc} is lower than {expected_pccs[0]}")

    last_layer_index = 31
    if llm_mode == "prefill":
        pytorch_layer_pres = (
            pytorch_layer_present[last_layer_index][0].squeeze(1),
            pytorch_layer_present[last_layer_index][1].squeeze(1),
        )
        tt_layer_pres = concat_device_out_layer_present(num_devices, tt_layer_present[last_layer_index], kv_len)
    elif llm_mode == "decode":
        pytorch_layer_pres = (
            pytorch_layer_present[last_layer_index][0].squeeze(1)[:, kv_cache_len, :],
            pytorch_layer_present[last_layer_index][1].squeeze(1)[:, kv_cache_len, :],
        )
        tt_layer_pres = concat_device_out_layer_present(
            num_devices, tt_layer_present[i], kv_cache_len, end_idx_only=True
        )
    tt_layer_pres_0 = tt_layer_pres[0].type(pytorch_layer_pres[0].dtype)
    _, _, device_pcc_k, _ = get_atol_rtol_pcc(pytorch_layer_pres[0], tt_layer_pres_0)

    tt_layer_pres_1 = tt_layer_pres[1].type(pytorch_layer_pres[1].dtype)
    _, _, device_pcc_v, _ = get_atol_rtol_pcc(pytorch_layer_pres[1], tt_layer_pres_1)

    logger.info(f"Device PCC K: {device_pcc_k}")
    logger.info(f"Device PCC V: {device_pcc_v}")

    if device_pcc_k < expected_pccs[1]:
        does_pass = False
        logger.warning(f"K Cache PCC {device_pcc_k} is lower than {expected_pccs[1]}")

    if device_pcc_v < expected_pccs[2]:
        does_pass = False
        logger.warning(f"V Cache PCC {device_pcc_v} is lower than {expected_pccs[2]}")

    assert does_pass, "PCC checks failed"


@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize(
    "llm_mode, num_layers, batch, seq_len, kv_cache_len, model_config_str, expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc, expected_inference_time",
    (
        ("prefill", 32, 1, 128, 0, "BFLOAT16-DRAM", 0.97, 0.99, 0.97, 0.1),
        ("prefill", 32, 1, 1024, 0, "BFLOAT16-DRAM", 0.99, 0.99, 0.98, 0.5),
        ("prefill", 32, 1, 2048, 0, "BFLOAT16-DRAM", 0.99, 0.99, 0.98, 1.1),
    ),
    ids=[
        "prefill_seq128_bf16_dram",
        "prefill_seq1024_bf16_dram",
        "prefill_seq2048_bf16_dram",
    ],
)
def test_perf_wh_bare_metal(
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    expected_inference_time,
    num_layers,
    expected_output_pcc,
    expected_k_cache_pcc,
    expected_v_cache_pcc,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    device,
    use_program_cache,
):
    model_config = get_model_config(model_config_str, seq_len)
    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    run_test_FalconCausalLM_end_to_end(
        [device],
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        num_layers,
        [expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc],
        model_config,
        model_config_str,
        tt_cache_path,
        model_location_generator,
        expected_inference_time,
    )


@pytest.mark.parametrize("seq_len, samples", [(128, 1800), (1024, 3000), (2048, 3000)])
def test_device_perf(seq_len, samples):
    command = f"pytest models/demos/falcon7b/tests/test_falcon_device_perf.py::test_perf_wh_bare_metal -k prefill_seq{seq_len}_bf16_dram"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    subdir = "falcon7b"
    post_processed_results = run_device_perf(command, subdir, 1, cols, seq_len)

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: samples}

    expected_results = check_device_perf(post_processed_results, 0.03, expected_perf_cols)

    prep_device_perf_report(
        model_name=f"falcon7b-prefill-bf16-dram-seq{seq_len}",
        batch_size=1,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
    )
