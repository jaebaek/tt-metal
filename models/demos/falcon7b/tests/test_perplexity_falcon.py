# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer
from tqdm import tqdm
import time
import ttnn
from models.demos.falcon7b.tt.falcon_causallm import TtFalconCausalLM
from models.demos.falcon7b.tt.model_config import get_model_config
from models.demos.falcon7b.tests.test_utils import initialize_kv_cache, load_hf_model
from models.datasets.llm_dataset_utils import prepare_textgen_dataset, prepare_textgen_dataloader
from models.utility_functions import is_wormhole_b0, get_devices_for_t3000, tt_tensors_to_torch_tensors


def calculate_perplexity(model, dataloader, llm_mode, batch_size, seq_len, kv_cache, configuration, use_hf_model=False):
    if llm_mode == "prefill" and not use_hf_model:
        assert batch_size == 1
    use_cache = True
    loss_func = torch.nn.CrossEntropyLoss()
    nlls = []
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader, desc="Evaluating batches"):
            if llm_mode == "prefill":
                if not use_hf_model:
                    user_id = 0
                    (
                        tt_prefill_input_ids,
                        tt_prefill_attention_mask,
                    ) = model.model_preprocessing(
                        "prefill", input_ids[user_id::batch_size], 0, num_input_tokens=seq_len
                    )
                    tt_logits, kv_cache = model(
                        input_ids=tt_prefill_input_ids,
                        llm_mode="prefill",
                        attention_mask=tt_prefill_attention_mask,
                        user_id=user_id,
                        layer_past=kv_cache,
                        layer_past_len=0,
                        use_cache=use_cache,
                    )
                    # Get outputs from all devices
                    logits = torch.concat(
                        [tt_out_torch.squeeze(1) for tt_out_torch in tt_tensors_to_torch_tensors(tt_logits)]
                    )
                    # Deallocate tt tensors
                    for i in range(len(tt_logits)):
                        tt_prefill_input_ids[i].deallocate()
                        if isinstance(tt_prefill_attention_mask[i], ttnn.experimental.tensor.Tensor):
                            tt_prefill_attention_mask[i].deallocate()
                        elif isinstance(tt_prefill_attention_mask[i], list):
                            for tt_attention_mask_element in tt_prefill_attention_mask[i]:
                                tt_attention_mask_element.deallocate()
                        tt_logits[i].deallocate()
                else:  # huggingface model
                    logits, _ = model(input_ids=input_ids, use_cache=use_cache, return_dict=False)

            elif llm_mode == "decode":
                logits = []
                layer_present = None
                for kv_cache_len in tqdm(range(seq_len), desc="Decoding tokens for current batch"):
                    decode_ids = input_ids[:, kv_cache_len].view(batch_size, 1)
                    if not use_hf_model:
                        (
                            tt_decode_input_ids,
                            tt_decode_attention_mask,
                        ) = model.model_preprocessing(
                            "decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1
                        )
                        tt_logits, kv_cache = model(
                            input_ids=tt_decode_input_ids,
                            llm_mode="decode",
                            attention_mask=tt_decode_attention_mask,
                            layer_past=kv_cache,
                            layer_past_len=kv_cache_len,
                            use_cache=use_cache,
                        )
                        # Get outputs from all devices
                        logits_cur = torch.concat(
                            [torch_logit.squeeze(1) for torch_logit in tt_tensors_to_torch_tensors(tt_logits)], dim=-2
                        )
                        logits.append(logits_cur.view(-1, 1, configuration.vocab_size))
                        # Deallocate tt tensors
                        for i in range(len(tt_logits)):
                            tt_decode_input_ids[i].deallocate()
                            tt_decode_attention_mask[i].deallocate()
                            tt_logits[i].deallocate()
                    else:  # huggingface model
                        logits_cur, layer_present = model(
                            input_ids=decode_ids, past_key_values=layer_present, use_cache=use_cache, return_dict=False
                        )
                        logits.append(logits_cur)

                logits = torch.cat(logits, dim=1)

            loss = loss_func(logits.view(batch_size * seq_len, configuration.vocab_size), labels.view(-1))
            nlls.append(loss.float())

    nll = torch.stack(nlls).mean()
    ppl = torch.exp(nll)
    return nll.item(), ppl.item()


def run_test_perplexity(
    llm_mode,
    batch_size,
    max_seq_len,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    devices,
    num_samples,
    expected_ppl,
    stride=None,
    model_version="tiiuae/falcon-7b-instruct",
    num_layers=32,
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    split="test",
    use_hf_model=False,
):
    # Set random reproducible seed
    torch.manual_seed(0)

    # Load HF model
    logger.info("Loading HuggingFace model...")
    hugging_face_reference_model, state_dict = load_hf_model(model_location_generator, model_version)
    configuration = hugging_face_reference_model.config

    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset = prepare_textgen_dataset(dataset_name, dataset_config, split)
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    encodings = tokenizer(dataset, return_tensors="pt")["input_ids"].squeeze(0)
    dataloader = prepare_textgen_dataloader(encodings, batch_size, max_seq_len, num_samples, stride)

    if not use_hf_model:
        # Load tt-metal model config
        model_config = get_model_config(model_config_str, max_seq_len)
        tt_cache_path = get_tt_cache_path(
            model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
        )

        # Load tt-metal model
        logger.info("Moving weights (all layers) to device; might take some time...")
        model = TtFalconCausalLM(
            devices,
            state_dict,
            "",
            num_layers,
            configuration,
            max_seq_len,
            model_config,
            tt_cache_path,
            max_seq_len,
        )

        # Initialize kvcache
        logger.info("Initializing kvcache...")
        kv_cache = initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, devices)
    else:
        model = hugging_face_reference_model
        kv_cache = None

    # Evaluate perplexity
    logger.info("Evaluating perplexity...")
    start = time.time()
    nll, ppl = calculate_perplexity(
        model, dataloader, llm_mode, batch_size, max_seq_len, kv_cache, configuration, use_hf_model=use_hf_model
    )
    logger.info(f"Perplexity evaluation time: {(time.time() - start):.2f} s")
    logger.info(f"Negative log-likelihood: {nll:.4f}")
    logger.info(f"Perplexity: {ppl:.4f}")

    if ppl > expected_ppl:
        assert False, f"Perplexity {ppl} is higher (worse) than {expected_ppl}"
    elif ppl < 0.95 * expected_ppl:
        assert False, f"Perplexity {ppl} is lower (better) than {expected_ppl}. Please update the expected perplexity."
    logger.info("Falcon Perplexity Check Passed!")


@pytest.mark.parametrize(
    "llm_mode, batch_size, max_seq_len, num_samples, expected_ppl",
    (
        ("prefill", 32, 1024, 64, 11.5),
        ("decode", 64, 1024, 64, 11.5),
    ),
    ids=[
        "prefill_seq1024",
        "decode_1024",
    ],
)
def test_perplexity_huggingface(
    llm_mode,
    batch_size,
    max_seq_len,
    num_samples,  # Total number of prompts to evaluate (all if None)
    expected_ppl,
    model_location_generator,
):
    run_test_perplexity(
        llm_mode,
        batch_size,
        max_seq_len,
        None,
        model_location_generator,
        None,
        None,
        num_samples,
        expected_ppl,
        use_hf_model=True,
    )


@pytest.mark.parametrize(
    "llm_mode, batch_size, max_seq_len, model_config_str, num_samples, expected_ppl",
    (
        ("prefill", 1, 1024, "BFLOAT16-DRAM", 64, 12.0),
        ("decode", 32, 1024, "BFLOAT16-L1_SHARDED", 64, 12.5),
    ),
    ids=[
        "prefill_seq1024_dram",
        "decode_1024_l1_sharded",
    ],
)
@pytest.mark.parametrize("async_mode", (True,))  # Option to run Falcon in Async mode
@pytest.mark.parametrize("num_devices", (1,))
def test_perplexity(
    llm_mode,
    batch_size,
    max_seq_len,
    model_config_str,
    num_samples,  # Total number of prompts to evaluate (all if None)
    expected_ppl,
    async_mode,
    num_devices,
    model_location_generator,
    get_tt_cache_path,
    all_devices,
    use_program_cache,
):
    assert is_wormhole_b0(), "Multi-chip is only supported for Wormhole B0"
    devices = get_devices_for_t3000(all_devices, num_devices)

    for device in devices:
        device.enable_async(async_mode)

    run_test_perplexity(
        llm_mode,
        batch_size,
        max_seq_len,
        model_config_str,
        model_location_generator,
        get_tt_cache_path,
        devices,
        num_samples,
        expected_ppl,
    )
