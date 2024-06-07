# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import tt_lib as ttl

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
)


def run_embeddings_tests(
    batch_size, num_embeddings, embedding_dim, num_rows, dtype, in0_mem_config, out_mem_config, device, tilized=False
):
    torch.manual_seed(1234)

    tensor = ttl.tensor
    dev = device

    input_rows_shape = [batch_size, 1, 1, num_rows]

    input_rows_torch = torch.randint(0, num_embeddings - 1, tuple(input_rows_shape))

    weights_shape = [1, 1, num_embeddings, embedding_dim]
    weights_torch = torch.randn(weights_shape).bfloat16()

    input_tensor = tensor.Tensor(input_rows_torch, ttl.tensor.DataType.UINT32).to(dev, in0_mem_config)
    weights_tensor = tensor.Tensor(weights_torch, dtype).to(dev, in0_mem_config)

    ttz = tensor.embeddings(input_tensor, weights_tensor, tilized, output_mem_config=out_mem_config)

    if tilized:
        tt_data = ttz.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    else:
        tt_data = ttz.cpu().to_torch()

    tt_got_back = torch.Tensor(tt_data).reshape((batch_size, 1, num_rows, embedding_dim))
    t_ref = torch.nn.functional.embedding(
        input_rows_torch.reshape((batch_size, num_rows)),
        weights_torch.reshape((num_embeddings, embedding_dim)),
    ).reshape((batch_size, 1, num_rows, embedding_dim))

    passing_pcc, output_pcc = comp_equal(t_ref, tt_got_back)
    logger.debug(f"Out passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")
    assert passing_pcc


@pytest.mark.parametrize(
    "out_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=["out_DRAM"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=["in0_DRAM"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "num_rows",
    (32, 256, 512),
    ids=["Num_Rows_32", "Num_Rows_256", "Num_Rows_512"],
)
@pytest.mark.parametrize(
    "num_embeddings",
    (512, 30522, 2048),
    ids=[
        "Bert_Position_Embeddings_512",
        "Bert_Word_Embeddings_30528",
        "Llama_Position_Embeddings",
    ],
)
@pytest.mark.parametrize(
    "embedding_dim",
    (768, 4096),
    ids=["Bert_Num_Cols_768", "Llama_Num_Cols"],
)
@pytest.mark.parametrize(
    "batch_size",
    (1, 8, 9),
    ids=["Batch_Size_1", "Batch_Size_8", "Batch_Size_9"],
)
@pytest.mark.parametrize(
    "tilized",
    (True,),
    ids=[
        "Tilized",
    ],
)
def test_embeddings(
    batch_size, num_embeddings, embedding_dim, num_rows, dtype, in0_mem_config, out_mem_config, device, tilized
):
    run_embeddings_tests(
        batch_size, num_embeddings, embedding_dim, num_rows, dtype, in0_mem_config, out_mem_config, device, tilized
    )


def run_embeddings_optional_output_tests(
    batch_size,
    num_embeddings,
    embedding_dim,
    num_rows,
    dtype,
    in0_mem_config,
    out_mem_config,
    device,
    pass_queue_id,
    tilized=False,
):
    torch.manual_seed(1234)

    tensor = ttl.tensor
    dev = device

    input_rows_shape = [batch_size, 1, 1, num_rows]

    input_rows_torch = torch.randint(0, num_embeddings - 1, tuple(input_rows_shape))

    weights_shape = [1, 1, num_embeddings, embedding_dim]
    weights_torch = torch.randn(weights_shape).bfloat16()

    input_tensor = tensor.Tensor(input_rows_torch, ttl.tensor.DataType.UINT32).to(dev, in0_mem_config)
    weights_tensor = tensor.Tensor(weights_torch, dtype).to(dev, in0_mem_config)

    out_torch = torch.randn([batch_size, 1, num_rows, embedding_dim]).bfloat16()
    out = tensor.Tensor(out_torch, dtype).to(dev, in0_mem_config)

    cq_id = 0
    if pass_queue_id:
        tensor.embeddings(
            cq_id, input_tensor, weights_tensor, tilized, output_mem_config=out_mem_config, output_tensor=out
        )
    else:
        tensor.embeddings(input_tensor, weights_tensor, tilized, output_mem_config=out_mem_config, output_tensor=out)

    if tilized:
        tt_data = out.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    else:
        tt_data = out.cpu().to_torch()

    tt_got_back = torch.Tensor(tt_data).reshape((batch_size, 1, num_rows, embedding_dim))
    t_ref = torch.nn.functional.embedding(
        input_rows_torch.reshape((batch_size, num_rows)),
        weights_torch.reshape((num_embeddings, embedding_dim)),
    ).reshape((batch_size, 1, num_rows, embedding_dim))

    passing_pcc, output_pcc = comp_equal(t_ref, tt_got_back)
    logger.debug(f"Out passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")
    assert passing_pcc


@pytest.mark.parametrize(
    "out_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=["out_DRAM"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=["in0_DRAM"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "num_rows",
    (32, 256, 512),
    ids=["Num_Rows_32", "Num_Rows_256", "Num_Rows_512"],
)
@pytest.mark.parametrize(
    "num_embeddings",
    (512, 30522, 2048),
    ids=[
        "Bert_Position_Embeddings_512",
        "Bert_Word_Embeddings_30528",
        "Llama_Position_Embeddings",
    ],
)
@pytest.mark.parametrize(
    "embedding_dim",
    (768, 4096),
    ids=["Bert_Num_Cols_768", "Llama_Num_Cols"],
)
@pytest.mark.parametrize(
    "batch_size",
    (1, 8, 9),
    ids=["Batch_Size_1", "Batch_Size_8", "Batch_Size_9"],
)
@pytest.mark.parametrize(
    "tilized",
    (False,),
    ids=[
        "Tilized",
    ],
)
@pytest.mark.parametrize("pass_queue_id", [True, False])
def test_embeddings_optional_output(
    batch_size,
    num_embeddings,
    embedding_dim,
    num_rows,
    dtype,
    in0_mem_config,
    out_mem_config,
    device,
    pass_queue_id,
    tilized,
):
    run_embeddings_optional_output_tests(
        batch_size,
        num_embeddings,
        embedding_dim,
        num_rows,
        dtype,
        in0_mem_config,
        out_mem_config,
        device,
        pass_queue_id,
        tilized,
    )
