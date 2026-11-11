# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``DumpAttention`` under the QuantSuite paged-attention backend: the output is
unchanged, and the dumped Q/K/V replay to the dumped output.

Needs a GPU and vLLM; skipped without either.
"""

import contextlib
import importlib.util
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from dump_test_utils import AttentionLayer, dump_session, load, read_manifest
from quant_suite_recipe.definition import PipelineRecipe, QuantAttentionRecipe
from quant_suite_recipe.runtime import QuantRecipeRuntime
from tensor_dump.recipe import DumpRecipe
from tensor_dump.runtime import RUNTIME

_HAS_VLLM: bool = importlib.util.find_spec("vllm") is not None
_LAYER = "model.layers.0.self_attn.attn"


@dataclass
class _PagedMetadata:
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    block_table: torch.Tensor
    decode_offset: torch.Tensor
    num_reqs: int
    num_actual_tokens: int


def _replay(directory: Path, record: dict[str, Any]) -> torch.Tensor:
    """The attention output recomputed in fp32 from the dumped Q, K and V."""
    replay = record["replay"]
    tensors = {
        name: load(directory, entry).float()
        for name, entry in record["tensors"].items()
    }
    group = replay["gqa_group"]
    q = tensors["q"].transpose(0, 1)
    k = tensors["k"].repeat_interleave(group, dim=1).transpose(0, 1)
    v = tensors["v"].repeat_interleave(group, dim=1).transpose(0, 1)
    causal = torch.arange(replay["q_len"])[:, None] + replay[
        "q_start_pos"
    ] >= torch.arange(replay["kv_len"])
    output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=causal, scale=replay["softmax_scale"]
    )
    return output.transpose(0, 1)


def _vllm_forward_context(
    attn_metadata: dict[str, Any],
) -> contextlib.AbstractContextManager[None]:
    """A vLLM forward context holding ``attn_metadata``, entered on use."""
    from vllm.forward_context import ForwardContext, override_forward_context

    context = ForwardContext(
        no_compile_layers={},
        attn_metadata=attn_metadata,
        slot_mapping={},
        virtual_engine=0,
    )
    return override_forward_context(context)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
@unittest.skipUnless(_HAS_VLLM, "needs vLLM")
class DumpAttentionGpuTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_size = 64
        self.block_size = 16
        self.num_blocks = 32

    def _randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=torch.bfloat16, device=self.device)

    def _metadata(self, q_lens: list[int], q_starts: list[int]) -> _PagedMetadata:
        seq_lens = [q_len + start for q_len, start in zip(q_lens, q_starts)]
        max_blocks = max(-(-length // self.block_size) for length in seq_lens)
        block_table = torch.zeros(
            len(seq_lens), max_blocks, dtype=torch.int32, device=self.device
        )
        next_block = 1
        for request, seq_len in enumerate(seq_lens):
            count = -(-seq_len // self.block_size)
            block_table[request, :count] = torch.arange(
                next_block, next_block + count, dtype=torch.int32
            )
            next_block += count
        starts = [0]
        slots = []
        for request, (q_len, q_start) in enumerate(zip(q_lens, q_starts)):
            starts.append(starts[-1] + q_len)
            for position in range(q_start, q_start + q_len):
                block = int(block_table[request, position // self.block_size])
                slots.append(block * self.block_size + position % self.block_size)
        return _PagedMetadata(
            slot_mapping=torch.tensor(slots, device=self.device),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            query_start_loc=torch.tensor(starts, device=self.device),
            block_table=block_table,
            decode_offset=torch.tensor(q_starts, device=self.device),
            num_reqs=len(q_lens),
            num_actual_tokens=sum(q_lens),
        )

    def _attention(
        self,
        executor: torch.nn.Module,
        q_lens: list[int],
        q_starts: list[int],
        inputs: tuple[torch.Tensor, ...],
        micro_batch_size: int,
    ) -> torch.Tensor:
        from quant_suite_vllm.attention.backend import apply_quantized_attention

        query, key, value, cached_key, cached_value = inputs
        metadata = self._metadata(q_lens, q_starts)
        kv_cache = torch.zeros(
            2,
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        for request, q_start in enumerate(q_starts):
            for position in range(q_start):
                block = int(metadata.block_table[request, position // self.block_size])
                offset = position % self.block_size
                kv_cache[0, block, offset] = cached_key[request, position]
                kv_cache[1, block, offset] = cached_value[request, position]
        one = torch.tensor(1.0, device=self.device)
        return apply_quantized_attention(
            query.clone(),
            key.clone(),
            value.clone(),
            kv_cache,
            "auto",
            one,
            one,
            self.num_kv_heads,
            self.head_size,
            self.num_heads,
            metadata,
            self.head_size**-0.5,
            executor,
            micro_batch_size=micro_batch_size,
        )

    def _inputs(
        self, q_lens: list[int], q_starts: list[int]
    ) -> tuple[torch.Tensor, ...]:
        tokens = sum(q_lens)
        cached = max(q_starts) or 1
        return (
            self._randn(tokens, self.num_heads, self.head_size),
            self._randn(tokens, self.num_kv_heads, self.head_size),
            self._randn(tokens, self.num_kv_heads, self.head_size),
            self._randn(len(q_lens), cached, self.num_kv_heads, self.head_size),
            self._randn(len(q_lens), cached, self.num_kv_heads, self.head_size),
        )

    def _layer(self) -> AttentionLayer:
        recipe = QuantAttentionRecipe(q=PipelineRecipe([DumpRecipe()]))
        return AttentionLayer(
            RUNTIME.prepare_attention(recipe),
            _LAYER,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
        )

    def _check_case(
        self, q_lens: list[int], q_starts: list[int], *, real_vllm: bool = False
    ) -> None:
        inputs = self._inputs(q_lens, q_starts)
        builtin = QuantRecipeRuntime().prepare_attention(QuantAttentionRecipe())
        expected = self._attention(builtin, q_lens, q_starts, inputs, 1)
        directory = Path(tempfile.mkdtemp())
        attn_metadata = {_LAYER: self._metadata(q_lens, q_starts)}
        with dump_session(directory, real_vllm=real_vllm) as forward:
            layer = self._layer()
            if real_vllm:
                context = _vllm_forward_context(attn_metadata)
            else:
                forward.real(attn_metadata)
                context = contextlib.nullcontext()
            with context:
                output = layer(
                    lambda e: self._attention(e, q_lens, q_starts, inputs, 1)
                )
        self.assertTrue(torch.equal(output, expected))
        records = [r for r in read_manifest(directory) if r["record"] == "attention"]
        self.assertEqual([r["request"] for r in records], list(range(len(q_lens))))
        token_start = 0
        for record, q_len in zip(records, q_lens):
            dumped = load(directory, record["tensors"]["output"])
            rows = output[token_start : token_start + q_len].cpu()
            self.assertTrue(torch.equal(dumped, rows))
            torch.testing.assert_close(
                dumped.float(), _replay(directory, record), atol=2e-2, rtol=2e-2
            )
            token_start += q_len

    def test_output_is_unchanged_and_dumps_replay(self) -> None:
        cases = {
            "prefill": ([7], [0]),
            "chunked_prefill": ([5], [7]),
            "decode": ([1], [9]),
            "two_requests": ([3, 1], [0, 4]),
        }
        for name, (q_lens, q_starts) in cases.items():
            with self.subTest(case=name):
                self._check_case(q_lens, q_starts)

    def test_inside_vllm_forward_context(self) -> None:
        self._check_case([3, 1], [0, 4], real_vllm=True)

    def test_two_requests_per_micro_batch_raise(self) -> None:
        q_lens, q_starts = [3, 2], [0, 4]
        inputs = self._inputs(q_lens, q_starts)
        with dump_session(Path(tempfile.mkdtemp())) as forward:
            layer = self._layer()
            forward.real({_LAYER: self._metadata(q_lens, q_starts)})
            with self.assertRaisesRegex(
                RuntimeError, "QUANT_SUITE_MHA_MICRO_BATCH_SIZE"
            ):
                layer(lambda e: self._attention(e, q_lens, q_starts, inputs, 2))


if __name__ == "__main__":
    unittest.main()
