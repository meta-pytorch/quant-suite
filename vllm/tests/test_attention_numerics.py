# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from dataclasses import dataclass

import torch
from quant_suite_core import (
    BlockSize,
    ElemDtype,
    ElemRoundMode,
    GlobalScaleGranularity,
    QuantConfig,
    QuantMode,
    ScaleDtype,
    ScaleNormMode,
    ScaleRoundMode,
)
from quant_suite_core.rht_generator import HadamardMode
from quant_suite_recipe.definition import (
    FakeQuantRecipe,
    PipelineRecipe,
    QuantAttentionRecipe,
    RHTRecipe,
)
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME
from quant_suite_vllm.attention.backend import apply_quantized_attention


@dataclass
class _AttentionMetadata:
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    block_table: torch.Tensor
    decode_offset: torch.Tensor
    num_reqs: int
    num_actual_tokens: int


def _sqnr(actual: torch.Tensor, reference: torch.Tensor) -> float:
    signal = reference.float().square().sum()
    noise = (reference.float() - actual.float()).square().sum()
    return float(10 * torch.log10(signal / noise))


def _nvfp4_recipe() -> FakeQuantRecipe:
    return FakeQuantRecipe(
        quant_config=QuantConfig(
            quant_mode=QuantMode.SYMM,
            scale_dtype=ScaleDtype.FP8,
            scale_norm_mode=ScaleNormMode.AMAX,
            scale_round_mode=ScaleRoundMode.NONE,
            global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
            elem_dtype=ElemDtype.E2M1,
            elem_round_mode=ElemRoundMode.NEAREST,
            block_size=BlockSize(1, 16),
            global_scale_dtype=ScaleDtype.FP32,
        )
    )


def _nvfp4_pipeline() -> PipelineRecipe:
    return PipelineRecipe(
        [
            _nvfp4_recipe(),
        ]
    )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class AttentionNumericsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.num_heads = 8
        self.num_kv_heads = 4
        self.head_size = 64
        self.scale = self.head_size**-0.5
        self.block_size = 16
        self.num_blocks = 32

    def _inputs(
        self,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.randn(
                num_tokens,
                self.num_heads,
                self.head_size,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.randn(
                num_tokens,
                self.num_kv_heads,
                self.head_size,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.randn(
                num_tokens,
                self.num_kv_heads,
                self.head_size,
                dtype=torch.bfloat16,
                device=self.device,
            ),
        )

    def _cached_inputs(
        self,
        decode_offsets: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        cached: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for length in decode_offsets:
            if length == 0:
                cached.append(None)
                continue
            cached.append(
                (
                    torch.randn(
                        length,
                        self.num_kv_heads,
                        self.head_size,
                        dtype=torch.bfloat16,
                        device=self.device,
                    ),
                    torch.randn(
                        length,
                        self.num_kv_heads,
                        self.head_size,
                        dtype=torch.bfloat16,
                        device=self.device,
                    ),
                )
            )
        return cached

    def _metadata(
        self,
        seq_lens: list[int],
        query_lengths: list[int],
        decode_offsets: list[int],
    ) -> _AttentionMetadata:
        max_blocks = max(
            (length + self.block_size - 1) // self.block_size for length in seq_lens
        )
        block_table = torch.zeros(
            len(seq_lens),
            max_blocks,
            dtype=torch.int32,
            device=self.device,
        )
        physical_block = 1
        for request, seq_len in enumerate(seq_lens):
            block_count = (seq_len + self.block_size - 1) // self.block_size
            block_table[request, :block_count] = torch.arange(
                physical_block,
                physical_block + block_count,
                dtype=torch.int32,
                device=self.device,
            )
            physical_block += block_count

        query_start_loc = [0]
        for length in query_lengths:
            query_start_loc.append(query_start_loc[-1] + length)

        slots: list[int] = []
        for request, (query_length, decode_offset) in enumerate(
            zip(query_lengths, decode_offsets)
        ):
            for token in range(query_length):
                position = decode_offset + token
                logical_block = position // self.block_size
                block_offset = position % self.block_size
                slots.append(
                    int(block_table[request, logical_block]) * self.block_size
                    + block_offset
                )

        return _AttentionMetadata(
            slot_mapping=torch.tensor(slots, dtype=torch.int64, device=self.device),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            query_start_loc=torch.tensor(
                query_start_loc,
                dtype=torch.int64,
                device=self.device,
            ),
            block_table=block_table,
            decode_offset=torch.tensor(
                decode_offsets,
                dtype=torch.int64,
                device=self.device,
            ),
            num_reqs=len(seq_lens),
            num_actual_tokens=sum(query_lengths),
        )

    def _run(
        self,
        recipe: QuantAttentionRecipe,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_lens: list[int],
        query_lengths: list[int],
        decode_offsets: list[int],
        cached: list[tuple[torch.Tensor, torch.Tensor] | None],
        micro_batch_size: int,
    ) -> torch.Tensor:
        metadata = self._metadata(seq_lens, query_lengths, decode_offsets)
        kv_cache = torch.zeros(
            2,
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        for request, cached_request in enumerate(cached):
            if cached_request is None:
                continue
            cached_key, cached_value = cached_request
            for position in range(cached_key.shape[0]):
                logical_block = position // self.block_size
                block_offset = position % self.block_size
                physical_block = int(metadata.block_table[request, logical_block])
                kv_cache[0, physical_block, block_offset] = cached_key[position]
                kv_cache[1, physical_block, block_offset] = cached_value[position]

        return apply_quantized_attention(
            query.clone(),
            key.clone(),
            value.clone(),
            kv_cache,
            "auto",
            torch.tensor(1.0, device=self.device),
            torch.tensor(1.0, device=self.device),
            self.num_kv_heads,
            self.head_size,
            self.num_heads,
            metadata,
            self.scale,
            QUANT_RECIPE_RUNTIME.prepare_attention(recipe),
            micro_batch_size=micro_batch_size,
        )

    def _compare_fake_quant(
        self,
        query_lengths: list[int],
        decode_offsets: list[int],
    ) -> None:
        seq_lens = [
            query_length + decode_offset
            for query_length, decode_offset in zip(query_lengths, decode_offsets)
        ]
        query, key, value = self._inputs(sum(query_lengths))
        cached = self._cached_inputs(decode_offsets)
        pipeline = _nvfp4_pipeline()
        quantized_recipe = QuantAttentionRecipe(
            q=pipeline,
            k=pipeline,
            v=pipeline,
            attn_weights=pipeline,
        )
        baseline = self._run(
            QuantAttentionRecipe(),
            query,
            key,
            value,
            seq_lens,
            query_lengths,
            decode_offsets,
            cached,
            len(seq_lens),
        )
        quantized = self._run(
            quantized_recipe,
            query,
            key,
            value,
            seq_lens,
            query_lengths,
            decode_offsets,
            cached,
            len(seq_lens),
        )

        self.assertFalse(torch.equal(quantized, baseline))
        self.assertGreaterEqual(_sqnr(quantized, baseline), 10.0)

    def test_prefill_with_nvfp4_fake_quantization(self) -> None:
        self._compare_fake_quant([16], [0])

    def test_decode_with_nvfp4_fake_quantization(self) -> None:
        self._compare_fake_quant([1], [15])

    def test_variable_length_with_nvfp4_fake_quantization(self) -> None:
        self._compare_fake_quant([16, 1], [0, 15])

    def test_prefill_with_rht_pipeline(self) -> None:
        query_lengths = [16]
        decode_offsets = [0]
        seq_lens = [16]
        query, key, value = self._inputs(16)
        cached = self._cached_inputs(decode_offsets)
        pipeline = PipelineRecipe(
            [
                RHTRecipe(
                    mode=HadamardMode.RIGHT,
                    block_size=self.head_size,
                    seed=0,
                ),
                _nvfp4_recipe(),
                RHTRecipe(
                    mode=HadamardMode.RIGHT_INV,
                    block_size=self.head_size,
                    seed=0,
                ),
            ]
        )
        baseline = self._run(
            QuantAttentionRecipe(),
            query,
            key,
            value,
            seq_lens,
            query_lengths,
            decode_offsets,
            cached,
            1,
        )
        transformed = self._run(
            QuantAttentionRecipe(q=pipeline, k=pipeline, v=pipeline),
            query,
            key,
            value,
            seq_lens,
            query_lengths,
            decode_offsets,
            cached,
            1,
        )

        self.assertTrue(torch.isfinite(transformed).all())
        self.assertFalse(torch.equal(transformed, baseline))
        self.assertGreaterEqual(_sqnr(transformed, baseline), 10.0)

    def _assert_batching_matches(
        self,
        query_lengths: list[int],
        decode_offsets: list[int],
        micro_batch_size: int,
    ) -> None:
        seq_lens = [
            query_length + decode_offset
            for query_length, decode_offset in zip(query_lengths, decode_offsets)
        ]
        query, key, value = self._inputs(sum(query_lengths))
        cached = self._cached_inputs(decode_offsets)
        sequential = self._run(
            QuantAttentionRecipe(),
            query,
            key,
            value,
            seq_lens,
            query_lengths,
            decode_offsets,
            cached,
            1,
        )
        batched = self._run(
            QuantAttentionRecipe(),
            query,
            key,
            value,
            seq_lens,
            query_lengths,
            decode_offsets,
            cached,
            micro_batch_size,
        )
        torch.testing.assert_close(batched, sequential, atol=0, rtol=0)

    def test_batched_matches_sequential_prefill(self) -> None:
        self._assert_batching_matches([16, 16, 16, 16], [0, 0, 0, 0], 4)

    def test_batched_matches_sequential_decode(self) -> None:
        self._assert_batching_matches([1, 1, 1, 1], [15, 15, 15, 15], 4)

    def test_batched_matches_sequential_variable_length(self) -> None:
        self._assert_batching_matches([16, 1, 16, 1], [0, 31, 0, 47], 4)

    def test_batched_partial_chunk(self) -> None:
        self._assert_batching_matches([16, 16, 16, 16, 16], [0, 0, 0, 0, 0], 2)
