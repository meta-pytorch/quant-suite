# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import torch
import torch.nn.functional as F
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
from quant_suite_recipe.api.implementation import QuantizedTensor
from quant_suite_recipe.definition import FakeQuantRecipe, QuantAttentionRecipe
from quant_suite_recipe.implementation.attention import QuantizedAttention
from quant_suite_recipe.implementation.quant_pipeline import Pipeline
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME
from quant_suite_vllm.attention.mla.implementation import (
    apply_quantized_mla_attention,
    apply_quantized_mla_decode,
    apply_quantized_mla_prefill,
    up_project_kv,
)


def _calculate_sqnr(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    signal_power = torch.mean(reference.float().square())
    noise_power = torch.mean((reference.float() - actual.float()).square())
    return 10 * torch.log10(signal_power / noise_power)


def _prepare_attention(
    q_pipeline: Pipeline | None = None,
    k_pipeline: Pipeline | None = None,
    v_pipeline: Pipeline | None = None,
    attn_weights_pipeline: Pipeline | None = None,
) -> QuantizedAttention:
    return QuantizedAttention(
        QuantAttentionRecipe(),
        q_pipeline if q_pipeline is not None else Pipeline(),
        k_pipeline if k_pipeline is not None else Pipeline(),
        v_pipeline if v_pipeline is not None else Pipeline(),
        (attn_weights_pipeline if attn_weights_pipeline is not None else Pipeline()),
        Pipeline(),
    )


class _RecordingAttention(QuantizedAttention):
    """Built-in executor that records each BMM call."""

    def __init__(self, *, supports_multi_request_bmm: bool = True) -> None:
        super().__init__(
            QuantAttentionRecipe(),
            Pipeline(),
            Pipeline(),
            Pipeline(),
            Pipeline(),
            Pipeline(),
        )
        self.supports_multi_request_bmm = supports_multi_request_bmm
        self.calls: list[tuple[torch.Size, torch.Size, bool]] = []

    def bmm(
        self,
        left: QuantizedTensor,
        right: QuantizedTensor,
        *,
        transpose_right: bool = False,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        assert left.tensor is not None and right.tensor is not None
        self.calls.append((left.tensor.shape, right.tensor.shape, transpose_right))
        return super().bmm(
            left, right, transpose_right=transpose_right, out_dtype=out_dtype
        )


def _create_nvfp4_quant_config() -> QuantConfig:
    """Create an NVFP4 fake quantization config for testing."""
    return QuantConfig(
        quant_mode=QuantMode.SYMM,
        scale_dtype=ScaleDtype.FP8,
        scale_norm_mode=ScaleNormMode.AMAX,
        scale_round_mode=ScaleRoundMode.NONE,
        global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
        global_scale_dtype=ScaleDtype.FP32,
        elem_dtype=ElemDtype.E2M1,
        elem_round_mode=ElemRoundMode.NEAREST,
        block_size=BlockSize(1, 16),
    )


@dataclass
class SimpleChunkedContext:
    """Chunked-context metadata for supported context-length representations."""

    seq_lens: torch.Tensor | None = None  # [num_chunks, num_prefills]
    context_lens: torch.Tensor | None = None


@dataclass
class SimpleMLAPrefillMetadata:
    """Prefill-specific metadata for testing forward_mha."""

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    max_query_len: int
    chunked_context: SimpleChunkedContext | None = None


@dataclass
class SimpleMLADecodeMetadata:
    """Decode-specific metadata for testing forward_mqa."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor


@dataclass
class SimpleMLAMetadata:
    """Minimal MLA attention metadata for testing.

    Mirrors the fields used by the MLA prefill and decode implementations.
    """

    num_reqs: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    decode: SimpleMLADecodeMetadata | None = None
    prefill: SimpleMLAPrefillMetadata | None = None


def _create_mla_kv_cache(
    num_blocks: int,
    block_size: int,
    head_size: int,
    device: str,
) -> torch.Tensor:
    """Create MLA paged KV cache: [num_blocks, block_size, head_size]."""
    return torch.zeros(
        num_blocks, block_size, head_size, dtype=torch.bfloat16, device=device
    )


def _populate_mla_cache(
    kv_cache: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    block_table: torch.Tensor,
    req_idx: int,
    block_size: int,
) -> None:
    """Write compressed [kv_c, k_pe] into paged MLA cache for a request.

    Args:
        kv_cache: [num_blocks, block_size, head_size]
        kv_c: [seq_len, kv_lora_rank]
        k_pe: [seq_len, qk_rope_head_dim]
        block_table: [num_reqs, max_blocks]
        req_idx: Request index.
        block_size: Block size.
    """
    combined = torch.cat([kv_c, k_pe], dim=-1)  # [seq_len, head_size]
    for i in range(combined.shape[0]):
        logical_block = i // block_size
        block_offset = i % block_size
        physical_block: int = int(block_table[req_idx, logical_block].item())
        kv_cache[physical_block, block_offset] = combined[i]


def _create_block_table(
    seq_lens: list[int],
    block_size: int,
    device: str,
) -> torch.Tensor:
    """Create sequential block table for paged attention."""
    num_requests = len(seq_lens)
    max_num_blocks = max((s + block_size - 1) // block_size for s in seq_lens)
    block_table = torch.zeros(
        num_requests, max_num_blocks, dtype=torch.int32, device=device
    )
    physical_block_idx = 0
    for req_idx, seq_len in enumerate(seq_lens):
        num_blocks = (seq_len + block_size - 1) // block_size
        for b in range(num_blocks):
            block_table[req_idx, b] = physical_block_idx
            physical_block_idx += 1
    return block_table


def _reference_mla_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    decode_offset: int,
) -> torch.Tensor:
    """Reference MLA attention using plain PyTorch (no quantization).

    Args:
        q: [num_heads, q_len, qk_head_dim]
        k: [num_heads, kv_len, qk_head_dim]
        v: [num_heads, kv_len, v_head_dim]
        scale: Attention scale factor.
        decode_offset: Causal offset.

    Returns:
        output: [q_len, num_heads, v_head_dim]
    """
    _, q_len, _ = q.shape
    kv_len = k.shape[1]

    attn_scores = torch.bmm(q.float(), k.float().transpose(-1, -2)) * scale

    q_indices = torch.arange(q_len, device=q.device).unsqueeze(1)
    kv_indices = torch.arange(kv_len, device=q.device).unsqueeze(0)
    causal_mask = (q_indices + decode_offset) >= kv_indices
    attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

    attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
    output = torch.bmm(attn_weights, v.float()).to(q.dtype)

    return output.transpose(0, 1).contiguous()


# ============================================================================
# Tests
# ============================================================================


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class ComputeMLAAttentionTest(unittest.TestCase):
    """Tests for apply_quantized_mla_attention.

    Tests cover:
    - No quantization: output matches reference implementation.
    - NVFP4 quantization: output differs but SQNR is reasonable.
    - Asymmetric head dims: qk_head_dim != v_head_dim.
    - Causal masking with decode_offset.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = "cuda"
        self.num_heads = 4
        self.qk_nope_head_dim = 16
        self.qk_rope_head_dim = 16
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = 16
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

    def test_no_quant_matches_reference(self) -> None:
        """No-quant output matches the plain PyTorch reference."""
        # Setup
        num_tokens, cache_len = 8, 8
        q = torch.randn(
            self.num_heads,
            num_tokens,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        k = torch.randn(
            self.num_heads,
            cache_len,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        v = torch.randn(
            self.num_heads,
            cache_len,
            self.v_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        # Execute
        output, _ = apply_quantized_mla_attention(
            query_t=q,
            key_t=k,
            value_t=v,
            num_tokens=num_tokens,
            cache_len=cache_len,
            decode_offset=0,
            scale=self.scale,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            quantized_attention=_prepare_attention(),
        )

        # Reference
        ref = _reference_mla_attention(q, k, v, self.scale, decode_offset=0)

        # Assert
        torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)

    def test_attention_preserves_dtypes(self) -> None:
        num_tokens = 2
        cache_len = 2
        q = torch.randn(
            self.num_heads,
            num_tokens,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        k = torch.randn_like(q)
        v = torch.randn(
            self.num_heads,
            cache_len,
            self.v_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        output, _ = apply_quantized_mla_attention(
            query_t=q,
            key_t=k,
            value_t=v,
            num_tokens=num_tokens,
            cache_len=cache_len,
            decode_offset=0,
            scale=self.scale,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            quantized_attention=_prepare_attention(),
        )

        self.assertEqual(output.dtype, torch.bfloat16)

    def test_nvfp4_quantization_sqnr(self) -> None:
        """NVFP4 quantized output differs but has reasonable SQNR."""
        # Setup
        num_tokens, cache_len = 16, 16
        q = torch.randn(
            self.num_heads,
            num_tokens,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        k = torch.randn(
            self.num_heads,
            cache_len,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        v = torch.randn(
            self.num_heads,
            cache_len,
            self.v_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        no_quant = Pipeline()
        nvfp4 = QUANT_RECIPE_RUNTIME.create_pipeline(
            FakeQuantRecipe(quant_config=_create_nvfp4_quant_config())
        )

        # Execute: no-quant baseline
        output_baseline, _ = apply_quantized_mla_attention(
            query_t=q.clone(),
            key_t=k.clone(),
            value_t=v.clone(),
            num_tokens=num_tokens,
            cache_len=cache_len,
            decode_offset=0,
            scale=self.scale,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            quantized_attention=_prepare_attention(
                no_quant, no_quant, no_quant, no_quant
            ),
        )

        # Execute: NVFP4 quantized
        output_quant, _ = apply_quantized_mla_attention(
            query_t=q.clone(),
            key_t=k.clone(),
            value_t=v.clone(),
            num_tokens=num_tokens,
            cache_len=cache_len,
            decode_offset=0,
            scale=self.scale,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            quantized_attention=_prepare_attention(nvfp4, nvfp4, nvfp4, nvfp4),
        )

        # Assert
        self.assertFalse(
            torch.equal(output_baseline, output_quant),
            "Quantized output should differ from baseline",
        )
        sqnr = _calculate_sqnr(output_quant, output_baseline)
        self.assertGreaterEqual(
            sqnr,
            10.0,
            f"SQNR is {sqnr} dB, expected >= 10 dB",
        )

    def test_decode_offset_causal_masking(self) -> None:
        """Decode token (num_tokens=1, large decode_offset) attends to all KV."""
        # Setup: single query token at the end of a long sequence
        cache_len = 16
        q = torch.randn(
            self.num_heads,
            1,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        k = torch.randn(
            self.num_heads,
            cache_len,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        v = torch.randn(
            self.num_heads,
            cache_len,
            self.v_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        # Execute
        output, debug = apply_quantized_mla_attention(
            query_t=q,
            key_t=k,
            value_t=v,
            num_tokens=1,
            cache_len=cache_len,
            decode_offset=cache_len - 1,
            scale=self.scale,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            quantized_attention=_prepare_attention(),
            return_debug=True,
        )

        # Assert: no -inf in attention scores (decode attends to all positions)
        attn_scores = debug["attn_scores"]  # [num_heads, 1, cache_len]
        self.assertFalse(
            torch.isinf(attn_scores).any(),
            "Decode token with full decode_offset should attend to all KV positions",
        )

        # Assert: output shape is correct
        self.assertEqual(output.shape, (1, self.num_heads, self.v_head_dim))

    def test_output_shape_asymmetric_dims(self) -> None:
        """Output uses v_head_dim even when qk_head_dim differs."""
        # Setup: qk_head_dim = 32, v_head_dim = 64 (asymmetric)
        qk_dim = 32
        v_dim = 64
        num_tokens, cache_len = 4, 8
        q = torch.randn(
            self.num_heads,
            num_tokens,
            qk_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        k = torch.randn(
            self.num_heads,
            cache_len,
            qk_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        v = torch.randn(
            self.num_heads,
            cache_len,
            v_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        # Execute
        output, _ = apply_quantized_mla_attention(
            query_t=q,
            key_t=k,
            value_t=v,
            num_tokens=num_tokens,
            cache_len=cache_len,
            decode_offset=0,
            scale=1.0 / math.sqrt(qk_dim),
            qk_head_dim=qk_dim,
            v_head_dim=v_dim,
            quantized_attention=_prepare_attention(),
        )

        # Assert
        self.assertEqual(output.shape, (num_tokens, self.num_heads, v_dim))


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class MLAAttentionExecutorTest(unittest.TestCase):
    """MLA multiplies through the prepared attention executor."""

    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        num_heads, num_tokens, cache_len = 3, 5, 7
        q = torch.randn(num_heads, num_tokens, 8)
        k = torch.randn(num_heads, cache_len, 8)
        v = torch.randn(num_heads, cache_len, 4)
        return q, k, v

    def _apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        quantized_attention: QuantizedAttention,
    ) -> torch.Tensor:
        output, _ = apply_quantized_mla_attention(
            query_t=q,
            key_t=k,
            value_t=v,
            num_tokens=q.shape[1],
            cache_len=k.shape[1],
            decode_offset=2,
            scale=0.25,
            qk_head_dim=q.shape[2],
            v_head_dim=v.shape[2],
            quantized_attention=quantized_attention,
        )
        return output

    def test_uses_the_prepared_executor(self) -> None:
        q, k, v = self._inputs()
        attention = _RecordingAttention()

        self._apply(q, k, v, attention)

        self.assertEqual(
            attention.calls,
            [
                (q.shape, k.shape, True),
                (torch.Size([3, 5, 7]), v.shape, False),
            ],
        )

    def test_built_in_executor_matches_torch_bmm(self) -> None:
        q, k, v = self._inputs()

        output = self._apply(q, k, v, _prepare_attention())

        scores = torch.bmm(q, k.transpose(-1, -2)) * 0.25
        causal = torch.arange(5).unsqueeze(1) + 2 >= torch.arange(7).unsqueeze(0)
        scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        expected = torch.bmm(weights, v).transpose(0, 1).contiguous()
        self.assertTrue(torch.equal(output, expected))


class UpProjectKVTest(unittest.TestCase):
    """Tests for up_project_kv()."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = "cuda"
        self.num_heads = 4
        self.kv_lora_rank = 32
        self.qk_nope_head_dim = 16
        self.qk_rope_head_dim = 16
        self.v_head_dim = 16

    def test_output_shapes(self) -> None:
        """Verify shapes: k_full and v_full are correctly formed."""
        seq_len = 8
        kv_c = torch.randn(
            seq_len, self.kv_lora_rank, dtype=torch.bfloat16, device=self.device
        )
        k_pe = torch.randn(
            seq_len, self.qk_rope_head_dim, dtype=torch.bfloat16, device=self.device
        )
        W_UK_T = torch.randn(
            self.num_heads,
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            dtype=torch.bfloat16,
            device=self.device,
        )
        W_UV = torch.randn(
            self.num_heads,
            self.kv_lora_rank,
            self.v_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        k_full, v_full = up_project_kv(kv_c, k_pe, W_UK_T, W_UV, self.num_heads)

        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.assertEqual(k_full.shape, (self.num_heads, seq_len, qk_head_dim))
        self.assertEqual(v_full.shape, (self.num_heads, seq_len, self.v_head_dim))

    def test_k_pe_passthrough(self) -> None:
        """The RoPE portion of k_full equals the original k_pe for every head."""
        seq_len = 4
        kv_c = torch.randn(
            seq_len, self.kv_lora_rank, dtype=torch.bfloat16, device=self.device
        )
        k_pe = torch.randn(
            seq_len, self.qk_rope_head_dim, dtype=torch.bfloat16, device=self.device
        )
        W_UK_T = torch.randn(
            self.num_heads,
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            dtype=torch.bfloat16,
            device=self.device,
        )
        W_UV = torch.randn(
            self.num_heads,
            self.kv_lora_rank,
            self.v_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        k_full, _ = up_project_kv(kv_c, k_pe, W_UK_T, W_UV, self.num_heads)

        # The last qk_rope_head_dim columns of k_full should equal k_pe
        for h in range(self.num_heads):
            torch.testing.assert_close(k_full[h, :, self.qk_nope_head_dim :], k_pe)


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class QuantizedMLAPrefillTest(unittest.TestCase):
    """Tests for the quantized MLA prefill path.

    Tests cover:
    - Single-request prefill with no quantization matches reference.
    - Multi-request prefill with variable sequence lengths.
    - NVFP4 quantization SQNR.
    - Micro-batched prefill (micro_batch_size > 1) matches per-request loop.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = "cuda"
        self.num_heads = 4
        self.kv_lora_rank = 32
        self.qk_nope_head_dim = 16
        self.qk_rope_head_dim = 16
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = 16
        self.head_size = self.kv_lora_rank + self.qk_rope_head_dim  # 48
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)
        self.block_size = 16
        self.num_blocks = 32

        # Random up-projection weights
        self.W_UK_T = torch.randn(
            self.num_heads,
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.W_UV = torch.randn(
            self.num_heads,
            self.kv_lora_rank,
            self.v_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

    def _run_prefill(
        self,
        seq_lens: list[int],
        query_lens: list[int],
        q_pipeline: Pipeline | None = None,
        k_pipeline: Pipeline | None = None,
        v_pipeline: Pipeline | None = None,
        attn_weights_pipeline: Pipeline | None = None,
        micro_batch_size: int | None = None,
        use_logical_kv_cache: bool = False,
        use_v030_context_metadata: bool = False,
        quantized_attention: QuantizedAttention | None = None,
    ) -> torch.Tensor:
        """Populate the cache, build metadata, and run MLA prefill."""
        num_prefills = len(seq_lens)
        total_q_tokens = sum(query_lens)

        block_table = _create_block_table(seq_lens, self.block_size, self.device)
        kv_cache = _create_mla_kv_cache(
            self.num_blocks, self.block_size, self.head_size, self.device
        )

        # Populate cache with random data for each request
        for req_idx, seq_len in enumerate(seq_lens):
            kv_c = torch.randn(
                seq_len, self.kv_lora_rank, dtype=torch.bfloat16, device=self.device
            )
            k_pe = torch.randn(
                seq_len, self.qk_rope_head_dim, dtype=torch.bfloat16, device=self.device
            )
            _populate_mla_cache(
                kv_cache, kv_c, k_pe, block_table, req_idx, self.block_size
            )

        # Build query_start_loc (all prefill, no decode)
        query_start = [0]
        for ql in query_lens:
            query_start.append(query_start[-1] + ql)

        q = torch.randn(
            total_q_tokens,
            self.num_heads,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        output = torch.zeros(
            total_q_tokens,
            self.num_heads * self.v_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        # Determine if chunked prefill (any seq_len > query_len)
        context_lens = [s - ql for s, ql in zip(seq_lens, query_lens)]
        chunked_context: SimpleChunkedContext | None = None
        if any(c > 0 for c in context_lens):
            context_lens_tensor = torch.tensor(
                context_lens, dtype=torch.int32, device=self.device
            )
            chunked_context = (
                SimpleChunkedContext(context_lens=context_lens_tensor)
                if use_v030_context_metadata
                else SimpleChunkedContext(seq_lens=context_lens_tensor.unsqueeze(0))
            )

        metadata = SimpleMLAMetadata(
            num_reqs=num_prefills,
            num_actual_tokens=total_q_tokens,
            query_start_loc=torch.tensor(
                query_start, dtype=torch.int64, device=self.device
            ),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            slot_mapping=torch.zeros(
                total_q_tokens, dtype=torch.int64, device=self.device
            ),
            block_table=block_table,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=num_prefills,
            prefill=SimpleMLAPrefillMetadata(
                block_table=block_table,
                query_start_loc=torch.tensor(
                    query_start, dtype=torch.int64, device=self.device
                ),
                max_query_len=max(query_lens),
                chunked_context=chunked_context,
            ),
        )

        apply_quantized_mla_prefill(
            q=q,
            kv_c_and_k_pe_cache=(
                kv_cache.unsqueeze(1) if use_logical_kv_cache else kv_cache
            ),
            attn_metadata=metadata,
            output=output,
            num_heads=self.num_heads,
            head_size=self.head_size,
            kv_lora_rank=self.kv_lora_rank,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            scale=self.scale,
            W_UK_T=self.W_UK_T,
            W_UV=self.W_UV,
            quantized_attention=(
                quantized_attention
                if quantized_attention is not None
                else _prepare_attention(
                    q_pipeline,
                    k_pipeline,
                    v_pipeline,
                    attn_weights_pipeline,
                )
            ),
            micro_batch_size=micro_batch_size,
        )
        return output

    def test_singleton_bmm_executor_gets_one_request_per_chunk(self) -> None:
        seq_lens = [16, 32, 24]
        attention = _RecordingAttention(supports_multi_request_bmm=False)

        torch.manual_seed(42)
        output = self._run_prefill(
            seq_lens, seq_lens, micro_batch_size=10, quantized_attention=attention
        )
        torch.manual_seed(42)
        output_baseline = self._run_prefill(seq_lens, seq_lens, micro_batch_size=1)

        self.assertEqual(len(attention.calls), 2 * len(seq_lens))
        for left_shape, _, _ in attention.calls:
            self.assertEqual(left_shape[0], self.num_heads)
        torch.testing.assert_close(output, output_baseline)

    def test_multi_request_variable_seq_lens(self) -> None:
        """Multiple requests with different sequence lengths."""
        output = self._run_prefill(seq_lens=[16, 32], query_lens=[16, 32])
        total = 16 + 32
        self.assertEqual(output.shape, (total, self.num_heads * self.v_head_dim))
        self.assertFalse(torch.all(output == 0))

    def test_nvfp4_vs_no_quant_sqnr(self) -> None:
        """NVFP4 quantized prefill has reasonable SQNR vs baseline."""
        seq_lens = [16]
        query_lens = [16]

        # Use the same seed for both runs so they get identical random data
        torch.manual_seed(123)
        output_baseline = self._run_prefill(seq_lens, query_lens)

        torch.manual_seed(123)
        nvfp4 = QUANT_RECIPE_RUNTIME.create_pipeline(
            FakeQuantRecipe(quant_config=_create_nvfp4_quant_config())
        )
        output_quant = self._run_prefill(
            seq_lens,
            query_lens,
            q_pipeline=nvfp4,
            k_pipeline=nvfp4,
            v_pipeline=nvfp4,
            attn_weights_pipeline=nvfp4,
        )

        self.assertFalse(torch.equal(output_baseline, output_quant))
        sqnr = _calculate_sqnr(output_quant, output_baseline)
        self.assertGreaterEqual(sqnr, 10.0, f"SQNR is {sqnr} dB, expected >= 10")

    def test_prefill_matches_reference(self) -> None:
        """No-quant prefill matches the PyTorch reference attention."""
        torch.manual_seed(42)
        # Use seq_len that is NOT a multiple of block_size to catch
        # partial-block zero-padding bugs.
        seq_len = 13
        block_table = _create_block_table([seq_len], self.block_size, self.device)
        kv_cache = _create_mla_kv_cache(
            self.num_blocks, self.block_size, self.head_size, self.device
        )

        # Generate KV data and populate cache
        kv_c = torch.randn(
            seq_len, self.kv_lora_rank, dtype=torch.bfloat16, device=self.device
        )
        k_pe = torch.randn(
            seq_len, self.qk_rope_head_dim, dtype=torch.bfloat16, device=self.device
        )
        _populate_mla_cache(kv_cache, kv_c, k_pe, block_table, 0, self.block_size)

        q = torch.randn(
            seq_len,
            self.num_heads,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        output = torch.zeros(
            seq_len,
            self.num_heads * self.v_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        metadata = SimpleMLAMetadata(
            num_reqs=1,
            num_actual_tokens=seq_len,
            query_start_loc=torch.tensor(
                [0, seq_len], dtype=torch.int64, device=self.device
            ),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=self.device),
            slot_mapping=torch.zeros(seq_len, dtype=torch.int64, device=self.device),
            block_table=block_table,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=1,
            prefill=SimpleMLAPrefillMetadata(
                block_table=block_table,
                query_start_loc=torch.tensor(
                    [0, seq_len], dtype=torch.int64, device=self.device
                ),
                max_query_len=seq_len,
            ),
        )

        apply_quantized_mla_prefill(
            q=q,
            kv_c_and_k_pe_cache=kv_cache,
            attn_metadata=metadata,
            output=output,
            num_heads=self.num_heads,
            head_size=self.head_size,
            kv_lora_rank=self.kv_lora_rank,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            scale=self.scale,
            W_UK_T=self.W_UK_T,
            W_UV=self.W_UV,
            quantized_attention=_prepare_attention(),
        )

        # Compute reference: up-project KV, then do standard attention
        k_full, v_full = up_project_kv(
            kv_c, k_pe, self.W_UK_T, self.W_UV, self.num_heads
        )
        q_t = q.transpose(0, 1)  # [num_heads, seq_len, qk_head_dim]
        ref = _reference_mla_attention(q_t, k_full, v_full, self.scale, decode_offset=0)
        ref_flat = ref.view(seq_len, self.num_heads * self.v_head_dim)

        # Tolerance accounts for bf16 precision in quant_bmm vs float32 reference.
        # The quantized MLA implementation operates in bf16 while
        # _reference_mla_attention uses float32, causing ~0.2 max abs diff.
        torch.testing.assert_close(output, ref_flat, atol=0.2, rtol=0.2)

    def test_prefill_micro_batch_size_one_matches_reference(self) -> None:
        """micro_batch_size=1 matches the PyTorch reference attention.

        Validates that the unified batched code path with mb=1 produces
        the same result as the reference implementation.
        """
        torch.manual_seed(42)
        seq_len = 13  # Not a multiple of block_size to catch partial-block bugs
        block_table = _create_block_table([seq_len], self.block_size, self.device)
        kv_cache = _create_mla_kv_cache(
            self.num_blocks, self.block_size, self.head_size, self.device
        )

        kv_c = torch.randn(
            seq_len, self.kv_lora_rank, dtype=torch.bfloat16, device=self.device
        )
        k_pe = torch.randn(
            seq_len, self.qk_rope_head_dim, dtype=torch.bfloat16, device=self.device
        )
        _populate_mla_cache(kv_cache, kv_c, k_pe, block_table, 0, self.block_size)

        q = torch.randn(
            seq_len,
            self.num_heads,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        output = torch.zeros(
            seq_len,
            self.num_heads * self.v_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        metadata = SimpleMLAMetadata(
            num_reqs=1,
            num_actual_tokens=seq_len,
            query_start_loc=torch.tensor(
                [0, seq_len], dtype=torch.int64, device=self.device
            ),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=self.device),
            slot_mapping=torch.zeros(seq_len, dtype=torch.int64, device=self.device),
            block_table=block_table,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=1,
            prefill=SimpleMLAPrefillMetadata(
                block_table=block_table,
                query_start_loc=torch.tensor(
                    [0, seq_len], dtype=torch.int64, device=self.device
                ),
                max_query_len=seq_len,
            ),
        )

        apply_quantized_mla_prefill(
            q=q,
            kv_c_and_k_pe_cache=kv_cache,
            attn_metadata=metadata,
            output=output,
            num_heads=self.num_heads,
            head_size=self.head_size,
            kv_lora_rank=self.kv_lora_rank,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            scale=self.scale,
            W_UK_T=self.W_UK_T,
            W_UV=self.W_UV,
            quantized_attention=_prepare_attention(),
            micro_batch_size=1,
        )

        k_full, v_full = up_project_kv(
            kv_c, k_pe, self.W_UK_T, self.W_UV, self.num_heads
        )
        q_t = q.transpose(0, 1)
        ref = _reference_mla_attention(q_t, k_full, v_full, self.scale, decode_offset=0)
        ref_flat = ref.view(seq_len, self.num_heads * self.v_head_dim)

        torch.testing.assert_close(output, ref_flat, atol=0.2, rtol=0.2)

    def test_prefill_micro_batch_size(self) -> None:
        """micro_batch_size=2 matches micro_batch_size=1 for prefill.

        Uses 3 requests with variable seq_lens, which also tests:
        - Non-divisible batch (3 / 2 → chunks of 2 and 1).
        - Variable sequence lengths within a micro-batch.
        """
        seq_lens = [16, 32, 24]
        query_lens = [16, 32, 24]

        torch.manual_seed(42)
        output_baseline = self._run_prefill(seq_lens, query_lens, micro_batch_size=1)

        torch.manual_seed(42)
        output_batched = self._run_prefill(seq_lens, query_lens, micro_batch_size=2)

        torch.testing.assert_close(output_batched, output_baseline)

    def test_prefill_micro_batch_size_larger_than_batch(self) -> None:
        """micro_batch_size larger than batch_size works (single chunk)."""
        seq_lens = [16, 32, 24]
        query_lens = [16, 32, 24]

        torch.manual_seed(42)
        output_baseline = self._run_prefill(seq_lens, query_lens, micro_batch_size=1)

        torch.manual_seed(42)
        output_batched = self._run_prefill(seq_lens, query_lens, micro_batch_size=10)

        torch.testing.assert_close(output_batched, output_baseline)

    def test_prefill_micro_batch_size_chunked_prefill(self) -> None:
        """micro_batch_size=2 with non-zero decode_offset (chunked prefill).

        Simulates chunked prefill where seq_lens > query_lens, meaning
        some tokens are already in the cache (context).
        """
        # seq_lens[i] = context_len + query_len
        # context_lens = [8, 16, 4], query_lens = [8, 16, 20]
        seq_lens = [16, 32, 24]
        query_lens = [8, 16, 20]

        torch.manual_seed(42)
        output_baseline = self._run_prefill(seq_lens, query_lens, micro_batch_size=1)

        torch.manual_seed(42)
        output_batched = self._run_prefill(seq_lens, query_lens, micro_batch_size=2)

        torch.testing.assert_close(output_batched, output_baseline)

    def test_v030_logical_cache_and_chunked_context(self) -> None:
        seq_lens = [16, 32]
        query_lens = [8, 16]

        torch.manual_seed(42)
        legacy_output = self._run_prefill(seq_lens, query_lens)
        torch.manual_seed(42)
        logical_output = self._run_prefill(
            seq_lens,
            query_lens,
            use_logical_kv_cache=True,
            use_v030_context_metadata=True,
        )

        torch.testing.assert_close(logical_output, legacy_output)


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class QuantizedMLADecodeTest(unittest.TestCase):
    """Tests for the quantized MLA decode path.

    Tests cover:
    - Single decode token with no quantization.
    - Tuple vs concatenated q input.
    - Batched decode (multiple requests).
    - NVFP4 quantization SQNR.
    - Micro-batched decode (micro_batch_size > 1) matches per-request loop.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = "cuda"
        self.num_heads = 4
        self.kv_lora_rank = 32
        self.qk_rope_head_dim = 16
        self.head_size = self.kv_lora_rank + self.qk_rope_head_dim
        self.scale = 1.0 / math.sqrt(self.head_size)
        self.block_size = 16
        self.num_blocks = 32

    def _build_decode_scenario(
        self,
        seq_lens: list[int],
        *,
        use_logical_kv_cache: bool = False,
    ) -> tuple[torch.Tensor, SimpleMLAMetadata]:
        """Build KV cache and metadata for a decode scenario.

        Returns:
            Tuple of (kv_cache, metadata).
        """
        batch_size = len(seq_lens)
        block_table = _create_block_table(seq_lens, self.block_size, self.device)
        kv_cache = _create_mla_kv_cache(
            self.num_blocks, self.block_size, self.head_size, self.device
        )

        for req_idx, seq_len in enumerate(seq_lens):
            kv_c = torch.randn(
                seq_len, self.kv_lora_rank, dtype=torch.bfloat16, device=self.device
            )
            k_pe = torch.randn(
                seq_len, self.qk_rope_head_dim, dtype=torch.bfloat16, device=self.device
            )
            _populate_mla_cache(
                kv_cache, kv_c, k_pe, block_table, req_idx, self.block_size
            )

        # Decode: each request produces 1 token
        query_start = list(range(batch_size + 1))
        metadata = SimpleMLAMetadata(
            num_reqs=batch_size,
            num_actual_tokens=batch_size,
            query_start_loc=torch.tensor(
                query_start, dtype=torch.int64, device=self.device
            ),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            slot_mapping=torch.zeros(batch_size, dtype=torch.int64, device=self.device),
            block_table=block_table,
            num_decodes=batch_size,
            num_decode_tokens=batch_size,
            num_prefills=0,
            decode=SimpleMLADecodeMetadata(
                block_table=block_table,
                seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            ),
        )
        return (kv_cache.unsqueeze(1) if use_logical_kv_cache else kv_cache), metadata

    def test_single_decode_no_quant(self) -> None:
        """Single decode request produces output in compressed space."""
        kv_cache, metadata = self._build_decode_scenario(seq_lens=[16])

        ql_nope = torch.randn(
            1,
            self.num_heads,
            self.kv_lora_rank,
            dtype=torch.bfloat16,
            device=self.device,
        )
        q_pe = torch.randn(
            1,
            self.num_heads,
            self.qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        attn_out, lse = apply_quantized_mla_decode(
            q=(ql_nope, q_pe),
            kv_c_and_k_pe_cache=kv_cache,
            attn_metadata=metadata,
            num_heads=self.num_heads,
            head_size=self.head_size,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            scale=self.scale,
            quantized_attention=_prepare_attention(),
        )

        self.assertEqual(attn_out.shape, (1, self.num_heads, self.kv_lora_rank))
        self.assertIsNone(lse)
        self.assertFalse(torch.all(attn_out == 0))

    def test_v030_logical_cache_matches_legacy_decode(self) -> None:
        torch.manual_seed(42)
        legacy_cache, legacy_metadata = self._build_decode_scenario(seq_lens=[16])
        torch.manual_seed(42)
        logical_cache, logical_metadata = self._build_decode_scenario(
            seq_lens=[16],
            use_logical_kv_cache=True,
        )
        ql_nope = torch.randn(
            1,
            self.num_heads,
            self.kv_lora_rank,
            dtype=torch.bfloat16,
            device=self.device,
        )
        q_pe = torch.randn(
            1,
            self.num_heads,
            self.qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        quantized_attention = _prepare_attention()

        legacy_output, _ = apply_quantized_mla_decode(
            q=(ql_nope, q_pe),
            kv_c_and_k_pe_cache=legacy_cache,
            attn_metadata=legacy_metadata,
            num_heads=self.num_heads,
            head_size=self.head_size,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            scale=self.scale,
            quantized_attention=quantized_attention,
        )
        logical_output, _ = apply_quantized_mla_decode(
            q=(ql_nope, q_pe),
            kv_c_and_k_pe_cache=logical_cache,
            attn_metadata=logical_metadata,
            num_heads=self.num_heads,
            head_size=self.head_size,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            scale=self.scale,
            quantized_attention=quantized_attention,
        )

        torch.testing.assert_close(logical_output, legacy_output)

    def test_tuple_vs_concatenated_q(self) -> None:
        """Tuple and concatenated inputs preserve ordering across chunks."""
        seq_lens = [16, 32, 24]
        kv_cache, metadata = self._build_decode_scenario(seq_lens=seq_lens)
        batch_size = len(seq_lens)

        ql_nope = torch.randn(
            batch_size,
            self.num_heads,
            self.kv_lora_rank,
            dtype=torch.bfloat16,
            device=self.device,
        )
        q_pe = torch.randn(
            batch_size,
            self.num_heads,
            self.qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        q_concat = torch.cat([ql_nope, q_pe], dim=-1)

        no_quant = Pipeline()
        common_kwargs = {
            "kv_c_and_k_pe_cache": kv_cache,
            "attn_metadata": metadata,
            "num_heads": self.num_heads,
            "head_size": self.head_size,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "scale": self.scale,
            "quantized_attention": _prepare_attention(
                no_quant, no_quant, no_quant, no_quant
            ),
            "micro_batch_size": 2,
        }

        out_tuple, _ = apply_quantized_mla_decode(q=(ql_nope, q_pe), **common_kwargs)
        out_concat, _ = apply_quantized_mla_decode(q=q_concat, **common_kwargs)

        torch.testing.assert_close(out_tuple, out_concat)

    def test_batched_decode(self) -> None:
        """Batched decode with two requests."""
        kv_cache, metadata = self._build_decode_scenario(seq_lens=[16, 32])

        ql_nope = torch.randn(
            2,
            self.num_heads,
            self.kv_lora_rank,
            dtype=torch.bfloat16,
            device=self.device,
        )
        q_pe = torch.randn(
            2,
            self.num_heads,
            self.qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        attn_out, _ = apply_quantized_mla_decode(
            q=(ql_nope, q_pe),
            kv_c_and_k_pe_cache=kv_cache,
            attn_metadata=metadata,
            num_heads=self.num_heads,
            head_size=self.head_size,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            scale=self.scale,
            quantized_attention=_prepare_attention(),
        )

        self.assertEqual(attn_out.shape, (2, self.num_heads, self.kv_lora_rank))
        self.assertFalse(torch.all(attn_out == 0))

    def test_nvfp4_vs_no_quant_sqnr(self) -> None:
        """NVFP4 quantized decode has reasonable SQNR vs baseline."""
        kv_cache, metadata = self._build_decode_scenario(seq_lens=[16])

        ql_nope = torch.randn(
            1,
            self.num_heads,
            self.kv_lora_rank,
            dtype=torch.bfloat16,
            device=self.device,
        )
        q_pe = torch.randn(
            1,
            self.num_heads,
            self.qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        no_quant = Pipeline()
        nvfp4 = QUANT_RECIPE_RUNTIME.create_pipeline(
            FakeQuantRecipe(quant_config=_create_nvfp4_quant_config())
        )

        common_kwargs = {
            "q": (ql_nope.clone(), q_pe.clone()),
            "kv_c_and_k_pe_cache": kv_cache,
            "attn_metadata": metadata,
            "num_heads": self.num_heads,
            "head_size": self.head_size,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "scale": self.scale,
        }

        out_baseline, _ = apply_quantized_mla_decode(
            **common_kwargs,
            quantized_attention=_prepare_attention(
                no_quant, no_quant, no_quant, no_quant
            ),
        )
        out_quant, _ = apply_quantized_mla_decode(
            **common_kwargs,
            quantized_attention=_prepare_attention(nvfp4, nvfp4, nvfp4, nvfp4),
        )

        self.assertFalse(torch.equal(out_baseline, out_quant))
        sqnr = _calculate_sqnr(out_quant, out_baseline)
        self.assertGreaterEqual(sqnr, 10.0, f"SQNR is {sqnr} dB, expected >= 10")

    def test_decode_matches_reference(self) -> None:
        """No-quant decode matches PyTorch attention in compressed space."""
        torch.manual_seed(42)
        seq_len = 16
        kv_cache, metadata = self._build_decode_scenario(seq_lens=[seq_len])

        ql_nope = torch.randn(
            1,
            self.num_heads,
            self.kv_lora_rank,
            dtype=torch.bfloat16,
            device=self.device,
        )
        q_pe = torch.randn(
            1,
            self.num_heads,
            self.qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        attn_out, _ = apply_quantized_mla_decode(
            q=(ql_nope, q_pe),
            kv_c_and_k_pe_cache=kv_cache,
            attn_metadata=metadata,
            num_heads=self.num_heads,
            head_size=self.head_size,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            scale=self.scale,
            quantized_attention=_prepare_attention(),
        )

        # Compute reference in compressed space
        # Gather KV from cache
        block_table = metadata.decode.block_table
        block_size = kv_cache.shape[1]
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        block_indices = block_table[0, :num_blocks_needed]
        kv_blocks = kv_cache[block_indices]
        kv_cached = kv_blocks.reshape(-1, self.head_size)[:seq_len]

        kv_c_cached = kv_cached[:, : self.kv_lora_rank]
        k_pe_cached = kv_cached[:, self.kv_lora_rank :]

        # Q in compressed space: [num_heads, 1, compressed_dim]
        q_compressed = torch.cat([ql_nope[0], q_pe[0]], dim=-1).unsqueeze(1)
        # K in compressed space: [num_heads, seq_len, compressed_dim]
        k_compressed = torch.cat([kv_c_cached, k_pe_cached], dim=-1)
        k_compressed = k_compressed.unsqueeze(0).expand(self.num_heads, -1, -1)
        # V in compressed space: [num_heads, seq_len, kv_lora_rank]
        v_compressed = kv_c_cached.unsqueeze(0).expand(self.num_heads, -1, -1)

        ref = _reference_mla_attention(
            q_compressed,
            k_compressed,
            v_compressed,
            self.scale,
            decode_offset=seq_len - 1,
        )
        # ref: [1, num_heads, kv_lora_rank]

        torch.testing.assert_close(attn_out, ref, atol=1e-2, rtol=1e-2)

    def test_decode_micro_batch_size_one_matches_reference(self) -> None:
        """micro_batch_size=1 matches PyTorch reference attention in compressed space.

        Validates that the unified batched code path with mb=1 produces
        the same result as the reference implementation.
        """
        torch.manual_seed(42)
        seq_len = 16
        kv_cache, metadata = self._build_decode_scenario(seq_lens=[seq_len])

        ql_nope = torch.randn(
            1,
            self.num_heads,
            self.kv_lora_rank,
            dtype=torch.bfloat16,
            device=self.device,
        )
        q_pe = torch.randn(
            1,
            self.num_heads,
            self.qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        attn_out, _ = apply_quantized_mla_decode(
            q=(ql_nope, q_pe),
            kv_c_and_k_pe_cache=kv_cache,
            attn_metadata=metadata,
            num_heads=self.num_heads,
            head_size=self.head_size,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            scale=self.scale,
            quantized_attention=_prepare_attention(),
            micro_batch_size=1,
        )

        # Compute reference in compressed space
        block_table = metadata.decode.block_table
        block_size = kv_cache.shape[1]
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        block_indices = block_table[0, :num_blocks_needed]
        kv_blocks = kv_cache[block_indices]
        kv_cached = kv_blocks.reshape(-1, self.head_size)[:seq_len]

        kv_c_cached = kv_cached[:, : self.kv_lora_rank]
        k_pe_cached = kv_cached[:, self.kv_lora_rank :]

        q_compressed = torch.cat([ql_nope[0], q_pe[0]], dim=-1).unsqueeze(1)
        k_compressed = torch.cat([kv_c_cached, k_pe_cached], dim=-1)
        k_compressed = k_compressed.unsqueeze(0).expand(self.num_heads, -1, -1)
        v_compressed = kv_c_cached.unsqueeze(0).expand(self.num_heads, -1, -1)

        ref = _reference_mla_attention(
            q_compressed,
            k_compressed,
            v_compressed,
            self.scale,
            decode_offset=seq_len - 1,
        )

        torch.testing.assert_close(attn_out, ref, atol=1e-2, rtol=1e-2)

    def test_batched_decode_micro_batch_size(self) -> None:
        """micro_batch_size=2 matches micro_batch_size=1 for decode.

        Uses 3 requests with variable seq_lens, which also tests:
        - Non-divisible batch (3 / 2 → chunks of 2 and 1).
        - Variable cache lengths within a micro-batch.
        """
        torch.manual_seed(42)
        seq_lens = [16, 32, 24]
        kv_cache, metadata = self._build_decode_scenario(seq_lens=seq_lens)
        batch_size = len(seq_lens)

        ql_nope = torch.randn(
            batch_size,
            self.num_heads,
            self.kv_lora_rank,
            dtype=torch.bfloat16,
            device=self.device,
        )
        q_pe = torch.randn(
            batch_size,
            self.num_heads,
            self.qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        no_quant = Pipeline()
        common_kwargs = {
            "kv_c_and_k_pe_cache": kv_cache,
            "attn_metadata": metadata,
            "num_heads": self.num_heads,
            "head_size": self.head_size,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "scale": self.scale,
            "quantized_attention": _prepare_attention(
                no_quant, no_quant, no_quant, no_quant
            ),
        }

        out_baseline, _ = apply_quantized_mla_decode(
            q=(ql_nope.clone(), q_pe.clone()),
            micro_batch_size=1,
            **common_kwargs,
        )
        out_batched, _ = apply_quantized_mla_decode(
            q=(ql_nope.clone(), q_pe.clone()),
            micro_batch_size=2,
            **common_kwargs,
        )

        torch.testing.assert_close(out_batched, out_baseline)

    def test_decode_micro_batch_size_larger_than_batch(self) -> None:
        """micro_batch_size larger than batch_size works (single chunk)."""
        torch.manual_seed(42)
        seq_lens = [16, 32, 24]
        kv_cache, metadata = self._build_decode_scenario(seq_lens=seq_lens)
        batch_size = len(seq_lens)

        ql_nope = torch.randn(
            batch_size,
            self.num_heads,
            self.kv_lora_rank,
            dtype=torch.bfloat16,
            device=self.device,
        )
        q_pe = torch.randn(
            batch_size,
            self.num_heads,
            self.qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        no_quant = Pipeline()
        common_kwargs = {
            "kv_c_and_k_pe_cache": kv_cache,
            "attn_metadata": metadata,
            "num_heads": self.num_heads,
            "head_size": self.head_size,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "scale": self.scale,
            "quantized_attention": _prepare_attention(
                no_quant, no_quant, no_quant, no_quant
            ),
        }

        out_baseline, _ = apply_quantized_mla_decode(
            q=(ql_nope.clone(), q_pe.clone()),
            micro_batch_size=1,
            **common_kwargs,
        )
        out_batched, _ = apply_quantized_mla_decode(
            q=(ql_nope.clone(), q_pe.clone()),
            micro_batch_size=10,
            **common_kwargs,
        )

        torch.testing.assert_close(out_batched, out_baseline)
