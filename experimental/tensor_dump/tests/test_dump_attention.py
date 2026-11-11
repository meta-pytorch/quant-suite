# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``DumpAttention`` at the ``bmm`` level, on operands laid out as the
QuantSuite attention backend lays out one GQA request. CPU only."""

import tempfile
import unittest
from pathlib import Path

import torch
from dump_test_utils import (
    AttentionLayer,
    dump_session,
    fake_metadata,
    load,
    read_manifest,
)
from quant_suite_recipe.api import QuantizedTensor
from quant_suite_recipe.definition import PipelineRecipe, QuantAttentionRecipe
from quant_suite_recipe.runtime import QuantRecipeRuntime
from tensor_dump._step import _VllmSettings
from tensor_dump.recipe import DumpRecipe
from tensor_dump.runtime import RUNTIME

_LAYER = "model.layers.0.self_attn.attn"
_HEADS = 8
_KV_HEADS = 2
_GROUP = _HEADS // _KV_HEADS
_HEAD_SIZE = 16


def _expand(kv: torch.Tensor) -> torch.Tensor:
    """``[L, Hkv, D]`` -> head-major ``[Hq, L, D]``, as the backend expands GQA."""
    return kv.repeat_interleave(_GROUP, dim=1).transpose(0, 1).contiguous()


def _attend(
    executor: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_start_pos: int,
) -> torch.Tensor:
    """One request's causal attention through ``executor.bmm``; ``[Hq, Lq, D]``."""
    scores = executor.bmm(
        QuantizedTensor(tensor=q.transpose(0, 1).contiguous()),
        QuantizedTensor(tensor=_expand(k)),
        transpose_right=True,
        out_dtype=torch.float32,
    )
    scores *= _HEAD_SIZE**-0.5
    causal = torch.arange(q.shape[0])[:, None] + q_start_pos >= torch.arange(k.shape[0])
    scores.masked_fill_(~causal, float("-inf"))
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return executor.bmm(
        QuantizedTensor(tensor=weights), QuantizedTensor(tensor=_expand(v))
    )


def _inputs(q_len: int, kv_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(q_len, _HEADS, _HEAD_SIZE, dtype=torch.bfloat16),
        torch.randn(kv_len, _KV_HEADS, _HEAD_SIZE, dtype=torch.bfloat16),
        torch.randn(kv_len, _KV_HEADS, _HEAD_SIZE, dtype=torch.bfloat16),
    )


def _layer(micro_batch_size: int | None = None) -> AttentionLayer:
    recipe = QuantAttentionRecipe(q=PipelineRecipe([DumpRecipe()]))
    return AttentionLayer(
        RUNTIME.prepare_attention(recipe),
        _LAYER,
        _HEADS,
        _KV_HEADS,
        _HEAD_SIZE,
        micro_batch_size,
    )


def _attention_records(directory: Path) -> list[dict[str, object]]:
    return [r for r in read_manifest(directory) if r["record"] == "attention"]


class DumpAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.directory = Path(tempfile.mkdtemp())

    def _check_files(self, phase: str, q_len: int, q_start_pos: int) -> None:
        kv_len = q_len + q_start_pos
        q, k, v = _inputs(q_len, kv_len)
        with dump_session(self.directory) as forward:
            layer = _layer()
            forward.real({_LAYER: fake_metadata([q_len], [q_start_pos])})
            output = layer(lambda e: _attend(e, q, k, v, q_start_pos))
        (record,) = [
            r for r in read_manifest(self.directory) if r["record"] == "attention"
        ]
        self.assertEqual(record["phase"], phase)
        expected = {"q": q, "k": k, "v": v, "output": output.transpose(0, 1)}
        self.assertEqual(sorted(record["tensors"]), sorted(expected))
        for tensor, entry in record["tensors"].items():
            saved = load(self.directory, entry)
            self.assertTrue(torch.equal(saved, expected[tensor]), tensor)
            self.assertEqual(entry["shape"], list(expected[tensor].shape))
        replay = record["replay"]
        self.assertEqual(
            (replay["q_len"], replay["kv_len"], replay["q_start_pos"]),
            (q_len, kv_len, q_start_pos),
        )
        self.assertEqual(replay["gqa_group"], _GROUP)
        self.assertEqual(replay["softmax_scale"], _HEAD_SIZE**-0.5)
        self.assertEqual(replay["dtype"], "bfloat16")

    def test_prefill_files_are_unpadded_unexpanded_nhd(self) -> None:
        self._check_files("prefill", q_len=5, q_start_pos=0)

    def test_decode_files_are_unpadded_unexpanded_nhd(self) -> None:
        self._check_files("decode", q_len=1, q_start_pos=5)

    def test_bmm_results_equal_base_executor(self) -> None:
        q, k, v = _inputs(4, 4)
        base = QuantRecipeRuntime().prepare_attention(QuantAttentionRecipe())
        expected = _attend(base, q, k, v, 0)
        with dump_session(self.directory) as forward:
            layer = _layer()
            forward.real({_LAYER: fake_metadata([4], [0])})
            output = layer(lambda e: _attend(e, q, k, v, 0))
        self.assertTrue(torch.equal(output, expected))

    def test_requests_of_one_step_map_to_their_files(self) -> None:
        q0, k0, v0 = _inputs(3, 3)
        q1, k1, v1 = _inputs(1, 5)
        with dump_session(self.directory) as forward:
            layer = _layer()
            forward.real({_LAYER: fake_metadata([3, 1], [0, 4])})
            output = layer(
                lambda e: torch.cat(
                    [_attend(e, q0, k0, v0, 0), _attend(e, q1, k1, v1, 4)], dim=1
                )
            )
        records = _attention_records(self.directory)
        self.assertEqual([r["request"] for r in records], [0, 1])
        outputs = output.transpose(0, 1).split([3, 1])
        sources = [(q0, k0, v0, outputs[0]), (q1, k1, v1, outputs[1])]
        for record, (q, k, v, out) in zip(records, sources):
            index = record["request"]
            with self.subTest(request=index):
                replay = record["replay"]
                self.assertEqual(
                    (replay["q_len"], replay["kv_len"], replay["q_start_pos"]),
                    ((3, 3, 0), (1, 5, 4))[index],
                )
                expected = {"q": q, "k": k, "v": v, "output": out}
                for name, entry in record["tensors"].items():
                    self.assertTrue(
                        entry["file"].endswith(f"/req{index:03d}.{name}.pt")
                    )
                    self.assertTrue(
                        torch.equal(load(self.directory, entry), expected[name]), name
                    )

    def test_prefill_then_decode_on_one_layer(self) -> None:
        q0, k0, v0 = _inputs(4, 4)
        q1, k1, v1 = _inputs(1, 5)
        with dump_session(self.directory, MAX_STEPS="2") as forward:
            layer = _layer()
            forward.real({_LAYER: fake_metadata([4], [0])})
            layer(lambda e: _attend(e, q0, k0, v0, 0))
            forward.real({_LAYER: fake_metadata([1], [4])})
            layer(lambda e: _attend(e, q1, k1, v1, 4))
        records = _attention_records(self.directory)
        self.assertEqual(
            [(r["step"], r["request"], r["phase"]) for r in records],
            [(0, 0, "prefill"), (1, 0, "decode")],
        )
        decode_q = records[1]["tensors"]["q"]
        self.assertEqual(decode_q["file"], f"step_000001/{_LAYER}/req000.q.pt")
        self.assertTrue(torch.equal(load(self.directory, decode_q), q1))
        self.assertTrue(
            torch.equal(load(self.directory, records[1]["tensors"]["k"]), k1)
        )

    def test_batched_products_raise_at_bind(self) -> None:
        # Binding happens at the first forward, even a dummy one.
        with dump_session(self.directory) as forward:
            layer = _layer(micro_batch_size=8)
            forward.dummy()
            with self.assertRaisesRegex(
                RuntimeError, "QUANT_SUITE_MHA_MICRO_BATCH_SIZE=1"
            ):
                layer(lambda e: torch.zeros(1))

    def test_single_request_products_bind(self) -> None:
        cases = {
            "micro_batch_size=1": (1, _VllmSettings()),
            "max_num_seqs=1": (8, _VllmSettings(max_num_seqs=1)),
        }
        for name, (micro_batch_size, settings) in cases.items():
            with self.subTest(case=name):
                with dump_session(self.directory, settings=settings) as forward:
                    layer = _layer(micro_batch_size=micro_batch_size)
                    forward.dummy()
                    self.assertEqual(layer(lambda e: torch.ones(1)).item(), 1.0)
                    self.assertIsNotNone(layer.quantized_attention._layout)

    def test_two_requests_in_one_product_raise(self) -> None:
        q, k, _ = _inputs(3, 3)
        q_operand = q.transpose(0, 1).repeat(2, 1, 1)
        k_operand = _expand(k).repeat(2, 1, 1)
        with dump_session(self.directory) as forward:
            layer = _layer()
            forward.real({_LAYER: fake_metadata([3, 3], [0, 0])})
            with self.assertRaisesRegex(
                RuntimeError, "QUANT_SUITE_MHA_MICRO_BATCH_SIZE"
            ):
                layer(
                    lambda e: e.bmm(
                        QuantizedTensor(tensor=q_operand),
                        QuantizedTensor(tensor=k_operand),
                        transpose_right=True,
                        out_dtype=torch.float32,
                    )
                )

    def test_pv_without_qk_raises(self) -> None:
        _, _, v = _inputs(3, 3)
        weights = torch.rand(_HEADS, 3, 3, dtype=torch.bfloat16)
        with dump_session(self.directory) as forward:
            layer = _layer()
            forward.real({_LAYER: fake_metadata([3], [0])})
            with self.assertRaisesRegex(RuntimeError, "without Q @ K"):
                layer(
                    lambda e: e.bmm(
                        QuantizedTensor(tensor=weights),
                        QuantizedTensor(tensor=_expand(v)),
                    )
                )

    def test_ungrouped_kv_heads_raise(self) -> None:
        q, _, _ = _inputs(3, 3)
        k_operand = torch.randn(_HEADS, 3, _HEAD_SIZE, dtype=torch.bfloat16)
        with dump_session(self.directory) as forward:
            layer = _layer()
            forward.real({_LAYER: fake_metadata([3], [0])})
            with self.assertRaisesRegex(RuntimeError, "does not repeat each KV head"):
                layer(
                    lambda e: e.bmm(
                        QuantizedTensor(tensor=q.transpose(0, 1).contiguous()),
                        QuantizedTensor(tensor=k_operand),
                        transpose_right=True,
                        out_dtype=torch.float32,
                    )
                )


if __name__ == "__main__":
    unittest.main()
