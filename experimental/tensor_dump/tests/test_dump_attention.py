# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention dumps from forward hooks on a stand-in for vLLM's ``Attention``
layer. CPU only."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from dump_test_utils import (
    dump_session,
    fake_metadata,
    FakeAttention,
    generic_metadata,
    load,
    read_manifest,
)
from quant_suite_recipe.definition import QuantLinearRecipe
from tensor_dump._step import _settings_from, _VllmSettings
from tensor_dump.runtime import RUNTIME

_LAYERS = ("model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn")
_HEADS = 8
_KV_HEADS = 2
_HEAD_SIZE = 16
_WINDOW = 4
_SOFT_CAP = 30.0


class _Model(torch.nn.Module):
    """Two attention layers, the second with a sliding window and a soft cap,
    and a linear executor without a ``dump`` stage, whose construction starts
    the dump session as in vLLM. The second layer is called with keywords."""

    def __init__(self) -> None:
        super().__init__()
        self.quantized_linear = RUNTIME.prepare_linear(QuantLinearRecipe())
        self.layers = torch.nn.ModuleList(
            [
                FakeAttention(_LAYERS[0], _HEADS, _KV_HEADS, _HEAD_SIZE),
                FakeAttention(
                    _LAYERS[1],
                    _HEADS,
                    _KV_HEADS,
                    _HEAD_SIZE,
                    sliding_window=_WINDOW,
                    logits_soft_cap=_SOFT_CAP,
                ),
            ]
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> list[torch.Tensor]:
        first, second = self.layers
        return [first(q, k, v), second(query=q, key=k, value=v)]


def _inputs(tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q, K and V of ``tokens`` new tokens, as vLLM passes them to the layer."""
    return (
        torch.randn(tokens, _HEADS * _HEAD_SIZE, dtype=torch.bfloat16),
        torch.randn(tokens, _KV_HEADS * _HEAD_SIZE, dtype=torch.bfloat16),
        torch.randn(tokens, _KV_HEADS * _HEAD_SIZE, dtype=torch.bfloat16),
    )


def _run(
    forward: Any,
    model: _Model,
    q_lens: list[int],
    q_starts: list[int],
) -> tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]:
    """One forward of ``model``; returns its inputs and outputs."""
    forward.real({name: generic_metadata(q_lens, q_starts) for name in _LAYERS})
    inputs = _inputs(sum(q_lens))
    return inputs, model(*inputs)


def _records(directory: Path, layer: str | None = None) -> list[dict[str, Any]]:
    return [
        r
        for r in read_manifest(directory)
        if r["record"] == "attention" and (layer is None or r["layer"] == layer)
    ]


def _heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], heads, _HEAD_SIZE)


class DumpAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.directory = Path(tempfile.mkdtemp())

    def test_files_and_replay_fields(self) -> None:
        with dump_session(self.directory, ATTENTION=".*") as forward:
            model = _Model()
            (q, k, v), outputs = _run(forward, model, [5], [0])
        manifest = read_manifest(self.directory)
        self.assertEqual(manifest[0]["format_version"], 2)
        self.assertEqual(manifest[0]["options"]["attention"], ".*")
        self.assertEqual([r for r in manifest if r["record"] == "linear"], [])
        expected = {
            "q": _heads(q, _HEADS),
            "k": _heads(k, _KV_HEADS),
            "v": _heads(v, _KV_HEADS),
        }
        for layer, output, window, soft_cap in zip(
            _LAYERS, outputs, (None, _WINDOW), (None, _SOFT_CAP)
        ):
            with self.subTest(layer=layer):
                (record,) = _records(self.directory, layer)
                self.assertEqual(record["phase"], "prefill")
                tensors = {**expected, "output": _heads(output, _HEADS)}
                self.assertEqual(sorted(record["tensors"]), sorted(tensors))
                for name, entry in record["tensors"].items():
                    self.assertEqual(
                        entry["file"], f"step_000000/{layer}/req000.{name}.pt"
                    )
                    self.assertTrue(
                        torch.equal(load(self.directory, entry), tensors[name]), name
                    )
                self.assertEqual(
                    record["replay"],
                    {
                        "q_start_pos": 0,
                        "q_len": 5,
                        "kv_len": 5,
                        "kv_rows": [0, 5],
                        "causal": True,
                        "scale": _HEAD_SIZE**-0.5,
                        "sliding_window": window,
                        "logits_soft_cap": soft_cap,
                        "num_q_heads": _HEADS,
                        "num_kv_heads": _KV_HEADS,
                        "head_size": _HEAD_SIZE,
                        "head_size_v": _HEAD_SIZE,
                        "layout": "NHD",
                        "dtype": "bfloat16",
                        "kv_cache_dtype": "auto",
                        "backend": "FAKE",
                    },
                )

    def test_outputs_are_unchanged(self) -> None:
        q, k, v = _inputs(3)
        with dump_session(None):
            expected = _Model()(q, k, v)
        with dump_session(self.directory, ATTENTION=".*") as forward:
            model = _Model()
            forward.real({name: generic_metadata([3], [0]) for name in _LAYERS})
            outputs = model(q, k, v)
        for output, reference in zip(outputs, expected):
            self.assertTrue(torch.equal(output, reference))
        self.assertEqual(len(_records(self.directory)), 2)

    def test_requests_of_one_step_map_to_their_rows(self) -> None:
        with dump_session(self.directory, ATTENTION=_LAYERS[0]) as forward:
            model = _Model()
            (q, k, _), (output, _) = _run(forward, model, [3, 1], [0, 4])
        records = _records(self.directory)
        self.assertEqual([r["request"] for r in records], [0, 1])
        for record, rows, replay in zip(
            records, (slice(0, 3), slice(3, 4)), ((0, 3, 3), (4, 1, 5))
        ):
            with self.subTest(request=record["request"]):
                r = record["replay"]
                self.assertEqual((r["q_start_pos"], r["q_len"], r["kv_len"]), replay)
                self.assertEqual(r["kv_rows"], [replay[0], replay[2]])
                tensors = record["tensors"]
                self.assertTrue(
                    torch.equal(
                        load(self.directory, tensors["q"]), _heads(q[rows], _HEADS)
                    )
                )
                self.assertTrue(
                    torch.equal(
                        load(self.directory, tensors["k"]),
                        _heads(k[rows], _KV_HEADS),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        load(self.directory, tensors["output"]),
                        _heads(output[rows], _HEADS),
                    )
                )

    def test_chunked_prefill_then_decode_hold_new_rows_only(self) -> None:
        steps = [([4], [0]), ([2], [4]), ([1], [6])]
        with dump_session(self.directory, ATTENTION=_LAYERS[0], MAX_STEPS="0") as f:
            model = _Model()
            keys = [
                _run(f, model, q_lens, q_starts)[0][1] for q_lens, q_starts in steps
            ]
        records = _records(self.directory)
        self.assertEqual(
            [(r["step"], r["phase"], r["replay"]["kv_rows"]) for r in records],
            [(0, "prefill", [0, 4]), (1, "prefill", [4, 6]), (2, "decode", [6, 7])],
        )
        self.assertEqual(
            [(r["replay"]["q_start_pos"], r["replay"]["kv_len"]) for r in records],
            [(0, 4), (4, 6), (6, 7)],
        )
        history = torch.cat(
            [load(self.directory, r["tensors"]["k"]) for r in records], dim=0
        )
        self.assertTrue(torch.equal(history, _heads(torch.cat(keys), _KV_HEADS)))

    def test_backend_metadata_fields_are_used(self) -> None:
        # Metadata with decode offsets and a padded token count, as some
        # backends provide.
        with dump_session(self.directory, ATTENTION=_LAYERS[0]) as forward:
            model = _Model()
            forward.real({name: fake_metadata([2, 1], [0, 3]) for name in _LAYERS})
            model(*_inputs(4))
        self.assertEqual(
            [r["replay"]["kv_rows"] for r in _records(self.directory)],
            [[0, 2], [3, 4]],
        )

    def test_selection_by_pattern_and_tensors(self) -> None:
        with dump_session(
            self.directory,
            ATTENTION=r"model\.layers\.1\.",
            ATTENTION_TENSORS="output, q",
        ) as forward:
            model = _Model()
            _run(forward, model, [2], [0])
        (record,) = _records(self.directory)
        self.assertEqual(record["layer"], _LAYERS[1])
        self.assertEqual(sorted(record["tensors"]), ["output", "q"])

    def test_unset_pattern_hooks_no_layer(self) -> None:
        with dump_session(self.directory) as forward:
            model = _Model()
            _run(forward, model, [2], [0])
            self.assertEqual(
                [len(layer._forward_hooks) for layer in model.layers], [0, 0]
            )
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_dummy_forward_writes_nothing(self) -> None:
        with dump_session(self.directory, ATTENTION=".*") as forward:
            model = _Model()
            forward.dummy()
            model(*_inputs(3))
            self.assertEqual(list(self.directory.iterdir()), [])
            _run(forward, model, [3], [0])
        self.assertEqual(
            sorted({r["step"] for r in _records(self.directory)}),
            [0],
        )

    def test_unsupported_layers_raise_at_first_forward(self) -> None:
        class Subclass(FakeAttention):
            pass

        def alibi(layer: Any) -> None:
            layer.impl.alibi_slopes = [1.0]

        def sinks(layer: Any) -> None:
            layer.has_sink = True

        def shared_kv(layer: Any) -> None:
            layer.kv_sharing_target_layer_name = "model.layers.0.self_attn.attn"

        def encoder(layer: Any) -> None:
            layer.attn_type = "encoder_only"

        cases = {
            "alibi": alibi,
            "sinks": sinks,
            "kv sharing": shared_kv,
            "encoder": encoder,
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                with dump_session(Path(tempfile.mkdtemp()), ATTENTION=".*") as forward:
                    model = _Model()
                    change(model.layers[1])
                    with self.assertRaisesRegex(
                        NotImplementedError, "QUANT_SUITE_TENSOR_DUMP_ATTENTION"
                    ):
                        _run(forward, model, [2], [0])
        with self.subTest(case="subclass"):
            with dump_session(Path(tempfile.mkdtemp()), ATTENTION=".*") as forward:
                model = _Model()
                model.layers[1] = Subclass(_LAYERS[1], _HEADS, _KV_HEADS, _HEAD_SIZE)
                with self.assertRaisesRegex(NotImplementedError, "Subclass"):
                    _run(forward, model, [2], [0])

    def test_prefix_caching_raises_when_attention_is_selected(self) -> None:
        settings = _VllmSettings(prefix_caching=True)
        with dump_session(self.directory, settings=settings, ATTENTION=".*"):
            with self.assertRaisesRegex(RuntimeError, "--no-enable-prefix-caching"):
                _Model()
        with dump_session(self.directory, settings=settings) as forward:
            _run(forward, _Model(), [2], [0])
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_prefix_caching_is_read_from_the_config(self) -> None:
        for enabled in (True, False):
            config = SimpleNamespace(
                use_v2_model_runner=False,
                speculative_config=None,
                cache_config=SimpleNamespace(enable_prefix_caching=enabled),
            )
            self.assertEqual(_settings_from(config, None).prefix_caching, enabled)
        self.assertIsNone(_settings_from(None, None).prefix_caching)


if __name__ == "__main__":
    unittest.main()
