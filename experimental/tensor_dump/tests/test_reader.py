# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``tensor_dump.reader`` on dumps the writer wrote: two runs into one
directory, with a chunked prefill, decodes, requests in sequence and two
requests in one step. Every test reads the dump as written and, in
``MergedReaderTest``, after ``tensor_dump.merge`` merged it. CPU only."""

import dataclasses
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from dump_test_utils import dump_session, FakeAttention, generic_metadata
from quant_suite_recipe.api import QuantizedTensor
from quant_suite_recipe.definition import PipelineRecipe, QuantLinearRecipe
from tensor_dump._writer import _FORMAT_VERSION
from tensor_dump.merge import _MERGED_FORMAT_VERSION, merge
from tensor_dump.reader import (
    _FORMAT_VERSIONS,
    _History,
    _MERGED_FORMAT_VERSIONS,
    _natural_key,
    _reference_attention,
    Request,
    TensorDump,
)
from tensor_dump.recipe import DumpRecipe
from tensor_dump.runtime import RUNTIME

_LINEAR = "model.layers.0.mlp.down_proj"
_FULL = "model.layers.0.self_attn.attn"
_WINDOWED = "model.layers.1.self_attn.attn"
_HEADS = 4
_KV_HEADS = 2
_HEAD_SIZE = 8
_WINDOW = 4

# Per step: (request names, q_len of each, q_start_pos of each).
_FIRST_RUN: list[tuple[list[str], list[int], list[int]]] = [
    (["a"], [4], [0]),
    (["a"], [2], [4]),
    (["a"], [1], [6]),
    (["a"], [1], [7]),
    (["b"], [3], [0]),
    (["b"], [1], [3]),
]
_SECOND_RUN: list[tuple[list[str], list[int], list[int]]] = [
    (["c", "d"], [5, 2], [0, 0]),
    (["c", "d"], [1, 1], [5, 2]),
    (["d"], [1], [3]),
]
_HEADS_OF = {"q": _HEADS, "k": _KV_HEADS, "v": _KV_HEADS, "output": _HEADS}


class _CausalAttention(FakeAttention):
    """Computes causal attention in fp32 over each request's keys and values,
    which it keeps across forwards; ``requests`` names the step's requests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.requests: list[str] = []
        self.q_lens: list[int] = []

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        q = query.view(-1, self.num_heads, self.head_size)
        k = key.view(-1, self.num_kv_heads, self.head_size)
        v = value.view(-1, self.num_kv_heads, self.head_size)
        group = self.num_heads // self.num_kv_heads
        outputs, first = [], 0
        for name, q_len in zip(self.requests, self.q_lens):
            rows = slice(first, first + q_len)
            first += q_len
            old_k, old_v = self.history.get(name, (k[:0], v[:0]))
            full_k, full_v = torch.cat([old_k, k[rows]]), torch.cat([old_v, v[rows]])
            self.history[name] = (full_k, full_v)
            start = full_k.shape[0] - q_len
            positions = torch.arange(q_len)[:, None] + start
            keys = torch.arange(full_k.shape[0])[None, :]
            mask = keys <= positions
            if self.sliding_window is not None:
                mask &= keys > positions - self.sliding_window
            output = torch.nn.functional.scaled_dot_product_attention(
                q[rows].float().transpose(0, 1),
                full_k.float().repeat_interleave(group, dim=1).transpose(0, 1),
                full_v.float().repeat_interleave(group, dim=1).transpose(0, 1),
                attn_mask=mask,
                scale=self.impl.scale,
            )
            outputs.append(output.transpose(0, 1).to(query.dtype))
        return torch.cat(outputs).reshape(first, -1)


class _Linear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.prefix = _LINEAR
        self.weight = weight
        self.bias = bias
        recipe = QuantLinearRecipe(activation=PipelineRecipe([DumpRecipe()]))
        self.quantized_linear = RUNTIME.prepare_linear(recipe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantized_linear(x, QuantizedTensor(tensor=self.weight), self.bias)


class _Model(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.linear = _Linear(weight, bias)
        self.attention = torch.nn.ModuleList(
            [
                _CausalAttention(_FULL, _HEADS, _KV_HEADS, _HEAD_SIZE),
                _CausalAttention(
                    _WINDOWED, _HEADS, _KV_HEADS, _HEAD_SIZE, sliding_window=_WINDOW
                ),
            ]
        )

    def forward(
        self,
        names: list[str],
        q_lens: list[int],
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        outputs = {"linear": self.linear(inputs["x"])}
        for layer, name in zip(self.attention, (_FULL, _WINDOWED)):
            assert isinstance(layer, _CausalAttention)
            layer.requests, layer.q_lens = names, q_lens
            outputs[name] = layer(inputs["q"], inputs["k"], inputs["v"])
        return outputs


@dataclass
class _Written:
    """What one step fed to the model and got back."""

    names: list[str]
    q_lens: list[int]
    inputs: dict[str, torch.Tensor]
    outputs: dict[str, torch.Tensor]


def _write(
    directory: Path,
    schedule: list[tuple[list[str], list[int], list[int]]],
    weight: torch.Tensor,
    bias: torch.Tensor,
    **env: str,
) -> list[_Written]:
    """One run of ``schedule``, appended to ``directory``."""
    written = []
    env.setdefault("ATTENTION", ".*")
    with dump_session(
        directory, attention_class=_CausalAttention, MAX_STEPS="0", **env
    ) as forward:
        model = _Model(weight, bias)
        for names, q_lens, q_starts in schedule:
            forward.real(
                {
                    name: generic_metadata(q_lens, q_starts)
                    for name in (_FULL, _WINDOWED)
                }
            )
            tokens = sum(q_lens)
            inputs = {
                "x": torch.randn(tokens, 4),
                "q": torch.randn(tokens, _HEADS * _HEAD_SIZE, dtype=torch.bfloat16),
                "k": torch.randn(tokens, _KV_HEADS * _HEAD_SIZE, dtype=torch.bfloat16),
                "v": torch.randn(tokens, _KV_HEADS * _HEAD_SIZE, dtype=torch.bfloat16),
            }
            outputs = model(names, q_lens, inputs)
            written.append(_Written(names, q_lens, inputs, outputs))
    return written


def _rows(step: _Written, name: str, tensor: torch.Tensor) -> torch.Tensor:
    """The rows of request ``name`` in one of the step's tensors."""
    index = step.names.index(name)
    first = sum(step.q_lens[:index])
    return tensor[first : first + step.q_lens[index]]


class ReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.directory = Path(tempfile.mkdtemp())
        self.weight = torch.randn(6, 4)
        self.bias = torch.randn(6)
        first = _write(self.directory, _FIRST_RUN, self.weight, self.bias)
        second = _write(self.directory, _SECOND_RUN, self.weight, self.bias)
        self.written = first + second
        self.dump = self._open(self.directory)

    def _open(self, directory: Path) -> TensorDump:
        """The dump in ``directory``, in the layout the test case reads."""
        return TensorDump(directory)

    def _request(self, name: str) -> Request:
        index = "abcd".index(name)
        return self.dump.requests()[index]

    def _history(self, name: str, layer_input: str, up_to: int) -> torch.Tensor:
        """Request ``name``'s rows of ``layer_input`` over steps 0..``up_to``."""
        rows = [
            _rows(step, name, step.inputs[layer_input])
            for step in self.written[: up_to + 1]
            if name in step.names
        ]
        return torch.cat(rows).view(-1, _KV_HEADS, _HEAD_SIZE)

    def test_runs_and_steps(self) -> None:
        self.assertEqual(self.dump.ranks, (0,))
        self.assertEqual(
            [(run.run, run.first_step) for run in self.dump.runs()], [(0, 0), (1, 6)]
        )
        self.assertEqual(
            [(step.step, step.run, step.phase) for step in self.dump.steps()],
            [
                (0, 0, "prefill"),
                (1, 0, "prefill"),
                (2, 0, "decode"),
                (3, 0, "decode"),
                (4, 0, "prefill"),
                (5, 0, "decode"),
                (6, 1, "prefill"),
                (7, 1, "decode"),
                (8, 1, "decode"),
            ],
        )

    def test_requests_are_segmented(self) -> None:
        requests = self.dump.requests()
        self.assertEqual(
            [
                (r.id, r.run, r.steps, r.complete, r.prompt_len, r.generated_len)
                for r in requests
            ],
            [
                (0, 0, (0, 1, 2, 3), True, 6, 2),
                (1, 0, (4, 5), True, 3, 1),
                (2, 1, (6, 7), True, 5, 1),
                (3, 1, (6, 7, 8), True, 2, 2),
            ],
        )
        self.assertEqual([piece.request for piece in requests[3].slices], [1, 1, 0])

    def test_linear_pairs(self) -> None:
        for step, written in enumerate(self.written):
            with self.subTest(step=step):
                tensors = self.dump.linear(_LINEAR, step)
                self.assertTrue(torch.equal(tensors.activation, written.inputs["x"]))
                self.assertTrue(torch.equal(tensors.output, written.outputs["linear"]))
                self.assertTrue(torch.equal(tensors.weight, self.weight))
                self.assertTrue(torch.equal(tensors.bias, self.bias))
        pairs = list(self.dump.linear_pairs(phase="decode", layer=r"model\.layers\.0"))
        self.assertEqual([pair.step for pair in pairs], [2, 3, 5, 7, 8])
        self.assertEqual(list(self.dump.linear_pairs(layer="lm_head")), [])
        self.assertEqual(
            [p.step for p in self.dump.linear_pairs(rank=0, steps={1, 6})], [1, 6]
        )

    def test_attention_rebuilds_the_history_of_every_step(self) -> None:
        for name in "abcd":
            request = self._request(name)
            for piece in request.slices:
                for layer in (_FULL, _WINDOWED):
                    with self.subTest(request=name, step=piece.step, layer=layer):
                        tensors = self.dump.attention(layer, request, piece.step)
                        step = self.written[piece.step]
                        self.assertTrue(
                            torch.equal(tensors.k, self._history(name, "k", piece.step))
                        )
                        self.assertTrue(
                            torch.equal(tensors.v, self._history(name, "v", piece.step))
                        )
                        expected_q = _rows(step, name, step.inputs["q"])
                        self.assertTrue(
                            torch.equal(
                                tensors.q, expected_q.view(-1, _HEADS, _HEAD_SIZE)
                            )
                        )
                        expected_out = _rows(step, name, step.outputs[layer])
                        self.assertTrue(
                            torch.equal(
                                tensors.output,
                                expected_out.view(-1, _HEADS, _HEAD_SIZE),
                            )
                        )
                        replay = tensors.replay
                        self.assertEqual(
                            (replay.q_start_pos, replay.kv_len),
                            (piece.q_start_pos, piece.kv_len),
                        )

    def _assert_history(self, layer: str, name: str, step: int) -> None:
        tensors = self.dump.attention(layer, self._request(name), step)
        self.assertTrue(torch.equal(tensors.k, self._history(name, "k", step)))
        self.assertTrue(torch.equal(tensors.v, self._history(name, "v", step)))

    def _history_bound(self) -> int:
        """The two attention layers times the most requests in one step."""
        return 2 * max(len(names) for names, _, _ in _FIRST_RUN + _SECOND_RUN)

    def test_history_cache_is_bounded_by_layers_and_concurrency(self) -> None:
        bound = self._history_bound()
        for name in "abcd":
            first_step = self._request(name).slices[0].step
            for layer in (_FULL, _WINDOWED):
                with self.subTest(request=name, layer=layer):
                    self._assert_history(layer, name, first_step)
                    self.assertLessEqual(len(self.dump._histories), bound)
        self.assertEqual(len(self.dump._histories), bound)

    def test_evicted_history_is_rebuilt(self) -> None:
        a = self._request("a")
        others = [self._request(name) for name in "bcd"]
        self.assertGreater(2 * (1 + len(others)), self._history_bound())
        for piece in a.slices:
            for layer in (_FULL, _WINDOWED):
                with self.subTest(step=piece.step, layer=layer):
                    self._assert_history(layer, "a", piece.step)
            for other in others:
                for layer in (_FULL, _WINDOWED):
                    self.dump.attention(layer, other, other.slices[0].step)
            for layer in (_FULL, _WINDOWED):
                self.assertNotIn((a.rank, layer, a.id), self.dump._histories)

    def test_histories_of_requests_in_one_step_stay_cached(self) -> None:
        requests = {name: self._request(name) for name in "cd"}
        steps = sorted({step for r in requests.values() for step in r.steps})
        cached: dict[tuple[int, str, int], _History] = {}
        for step in steps:
            for name, request in requests.items():
                if step not in request.steps:
                    continue
                position = request.steps.index(step)
                for layer in (_FULL, _WINDOWED):
                    with self.subTest(request=name, step=step, layer=layer):
                        key = (request.rank, layer, request.id)
                        if position > 0:
                            self.assertIs(self.dump._histories.get(key), cached[key])
                            self.assertEqual(cached[key].slices, position)
                        self._assert_history(layer, name, step)
                        if position < len(request.slices) - 1:
                            cached[key] = self.dump._histories[key]
                            self.assertEqual(cached[key].slices, position + 1)

    def test_reference_attention_reproduces_both_layers(self) -> None:
        request = self._request("a")
        for layer, window in ((_FULL, None), (_WINDOWED, _WINDOW)):
            with self.subTest(layer=layer):
                tensors = self.dump.attention(layer, request, 3)
                self.assertEqual(tensors.replay.sliding_window, window)
                self.assertGreater(tensors.replay.kv_len, _WINDOW)
                q, output = tensors.q, tensors.output
                assert q is not None and output is not None
                torch.testing.assert_close(
                    _reference_attention(q, tensors.k, tensors.v, tensors.replay),
                    output.float(),
                    atol=2e-2,
                    rtol=2e-2,
                )

    def test_missing_history_raises(self) -> None:
        directory = Path(tempfile.mkdtemp())
        _write(directory, _FIRST_RUN, self.weight, self.bias, STEPS="1:")
        dump = self._open(directory)
        first = dump.requests()[0]
        self.assertFalse(first.complete)
        self.assertEqual(first.steps, (1, 2, 3))
        with self.assertRaisesRegex(ValueError, r"rows \[0, 4\)"):
            dump.attention(_FULL, first, 2)

        directory = Path(tempfile.mkdtemp())
        _write(
            directory, _FIRST_RUN, self.weight, self.bias, ATTENTION_TENSORS="q,output"
        )
        dump = self._open(directory)
        with self.assertRaisesRegex(ValueError, "wrote no k and v"):
            dump.attention(_FULL, dump.requests()[0], 1)

        directory = Path(tempfile.mkdtemp())
        _write(directory, _FIRST_RUN, self.weight, self.bias, ATTENTION=_WINDOWED)
        dump = self._open(directory)
        with self.assertRaises(KeyError):
            dump.attention(_FULL, dump.requests()[0], 0)

    def test_ambiguous_requests_raise(self) -> None:
        directory = Path(tempfile.mkdtemp())
        schedule = [(["e", "f"], [2, 2], [0, 0]), (["e", "f"], [1, 1], [2, 2])]
        _write(directory, schedule, self.weight, self.bias)
        with self.assertRaisesRegex(ValueError, "cannot be told apart"):
            self._open(directory).requests()

    def test_unsupported_format_raises(self) -> None:
        self.assertEqual(_FORMAT_VERSIONS, frozenset({_FORMAT_VERSION}))
        manifest = self.directory / "rank_0" / "manifest.jsonl"
        lines = manifest.read_text().splitlines()
        run = json.loads(lines[0])
        run["format_version"] = 1
        manifest.write_text("\n".join([json.dumps(run), *lines[1:]]) + "\n")
        with self.assertRaisesRegex(ValueError, "format 1"):
            TensorDump(self.directory)

    def test_malformed_field_raises_with_its_line(self) -> None:
        manifest = self.directory / "rank_0" / "manifest.jsonl"
        lines = manifest.read_text().splitlines()
        records = [json.loads(line) for line in lines]
        number = next(i for i, r in enumerate(records) if r["record"] == "step")
        records[number]["num_actual_tokens"] = "many"
        lines[number] = json.dumps(records[number], sort_keys=True)
        manifest.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(
            ValueError, f"line {number + 1} of {manifest} is not a valid record"
        ):
            TensorDump(self.directory)

    def test_kv_len_that_disagrees_with_the_history_raises(self) -> None:
        manifest = self.directory / "rank_0" / "manifest.jsonl"
        lines = manifest.read_text().splitlines()
        records = [json.loads(line) for line in lines]
        number = next(
            i for i, r in enumerate(records) if r["record"] == "step" and r["step"] == 3
        )
        piece = records[number]["requests"][0]
        self.assertEqual((piece["phase"], piece["kv_len"]), ("decode", 8))
        piece["kv_len"] = 9
        lines[number] = json.dumps(records[number], sort_keys=True)
        manifest.write_text("\n".join(lines) + "\n")
        dump = TensorDump(self.directory)
        request = dump.requests()[0]
        self.assertEqual(request.steps, (0, 1, 2, 3))
        dump.attention(_FULL, request, 2)
        with self.assertRaisesRegex(ValueError, "kv_len 9 at step 3.* rebuilds 8 rows"):
            dump.attention(_FULL, request, 3)

    def test_shape_mismatch_raises_on_load(self) -> None:
        pair = next(self.dump.linear_pairs(steps={0}))
        activation = pair.activation
        assert activation is not None
        tampered = dataclasses.replace(activation, shape=(99,))
        with self.assertRaisesRegex(ValueError, "manifest entry records"):
            tampered.load()

    def test_unreadable_file_raises_on_load(self) -> None:
        pair = next(self.dump.linear_pairs(steps={0}))
        activation = pair.activation
        assert activation is not None
        original = activation.path.read_bytes()
        for content in (b"not a torch file", original[: len(original) // 2], b""):
            with self.subTest(size=len(content)):
                activation.path.write_bytes(content)
                with self.assertRaisesRegex(
                    ValueError, f"{activation.path}.*cannot be read as a torch file"
                ):
                    activation.load()

    def test_phase_tensors_stack_each_phase_in_step_order(self) -> None:
        for request in self.dump.requests():
            name = "abcd"[request.id]
            for phase in ("prefill", "decode"):
                steps = [
                    (self.written[piece.step], piece)
                    for piece in request.slices
                    if piece.phase == phase
                ]
                linear = self.dump.phase_tensors(_LINEAR, request, phase)
                self.assertEqual(sorted(linear), ["input", "output"])
                inputs = [_rows(step, name, step.inputs["x"]) for step, _ in steps]
                outputs = [
                    _rows(step, name, step.outputs["linear"]) for step, _ in steps
                ]
                self.assertTrue(torch.equal(linear["input"], torch.cat(inputs)))
                self.assertTrue(torch.equal(linear["output"], torch.cat(outputs)))
                for layer in (_FULL, _WINDOWED):
                    with self.subTest(request=name, phase=phase, layer=layer):
                        tensors = self.dump.phase_tensors(
                            layer, request.id, phase, rank=0
                        )
                        self.assertEqual(sorted(tensors), ["k", "output", "q", "v"])
                        for key, tensor in tensors.items():
                            rows = [
                                _rows(
                                    step,
                                    name,
                                    step.outputs[layer]
                                    if key == "output"
                                    else step.inputs[key],
                                )
                                for step, _ in steps
                            ]
                            expected = torch.cat(rows).view(
                                -1, _HEADS_OF[key], _HEAD_SIZE
                            )
                            self.assertTrue(torch.equal(tensor, expected))
                        self.assertEqual(
                            tensors["q"].shape[0], sum(p.q_len for _, p in steps)
                        )
        with self.assertRaises(KeyError):
            self.dump.phase_tensors(_FULL, 99, "prefill")
        with self.assertRaisesRegex(ValueError, "mixed"):
            self.dump.phase_tensors(_FULL, 0, "mixed")

    def test_full_keys_are_the_phase_rows_up_to_a_step(self) -> None:
        request = self._request("a")
        prefill = self.dump.phase_tensors(_FULL, request, "prefill")
        decode = self.dump.phase_tensors(_FULL, request, "decode")
        decodes = [piece.step for piece in request.slices if piece.phase == "decode"]
        for row, step in enumerate(decodes):
            tensors = self.dump.attention(_FULL, request, step)
            for key in ("k", "v"):
                full = torch.cat([prefill[key], decode[key][: row + 1]])
                self.assertTrue(torch.equal(full, getattr(tensors, key)))

    def test_attention_layers(self) -> None:
        self.assertEqual(self.dump.attention_layers(), (_FULL, _WINDOWED))
        self.assertEqual(self.dump.attention_layers(rank=0), (_FULL, _WINDOWED))
        with self.assertRaises(KeyError):
            self.dump.attention_layers(rank=1)
        names = ["layers.10.attn", "layers.2.attn", "layers.1.attn"]
        self.assertEqual(
            sorted(names, key=_natural_key),
            ["layers.1.attn", "layers.2.attn", "layers.10.attn"],
        )


class MergedReaderTest(ReaderTest):
    """The same reads on the merged dump, its per-step files deleted."""

    def _open(self, directory: Path) -> TensorDump:
        merged = directory.with_name(directory.name + "_merged")
        merge(directory, merged, delete_source=True)
        self.assertFalse((directory / "rank_0").exists())
        return TensorDump(merged)

    def _manifest(self) -> Path:
        return self.directory.with_name(self.directory.name + "_merged") / (
            "rank_0/manifest.json"
        )

    def test_unsupported_format_raises(self) -> None:
        self.assertEqual(_MERGED_FORMAT_VERSIONS, frozenset({_MERGED_FORMAT_VERSION}))
        manifest = json.loads(self._manifest().read_text())
        manifest["merged_format_version"] = 9
        self._manifest().write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "merged format 9"):
            TensorDump(self._manifest().parents[1])

    def test_malformed_field_raises_with_its_line(self) -> None:
        manifest = json.loads(self._manifest().read_text())
        manifest["steps"][0]["num_actual_tokens"] = "many"
        self._manifest().write_text(json.dumps(manifest))
        with self.assertRaisesRegex(
            ValueError, f"{self._manifest()} is not a valid merged manifest"
        ):
            TensorDump(self._manifest().parents[1])

    def test_kv_len_that_disagrees_with_the_history_raises(self) -> None:
        manifest = json.loads(self._manifest().read_text())
        entry = next(r for r in manifest["requests"] if r["request"] == 0)
        segments = entry["phases"]["decode"]["segments"]
        segment = next(s for s in segments if s["step"] == 3)
        self.assertEqual(segment["kv_len"], 8)
        segment["kv_len"] = 9
        self._manifest().write_text(json.dumps(manifest))
        dump = TensorDump(self._manifest().parents[1])
        request = dump.requests()[0]
        self.assertEqual(request.steps, (0, 1, 2, 3))
        dump.attention(_FULL, request, 2)
        with self.assertRaisesRegex(ValueError, "kv_len 9 at step 3.* rebuilds 8 rows"):
            dump.attention(_FULL, request, 3)


if __name__ == "__main__":
    unittest.main()
