# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``tensor_dump.merge`` on dumps the writer wrote: two runs into one
directory, with a chunked prefill, decodes, requests in sequence, two requests
in one step and a mixed step. CPU only."""

import io
import json
import shutil
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import call, Mock, patch

import torch
from dump_test_utils import dump_session, FakeAttention, generic_metadata
from quant_suite_recipe.api import QuantizedTensor
from quant_suite_recipe.definition import PipelineRecipe, QuantLinearRecipe
from tensor_dump import merge as merge_module
from tensor_dump.merge import merge
from tensor_dump.reader import TensorDump
from tensor_dump.recipe import DumpRecipe
from tensor_dump.runtime import RUNTIME

_LINEARS = ("model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj")
_FULL = "model.layers.0.self_attn.attn"
_WINDOWED = "model.layers.1.self_attn.attn"
_ATTENTION = (_FULL, _WINDOWED)
_HEADS = 4
_KV_HEADS = 2
_HEAD_SIZE = 8
_WINDOW = 4

_Schedule = list[tuple[list[str], list[int], list[int]]]

# Per step: (request names, q_len of each, q_start_pos of each).
_FIRST_RUN: _Schedule = [
    (["a"], [4], [0]),
    (["a"], [3], [4]),
    (["a"], [1], [7]),
    (["a"], [1], [8]),
    (["b"], [3], [0]),
    (["b"], [1], [3]),
]
_SECOND_RUN: _Schedule = [
    (["c", "d"], [5, 2], [0, 0]),
    (["c", "d"], [1, 1], [5, 2]),
    (["d", "e"], [1, 3], [3, 0]),
    (["e"], [1], [3]),
]
_NAMES = "abcde"


class _Linear(torch.nn.Module):
    def __init__(
        self, prefix: str, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.weight = weight
        self.bias = bias
        recipe = QuantLinearRecipe(activation=PipelineRecipe([DumpRecipe()]))
        self.quantized_linear = RUNTIME.prepare_linear(recipe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantized_linear(x, QuantizedTensor(tensor=self.weight), self.bias)


class _Model(torch.nn.Module):
    def __init__(self, weights: list[tuple[torch.Tensor, torch.Tensor | None]]) -> None:
        super().__init__()
        self.linears = torch.nn.ModuleList(
            [
                _Linear(name, weight, bias)
                for name, (weight, bias) in zip(_LINEARS, weights)
            ]
        )
        self.attention = torch.nn.ModuleList(
            [
                FakeAttention(_FULL, _HEADS, _KV_HEADS, _HEAD_SIZE),
                FakeAttention(
                    _WINDOWED, _HEADS, _KV_HEADS, _HEAD_SIZE, sliding_window=_WINDOW
                ),
            ]
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = {
            name: linear(inputs["x"]) for name, linear in zip(_LINEARS, self.linears)
        }
        for name, layer in zip(_ATTENTION, self.attention):
            outputs[name] = layer(inputs["q"], inputs["k"], inputs["v"])
        return outputs


@dataclass
class _Written:
    """What one step fed to the model and got back."""

    names: list[str]
    q_lens: list[int]
    q_starts: list[int]
    inputs: dict[str, torch.Tensor]
    outputs: dict[str, torch.Tensor]

    def rows(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        index = self.names.index(name)
        first = sum(self.q_lens[:index])
        return tensor[first : first + self.q_lens[index]]

    def phase(self, name: str) -> str:
        index = self.names.index(name)
        decode = self.q_lens[index] == 1 and self.q_starts[index] > 0
        return "decode" if decode else "prefill"


def _write(
    directory: Path,
    schedule: _Schedule,
    weights: list[tuple[torch.Tensor, torch.Tensor | None]],
    **env: str,
) -> list[_Written]:
    """One run of ``schedule``, appended to ``directory``."""
    written = []
    env.setdefault("MAX_STEPS", "0")
    with dump_session(directory, ATTENTION=".*", **env) as forward:
        model = _Model(weights)
        for names, q_lens, q_starts in schedule:
            forward.real(
                {name: generic_metadata(q_lens, q_starts) for name in _ATTENTION}
            )
            tokens = sum(q_lens)
            inputs = {
                "x": torch.randn(tokens, 4),
                "q": torch.randn(tokens, _HEADS * _HEAD_SIZE, dtype=torch.bfloat16),
                "k": torch.randn(tokens, _KV_HEADS * _HEAD_SIZE, dtype=torch.bfloat16),
                "v": torch.randn(tokens, _KV_HEADS * _HEAD_SIZE, dtype=torch.bfloat16),
            }
            outputs = model(inputs)
            written.append(_Written(names, q_lens, q_starts, inputs, outputs))
    return written


def _files(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "rank_0" / "manifest.json").read_text())


def _listed_files(directory: Path) -> list[str]:
    """Every merged file the destination manifest lists, sorted."""
    manifest = _manifest(directory)
    files = [entry["file"] for entry in manifest["weights"].values()]
    for request in manifest["requests"]:
        for phase in request["phases"].values():
            files.extend(entry["file"] for entry in phase["layers"].values())
    return sorted(files)


def _merged(directory: Path, request: int, phase: str, layer: str) -> dict[str, Any]:
    path = directory / "rank_0" / f"req_{request:04d}" / phase / f"{layer}.pt"
    return torch.load(path, weights_only=True)


@contextmanager
def _corrupt_reads(fragment: str) -> Iterator[None]:
    """Merged files whose path contains ``fragment`` read back with one bit
    flipped."""
    load = merge_module._load_file

    def corrupted(path: Path) -> object:
        data = load(path)
        if fragment in str(path) and isinstance(data, dict):
            name = sorted(data)[0]
            tensor = data[name].clone()
            raw = tensor.view(-1).view(torch.uint8)
            raw[0] ^= 1
            data[name] = tensor
        return data

    with patch.object(merge_module, "_load_file", side_effect=corrupted):
        yield


class MergeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.root = Path(tempfile.mkdtemp())
        self.source = self.root / "dump"
        self.destination = self.root / "merged"
        self.weights = [
            (torch.randn(6, 4), torch.randn(6)),
            (torch.randn(5, 4), None),
        ]
        first = _write(self.source, _FIRST_RUN, self.weights)
        second = _write(self.source, _SECOND_RUN, self.weights)
        self.written = first + second

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _expected(self, name: str, phase: str) -> dict[str, dict[str, torch.Tensor]]:
        """Request ``name``'s rows of ``phase`` of every layer, in step order."""
        steps = [
            step
            for step in self.written
            if name in step.names and step.phase(name) == phase
        ]
        expected: dict[str, dict[str, torch.Tensor]] = {}
        for layer in _LINEARS:
            expected[layer] = {
                "input": torch.cat(
                    [step.rows(name, step.inputs["x"]) for step in steps]
                ),
                "output": torch.cat(
                    [step.rows(name, step.outputs[layer]) for step in steps]
                ),
            }
        for layer in _ATTENTION:
            tensors = {
                "q": [step.rows(name, step.inputs["q"]) for step in steps],
                "k": [step.rows(name, step.inputs["k"]) for step in steps],
                "v": [step.rows(name, step.inputs["v"]) for step in steps],
                "output": [step.rows(name, step.outputs[layer]) for step in steps],
            }
            heads = {"q": _HEADS, "k": _KV_HEADS, "v": _KV_HEADS, "output": _HEADS}
            expected[layer] = {
                key: torch.cat(rows).view(-1, heads[key], _HEAD_SIZE)
                for key, rows in tensors.items()
            }
        return expected

    def _verified_files(self, delete_source: bool) -> list[str]:
        """Merge, and return the merged files ``_verify`` read, sorted."""
        with patch.object(merge_module, "_verify", wraps=merge_module._verify) as spy:
            merge(self.source, self.destination, delete_source=delete_source)
        rank = self.destination / "rank_0"
        return sorted(
            str(Path(entry.args[0]).relative_to(rank)) for entry in spy.call_args_list
        )

    def _assert_merged_matches(self, names: str = _NAMES) -> None:
        for index, name in enumerate(_NAMES):
            if name not in names:
                continue
            for phase in ("prefill", "decode"):
                expected = self._expected(name, phase)
                for layer, tensors in expected.items():
                    with self.subTest(request=name, phase=phase, layer=layer):
                        merged = _merged(self.destination, index, phase, layer)
                        self.assertEqual(sorted(merged), sorted(tensors))
                        for key, tensor in tensors.items():
                            self.assertEqual(merged[key].dtype, tensor.dtype)
                            self.assertTrue(torch.equal(merged[key], tensor))

    def test_merged_files_hold_each_request_phase_and_layer(self) -> None:
        merge(self.source, self.destination)
        self._assert_merged_matches()
        rank = self.destination / "rank_0"
        for layer, (weight, bias) in zip(_LINEARS, self.weights):
            stored = torch.load(rank / "weights" / f"{layer}.pt", weights_only=True)
            self.assertTrue(torch.equal(stored["weight"], weight))
            if bias is None:
                self.assertNotIn("bias", stored)
            else:
                self.assertTrue(torch.equal(stored["bias"], bias))
        self.assertTrue((self.source / "rank_0" / "manifest.jsonl").is_file())

    def test_reader_reads_both_layouts_alike(self) -> None:
        written = TensorDump(self.source)
        layers = _LINEARS + _ATTENTION
        expected = {
            (request.id, phase, layer): written.phase_tensors(layer, request, phase)
            for request in written.requests()
            for phase in ("prefill", "decode")
            for layer in layers
        }
        steps, requests = written.steps(), written.requests()
        runs = written.runs()
        linear = {
            (pair.layer, pair.step): written.load_linear(pair)
            for pair in written.linear_pairs()
        }
        merge(self.source, self.destination, delete_source=True)
        merged = TensorDump(self.destination)
        self.assertEqual(merged.steps(), steps)
        self.assertEqual(merged.requests(), requests)
        self.assertEqual(merged.runs(), runs)
        self.assertEqual(merged.attention_layers(), written.attention_layers())
        for (request, phase, layer), tensors in expected.items():
            with self.subTest(request=request, phase=phase, layer=layer):
                actual = merged.phase_tensors(layer, request, phase)
                self.assertEqual(sorted(actual), sorted(tensors))
                for key, tensor in tensors.items():
                    self.assertTrue(torch.equal(actual[key], tensor))
        pairs = list(merged.linear_pairs())
        self.assertEqual([(pair.layer, pair.step) for pair in pairs], list(linear))
        for pair in pairs:
            loaded = merged.load_linear(pair)
            for field in ("activation", "weight", "bias", "output"):
                before = getattr(linear[(pair.layer, pair.step)], field)
                after = getattr(loaded, field)
                if before is None:
                    self.assertIsNone(after)
                else:
                    self.assertTrue(torch.equal(after, before))

    def test_merged_files_match_the_per_step_reader(self) -> None:
        merge(self.source, self.destination)
        dump = TensorDump(self.source)
        for request in dump.requests():
            rows = {"prefill": 0, "decode": 0}
            for piece in request.slices:
                start = rows[piece.phase]
                rows[piece.phase] += piece.q_len
                for layer in _LINEARS:
                    linear = dump.linear(layer, piece.step)
                    merged = _merged(self.destination, request.id, piece.phase, layer)
                    first = piece.token_start
                    for key, tensor in (
                        ("input", linear.activation),
                        ("output", linear.output),
                    ):
                        assert tensor is not None
                        self.assertTrue(
                            torch.equal(
                                merged[key][start : start + piece.q_len],
                                tensor[first : first + piece.q_len],
                            )
                        )
                for layer in _ATTENTION:
                    attention = dump.attention(layer, request, piece.step)
                    merged = _merged(self.destination, request.id, piece.phase, layer)
                    q = attention.q
                    assert q is not None
                    self.assertTrue(
                        torch.equal(merged["q"][start : start + piece.q_len], q)
                    )
                    new_rows = slice(piece.q_start_pos, piece.kv_len)
                    self.assertTrue(
                        torch.equal(
                            merged["k"][start : start + piece.q_len],
                            attention.k[new_rows],
                        )
                    )

    def test_manifest_maps_rows_to_steps(self) -> None:
        merge(self.source, self.destination)
        manifest = _manifest(self.destination)
        self.assertEqual(manifest["merged_format_version"], 1)
        self.assertEqual(manifest["source_format_versions"], [2])
        self.assertEqual([run["run"] for run in manifest["runs"]], [0, 1])
        self.assertEqual([step["step"] for step in manifest["steps"]], list(range(10)))
        self.assertEqual(manifest["steps"][8]["phase"], "mixed")
        self.assertFalse(manifest["source_deleted"])
        requests = manifest["requests"]
        self.assertEqual(
            [
                (
                    r["request"],
                    r["run"],
                    r["steps"],
                    r["prompt_len"],
                    r["generated_len"],
                    r["complete"],
                )
                for r in requests
            ],
            [
                (0, 0, [0, 1, 2, 3], 7, 2, True),
                (1, 0, [4, 5], 3, 1, True),
                (2, 1, [6, 7], 5, 1, True),
                (3, 1, [6, 7, 8], 2, 2, True),
                (4, 1, [8, 9], 3, 1, True),
            ],
        )
        prefill = requests[0]["phases"]["prefill"]
        self.assertEqual(prefill["rows"], 7)
        self.assertEqual(
            [
                (s["row_start"], s["q_len"], s["step"], s["run"], s["q_start_pos"])
                for s in prefill["segments"]
            ],
            [(0, 4, 0, 0, 0), (4, 3, 1, 0, 4)],
        )
        decode = requests[3]["phases"]["decode"]
        self.assertEqual(
            [
                (s["step"], s["request"], s["token_start"], s["q_start_pos"])
                for s in decode["segments"]
            ],
            [(7, 1, 1, 2), (8, 0, 0, 3)],
        )
        self.assertEqual(
            [s["step"] for s in requests[4]["phases"]["prefill"]["segments"]], [8]
        )
        layers = decode["layers"]
        self.assertEqual(sorted(layers), sorted(_LINEARS + _ATTENTION))
        self.assertEqual(layers[_LINEARS[0]]["kind"], "linear")
        self.assertEqual(
            layers[_LINEARS[0]]["tensors"],
            {
                "input": {"shape": [2, 4], "dtype": "float32"},
                "output": {"shape": [2, 6], "dtype": "float32"},
            },
        )
        attention = layers[_WINDOWED]
        self.assertEqual(attention["kind"], "attention")
        self.assertEqual(
            attention["tensors"]["k"],
            {"shape": [2, _KV_HEADS, _HEAD_SIZE], "dtype": "bfloat16"},
        )
        replay = attention["replay"]
        self.assertEqual(replay["sliding_window"], _WINDOW)
        self.assertEqual(replay["num_q_heads"], _HEADS)
        self.assertEqual(replay["scale"], _HEAD_SIZE**-0.5)
        self.assertNotIn("kv_len", replay)
        self.assertIsNone(layers[_FULL]["replay"]["sliding_window"])
        weight = manifest["weights"][_LINEARS[1]]
        self.assertEqual(weight["file"], f"weights/{_LINEARS[1]}.pt")
        self.assertEqual(sorted(weight["tensors"]), ["weight"])
        self.assertEqual(weight["step"], 0)

    def test_full_keys_of_a_decode_row_are_a_slice(self) -> None:
        merge(self.source, self.destination)
        dump = TensorDump(self.source)
        request = dump.requests()[0]
        self.assertTrue(
            _manifest(self.destination)["requests"][0]["prefill_before_decode"]
        )
        for layer in _ATTENTION:
            prefill = _merged(self.destination, 0, "prefill", layer)
            decode = _merged(self.destination, 0, "decode", layer)
            steps = [piece.step for piece in request.slices if piece.phase == "decode"]
            for row, step in enumerate(steps):
                with self.subTest(layer=layer, row=row):
                    expected = dump.attention(layer, request, step)
                    for key in ("k", "v"):
                        full = torch.cat([prefill[key], decode[key][: row + 1]])
                        self.assertTrue(torch.equal(full, getattr(expected, key)))

    def test_replay_fields_that_change_are_stored_per_row_range(self) -> None:
        path = self.source / "rank_0" / "manifest.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for record in records:
            if (
                record["record"] == "attention"
                and record["layer"] == _FULL
                and record["step"] == 3
            ):
                record["replay"]["scale"] = 0.5
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
        merge(self.source, self.destination)
        layers = _manifest(self.destination)["requests"][0]["phases"]["decode"][
            "layers"
        ]
        ranges = layers[_FULL]["replay_ranges"]
        self.assertNotIn("replay", layers[_FULL])
        self.assertEqual(
            [(r["rows"], r["replay"]["scale"]) for r in ranges],
            [([0, 1], _HEAD_SIZE**-0.5), ([1, 2], 0.5)],
        )
        self.assertIn("replay", layers[_WINDOWED])

    def test_malformed_run_option_raises(self) -> None:
        path = self.source / "rank_0" / "manifest.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for record in records:
            if record["record"] == "run":
                record["options"]["max_bytes"] = 1.5
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
        with self.assertRaisesRegex(ValueError, "max_bytes=1.5"):
            merge(self.source, self.destination)

    def test_delete_source_removes_the_merged_dump(self) -> None:
        merge(self.source, self.destination, delete_source=True)
        self.assertEqual(_files(self.source), {})
        self.assertTrue(_manifest(self.destination)["source_deleted"])
        self._assert_merged_matches()
        merge(self.source, self.destination, delete_source=True)
        self._assert_merged_matches()

    def test_failed_verification_deletes_nothing_unverified(self) -> None:
        before = _files(self.source)
        with _corrupt_reads("weights"), self.assertRaisesRegex(RuntimeError, "differ"):
            merge(self.source, self.destination, delete_source=True)
        self.assertEqual(_files(self.source), before)

        with _corrupt_reads("req_0001"), self.assertRaisesRegex(RuntimeError, "differ"):
            merge(self.source, self.destination, delete_source=True)
        remaining = _files(self.source)
        self.assertIn("rank_0/manifest.jsonl", remaining)
        steps = {name.split("/")[1] for name in remaining if "/step_" in name}
        self.assertEqual(steps, {f"step_{step:06d}" for step in range(4, 10)})
        self.assertFalse(any("/weights/" in name for name in remaining))
        for name, data in remaining.items():
            if name != "rank_0/manifest.jsonl":
                self.assertEqual(data, before[name])
        self.assertEqual(
            [r["request"] for r in _manifest(self.destination)["requests"]], [0]
        )

        merge(self.source, self.destination, delete_source=True)
        self.assertEqual(_files(self.source), {})
        self._assert_merged_matches()

    def test_resumed_delete_reverifies_merged_files(self) -> None:
        merge(self.source, self.destination)
        path = self.destination / "rank_0" / "req_0001" / "decode" / f"{_LINEARS[0]}.pt"
        data = torch.load(path, weights_only=True)
        data["output"].view(-1).view(torch.uint8)[0] ^= 1
        torch.save(data, path)
        before = _files(self.source)
        with self.assertRaisesRegex(RuntimeError, "differ"):
            merge(self.source, self.destination, delete_source=True)
        remaining = _files(self.source)
        self.assertIn("rank_0/manifest.jsonl", remaining)
        for name, content in before.items():
            if "/step_000004/" in name or "/step_000005/" in name:
                self.assertEqual(remaining.get(name), content)
        self.assertFalse(_manifest(self.destination)["source_deleted"])

    def test_resumed_delete_deletes_after_reverifying(self) -> None:
        merge(self.source, self.destination)
        verified = self._verified_files(delete_source=True)
        self.assertEqual(verified, _listed_files(self.destination))
        self.assertEqual(_files(self.source), {})
        self.assertTrue(_manifest(self.destination)["source_deleted"])
        self._assert_merged_matches()

    def test_files_written_in_one_pass_are_verified_once(self) -> None:
        verified = self._verified_files(delete_source=True)
        self.assertEqual(verified, _listed_files(self.destination))
        self.assertEqual(_files(self.source), {})

    def test_command_line_caps_torch_threads(self) -> None:
        src, dst = str(self.source), str(self.destination)
        cases = [([], 16, 4), ([], 2, 2), ([], None, 1), (["--threads", "8"], 2, 8)]
        for extra, cpus, threads in cases:
            with self.subTest(extra=extra, cpus=cpus):
                calls = Mock()
                with (
                    patch.object(merge_module, "merge", calls.merge),
                    patch.object(torch, "set_num_threads", calls.set_num_threads),
                    patch.object(merge_module.os, "cpu_count", return_value=cpus),
                ):
                    merge_module._main([src, dst, *extra])
                self.assertEqual(
                    calls.mock_calls,
                    [
                        call.set_num_threads(threads),
                        call.merge(src, dst, delete_source=False),
                    ],
                )
        with (
            patch.object(merge_module, "merge") as run,
            self.assertRaises(SystemExit),
            redirect_stderr(io.StringIO()),
        ):
            merge_module._main([src, dst, "--threads", "0"])
        run.assert_not_called()

    def test_resume_skips_merged_requests(self) -> None:
        merge(self.source, self.destination)
        with patch.object(merge_module, "_save", wraps=merge_module._save) as save:
            merge(self.source, self.destination)
        self.assertEqual(save.call_count, 0)

        path = self.destination / "rank_0" / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["requests"] = manifest["requests"][:-1]
        path.write_text(json.dumps(manifest, sort_keys=True))
        with patch.object(merge_module, "_save", wraps=merge_module._save) as save:
            merge(self.source, self.destination)
        written = {
            Path(call.args[0]).relative_to(path.parent) for call in save.call_args_list
        }
        self.assertEqual(
            {str(p) for p in written},
            {
                f"req_0004/{phase}/{layer}.pt"
                for phase in ("prefill", "decode")
                for layer in _LINEARS + _ATTENTION
            }
            | {"manifest.json"},
        )
        self.assertEqual(
            [r["request"] for r in _manifest(self.destination)["requests"]],
            list(range(5)),
        )
        self._assert_merged_matches()

    def test_destination_of_another_dump_raises(self) -> None:
        merge(self.source, self.destination)
        other = self.root / "other"
        _write(other, _SECOND_RUN, self.weights)
        with self.assertRaisesRegex(RuntimeError, "another dump"):
            merge(other, self.destination)
        with self.assertRaisesRegex(ValueError, "is the dump directory"):
            merge(self.source, self.source)
        with self.assertRaisesRegex(ValueError, "holds no"):
            merge(self.root / "missing", self.root / "missing_merged")

    def test_incomplete_request(self) -> None:
        source = self.root / "partial"
        written = _write(source, _FIRST_RUN, self.weights, STEPS="1:", MAX_STEPS="2")
        merge_module._main([str(source), str(self.destination), "--delete-source"])
        (request,) = _manifest(self.destination)["requests"]
        self.assertEqual(
            (
                request["steps"],
                request["starts_at_zero"],
                request["may_continue"],
                request["complete"],
            ),
            ([1, 2], False, True, False),
        )
        self.assertEqual(
            [s["q_start_pos"] for s in request["phases"]["prefill"]["segments"]], [4]
        )
        prefill = _merged(self.destination, 0, "prefill", _FULL)
        decode = _merged(self.destination, 0, "decode", _FULL)
        self.assertEqual(prefill["k"].shape, (3, _KV_HEADS, _HEAD_SIZE))
        self.assertEqual(decode["q"].shape, (1, _HEADS, _HEAD_SIZE))
        self.assertTrue(
            torch.equal(
                prefill["k"], written[1].inputs["k"].view(-1, _KV_HEADS, _HEAD_SIZE)
            )
        )
        self.assertEqual(_files(source), {})


if __name__ == "__main__":
    unittest.main()
