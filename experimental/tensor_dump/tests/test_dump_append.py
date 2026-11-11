# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runs appended to one dump directory, weights recorded once across them, and
the byte budget. CPU only."""

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch
from dump_test_utils import dump_session, fake_metadata, load, read_manifest
from quant_suite_recipe.api import QuantizedTensor
from quant_suite_recipe.definition import PipelineRecipe, QuantLinearRecipe
from tensor_dump.recipe import DumpRecipe
from tensor_dump.runtime import RUNTIME

_NAME = "model.layers.0.mlp.down_proj"
_ATTN = "model.layers.0.self_attn.attn"


class _Linear(torch.nn.Module):
    """A linear layer with a ``dump`` stage that writes every tensor."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.prefix = _NAME
        self.weight = weight
        self.bias = bias
        recipe = QuantLinearRecipe(activation=PipelineRecipe([DumpRecipe()]))
        self.quantized_linear = RUNTIME.prepare_linear(recipe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantized_linear(x, QuantizedTensor(tensor=self.weight), self.bias)


def _records(directory: Path, kind: str) -> list[dict[str, Any]]:
    return [record for record in read_manifest(directory) if record["record"] == kind]


def _rank_files(directory: Path) -> dict[str, bytes]:
    root = directory / "rank_0"
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class DumpAppendTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.directory = Path(tempfile.mkdtemp())
        self.weight = torch.randn(6, 4)
        self.bias = torch.randn(6)

    def _run(
        self,
        steps: int,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        **env: str,
    ) -> None:
        """One run of a prefill and ``steps - 1`` decodes."""
        env.setdefault("MAX_STEPS", "0")
        with dump_session(self.directory, **env) as forward:
            model = _Linear(
                self.weight if weight is None else weight,
                self.bias if bias is None else bias,
            )
            dtype = model.weight.dtype
            for step in range(steps):
                if step == 0:
                    forward.real({_ATTN: fake_metadata([3], [0])})
                    model(torch.randn(3, 4, dtype=dtype))
                else:
                    forward.real({_ATTN: fake_metadata([1], [2 + step])})
                    model(torch.randn(1, 4, dtype=dtype))

    def test_second_run_continues_step_ids(self) -> None:
        self._run(2)
        before = _rank_files(self.directory)
        self._run(2)
        runs = _records(self.directory, "run")
        self.assertEqual([(r["run"], r["first_step"]) for r in runs], [(0, 0), (1, 2)])
        self.assertEqual(
            [r["step"] for r in _records(self.directory, "step")], [0, 1, 2, 3]
        )
        linear = _records(self.directory, "linear")
        self.assertEqual([r["step"] for r in linear], [0, 1, 2, 3])
        self.assertEqual(
            linear[2]["tensors"]["input"]["file"], f"step_000002/{_NAME}/input.pt"
        )
        after = _rank_files(self.directory)
        for name, content in before.items():
            if name != "manifest.jsonl":
                self.assertEqual(after[name], content, name)

    def test_matching_weight_is_checked_not_written_again(self) -> None:
        self._run(1)
        weight_file = self.directory / "rank_0" / "weights" / _NAME / "weight.pt"
        content = weight_file.read_bytes()
        self._run(1)
        (record,) = _records(self.directory, "weight")
        self.assertEqual(record["step"], 0)
        self.assertEqual(sorted(record["tensors"]), ["bias", "weight"])
        entry = record["tensors"]["weight"]
        saved = load(self.directory, entry)
        self.assertEqual(
            entry["sha256"],
            hashlib.sha256(saved.contiguous().numpy().tobytes()).hexdigest(),
        )
        self.assertEqual(weight_file.read_bytes(), content)
        self.assertEqual(len(_records(self.directory, "linear")), 2)

    def test_different_weight_raises(self) -> None:
        cases = {
            "values": {"weight": torch.randn(6, 4)},
            "shape": {"weight": torch.randn(7, 4), "bias": torch.randn(7)},
            "dtype": {
                "weight": torch.randn(6, 4, dtype=torch.float64),
                "bias": torch.randn(6, dtype=torch.float64),
            },
            "bias": {"bias": torch.randn(6) + 1},
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                self.directory = Path(tempfile.mkdtemp())
                self._run(1)
                weights = self.directory / "rank_0" / "weights"
                content = _rank_files(self.directory)
                with self.assertRaisesRegex(RuntimeError, "use a new directory"):
                    self._run(1, **change)
                files = _rank_files(self.directory)
                for path in content:
                    if path.startswith("weights"):
                        self.assertEqual(files[path], content[path], path)
                self.assertEqual(len(list(weights.rglob("*.pt"))), 2)

    def test_weight_file_without_record_raises(self) -> None:
        stray = self.directory / "rank_0" / "weights" / _NAME / "weight.pt"
        stray.parent.mkdir(parents=True)
        stray.write_bytes(b"")
        with self.assertRaisesRegex(RuntimeError, "no weight record"):
            self._run(1)
        self.assertEqual(stray.read_bytes(), b"")

    def test_manifest_of_another_format_raises(self) -> None:
        manifest = self.directory / "rank_0" / "manifest.jsonl"
        manifest.parent.mkdir()
        manifest.write_text(json.dumps({"record": "run", "format_version": 1}) + "\n")
        with self.assertRaisesRegex(RuntimeError, "format 1"):
            self._run(1)

    def test_truncated_manifest_raises(self) -> None:
        self._run(1)
        manifest = self.directory / "rank_0" / "manifest.jsonl"
        with manifest.open("a") as file:
            file.write('{"record": "step", "st')
        with self.assertRaisesRegex(RuntimeError, "not JSON"):
            self._run(1)

    def test_manifest_without_final_newline_raises(self) -> None:
        self._run(1)
        manifest = self.directory / "rank_0" / "manifest.jsonl"
        manifest.write_bytes(manifest.read_bytes().rstrip(b"\n"))
        before = _rank_files(self.directory)
        with self.assertRaisesRegex(RuntimeError, "does not end with a newline"):
            self._run(1)
        self.assertEqual(_rank_files(self.directory), before)

    def test_missing_file_without_size_raises(self) -> None:
        self._run(1)
        records = read_manifest(self.directory)
        (linear,) = (record for record in records if record["record"] == "linear")
        entry = linear["tensors"]["input"]
        del entry["bytes"]
        root = self.directory / "rank_0"
        (root / "manifest.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        (root / entry["file"]).unlink()
        with self.assertRaisesRegex(RuntimeError, re.escape(entry["file"])):
            self._run(1)

    def test_budget_stops_new_steps_with_one_log(self) -> None:
        with self.assertLogs("tensor_dump._step", level="INFO") as logs:
            self._run(3, MAX_BYTES="1")
        self.assertEqual([r["step"] for r in _records(self.directory, "linear")], [0])
        budget_logs = [line for line in logs.output if "MAX_BYTES" in line]
        self.assertEqual(len(budget_logs), 1)

    def test_bytes_count_every_file_and_earlier_runs(self) -> None:
        self._run(2)
        files = _rank_files(self.directory)
        total = sum(len(content) for content in files.values())
        listed = sum(
            entry["bytes"]
            for record in read_manifest(self.directory)
            for entry in record.get("tensors", {}).values()
        )
        self.assertEqual(listed + len(files["manifest.jsonl"]), total)

        self._run(2, MAX_BYTES=str(total))
        self.assertEqual(_rank_files(self.directory), files)

        # Below the budget at the step's start, so the whole step is written.
        self._run(2, MAX_BYTES=str(total + 1))
        self.assertEqual(
            [r["step"] for r in _records(self.directory, "step")], [0, 1, 2]
        )


if __name__ == "__main__":
    unittest.main()
