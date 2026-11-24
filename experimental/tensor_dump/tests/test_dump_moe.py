# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE dumps from forward hooks on a stand-in for vLLM's ``MoERunner``, read
back as written and merged. CPU only."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from dump_test_utils import (
    dump_session,
    fake_metadata,
    fake_moe_config,
    FakeMoERunner,
    finalize,
    flashinfer_metadata,
    load,
    read_manifest,
)
from quant_suite_recipe.definition import QuantLinearRecipe
from tensor_dump import _moe, _session, _step
from tensor_dump._step import _VllmSettings
from tensor_dump.merge import merge
from tensor_dump.reader import TensorDump
from tensor_dump.runtime import RUNTIME

_LAYERS = ("model.layers.0.feed_forward.experts", "model.layers.1.feed_forward.experts")
_ATTN = "model.layers.0.self_attn.attn"
_EXPERTS = 6
_HIDDEN = 8
_TOP_K = 2
_ALL = "input,router_logits,topk_ids,topk_weights,gemm2,output"


class _Model(torch.nn.Module):
    """Two MoE layers, the second called with keywords, and a linear executor
    without a ``dump`` stage, whose construction starts the dump session as in
    vLLM."""

    def __init__(
        self, *configs: Any, monolithic: bool = True, hook: bool = True
    ) -> None:
        super().__init__()
        self.quantized_linear = RUNTIME.prepare_linear(QuantLinearRecipe())
        configs = configs or (None, None)
        self.layers = torch.nn.ModuleList(
            [
                FakeMoERunner(
                    layer, config, monolithic=monolithic, hook=hook, seed=index
                )
                for index, (layer, config) in enumerate(zip(_LAYERS, configs))
            ]
        )

    def forward(self, x: torch.Tensor, logits: torch.Tensor) -> list[torch.Tensor]:
        first, second = self.layers
        return [first(x, logits), second(hidden_states=x, router_logits=logits)]


def _inputs(tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(tokens, _HIDDEN), torch.randn(tokens, _EXPERTS)


def _run(
    forward: Any,
    model: _Model,
    q_lens: list[int],
    q_starts: list[int],
    *,
    padding: int = 0,
) -> tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]]:
    """One real forward of ``model`` over the requests' tokens and
    ``padding`` more rows; returns its inputs and outputs."""
    forward.real({_ATTN: fake_metadata(q_lens, q_starts)})
    inputs = _inputs(sum(q_lens) + padding)
    return inputs, model(*inputs)


def _records(directory: Path, kind: str = "moe", rank: int = 0) -> list[dict[str, Any]]:
    return [r for r in read_manifest(directory, rank) if r["record"] == kind]


def _hook(layer: FakeMoERunner) -> Any:
    return layer.routed_experts.quant_method.moe_kernel.fused_experts.post_gemm2_hook


class DumpMoETest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.directory = Path(tempfile.mkdtemp())

    def _session(self, **env: str) -> Any:
        return dump_session(self.directory, MOE=r".*\.experts$", **env)

    def test_default_tensors_are_the_layer_and_kernel_values(self) -> None:
        with self._session() as forward:
            model = _Model()
            (x, logits), outputs = _run(forward, model, [3], [0])
        manifest = read_manifest(self.directory)
        self.assertEqual(manifest[0]["options"]["moe"], r".*\.experts$")
        records = _records(self.directory)
        self.assertEqual([r["layer"] for r in records], list(_LAYERS))
        for record, layer, output in zip(records, model.layers, outputs):
            with self.subTest(layer=layer.layer_name):
                ids, weights, _ = layer.last_routes
                expected = {
                    "input": x,
                    "router_logits": logits,
                    "topk_ids": ids.to(torch.int16),
                    "topk_weights": weights,
                    "output": output,
                }
                self.assertEqual(sorted(record["tensors"]), sorted(expected))
                self.assertIsNone(record["replicated_on"])
                self.assertEqual(
                    record["replicated_tensors"], ["input", "output", "router_logits"]
                )
                self.assertEqual(record["topk_weights_source"], "post_gemm2_hook")
                for name, entry in record["tensors"].items():
                    self.assertEqual(
                        entry["file"], f"step_000000/{layer.layer_name}/{name}.pt"
                    )
                    self.assertTrue(
                        torch.equal(load(self.directory, entry), expected[name]), name
                    )

    def test_gemm2_rows_are_route_major_and_outputs_unchanged(self) -> None:
        x, logits = _inputs(4)
        with dump_session(None):
            expected = _Model()(x, logits)
        with self._session(MOE_TENSORS="gemm2,output") as forward:
            model = _Model()
            forward.real({_ATTN: fake_metadata([4], [0])})
            outputs = model(x, logits)
        for output, reference in zip(outputs, expected):
            self.assertTrue(torch.equal(output, reference))
        for record, layer in zip(_records(self.directory), model.layers):
            with self.subTest(layer=layer.layer_name):
                gemm2 = load(self.directory, record["tensors"]["gemm2"])
                self.assertEqual(list(gemm2.shape), [4, _TOP_K, _HIDDEN])
                self.assertTrue(torch.equal(gemm2, layer.last_routes[2]))

    def test_gemm2_rows_reach_the_host_in_chunks(self) -> None:
        # One token per chunk: the rows are gathered and copied token by token.
        with patch.object(_moe, "_GEMM2_CHUNK_BYTES", 1):
            with self._session(MOE_TENSORS="gemm2") as forward:
                model = _Model()
                _run(forward, model, [5], [0])
        for record, layer in zip(_records(self.directory), model.layers):
            gemm2 = load(self.directory, record["tensors"]["gemm2"])
            self.assertTrue(torch.equal(gemm2, layer.last_routes[2]))

    def test_rows_are_cut_to_the_steps_tokens(self) -> None:
        with self._session(MOE_TENSORS=_ALL) as forward:
            model = _Model()
            (x, _), _ = _run(forward, model, [2, 1], [0, 5], padding=3)
        for record in _records(self.directory):
            with self.subTest(layer=record["layer"]):
                shapes = {name: e["shape"][0] for name, e in record["tensors"].items()}
                self.assertEqual(set(shapes.values()), {3})
        record = _records(self.directory)[0]
        self.assertTrue(
            torch.equal(load(self.directory, record["tensors"]["input"]), x[:3])
        )

    def test_dummy_forward_and_unselected_steps_write_nothing(self) -> None:
        with self._session(STEPS="1", MOE_TENSORS=_ALL) as forward:
            model = _Model()
            forward.dummy()
            model(*_inputs(3))
            _run(forward, model, [3], [0])
            self.assertFalse(self.directory.exists() and any(self.directory.iterdir()))
            _, outputs = _run(forward, model, [1], [3])
        records = _records(self.directory)
        self.assertEqual({r["step"] for r in records}, {1})
        self.assertTrue(
            torch.equal(
                load(self.directory, records[0]["tensors"]["output"]), outputs[0]
            )
        )
        ids = load(self.directory, records[0]["tensors"]["topk_ids"])
        self.assertTrue(
            torch.equal(ids, model.layers[0].last_routes[0].to(torch.int16))
        )

    def test_layout_is_recorded_once_per_layer(self) -> None:
        with self._session(MAX_STEPS="0") as forward:
            model = _Model()
            _run(forward, model, [3], [0])
            _run(forward, model, [1], [3])
        layouts = _records(self.directory, "moe_layout")
        self.assertEqual([r["layer"] for r in layouts], list(_LAYERS))
        layout = layouts[0]
        self.assertEqual(layout["step"], 0)
        self.assertEqual(
            {key: layout[key] for key in ("num_experts", "top_k", "hidden_size")},
            {"num_experts": _EXPERTS, "top_k": _TOP_K, "hidden_size": _HIDDEN},
        )
        self.assertEqual(
            (
                layout["tp_rank"],
                layout["tp_size"],
                layout["intermediate_size_per_partition"],
            ),
            (0, 1, 4),
        )
        self.assertEqual(
            (layout["routing_method"], layout["scoring_func"], layout["activation"]),
            ("MiniMax2", "sigmoid", "silu"),
        )
        self.assertTrue(layout["renormalize"])
        self.assertTrue(layout["has_e_score_correction_bias"])
        self.assertEqual(layout["post_gemm2_hook"], "function")
        self.assertTrue(layout["output_reduced"])
        self.assertNotIn("replicated_tensors", layout)
        self.assertNotIn("topk_weights_source", layout)
        bias = load(self.directory, layout["tensors"]["e_score_correction_bias"])
        self.assertTrue(
            torch.equal(bias, model.layers[0].routed_experts.e_score_correction_bias)
        )
        self.assertEqual(
            layout["tensors"]["e_score_correction_bias"]["file"],
            f"moe_layouts/{_LAYERS[0]}/e_score_correction_bias.pt",
        )

    def test_replicated_tensors_are_written_by_tp_rank_0_only(self) -> None:
        configs = [fake_moe_config(tp_rank=1, tp_size=2)] * 2
        with self._session(MOE_TENSORS=_ALL) as forward:
            model = _Model(*configs)
            _run(forward, model, [3], [0])
        for record in _records(self.directory):
            self.assertEqual(
                sorted(record["tensors"]), ["gemm2", "topk_ids", "topk_weights"]
            )
            self.assertEqual(record["replicated_on"], 0)

    def test_tp_rank_0_writes_every_tensor(self) -> None:
        configs = [fake_moe_config(tp_rank=0, tp_size=2)] * 2
        with self._session(MOE_TENSORS=_ALL) as forward:
            model = _Model(*configs)
            _run(forward, model, [3], [0])
        for record in _records(self.directory):
            self.assertEqual(sorted(record["tensors"]), sorted(_ALL.split(",")))

    def test_unreduced_output_is_written_on_every_rank(self) -> None:
        config = fake_moe_config(tp_rank=1, tp_size=2)
        config.skip_final_all_reduce = True
        with self._session(MOE_TENSORS="input,output") as forward:
            model = _Model(config, config)
            _run(forward, model, [3], [0])
        for record in _records(self.directory):
            self.assertEqual(sorted(record["tensors"]), ["output"])
            self.assertEqual(record["replicated_tensors"], ["input"])
        (layout, _) = _records(self.directory, "moe_layout")
        self.assertFalse(layout["output_reduced"])

    def test_modular_kernel_records_router_ids_and_weights(self) -> None:
        with self._session(MOE_TENSORS="topk_ids,topk_weights") as forward:
            model = _Model(monolithic=False, hook=False)
            _run(forward, model, [3], [0])
        for record, layer in zip(_records(self.directory), model.layers):
            ids, weights, _ = layer.last_routes
            self.assertTrue(
                torch.equal(load(self.directory, record["tensors"]["topk_ids"]), ids)
            )
            self.assertTrue(
                torch.equal(
                    load(self.directory, record["tensors"]["topk_weights"]), weights
                )
            )
        record = _records(self.directory)[0]
        self.assertEqual(record["topk_weights_source"], "experts_call")

    def test_gemm2_without_a_hook_raises(self) -> None:
        with self._session(MOE_TENSORS="gemm2") as forward:
            model = _Model(monolithic=False, hook=False)
            with self.assertRaisesRegex(NotImplementedError, "post_gemm2_hook"):
                _run(forward, model, [3], [0])

    def test_kernel_without_weights_omits_them(self) -> None:
        with self._session() as forward:
            model = _Model(hook=False)
            with self.assertLogs("tensor_dump._moe", "WARNING"):
                _run(forward, model, [3], [0])
        for record in _records(self.directory):
            self.assertNotIn("topk_weights", record["tensors"])
            self.assertIn("topk_ids", record["tensors"])

    def test_unsupported_variants_raise(self) -> None:
        def ep() -> Any:
            return fake_moe_config(use_ep=True, ep_size=2)

        def sequence_parallel() -> Any:
            return fake_moe_config(is_sequence_parallel=True)

        cases = {
            "expert parallelism": (ep, "", None),
            "sequence-parallel": (sequence_parallel, "", None),
            # The default tensors include output and router_logits.
            "shared experts": (fake_moe_config, "", "shared_experts"),
            "router logits": (fake_moe_config, "", "gate"),
            "transforms": (fake_moe_config, "input", "routed_input_transform"),
        }
        for reason, (config, tensors, attribute) in cases.items():
            with self.subTest(reason=reason):
                env = {"MOE_TENSORS": tensors} if tensors else {}
                with self._session(**env) as forward:
                    model = _Model(config(), config())
                    if attribute is not None:
                        setattr(model.layers[0], attribute, torch.nn.Identity())
                    with self.assertRaisesRegex(NotImplementedError, reason):
                        _run(forward, model, [2], [0])

    def test_an_existing_routing_capture_raises(self) -> None:
        with self._session() as forward:
            model = _Model()
            model.layers[0].routed_experts.routing_sink = SimpleNamespace()
            with self.assertRaisesRegex(NotImplementedError, "routed-experts capture"):
                _run(forward, model, [2], [0])

    def test_a_layer_name_without_one_index_raises(self) -> None:
        with self._session() as forward:
            model = _Model()
            with patch.object(
                _moe, "_bind_routing_capture", side_effect=AssertionError("index")
            ):
                with self.assertRaisesRegex(NotImplementedError, "cannot be bound"):
                    _run(forward, model, [2], [0])

    def test_experts_run_without_the_layer_forward_raise(self) -> None:
        with self._session(MOE_TENSORS="gemm2", MAX_STEPS="0") as forward:
            model = _Model()
            _run(forward, model, [2], [0])
            forward.real({_ATTN: fake_metadata([1], [2])})
            layer = model.layers[0]
            unfinalized = SimpleNamespace(
                gemm2_permuted=torch.zeros(2, _HIDDEN),
                expert_weights=torch.ones(1, _TOP_K),
                expanded_idx_to_permuted_idx=torch.zeros(1, _TOP_K, dtype=torch.int32),
            )
            with self.assertRaisesRegex(RuntimeError, "fast path"):
                _hook(layer)(unfinalized, torch.empty(1, _HIDDEN))

    def test_unmatched_pattern_warns_once(self) -> None:
        with dump_session(self.directory, MOE="nothing", MAX_STEPS="0") as forward:
            model = _Model()
            session = _session._get()
            assert session is not None
            with self.assertLogs("tensor_dump._moe", "WARNING") as logs:
                for start in range(3):
                    _run(forward, model, [1], [start + 1])
                    # As the linear executors do at every forward.
                    session.tracker.current()
        self.assertEqual(len(logs.records), 1)
        self.assertIn("matched no vLLM MoERunner", logs.output[0])

    def test_ranks_without_tp_rank_0_raise(self) -> None:
        with self._session(RANKS="1"):
            with patch.object(_step, "_distributed_rank", return_value=(1, 2)):
                with patch.object(_step, "_tensor_parallel", return_value=(1, 2)):
                    with self.assertRaisesRegex(RuntimeError, "rank 0"):
                        _Model()

    def test_context_parallelism_raises(self) -> None:
        settings = _VllmSettings(context_parallel=True)
        with dump_session(self.directory, settings=settings, MOE=".*"):
            with self.assertRaisesRegex(NotImplementedError, "context parallelism"):
                _Model()

    def test_reset_restores_the_layer(self) -> None:
        with self._session(MOE_TENSORS=_ALL) as forward:
            model = _Model()
            _run(forward, model, [2], [0])
            self.assertIsNot(_hook(model.layers[0]), finalize)
        self.assertIs(_hook(model.layers[0]), finalize)
        self.assertIsNone(model.layers[0].routed_experts.routing_sink)
        self.assertEqual(model.layers[0]._forward_hooks, {})

    def test_unset_pattern_hooks_no_layer(self) -> None:
        with dump_session(self.directory) as forward:
            model = _Model()
            _run(forward, model, [2], [0])
            self.assertEqual(
                [len(layer._forward_hooks) for layer in model.layers], [0, 0]
            )
        self.assertIsNone(model.layers[0].routed_experts.routing_sink)


class MoEReadAndMergeTest(unittest.TestCase):
    """Two steps: a decode and a prompt in one step, decodes first, then two
    decodes."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        root = Path(tempfile.mkdtemp())
        self.source = root / "dump"
        self.destination = root / "merged"
        self.outputs: list[list[torch.Tensor]] = []
        self.inputs: list[torch.Tensor] = []
        with dump_session(self.source, MOE=".*", MOE_TENSORS=_ALL, MAX_STEPS="0") as f:
            self.model = _Model()
            for q_lens, q_starts in (([1, 3], [4, 0]), ([1, 1], [5, 3])):
                f.real({_ATTN: flashinfer_metadata(q_lens, q_starts)})
                x, logits = _inputs(sum(q_lens))
                self.inputs.append(x)
                self.outputs.append(self.model(x, logits))

    def _source_files(self) -> list[str]:
        root = self.source / "rank_0"
        if not root.exists():
            return []
        return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())

    def test_reader_returns_the_written_tensors(self) -> None:
        dump = TensorDump(self.source)
        self.assertEqual(dump.moe_layers(), _LAYERS)
        moe = dump.moe(_LAYERS[1], 1)
        self.assertTrue(torch.equal(moe.input, self.inputs[1]))
        self.assertTrue(torch.equal(moe.output, self.outputs[1][1]))
        self.assertEqual(list(moe.gemm2.shape), [2, _TOP_K, _HIDDEN])
        self.assertEqual(moe.layout.fields["top_k"], _TOP_K)
        self.assertTrue(
            torch.equal(
                moe.layout.e_score_correction_bias.load(),
                self.model.layers[1].routed_experts.e_score_correction_bias,
            )
        )
        with self.assertRaises(KeyError):
            dump.moe(_LAYERS[0], 2)

    def test_phase_tensors_follow_each_requests_rows(self) -> None:
        dump = TensorDump(self.source)
        requests = dump.requests()
        # The prompt of step 0 is the second request in its batch, after a
        # decode, and continues as the decode of step 1's second row.
        prompt = next(r for r in requests if r.slices[0].q_start_pos == 0)
        prefill = dump.phase_tensors(_LAYERS[0], prompt, "prefill")
        self.assertTrue(torch.equal(prefill["input"], self.inputs[0][1:4]))
        decode = dump.phase_tensors(_LAYERS[0], prompt, "decode")
        self.assertTrue(torch.equal(decode["output"], self.outputs[1][0][1:2]))
        self.assertEqual(list(decode["gemm2"].shape), [1, _TOP_K, _HIDDEN])

    def test_merged_dump_reads_alike(self) -> None:
        merge(self.source, self.destination)
        written, merged = TensorDump(self.source), TensorDump(self.destination)
        self.assertEqual(merged.moe_layers(), _LAYERS)
        for layer in _LAYERS:
            self.assertEqual(
                dict(merged.moe_layout(layer).fields),
                dict(written.moe_layout(layer).fields),
            )
            for step in (0, 1):
                with self.subTest(layer=layer, step=step):
                    a, b = written.moe(layer, step), merged.moe(layer, step)
                    for name in _ALL.split(","):
                        self.assertTrue(torch.equal(getattr(a, name), getattr(b, name)))
        manifest = json.loads(
            (self.destination / "rank_0" / "manifest.json").read_text()
        )
        layout = manifest["moe_layouts"][_LAYERS[0]]
        self.assertEqual(layout["file"], f"moe_layouts/{_LAYERS[0]}.pt")
        request = manifest["requests"][0]
        kinds = {
            entry["kind"]
            for phase in request["phases"].values()
            for entry in phase["layers"].values()
        }
        self.assertEqual(kinds, {"moe"})

    def test_delete_source_leaves_no_moe_file(self) -> None:
        self.assertTrue(
            any(name.startswith("moe_layouts/") for name in self._source_files())
        )
        merge(self.source, self.destination, delete_source=True)
        self.assertEqual(self._source_files(), [])
        self.assertFalse((self.source / "rank_0").exists())
        merged = TensorDump(self.destination)
        self.assertTrue(torch.equal(merged.moe(_LAYERS[0], 0).input, self.inputs[0]))

    def test_unknown_record_kind_raises(self) -> None:
        manifest = self.source / "rank_0" / "manifest.jsonl"
        with manifest.open("a") as stream:
            stream.write(json.dumps({"record": "router", "step": 0}) + "\n")
        with self.assertRaisesRegex(ValueError, "unknown record kind"):
            merge(self.source, self.destination, delete_source=True)
        self.assertTrue(manifest.exists())

    def test_format_2_dumps_stay_readable(self) -> None:
        manifest = self.source / "rank_0" / "manifest.jsonl"
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        records[0]["format_version"] = 2
        manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
        self.assertEqual(TensorDump(self.source).moe_layers(), _LAYERS)

    def test_appended_run_checks_the_layout(self) -> None:
        # Other tensors: the layout does not depend on them.
        with dump_session(self.source, MOE=".*", MOE_TENSORS="topk_ids,gemm2") as f:
            model = _Model()
            _run(f, model, [2], [0])
        self.assertEqual(len(_records(self.source, "moe_layout")), 2)
        self.assertEqual(
            sorted(_records(self.source)[-1]["tensors"]), ["gemm2", "topk_ids"]
        )
        with dump_session(self.source, MOE=".*") as forward:
            model = _Model()
            model.layers[0].routed_experts.e_score_correction_bias += 1
            with self.assertRaisesRegex(RuntimeError, "use a new directory"):
                _run(forward, model, [2], [0])


class MoETensorParallelReadTest(unittest.TestCase):
    """Two tensor-parallel ranks in one dump: rank 0 writes one step, as when
    its byte budget runs out first, and rank 1 writes two."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.directory = Path(tempfile.mkdtemp())
        self.inputs: list[torch.Tensor] = []
        for rank, max_steps in ((0, "1"), (1, "0")):
            configs = [fake_moe_config(tp_rank=rank, tp_size=2)] * 2
            with dump_session(
                self.directory, MOE=".*", MOE_TENSORS=_ALL, MAX_STEPS=max_steps
            ) as forward:
                with patch.object(_step, "_distributed_rank", return_value=(rank, 2)):
                    model = _Model(*configs)
                    torch.manual_seed(1)
                    (x, _), _ = _run(forward, model, [3], [0])
                    _run(forward, model, [1], [3])
            self.inputs.append(x)

    def test_other_ranks_read_the_replicated_tensors_of_rank_0(self) -> None:
        dump = TensorDump(self.directory)
        self.assertEqual(dump.ranks, (0, 1))
        moe = dump.moe(_LAYERS[0], 0, rank=1)
        self.assertEqual(moe.replicated_tensors, ("input", "output", "router_logits"))
        self.assertTrue(torch.equal(moe.input, self.inputs[1]))
        self.assertIsNotNone(moe.gemm2)

    def test_a_rank_with_only_replicated_tensors_reads_them_from_rank_0(
        self,
    ) -> None:
        root = Path(tempfile.mkdtemp())
        source, merged = root / "dump", root / "merged"
        inputs: list[torch.Tensor] = []
        for rank in (0, 1):
            configs = [fake_moe_config(tp_rank=rank, tp_size=2)] * 2
            with dump_session(
                source, MOE=".*", MOE_TENSORS="input,output", MAX_STEPS="0"
            ) as forward:
                with patch.object(_step, "_distributed_rank", return_value=(rank, 2)):
                    model = _Model(*configs)
                    torch.manual_seed(1)
                    (x, _), _ = _run(forward, model, [3], [0])
            inputs.append(x)
        for record in _records(source, rank=1):
            self.assertEqual(record["tensors"], {})
        merge(source, merged)
        for dump in (TensorDump(source), TensorDump(merged)):
            moe = dump.moe(_LAYERS[0], 0, rank=1)
            self.assertEqual(moe.replicated_tensors, ("input", "output"))
            self.assertTrue(torch.equal(moe.input, inputs[0]))
            self.assertIsNotNone(moe.output)
            self.assertIsNone(moe.gemm2)

    def test_a_step_rank_0_did_not_write_raises(self) -> None:
        dump = TensorDump(self.directory)
        with self.assertRaisesRegex(ValueError, "byte budget"):
            dump.moe(_LAYERS[0], 1, rank=1)


if __name__ == "__main__":
    unittest.main()
