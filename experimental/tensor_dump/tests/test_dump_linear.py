# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``DumpLinear`` writes what its ``dump`` stage selects, and computes what the
built-in executor computes. CPU only."""

import contextlib
import dataclasses
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch
from dump_test_utils import (
    dump_session,
    fake_metadata,
    FakeFlashInferMetadata,
    FakeForward,
    FakeForwardContext,
    FakeWrapperMetadata,
    flashinfer_metadata,
    generic_metadata,
    load,
    read_manifest,
)
from quant_suite_recipe.api import QuantizedTensor
from quant_suite_recipe.definition import PipelineRecipe, QuantLinearRecipe
from quant_suite_recipe.runtime import QuantRecipeRuntime
from tensor_dump import _session
from tensor_dump._step import _settings_from, _VllmSettings
from tensor_dump.recipe import DumpRecipe
from tensor_dump.runtime import RUNTIME

_NAME = "model.layers.0.mlp.down_proj"
_ATTN = "model.layers.0.self_attn.attn"


class _Linear(torch.nn.Module):
    """The attributes of a vLLM linear layer that the dump reads."""

    def __init__(self, executor: torch.nn.Module) -> None:
        super().__init__()
        self.prefix = _NAME
        self.weight = torch.randn(6, 4)
        self.bias = torch.randn(6)
        self.quantized_linear = executor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantized_linear(x, QuantizedTensor(tensor=self.weight), self.bias)


class _Model(torch.nn.Module):
    def __init__(self, executor: torch.nn.Module) -> None:
        super().__init__()
        self.layer = _Linear(executor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class _TwoLayers(torch.nn.Module):
    """Two linear layers that write their input; ``second`` picks the one a
    forward runs."""

    def __init__(self, second_name: str) -> None:
        super().__init__()
        self.first = _model(tensors=("input",)).layer
        self.second = _model(tensors=("input",)).layer
        self.second.prefix = second_name

    def forward(self, x: torch.Tensor, second: bool = False) -> torch.Tensor:
        return self.second(x) if second else self.first(x)


def _model(tensors: tuple[str, ...] | None = ("input", "weight", "output")) -> _Model:
    stages = [] if tensors is None else [DumpRecipe(tensors=tensors)]
    recipe = QuantLinearRecipe(activation=PipelineRecipe(stages))
    return _Model(RUNTIME.prepare_linear(recipe))


def _prefill(
    forward: FakeForward, model: torch.nn.Module, tokens: int = 3
) -> torch.Tensor:
    forward.real({_ATTN: fake_metadata([tokens], [0])})
    return model(torch.randn(tokens, 4))


def _decode(forward: FakeForward, model: _Model, position: int = 3) -> torch.Tensor:
    forward.real({_ATTN: fake_metadata([1], [position])})
    return model(torch.randn(1, 4))


def _vllm_config(use_v2_model_runner: bool) -> SimpleNamespace:
    """The fields of vLLM's config that the dump reads."""
    return SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        speculative_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )


def _records(directory: Path, kind: str) -> list[dict[str, object]]:
    return [record for record in read_manifest(directory) if record["record"] == kind]


class DumpLinearTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.directory = Path(tempfile.mkdtemp())

    def test_output_equals_builtin_executor(self) -> None:
        with dump_session(self.directory) as forward:
            model = _model()
            x = torch.randn(3, 4)
            forward.real({_ATTN: fake_metadata([3], [0])})
            output = model(x)
        layer = model.layer
        builtin = QuantRecipeRuntime().prepare_linear(QuantLinearRecipe())
        expected = builtin(x, QuantizedTensor(tensor=layer.weight), layer.bias)
        self.assertTrue(torch.equal(output, expected))

    def test_files_match_manifest(self) -> None:
        with dump_session(self.directory) as forward:
            model = _model()
            x = torch.randn(3, 4)
            forward.real({_ATTN: fake_metadata([3], [0])})
            output = model(x)
        records = read_manifest(self.directory)
        self.assertEqual(
            [record["record"] for record in records],
            ["run", "step", "linear", "weight"],
        )
        (linear,) = _records(self.directory, "linear")
        (weight,) = _records(self.directory, "weight")
        self.assertEqual(linear["layer"], _NAME)
        expected = {
            "input": x,
            "output": output,
            "weight": model.layer.weight,
            "bias": model.layer.bias,
        }
        for record in (linear, weight):
            for name, entry in record["tensors"].items():
                with self.subTest(tensor=name):
                    saved = load(self.directory, entry)
                    self.assertEqual(entry["shape"], list(saved.shape))
                    self.assertEqual(entry["dtype"], str(saved.dtype).split(".")[-1])
                    self.assertTrue(torch.equal(saved, expected[name]))

    def test_weight_is_written_once(self) -> None:
        with dump_session(self.directory, MAX_STEPS="2") as forward:
            model = _model()
            _prefill(forward, model)
            _decode(forward, model)
        self.assertEqual(len(_records(self.directory, "linear")), 2)
        (weight,) = _records(self.directory, "weight")
        self.assertEqual(weight["step"], 0)

    def test_dummy_forward_writes_nothing_and_takes_no_step(self) -> None:
        with dump_session(self.directory) as forward:
            model = _model()
            forward.dummy()
            model(torch.randn(5, 4))
            self.assertEqual(list(self.directory.iterdir()), [])
            _prefill(forward, model)
        (step,) = _records(self.directory, "step")
        self.assertEqual(step["step"], 0)

    def test_forward_without_kv_slots_writes_nothing_and_takes_no_step(self) -> None:
        # The mixed batch of vLLM's FlashInfer warmup: 8 decodes, one prefill.
        warmup = dataclasses.replace(
            flashinfer_metadata([1] * 8 + [8], [0] * 8 + [1]),
            slot_mapping=torch.full((16,), -1),
        )
        with dump_session(self.directory) as forward:
            model = _model()
            forward.real({_ATTN: warmup})
            model(torch.randn(16, 4))
            self.assertEqual(list(self.directory.iterdir()), [])
            # Padding after the request's tokens has no slot either.
            forward.real(
                {
                    _ATTN: dataclasses.replace(
                        fake_metadata([3], [0]),
                        slot_mapping=torch.tensor([0, 1, 2, -1]),
                    )
                }
            )
            model(torch.randn(3, 4))
        (step,) = _records(self.directory, "step")
        self.assertEqual(step["step"], 0)
        self.assertEqual(
            [(r["q_len"], r["q_start_pos"]) for r in step["requests"]], [(3, 0)]
        )
        self.assertEqual(len(_records(self.directory, "linear")), 1)

    def test_forward_without_kv_slots_takes_a_step_under_context_parallelism(
        self,
    ) -> None:
        # A rank may hold the KV-cache slots of none of a step's tokens.
        settings = _VllmSettings(context_parallel=True)
        metadata = dataclasses.replace(
            fake_metadata([1], [4]), slot_mapping=torch.tensor([-1])
        )
        with dump_session(self.directory, settings=settings) as forward:
            model = _model()
            forward.real({_ATTN: metadata})
            model(torch.randn(1, 4))
        (step,) = _records(self.directory, "step")
        self.assertEqual(step["step"], 0)

    def test_context_parallel_is_read_from_the_config(self) -> None:
        config = _vllm_config(use_v2_model_runner=False)
        self.assertFalse(_settings_from(config, None).context_parallel)
        for name in ("decode_context_parallel_size", "prefill_context_parallel_size"):
            config.parallel_config = SimpleNamespace(**{name: 2})
            self.assertTrue(_settings_from(config, None).context_parallel)
        config.parallel_config = SimpleNamespace(
            decode_context_parallel_size=None, prefill_context_parallel_size=None
        )
        self.assertFalse(_settings_from(config, None).context_parallel)
        self.assertIsNone(_settings_from(None, None).context_parallel)

    def test_disabled_writes_nothing(self) -> None:
        # Every other variable is set and the working directory is the would-be
        # output root, so a write anywhere relative to it would show up.
        with (
            contextlib.chdir(self.directory),
            dump_session(None, STEPS="0:", MAX_STEPS="0") as forward,
        ):
            model = _model()
            _prefill(forward, model)
            _decode(forward, model)
            self.assertIsNone(_session._get())
        forward.provider.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_layer_without_dump_stage_writes_no_tensor(self) -> None:
        with dump_session(self.directory) as forward:
            model = _model(tensors=None)
            _prefill(forward, model)
        self.assertEqual(_records(self.directory, "linear"), [])
        self.assertEqual(_records(self.directory, "weight"), [])
        self.assertEqual(
            sorted(path.name for path in (self.directory / "rank_0").iterdir()),
            ["manifest.jsonl"],
        )

    def test_single_token_requests_are_decode(self) -> None:
        with dump_session(self.directory) as forward:
            model = _model()
            forward.real({_ATTN: fake_metadata([1, 1], [4, 7])})
            model(torch.randn(2, 4))
        (step,) = _records(self.directory, "step")
        self.assertEqual(step["phase"], "decode")
        self.assertEqual(
            [(r["phase"], r["token_start"]) for r in step["requests"]],
            [("decode", 0), ("decode", 1)],
        )

    def test_step_filters(self) -> None:
        cases = {
            "max_steps": ({"MAX_STEPS": "1"}, [0]),
            "phase": ({"PHASE": "decode", "MAX_STEPS": "0"}, [1, 2]),
            "steps": ({"STEPS": "1:", "MAX_STEPS": "0"}, [1, 2]),
            "steps_and_cap": ({"STEPS": "0,2", "MAX_STEPS": "1"}, [0]),
        }
        for name, (env, expected_steps) in cases.items():
            with self.subTest(case=name):
                directory = Path(tempfile.mkdtemp())
                with dump_session(directory, **env) as forward:
                    model = _model()
                    _prefill(forward, model)
                    _decode(forward, model, position=3)
                    _decode(forward, model, position=4)
                steps = [record["step"] for record in _records(directory, "linear")]
                self.assertEqual(steps, expected_steps)

    def test_unselected_rank_writes_nothing(self) -> None:
        with dump_session(self.directory, RANKS="1") as forward:
            model = _model()
            _prefill(forward, model)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_metadata_without_backend_fields(self) -> None:
        # Only query_start_loc and seq_lens, as every attention backend has.
        with dump_session(self.directory) as forward:
            model = _model()
            forward.real({_ATTN: generic_metadata([1, 1], [4, 7])})
            model(torch.randn(2, 4))
        (step,) = _records(self.directory, "step")
        self.assertEqual(step["num_actual_tokens"], 2)
        self.assertEqual(
            [(r["q_len"], r["kv_len"], r["q_start_pos"]) for r in step["requests"]],
            [(1, 5, 4), (1, 8, 7)],
        )
        (linear,) = _records(self.directory, "linear")
        self.assertEqual(linear["tensors"]["input"]["shape"], [2, 4])

    def test_metadata_entry_without_request_fields_is_skipped(self) -> None:
        with dump_session(self.directory) as forward:
            model = _model()
            forward.real(
                {"mamba": SimpleNamespace(), _ATTN: generic_metadata([3], [0])}
            )
            model(torch.randn(3, 4))
        (step,) = _records(self.directory, "step")
        self.assertEqual(step["requests"][0]["q_len"], 3)

    def test_metadata_without_request_fields_raises(self) -> None:
        with dump_session(self.directory) as forward:
            model = _model()
            forward.real({_ATTN: SimpleNamespace(slot_mapping=None)})
            with self.assertRaisesRegex(RuntimeError, "SimpleNamespace"):
                model(torch.randn(3, 4))

    def test_flashinfer_metadata_requests(self) -> None:
        # (q_len, kv_len, q_start_pos, token_start, phase) per request.
        cases = {
            "decode_only": (
                flashinfer_metadata([1, 1], [4, 7]),
                "decode",
                [(1, 5, 4, 0, "decode"), (1, 8, 7, 1, "decode")],
            ),
            "ragged_decode": (
                flashinfer_metadata([1, 1], [4, 7], ragged_decode=True),
                "decode",
                [(1, 5, 4, 0, "decode"), (1, 8, 7, 1, "decode")],
            ),
            "prefill_only": (
                flashinfer_metadata([3, 2], [0, 0]),
                "prefill",
                [(3, 3, 0, 0, "prefill"), (2, 2, 0, 3, "prefill")],
            ),
            "mixed_decodes_first": (
                flashinfer_metadata([1, 1, 3], [4, 7, 0]),
                "mixed",
                [
                    (1, 5, 4, 0, "decode"),
                    (1, 8, 7, 1, "decode"),
                    (3, 3, 0, 2, "prefill"),
                ],
            ),
            "chunked_prefill": (
                flashinfer_metadata([1, 4, 2], [9, 6, 0]),
                "mixed",
                [
                    (1, 10, 9, 0, "decode"),
                    (4, 10, 6, 1, "prefill"),
                    (2, 2, 0, 5, "prefill"),
                ],
            ),
        }
        for name, (metadata, phase, expected) in cases.items():
            with self.subTest(case=name):
                directory = Path(tempfile.mkdtemp())
                with dump_session(directory) as forward:
                    model = _model()
                    forward.real({_ATTN: metadata})
                    model(torch.randn(metadata.num_actual_tokens, 4))
                (step,) = _records(directory, "step")
                self.assertEqual(step["phase"], phase)
                self.assertEqual(step["num_actual_tokens"], metadata.num_actual_tokens)
                self.assertEqual(
                    [r["request"] for r in step["requests"]],
                    list(range(len(expected))),
                )
                self.assertEqual(
                    [
                        (
                            r["q_len"],
                            r["kv_len"],
                            r["q_start_pos"],
                            r["token_start"],
                            r["phase"],
                        )
                        for r in step["requests"]
                    ],
                    expected,
                )

    def test_flashinfer_paths_without_lengths_raise(self) -> None:
        def native_prefill() -> FakeFlashInferMetadata:
            metadata = flashinfer_metadata([1, 3], [4, 0])
            metadata.prefill = FakeWrapperMetadata()
            return metadata

        def native_decode() -> FakeFlashInferMetadata:
            metadata = flashinfer_metadata([1, 3], [4, 0])
            metadata.decode = FakeWrapperMetadata()
            return metadata

        def context_parallel_decode() -> FakeFlashInferMetadata:
            metadata = flashinfer_metadata([1], [4])
            metadata.decode.dcp_query_start_loc = torch.tensor([0, 1])
            return metadata

        def multi_token_decode() -> FakeFlashInferMetadata:
            # Two decodes of two query tokens each, without query offsets.
            metadata = flashinfer_metadata([1, 1], [4, 7])
            metadata.decode.q_len_per_req = 2
            metadata.num_decode_tokens = 4
            metadata.num_actual_tokens = 4
            return metadata

        def cascade() -> FakeFlashInferMetadata:
            metadata = flashinfer_metadata([3], [0])
            metadata.use_cascade = True
            return metadata

        def counts_disagree() -> FakeFlashInferMetadata:
            metadata = flashinfer_metadata([1, 1], [4, 7])
            metadata.num_decode_tokens = 3
            return metadata

        cases = {
            "native_prefill": (
                native_prefill,
                "FakeWrapperMetadata.*use_trtllm_attention=1",
            ),
            "native_decode": (
                native_decode,
                "FakeWrapperMetadata.*--attention-backend FLASH_ATTN",
            ),
            "context_parallel": (
                context_parallel_decode,
                "--decode-context-parallel-size 1",
            ),
            "multi_token_decode": (
                multi_token_decode,
                "q_cu_seq_lens is None and its q_len_per_req is 2",
            ),
            "cascade": (cascade, "--disable-cascade-attn"),
            "counts_disagree": (counts_disagree, "not over its 3 tokens"),
        }
        for name, (make, message) in cases.items():
            with self.subTest(case=name):
                metadata = make()
                with dump_session(Path(tempfile.mkdtemp())) as forward:
                    model = _model()
                    forward.real({_ATTN: metadata})
                    with self.assertRaisesRegex(RuntimeError, message):
                        model(torch.randn(metadata.num_actual_tokens, 4))

    def test_query_start_loc_metadata_is_preferred_to_flashinfer(self) -> None:
        with dump_session(self.directory) as forward:
            model = _model()
            forward.real(
                {
                    "flashinfer": flashinfer_metadata([3], [0]),
                    _ATTN: generic_metadata([3], [4]),
                }
            )
            model(torch.randn(3, 4))
        (step,) = _records(self.directory, "step")
        self.assertEqual(step["requests"][0]["q_start_pos"], 4)

    def test_v2_model_runner_raises_at_startup(self) -> None:
        settings = _VllmSettings(v2_model_runner=True)
        with dump_session(self.directory, settings=settings):
            with self.assertRaisesRegex(RuntimeError, "V2 model runner"):
                _model()

    def test_v2_model_runner_selected_by_config_raises(self) -> None:
        # The config selects the V2 runner although the variable is unset.
        settings = _settings_from(_vllm_config(use_v2_model_runner=True), None)
        with dump_session(self.directory, settings=settings):
            with self.assertRaisesRegex(RuntimeError, "VLLM_USE_V2_MODEL_RUNNER=0"):
                _model()

    def test_v1_model_runner_selected_by_config_starts(self) -> None:
        settings = _settings_from(_vllm_config(use_v2_model_runner=False), True)
        self.assertFalse(settings.v2_model_runner)
        self.assertFalse(settings.prefix_caching)
        with dump_session(self.directory, settings=settings) as forward:
            model = _model()
            _prefill(forward, model)
        self.assertEqual(len(_records(self.directory, "linear")), 1)

    def test_no_config_and_unset_variable_starts(self) -> None:
        settings = _settings_from(None, None)
        self.assertEqual(settings, _VllmSettings())
        with dump_session(self.directory, settings=settings) as forward:
            model = _model()
            _prefill(forward, model)
        self.assertEqual(len(_records(self.directory, "linear")), 1)
        self.assertTrue(_settings_from(None, True).v2_model_runner)

    def test_speculative_decoding_raises_at_startup(self) -> None:
        settings = _VllmSettings(speculative=True)
        with dump_session(self.directory, settings=settings):
            with self.assertRaisesRegex(NotImplementedError, "speculative"):
                _model()

    def test_nested_context_keeps_the_outer_step(self) -> None:
        other = "model.layers.1.mlp.down_proj"
        with dump_session(self.directory, MAX_STEPS="0") as forward:
            model = _TwoLayers(other)
            _prefill(forward, model)
            outer = forward.context
            _prefill(forward, model)
            forward.context = outer
            model(torch.randn(3, 4), second=True)
        self.assertEqual(
            [(r["step"], r["layer"]) for r in _records(self.directory, "linear")],
            [(0, _NAME), (1, _NAME), (0, other)],
        )
        self.assertEqual(len(_records(self.directory, "step")), 2)

    def test_layer_recorded_twice_in_a_step_raises(self) -> None:
        with dump_session(self.directory) as forward:
            model = _model(tensors=("input",))
            _prefill(forward, model)
            with self.assertRaisesRegex(RuntimeError, "ran twice in one step"):
                model(torch.randn(3, 4))

    def test_stale_context_callback_keeps_the_live_entry(self) -> None:
        # A callback of a dead context whose id the live context now holds.
        with dump_session(self.directory, MAX_STEPS="0") as forward:
            model = _model(tensors=("input",))
            _prefill(forward, model)
            session = _session._get()
            assert session is not None
            tracker = session.tracker
            stale = weakref.ref(FakeForwardContext(None))
            tracker._forget(id(forward.context))(stale)
            step = tracker.current()
            assert step is not None
            self.assertEqual(step.step, 0)
        self.assertEqual(len(_records(self.directory, "step")), 1)

    def test_cuda_graph_forward_raises(self) -> None:
        with dump_session(self.directory) as forward:
            model = _model()
            forward.context = FakeForwardContext(
                {_ATTN: fake_metadata([3], [0])},
                cudagraph_runtime_mode=SimpleNamespace(name="PIECEWISE"),
            )
            with self.assertRaisesRegex(RuntimeError, "--enforce-eager"):
                model(torch.randn(3, 4))

    def test_dual_batch_overlap_raises(self) -> None:
        with dump_session(self.directory) as forward:
            model = _model()
            forward.context = FakeForwardContext(
                [{_ATTN: fake_metadata([3], [0])}, {_ATTN: fake_metadata([3], [0])}]
            )
            with self.assertRaises(NotImplementedError):
                model(torch.randn(3, 4))


if __name__ == "__main__":
    unittest.main()
