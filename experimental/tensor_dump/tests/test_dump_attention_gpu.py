# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention dumps on a GPU, with vLLM's FlashAttention kernel computing the
attention of two requests over a chunked prefill and decodes.

The outputs are unchanged, the dumped new-token keys and values rebuild each
request's history, and every record replays to its dumped output with the
recorded scale and sliding window.

Needs a GPU and vLLM; skipped without either.
"""

import contextlib
import importlib.util
import tempfile
import unittest
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from dump_test_utils import dump_session, FakeAttention, load, read_manifest
from quant_suite_recipe.definition import QuantLinearRecipe
from tensor_dump.runtime import RUNTIME

_HAS_VLLM: bool = importlib.util.find_spec("vllm") is not None
_LAYERS = ("model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn")
_HEADS = 8
_KV_HEADS = 2
_HEAD_SIZE = 64
_WINDOW = 5

# Per step: (request ids, q_len of each, q_start_pos of each). Request 0's
# prompt of 11 tokens is prefilled in two chunks.
_SCHEDULE: list[tuple[list[int], list[int], list[int]]] = [
    ([0, 1], [6, 4], [0, 0]),
    ([0, 1], [5, 1], [6, 4]),
    ([0, 1], [1, 1], [11, 5]),
]


@dataclass
class _Metadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor


class _FlashAttention(FakeAttention):
    """Runs vLLM's FlashAttention kernel over each request's keys and values,
    which the layer keeps across forwards. ``schedule`` sets the step's
    requests before each forward."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.requests: list[int] = []
        self.q_lens: list[int] = []

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        q = query.view(-1, self.num_heads, self.head_size)
        k = key.view(-1, self.num_kv_heads, self.head_size)
        v = value.view(-1, self.num_kv_heads, self.head_size)
        keys, values, cu_q, cu_k = [], [], [0], [0]
        for request, q_len in zip(self.requests, self.q_lens):
            first = cu_q[-1]
            new_k, new_v = k[first : first + q_len], v[first : first + q_len]
            if request in self.history:
                old_k, old_v = self.history[request]
                new_k, new_v = torch.cat([old_k, new_k]), torch.cat([old_v, new_v])
            self.history[request] = (new_k, new_v)
            keys.append(new_k)
            values.append(new_v)
            cu_q.append(first + q_len)
            cu_k.append(cu_k[-1] + new_k.shape[0])
        window = self.sliding_window
        output = flash_attn_varlen_func(
            q[: cu_q[-1]],
            torch.cat(keys),
            torch.cat(values),
            max_seqlen_q=max(self.q_lens),
            cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=q.device),
            max_seqlen_k=max(b - a for a, b in zip(cu_k, cu_k[1:])),
            cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=q.device),
            softmax_scale=self.impl.scale,
            causal=True,
            window_size=[-1, -1] if window is None else [window - 1, 0],
        )
        return output.reshape(cu_q[-1], -1)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantized_linear = RUNTIME.prepare_linear(QuantLinearRecipe())
        self.layers = torch.nn.ModuleList(
            [
                _FlashAttention(_LAYERS[0], _HEADS, _KV_HEADS, _HEAD_SIZE),
                _FlashAttention(
                    _LAYERS[1],
                    _HEADS,
                    _KV_HEADS,
                    _HEAD_SIZE,
                    sliding_window=_WINDOW,
                ),
            ]
        )

    def layer(self, index: int) -> _FlashAttention:
        layer = self.layers[index]
        assert isinstance(layer, _FlashAttention)
        return layer

    def schedule(self, requests: list[int], q_lens: list[int]) -> None:
        for index in range(len(self.layers)):
            self.layer(index).requests = requests
            self.layer(index).q_lens = q_lens

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> list[torch.Tensor]:
        return [layer(q, k, v) for layer in self.layers]


def _vllm_forward_context(
    attn_metadata: dict[str, Any],
) -> contextlib.AbstractContextManager[None]:
    """A vLLM forward context holding ``attn_metadata``, entered on use."""
    from vllm.forward_context import ForwardContext, override_forward_context

    kwargs: dict[str, Any] = {}
    if any(field.name == "virtual_engine" for field in fields(ForwardContext)):
        kwargs["virtual_engine"] = 0
    context = ForwardContext(
        no_compile_layers={}, attn_metadata=attn_metadata, slot_mapping={}, **kwargs
    )
    return override_forward_context(context)


def _replay(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    replay: dict[str, Any],
) -> torch.Tensor:
    """The attention output in fp32 from Q ``[q_len, Hq, D]`` and the full keys
    and values ``[kv_len, Hkv, D]``, using the record's fields."""
    group = replay["num_q_heads"] // replay["num_kv_heads"]
    positions = torch.arange(replay["q_len"])[:, None] + replay["q_start_pos"]
    keys = torch.arange(replay["kv_len"])[None, :]
    mask = keys <= positions
    window = replay["sliding_window"]
    if window is not None:
        mask &= keys > positions - window
    output = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(0, 1),
        k.float().repeat_interleave(group, dim=1).transpose(0, 1),
        v.float().repeat_interleave(group, dim=1).transpose(0, 1),
        attn_mask=mask,
        scale=replay["scale"],
    )
    return output.transpose(0, 1)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
@unittest.skipUnless(_HAS_VLLM, "needs vLLM")
class DumpAttentionGpuTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.inputs = [
            tuple(
                torch.randn(
                    sum(q_lens),
                    heads * _HEAD_SIZE,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                for heads in (_HEADS, _KV_HEADS, _KV_HEADS)
            )
            for _, q_lens, _ in _SCHEDULE
        ]

    def _metadata(self, q_lens: list[int], q_starts: list[int]) -> _Metadata:
        starts = [0]
        for q_len in q_lens:
            starts.append(starts[-1] + q_len)
        return _Metadata(
            query_start_loc=torch.tensor(starts, dtype=torch.int32, device=self.device),
            seq_lens=torch.tensor(
                [q_len + start for q_len, start in zip(q_lens, q_starts)],
                dtype=torch.int32,
                device=self.device,
            ),
        )

    def _run(
        self, directory: Path | None, *, real_vllm: bool
    ) -> tuple[_Model, list[list[torch.Tensor]]]:
        env = {} if directory is None else {"ATTENTION": ".*", "MAX_STEPS": "0"}
        outputs = []
        with dump_session(
            directory,
            real_vllm=real_vllm,
            attention_class=_FlashAttention,
            **env,
        ) as forward:
            model = _Model()
            for (requests, q_lens, q_starts), inputs in zip(_SCHEDULE, self.inputs):
                model.schedule(requests, q_lens)
                metadata = {name: self._metadata(q_lens, q_starts) for name in _LAYERS}
                if real_vllm:
                    context = _vllm_forward_context(metadata)
                else:
                    forward.real(metadata)
                    context = contextlib.nullcontext()
                with context:
                    outputs.append(model(*inputs))
        return model, outputs

    def _check(self, *, real_vllm: bool) -> None:
        _, expected = self._run(None, real_vllm=False)
        directory = Path(tempfile.mkdtemp())
        model, outputs = self._run(directory, real_vllm=real_vllm)
        for step_outputs, step_expected in zip(outputs, expected):
            for output, reference in zip(step_outputs, step_expected):
                self.assertTrue(torch.equal(output, reference))
        records = [r for r in read_manifest(directory) if r["record"] == "attention"]
        self.assertEqual(len(records), 2 * sum(len(s[0]) for s in _SCHEDULE))
        for layer_index, layer in enumerate(_LAYERS):
            history: dict[int, list[torch.Tensor]] = {}
            for record in (r for r in records if r["layer"] == layer):
                with self.subTest(
                    layer=layer, step=record["step"], request=record["request"]
                ):
                    replay = record["replay"]
                    self.assertEqual(
                        replay["sliding_window"], (None, _WINDOW)[layer_index]
                    )
                    request = _SCHEDULE[record["step"]][0][record["request"]]
                    tensors = {
                        name: load(directory, entry)
                        for name, entry in record["tensors"].items()
                    }
                    rows = history.setdefault(request, [])
                    self.assertEqual(
                        replay["kv_rows"][0], sum(t.shape[0] for t in rows[::2])
                    )
                    rows += [tensors["k"], tensors["v"]]
                    k = torch.cat(rows[::2])
                    v = torch.cat(rows[1::2])
                    self.assertEqual(k.shape[0], replay["kv_len"])
                    torch.testing.assert_close(
                        tensors["output"].float(),
                        _replay(tensors["q"], k, v, replay),
                        atol=2e-2,
                        rtol=2e-2,
                    )
            for request, rows in history.items():
                full_k, full_v = model.layer(layer_index).history[request]
                self.assertTrue(torch.equal(torch.cat(rows[::2]), full_k.cpu()))
                self.assertTrue(torch.equal(torch.cat(rows[1::2]), full_v.cpu()))

    def test_flash_attention_dumps_rebuild_kv_and_replay(self) -> None:
        self._check(real_vllm=False)

    def test_inside_vllm_forward_context(self) -> None:
        self._check(real_vllm=True)


if __name__ == "__main__":
    unittest.main()
