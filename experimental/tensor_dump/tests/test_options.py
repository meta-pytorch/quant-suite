# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The ``QUANT_SUITE_TENSOR_DUMP_*`` environment variables. CPU only."""

import unittest
from pathlib import Path

from tensor_dump._options import _DumpOptions, _options_from_environ

_PREFIX = "QUANT_SUITE_TENSOR_DUMP_"


def _parse(**env: str) -> _DumpOptions | None:
    return _options_from_environ(
        {f"{_PREFIX}{key}": value for key, value in env.items()}
    )


class OptionsTest(unittest.TestCase):
    def test_unset_or_empty_dir_disables_dumping(self) -> None:
        self.assertIsNone(_parse())
        self.assertIsNone(_parse(DIR=""))
        self.assertIsNone(_parse(DIR="", STEPS="not a step"))

    def test_defaults(self) -> None:
        options = _parse(DIR="/tmp/dump")
        assert options is not None
        self.assertEqual(options.directory, Path("/tmp/dump"))
        self.assertEqual(options.steps, ((0, None),))
        self.assertEqual(options.phases, frozenset({"prefill", "decode", "mixed"}))
        self.assertEqual(options.max_steps, 1)
        self.assertIsNone(options.ranks)
        self.assertTrue(options.selects_rank(7))
        self.assertIsNone(options.attention)
        self.assertEqual(
            options.attention_tensors, frozenset({"q", "k", "v", "output"})
        )
        empty = _parse(DIR="/tmp/dump", ATTENTION="")
        assert empty is not None
        self.assertIsNone(empty.attention)

    def test_grammar(self) -> None:
        options = _parse(
            DIR="d",
            STEPS="0, 5:8,10:",
            PHASE="decode,mixed",
            MAX_STEPS="0",
            RANKS="0,3",
            ATTENTION=r"model\.layers\.(0|7)\.",
            ATTENTION_TENSORS="q, output",
        )
        assert options is not None
        self.assertEqual(options.steps, ((0, 1), (5, 8), (10, None)))
        self.assertEqual(
            [step for step in range(12) if options.selects_step(step)],
            [0, 5, 6, 7, 10, 11],
        )
        self.assertEqual(options.phases, frozenset({"decode", "mixed"}))
        self.assertEqual(options.max_steps, 0)
        self.assertEqual(options.ranks, frozenset({0, 3}))
        self.assertFalse(options.selects_rank(1))
        attention = options.attention
        assert attention is not None
        self.assertTrue(attention.match("model.layers.7.self_attn.attn"))
        self.assertFalse(attention.match("model.layers.1.self_attn.attn"))
        self.assertEqual(options.attention_tensors, frozenset({"q", "output"}))

    def test_describe_round_trips(self) -> None:
        options = _parse(
            DIR="d",
            STEPS="2,5:8,10:",
            PHASE="all",
            RANKS="1",
            ATTENTION="layers",
            ATTENTION_TENSORS="v,q",
        )
        assert options is not None
        description = options.describe()
        self.assertEqual(
            description,
            {
                "steps": "2,5:8,10:",
                "phase": "all",
                "max_steps": 1,
                "ranks": "1",
                "attention": "layers",
                "attention_tensors": "q,v",
            },
        )
        self.assertEqual(
            _parse(
                DIR="d",
                STEPS=str(description["steps"]),
                PHASE=str(description["phase"]),
                MAX_STEPS=str(description["max_steps"]),
                RANKS=str(description["ranks"]),
                ATTENTION=str(description["attention"]),
                ATTENTION_TENSORS=str(description["attention_tensors"]),
            ),
            options,
        )

    def test_malformed_values_name_the_variable(self) -> None:
        cases = {
            "STEPS": ["", "a", "-1", "3:3", "5:2", "1,,2", "1:2:3"],
            "PHASE": ["", "warmup", "prefill,"],
            "MAX_STEPS": ["", "-1", "1.5", "many"],
            "RANKS": ["", "x", "0,-1", "all,0"],
            "ATTENTION": ["(", "layers.[0"],
            "ATTENTION_TENSORS": ["", "p", "q,,k", "input"],
        }
        for suffix, values in cases.items():
            for value in values:
                with self.subTest(variable=suffix, value=value):
                    with self.assertRaisesRegex(ValueError, f"{_PREFIX}{suffix}"):
                        _parse(DIR="d", **{suffix: value})


if __name__ == "__main__":
    unittest.main()
