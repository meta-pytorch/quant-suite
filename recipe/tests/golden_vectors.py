# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Refactor check for ``quant_suite_core``: record the presets, refactor, compare.

A local tool, run by hand, on one host
--------------------------------------
This is not a regression test. It is the instrument for validating a refactor
of ``quant_suite_core``, and it only means anything when both halves run on the
same machine:

1. ``record`` against the current implementation, writing the digest table to
   ``golden_vectors_data.py``.
2. Perform the refactor.
3. ``check`` against the refactored implementation: every case is re-run and
   compared with the recorded table.

The entry point is ``main()``, which takes one mode argument: ``check``
compares against the recorded table, ``record`` writes it.
``QUANT_SUITE_GOLDEN_OUT`` points both modes at a different file -- ``record``
writes it, ``check`` reads it -- which is how to rehearse a comparison without
touching the committed table.

Running it
----------
Install ``quant_suite_core`` and ``quant_suite_recipe`` and run the module from
this directory, which holds its sibling imports, on a host with a CUDA device::

    cd recipe/tests
    python golden_vectors.py check             # compare with the table
    python golden_vectors.py record            # rewrite the preset table
    python golden_vectors.py record-extended   # rewrite only the extended table

``python -m golden_vectors <mode>`` from the same directory is equivalent.

Deliberately not wired to CI
----------------------------
A digest is the bitwise output of a particular CUDA kernel, so it belongs to the
GPU that produced it; ``golden_vectors_data.GENERATED_ON`` names that GPU. A
check on a different GPU model would fail for a reason that has nothing to do
with the refactor under test, so a check is only meaningful against a table
recorded on the same GPU model. It exposes no test protocol and is run by hand.
Keep it that way.

Why digests, and not another equivalence check
----------------------------------------------
``qdq_triton.py`` passes ``config.<enum>.value`` into Triton as ``tl.constexpr``
strings and ``triton_utils.py`` branches on the same bare string literals. No
shared symbol ties the two sides together, so a refactor that renames or
relocates a format on one side and misses the other produces *silently wrong
numbers*, not an ImportError -- neither the build nor the type checker sees it. The
unit tests of ``quant_suite_core`` compare the backends against each other or
against a property, so a change that moves every backend the same way passes all of them.

This harness pins the actual bytes instead. It hashes the dequantized output of
each backend for every fake-quant/RHT preset combination selected below. Any
bitwise change on the path this harness actually calls shows up here.

The path this harness calls is the explicit recipe pipeline
------------------------------------------------------------
Each case is an executable ``PipelineRecipe``. Every quant-only fake-quant
preset is crossed with three RHT presets: no transform, left Hadamard-16, and
right Hadamard-16. The recipe always contains ``[RHTRecipe, FakeQuantRecipe]``;
runtime pipeline construction prunes the no-op RHT. No RHT state lives in an MX
config.

This matters because ``fake_quantize`` and its eager/Triton backends are quant-only;
calling them directly skips the transform, so ``mxfp4``, ``mxfp4_l_had16`` and
``mxfp4_r_had16`` would share digests. The explicit RHT stage makes left and
right multiplication by ``R`` observable. Its
``RHTRecipe.seed`` is set to ``_RHT_SEED``, independently of the
``torch.manual_seed`` state used by stochastic rounding.

A ``GoldenSuite`` is a preset table plus the config builder that can parse it and
the digests it was recorded into. ``DEFAULT_SUITE`` is ``FAKE_PRESETS`` against
``QuantConfig.from_dict``; ``run`` takes any ``GoldenSuite``. Inputs, backends, RHT
presets and the digest format are shared, so a row moves between tables
unchanged.

Backends covered, one entry per ``(preset, backend, dtype)``
------------------------------------------------------------
* ``qdq`` -- an explicit pipeline whose quantize stage has ``use_triton=False``.
* ``qdq_triton_mega`` -- the same pipeline with ``use_triton=True``.

Every preset is recorded against both, so each gets exactly one Triton golden --
the one a Triton caller would execute. ``is_triton_mega_supported`` is still
checked up front: it declines ``mbs_dynamic`` with a non-E8M0 scale, which no
preset selects, and a preset that starts declining would otherwise surface as a
``NotImplementedError`` partway through a run rather than as a clear refusal.

Inputs
------
One shape, two dtypes, three fixed seeds. ``torch.manual_seed(_INPUT_SEED)``
then ``x = torch.randn((32, 384))`` on CUDA, then ``torch.manual_seed(_QUANT_SEED)``
immediately before the backend call. The third, ``_RHT_SEED``, never goes
through ``torch.manual_seed`` -- it is placed in each explicit ``RHTRecipe``.

* 32 rows: two whole ``16x16`` blocks down the rows.
* 384 columns: a multiple of 16, 32 and 128, so every preset block size and the
  MBS16/MBS128 macro-block sizes tile exactly, and MBS128 gets three macro
  blocks per row rather than a degenerate one. 384 is deliberately not a
  multiple of 256; 256-aligned sizes hit faster paths in some kernels and are
  unrepresentative.

  The ``mbs256`` presets are the exception, and deliberately so: 384 pads to
  512, i.e. one and a half macro blocks with the remainder zero-filled. That is
  the only coverage the suite has of the MBS zero-padding path, which both
  backends implement separately.
* Re-seeding immediately before the call pins what the stochastic-rounding paths
  read: eager ``qdq`` draws its random bits from ``torch.randint`` and the Triton
  kernels derive their seed from the CUDA generator's seed and offset. Both then
  depend on ``_QUANT_SEED`` alone, not on how many draws building the input
  consumed.

Stochastic-rounding presets are pinned, not excluded
----------------------------------------------------
Eighteen configurations carry ``elem_round_mode: stochastic`` after the RHT
cross product. Under the seeding protocol above each backend was measured to be
bit-identical across repeated runs, so its output is a stable golden. What is
not well defined for those configurations is
comparing one backend against *another* -- ``torch.randint`` and the Triton
in-kernel RNG are unrelated streams -- and this tool never does that: every row
is one backend against its own record. ``record`` runs each case twice and
refuses to write a row whose two runs disagree, so a change that makes a backend
nondeterministic blocks the recording instead of baking in a flaky value.

Extended cases: the layouts the preset sweep does not reach
----------------------------------------------------------
The sweep above only ever feeds one 2-D, block-aligned CUDA tensor. A suite may
also list ``ExtendedConfig`` entries, each run directly through ``fake_quantize``
(no RHT stage) over ``_EXTENDED_LAYOUTS`` -- 0-D, 1-D, 3-D and 4-D inputs, rows
and columns that need padding, ``transpose=True`` and ``return_quant_info=True``
-- for the eager backend on CPU and CUDA and for the mega kernel on CUDA. Their
input is drawn on the CPU from ``_INPUT_SEED`` and seeded with an Inf, a -Inf, a
NaN and an all-zero run, so the CPU and CUDA rows see identical values.

They live in their own table, ``EXTENDED_GOLDEN_VECTOR_TABLE``, written by the
``record-extended`` mode, which leaves every other byte of the data file alone:
adding these cases never re-records the preset sweep. ``check`` covers both.
A ``return_quant_info`` row hashes every ``QuantInfo`` tensor, not just the
dequantized one.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import replace
from types import ModuleType
from typing import Any, Callable, NamedTuple

import golden_vectors_data
import torch
from evaluation_pipeline_configs import RHT_PRESETS
from quant_suite_core import fake_quantize, QuantConfig
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.fake_quant import is_triton_mega_supported
from quant_suite_recipe.definition import FakeQuantRecipe, PipelineRecipe, RHTRecipe
from quant_suite_recipe.runtime import QuantRecipeRuntime


_BUILTIN_RUNTIME = QuantRecipeRuntime()

_INPUT_SEED: int = 20260922
_QUANT_SEED: int = 7
_RHT_SEED: int = 20260929
_SHAPE: tuple[int, int] = (32, 384)
_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.bfloat16)

_EAGER_BACKEND: str = "qdq"
_MEGA_BACKEND: str = "qdq_triton_mega"

# The quantize stage's ``use_triton`` flag selects the backend while the rest of
# the pipeline remains identical.
_USE_TRITON: dict[str, bool] = {
    _EAGER_BACKEND: False,
    _MEGA_BACKEND: True,
}

# Redirects the table file for both modes: ``record`` writes it, ``check`` reads
# it. Unset, both use the suite's committed data module.
_GOLDEN_PATH_ENV: str = "QUANT_SUITE_GOLDEN_OUT"


_LICENSE_HEADER: str = (
    "# Copyright (c) Meta Platforms, Inc. and affiliates.\n"
    "# All rights reserved.\n"
    "#\n"
    "# This source code is licensed under the BSD-style license found in the\n"
    "# LICENSE file in the root directory of this source tree.\n"
)


class ExtendedConfig(NamedTuple):
    """A config run over ``_EXTENDED_LAYOUTS`` instead of the preset sweep."""

    name: str
    config: Mapping[str, Any]
    dtypes: tuple[torch.dtype, ...]


class ExtendedLayout(NamedTuple):
    shape: tuple[int, ...]
    transpose: bool = False
    return_quant_info: bool = False


class GoldenSuite(NamedTuple):
    """A preset table, the config builder that can parse it, its digests, and
    the header ``record`` writes at the top of the digest file.

    Inputs, backends, RHT presets, seeding and the digest format are shared by
    every suite, so tables are directly comparable and a row can move between
    them unchanged.
    """

    presets: Mapping[str, Mapping[str, Any]]
    build_config: Callable[[Mapping[str, Any]], QuantConfig]
    table: ModuleType
    header: str
    extended: tuple[ExtendedConfig, ...] = ()


_MX_1X32: dict[str, str] = {
    "elem_round_mode": "nearest",
    "block_size": "1x32",
    "scale_dtype": "e8m0",
    "scale_norm_mode": "pow2max",
}

# Four representative MX configs (e4m3, e2m3, e2m1 and int8 elements), then
# presets covering the two-level FP8 scale (tensor- and rowwise), 2-D blocks and
# MBS.
_EXTENDED: tuple[ExtendedConfig, ...] = (
    ExtendedConfig(
        "mxfp8_e4m3_oas_s",
        {**_MX_1X32, "elem_dtype": "e4m3", "scale_round_mode": "oas_s"},
        _DTYPES,
    ),
    ExtendedConfig(
        "mxfp6_e2m3_oas",
        {**_MX_1X32, "elem_dtype": "e2m3", "scale_round_mode": "oas"},
        _DTYPES,
    ),
    ExtendedConfig(
        "mxfp4_e2m1_oas",
        {**_MX_1X32, "elem_dtype": "e2m1", "scale_round_mode": "oas"},
        _DTYPES,
    ),
    ExtendedConfig(
        "mxint8_oas",
        {**_MX_1X32, "elem_dtype": "int8", "scale_round_mode": "oas"},
        _DTYPES,
    ),
    *(
        ExtendedConfig(name, FAKE_PRESETS[name], (torch.bfloat16,))
        for name in ("nvfp4", "rowwise_nvfp4", "mxfp4_16x16", "mxfp4_mbs128")
    ),
)

DEFAULT_SUITE: GoldenSuite = GoldenSuite(
    presets=FAKE_PRESETS,
    build_config=QuantConfig.from_dict,
    table=golden_vectors_data,
    header=_LICENSE_HEADER,
    extended=_EXTENDED,
)


class GoldenRecord(NamedTuple):
    """One recorded output. Only ``shape``/``dtype``/``sha256`` are compared;
    the rest exist so a digest mismatch can be read without a debugger."""

    shape: str
    dtype: str
    sha256: str
    minimum: str
    maximum: str
    mean: str
    n_nan: str
    n_inf: str

    def identity(self) -> tuple[str, str, str]:
        return (self.shape, self.dtype, self.sha256)

    def diagnostics(self) -> str:
        return (
            f"min={self.minimum} max={self.maximum} mean={self.mean} "
            f"n_nan={self.n_nan} n_inf={self.n_inf}"
        )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _case_key(preset: str, backend: str, dtype: torch.dtype) -> str:
    return f"{preset}|{backend}|{_dtype_name(dtype)}"


def _cases(suite: GoldenSuite) -> list[tuple[str, str, RHTRecipe]]:
    """Every suite preset crossed with each RHT preset, sorted by case name."""
    return sorted(
        (
            preset if rht_name == "none" else f"{preset}_{rht_name}",
            preset,
            rht_preset,
        )
        for preset in suite.presets
        for rht_name, rht_preset in RHT_PRESETS.items()
    )


def _expected_keys(suite: GoldenSuite) -> set[str]:
    return {
        _case_key(name, backend, dtype)
        for name, _, _ in _cases(suite)
        for backend in _USE_TRITON
        for dtype in _DTYPES
    }


def _parse_table(table: str) -> dict[str, GoldenRecord]:
    records: dict[str, GoldenRecord] = {}
    for raw_line in table.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, *fields = line.split()
        if len(fields) != len(GoldenRecord._fields):
            raise ValueError(
                f"golden table row {key!r} has {len(fields)} fields, expected "
                f"{len(GoldenRecord._fields)}: {' '.join(GoldenRecord._fields)}"
            )
        records[key] = GoldenRecord(*fields)
    return records


def _run_case(
    suite: GoldenSuite,
    name: str,
    fake_preset: str,
    rht_preset: RHTRecipe,
    backend: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    recipe = PipelineRecipe(
        [
            replace(rht_preset, seed=_RHT_SEED),
            FakeQuantRecipe(
                quant_config=suite.build_config(suite.presets[fake_preset]),
                use_triton=_USE_TRITON[backend],
            ),
        ]
    )
    pipeline = _BUILTIN_RUNTIME.create_pipeline(
        recipe,
        tensor_name=name,
    )
    torch.manual_seed(_INPUT_SEED)
    x = torch.randn(_SHAPE, dtype=dtype, device="cuda")
    # Re-seed here rather than only before the input draw: this is the RNG state
    # the stochastic-rounding paths read, and pinning it is what makes those
    # presets reproducible.
    torch.manual_seed(_QUANT_SEED)
    result = _BUILTIN_RUNTIME.apply_pipeline(pipeline, x)
    assert result.tensor is not None
    return result.tensor


# Bit pattern written over every NaN before hashing: (same-width int dtype, bits).
_CANONICAL_NAN: dict[torch.dtype, tuple[torch.dtype, int]] = {
    torch.float32: (torch.int32, 0x7FC00000),
    torch.bfloat16: (torch.int16, 0x7FC0),
    torch.float16: (torch.int16, 0x7E00),
}


def _digest_bytes(host: torch.Tensor) -> bytes:
    """Raw bytes of a contiguous CPU tensor, with every NaN set to its ``_CANONICAL_NAN`` bits."""
    flat = host.reshape(-1)
    if flat.is_floating_point():
        int_dtype, bits = _CANONICAL_NAN[flat.dtype]
        canonical = flat.view(int_dtype).clone()
        canonical[torch.isnan(flat)] = bits
        flat = canonical
    return flat.view(torch.uint8).numpy().tobytes()


def _summarize(out: torch.Tensor) -> GoldenRecord:
    host = out.detach().cpu().contiguous()
    raw = _digest_bytes(host)
    values = host.float()
    return GoldenRecord(
        shape="x".join(str(dim) for dim in host.shape),
        dtype=_dtype_name(host.dtype),
        sha256=hashlib.sha256(raw).hexdigest(),
        minimum=f"{values.amin().item():.6g}",
        maximum=f"{values.amax().item():.6g}",
        mean=f"{values.mean().item():.6g}",
        n_nan=str(int(torch.isnan(values).sum().item())),
        n_inf=str(int(torch.isinf(values).sum().item())),
    )


# ---------------------------------------------------------------------------
# Extended cases
# ---------------------------------------------------------------------------

# 33 and 100 are multiples of neither 16 nor 32, so both dims pad.
_EXTENDED_LAYOUTS: dict[str, ExtendedLayout] = {
    "0d": ExtendedLayout(()),
    "1d": ExtendedLayout((100,)),
    "3d": ExtendedLayout((3, 5, 96)),
    "4d": ExtendedLayout((2, 3, 4, 64)),
    "uneven": ExtendedLayout((33, 100)),
    "transpose": ExtendedLayout((48, 100), transpose=True),
    "quant_info": ExtendedLayout((33, 100), return_quant_info=True),
}

# (backend, device). The mega kernel has no CPU path.
_EXTENDED_TARGETS: tuple[tuple[str, str], ...] = (
    (_EAGER_BACKEND, "cpu"),
    (_EAGER_BACKEND, "cuda"),
    (_MEGA_BACKEND, "cuda"),
)

_EXTENDED_TABLE_HEADER: str = (
    "# Extended cases, written only by `golden_vectors record-extended`. Key:\n"
    "# `ext:<config>|<layout>|<backend>@<device>|<input dtype>`; layouts and\n"
    "# configs are defined in golden_vectors.py.\n"
)


class _ExtendedCase(NamedTuple):
    key: str
    config: ExtendedConfig
    layout: ExtendedLayout
    backend: str
    device: str
    dtype: torch.dtype


def _extended_cases(suite: GoldenSuite) -> list[_ExtendedCase]:
    return sorted(
        (
            _ExtendedCase(
                f"ext:{cfg.name}|{layout_name}|{backend}@{device}|{_dtype_name(dtype)}",
                cfg,
                layout,
                backend,
                device,
                dtype,
            )
            for cfg in suite.extended
            for layout_name, layout in _EXTENDED_LAYOUTS.items()
            for backend, device in _EXTENDED_TARGETS
            for dtype in cfg.dtypes
        ),
        key=lambda case: case.key,
    )


def _extended_input(shape: tuple[int, ...]) -> torch.Tensor:
    torch.manual_seed(_INPUT_SEED)
    x = torch.randn(shape, dtype=torch.float32)
    flat = x.view(-1)
    if flat.numel() >= 96:
        flat[0] = float("nan")
        flat[3] = float("inf")
        flat[5] = float("-inf")
        flat[40:72] = 0.0
    return x


def _run_extended_case(suite: GoldenSuite, case: _ExtendedCase) -> list[torch.Tensor]:
    x = _extended_input(case.layout.shape).to(case.dtype).to(case.device)
    torch.manual_seed(_QUANT_SEED)
    out = fake_quantize(
        x,
        case.config.name,
        suite.build_config(case.config.config),
        transpose=case.layout.transpose,
        use_triton=_USE_TRITON[case.backend],
        return_quant_info=case.layout.return_quant_info,
    )
    if not case.layout.return_quant_info:
        return [out]
    return [
        t
        for t in (
            out.dequant_data,
            out.quantized_val,
            out.local_unscale,
            out.global_scale,
        )
        if t is not None
    ]


def _summarize_all(outs: list[torch.Tensor]) -> GoldenRecord:
    """``_summarize`` of the first tensor, with a digest over all of them."""
    first = _summarize(outs[0])
    if not first.shape:
        # A 0-D output joins to an empty shape field, which the table cannot hold.
        first = first._replace(shape="scalar")
    if len(outs) == 1:
        return first
    digest = hashlib.sha256()
    for out in outs:
        host = out.detach().cpu().contiguous()
        digest.update(f"{tuple(host.shape)}|{host.dtype}|".encode())
        digest.update(_digest_bytes(host))
    return first._replace(sha256=digest.hexdigest())


def _render_extended_block(rows: str, generated_on: str) -> str:
    return (
        f"\n{_EXTENDED_TABLE_HEADER}"
        f'EXTENDED_GENERATED_ON: str = "{generated_on}"\n'
        f'EXTENDED_GOLDEN_VECTOR_TABLE: str = """\n{rows}"""\n'
    )


def record_extended(suite: GoldenSuite) -> int:
    """Record only the extended table, leaving the rest of the file untouched."""
    recorded: dict[str, GoldenRecord] = {}
    nondeterministic: list[str] = []
    for case in _extended_cases(suite):
        first = _summarize_all(_run_extended_case(suite, case))
        second = _summarize_all(_run_extended_case(suite, case))
        if first.identity() == second.identity():
            recorded[case.key] = first
        else:
            nondeterministic.append(case.key)
    if nondeterministic:
        print(
            "[record-extended] REFUSING to write: these cases were not "
            f"bit-identical across two runs: {nondeterministic}"
        )
        return 1

    dest = _golden_path(suite)
    with open(dest) as handle:
        text = handle.read()
    marker = text.find(f"\n{_EXTENDED_TABLE_HEADER}")
    if marker != -1:
        text = text[:marker]
    rows = "".join(f"{key} {' '.join(recorded[key])}\n" for key in sorted(recorded))
    with open(dest, "w") as handle:
        handle.write(text + _render_extended_block(rows, torch.cuda.get_device_name()))
    print(f"[record-extended] wrote {len(recorded)} records to {dest}")
    return 0


def _check_extended(
    suite: GoldenSuite, golden: ModuleType, running_on: str
) -> tuple[int, list[str]]:
    """(case count, problems) for the extended table."""
    cases = _extended_cases(suite)
    if not cases:
        return 0, []
    raw = getattr(golden, "EXTENDED_GOLDEN_VECTOR_TABLE", None)
    if raw is None:
        return len(cases), [
            f"{_golden_path(suite)} has no EXTENDED_GOLDEN_VECTOR_TABLE; run "
            "`record-extended`"
        ]
    table = _parse_table(raw)
    expected_keys = {case.key for case in cases}
    if set(table) != expected_keys:
        return len(cases), [
            f"the extended table does not cover the extended cases: "
            f"missing={sorted(expected_keys - set(table))} "
            f"unexpected={sorted(set(table) - expected_keys)}"
        ]
    failures: list[str] = []
    for case in cases:
        expected = table[case.key]
        actual = _summarize_all(_run_extended_case(suite, case))
        if actual.identity() != expected.identity():
            failures.append(
                f"{case.key}: quantization output changed (recorded on "
                f"{golden.EXTENDED_GENERATED_ON!r}, running on {running_on!r}).\n"
                f"  expected shape={expected.shape} dtype={expected.dtype} "
                f"sha256={expected.sha256}\n"
                f"    actual shape={actual.shape} dtype={actual.dtype} "
                f"sha256={actual.sha256}\n"
                f"  expected {expected.diagnostics()}\n"
                f"    actual {actual.diagnostics()}"
            )
    return len(cases), failures


# ---------------------------------------------------------------------------
# The table file
# ---------------------------------------------------------------------------


def _golden_path(suite: GoldenSuite) -> str:
    return os.environ.get(_GOLDEN_PATH_ENV) or os.path.realpath(suite.table.__file__)


def _load_golden(suite: GoldenSuite) -> ModuleType:
    """The table to compare against. Loading an override as a module, rather
    than scraping the text out of it, keeps reading and writing on one format."""
    override = os.environ.get(_GOLDEN_PATH_ENV)
    if not override:
        return suite.table
    spec = importlib.util.spec_from_file_location("golden_vectors_override", override)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{_GOLDEN_PATH_ENV}={override!r} is not a loadable module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _destination_doc(suite: GoldenSuite) -> str:
    """The prose of the file ``record`` is about to overwrite -- the override
    when one is set -- so a human can edit it without the next recording
    reverting them. A first recording to a path that does not exist yet has
    nothing to carry over, so the committed module supplies the prose."""
    if not os.path.exists(_golden_path(suite)):
        return suite.table.__doc__ or ""
    return _load_golden(suite).__doc__ or ""


def _backend_split_problems(suite: GoldenSuite) -> list[str]:
    """The mega kernel is the only Triton backend, so every config must be one it
    accepts. Checked from the predicate, never a hardcoded list, so this stays
    correct if the predicate changes."""
    declined = sorted(
        name
        for name, fake_preset, _ in _cases(suite)
        if not is_triton_mega_supported(suite.build_config(suite.presets[fake_preset]))
    )
    if not declined:
        return []
    return [
        f"{_MEGA_BACKEND} declines these configs, so they have no Triton golden "
        f"and no backend to record against: {declined}"
    ]


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------


def _record_case(
    suite: GoldenSuite,
    name: str,
    fake_preset: str,
    rht_preset: RHTRecipe,
    backend: str,
    dtype: torch.dtype,
) -> GoldenRecord | None:
    """Run a case twice; return it only if the two runs agree bit for bit."""
    first = _summarize(_run_case(suite, name, fake_preset, rht_preset, backend, dtype))
    second = _summarize(_run_case(suite, name, fake_preset, rht_preset, backend, dtype))
    return first if first.identity() == second.identity() else None


def _as_docstring_body(doc: str) -> str:
    """Escape only what could close a ``\"\"\"`` literal early -- a run of two
    or more quotes, and a trailing one abutting the delimiter -- so hand-edited
    prose keeps its line breaks and an ordinary quote in it stays unmarked."""
    return re.sub(r'"(?=")|"\Z', r'\\"', doc.replace("\\", "\\\\"))


def _render_golden_module(suite: GoldenSuite, recorded: dict[str, GoldenRecord]) -> str:
    doc = _as_docstring_body(_destination_doc(suite))
    rows = "".join(f"{key} {' '.join(recorded[key])}\n" for key in sorted(recorded))
    return (
        f"{suite.header}\n"
        f'"""{doc}"""\n'
        "\n"
        "# GPU the table was recorded on. Reported on failure: a mismatch on a"
        " different\n"
        "# architecture is a different question from a mismatch on this one.\n"
        f'GENERATED_ON: str = "{torch.cuda.get_device_name()}"\n'
        "\n"
        f'GOLDEN_VECTOR_TABLE: str = """\n{rows}"""\n' + _carried_extended_block(suite)
    )


def _carried_extended_block(suite: GoldenSuite) -> str:
    """The extended table of the file ``record`` is about to overwrite, so
    re-recording the preset sweep does not drop it."""
    if not os.path.exists(_golden_path(suite)):
        return ""
    golden = _load_golden(suite)
    rows = getattr(golden, "EXTENDED_GOLDEN_VECTOR_TABLE", None)
    if rows is None:
        return ""
    return _render_extended_block(rows.lstrip("\n"), golden.EXTENDED_GENERATED_ON)


def record(suite: GoldenSuite) -> int:
    problems = _backend_split_problems(suite)
    if problems:
        print("\n".join(problems))
        return 1

    recorded: dict[str, GoldenRecord] = {}
    nondeterministic: list[str] = []
    for name, fake_preset, rht_preset in _cases(suite):
        for backend in _USE_TRITON:
            for dtype in _DTYPES:
                case = _record_case(
                    suite,
                    name,
                    fake_preset,
                    rht_preset,
                    backend,
                    dtype,
                )
                if case is None:
                    nondeterministic.append(_case_key(name, backend, dtype))
                else:
                    recorded[_case_key(name, backend, dtype)] = case

    if nondeterministic:
        print(
            "[record] REFUSING to write: these cases were not bit-identical across "
            f"two runs, so they have no stable golden: {nondeterministic}"
        )
        return 1
    missing = sorted(_expected_keys(suite) - set(recorded))
    if missing:
        print(
            f"[record] REFUSING to write: {len(missing)} cases never ran, "
            f"first few: {missing[:5]}"
        )
        return 1

    dest = _golden_path(suite)
    # Rendered before the open: it carries over the prose of the file the open
    # is about to truncate.
    rendered = _render_golden_module(suite, recorded)
    with open(dest, "w") as handle:
        handle.write(rendered)
    print(f"[record] wrote {len(recorded)} records to {dest}")
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def _describe_mismatch(
    preset: str,
    backend: str,
    dtype: torch.dtype,
    expected: GoldenRecord,
    actual: GoldenRecord,
    recorded_on: str,
    running_on: str,
) -> str:
    return (
        f"{preset} / {backend} / {_dtype_name(dtype)}: quantization output changed "
        f"(recorded on {recorded_on!r}, running on {running_on!r}).\n"
        f"  expected shape={expected.shape} dtype={expected.dtype} "
        f"sha256={expected.sha256}\n"
        f"    actual shape={actual.shape} dtype={actual.dtype} "
        f"sha256={actual.sha256}\n"
        f"  expected {expected.diagnostics()}\n"
        f"    actual {actual.diagnostics()}"
    )


def check(suite: GoldenSuite) -> int:
    golden = _load_golden(suite)
    table = _parse_table(golden.GOLDEN_VECTOR_TABLE)
    expected_keys = _expected_keys(suite)

    problems = _backend_split_problems(suite)
    if set(table) != expected_keys:
        # Caught up front so a preset added or removed since the recording reads
        # as a stale table rather than as a KeyError mid-run.
        problems.append(
            f"the table in {_golden_path(suite)} does not cover the cases this "
            f"tool runs: missing={sorted(expected_keys - set(table))} "
            f"unexpected={sorted(set(table) - expected_keys)}"
        )
    if problems:
        print("\n".join(problems))
        return 1

    running_on = torch.cuda.get_device_name()
    failures: list[str] = []
    for name, fake_preset, rht_preset in _cases(suite):
        for backend in _USE_TRITON:
            for dtype in _DTYPES:
                expected = table[_case_key(name, backend, dtype)]
                actual = _summarize(
                    _run_case(suite, name, fake_preset, rht_preset, backend, dtype)
                )
                if actual.identity() == expected.identity():
                    continue
                failures.append(
                    _describe_mismatch(
                        name,
                        backend,
                        dtype,
                        expected,
                        actual,
                        golden.GENERATED_ON,
                        running_on,
                    )
                )

    n_extended, extended_failures = _check_extended(suite, golden, running_on)
    failures += extended_failures
    total = len(expected_keys) + n_extended
    if failures:
        print("\n".join(failures))
        print(
            f"[check] FAILED: {len(failures)} of {total} cases differ "
            f"from {_golden_path(suite)}"
        )
        return 1
    print(
        f"[check] OK: {total} cases ({len(expected_keys)} preset sweep + "
        f"{n_extended} extended) match {_golden_path(suite)} "
        f"(recorded on {golden.GENERATED_ON!r}, running on {running_on!r})"
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(suite: GoldenSuite, prog: str = "golden_vectors") -> None:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Record every fake-quant evaluation pipeline of this suite, or "
            "check it against the recorded table. Both halves must run on the "
            "same host; see the module docstring."
        ),
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("record", help="run every case and write the digest table")
    modes.add_parser("check", help="run every case and compare with the digest table")
    modes.add_parser(
        "record-extended",
        help="run only the extended cases and write their table, leaving the rest "
        "of the file untouched",
    )
    mode = parser.parse_args().mode

    if not torch.cuda.is_available():
        # Without this a CPU-only host runs zero cases and reports success.
        sys.exit(
            "no CUDA device available: this tool compares the bitwise output of "
            "CUDA kernels, so a CPU-only run would check nothing and still pass"
        )

    if mode == "record-extended":
        sys.exit(record_extended(suite))
    sys.exit(record(suite) if mode == "record" else check(suite))


def main() -> None:
    run(DEFAULT_SUITE)


if __name__ == "__main__":
    main()
