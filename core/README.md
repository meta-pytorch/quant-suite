# quant_suite_core

`quant_suite_core` provides PyTorch tensor-level quantization building blocks:
configurable fake QDQ for MXFP4/6/8, MXINT8, NVFP4, and related low-precision
formats; native packed MXFP4, MXFP8, and NVFP4 quantization and dequantization;
and packing, Triton, Hadamard, and randomized Hadamard transform (RHT) utilities.

The overflow-aware scale rounding (`ScaleRoundMode.OAS`, `OAS_S`) and the macro
block scale (`GlobalScaleGranularity.MBS`) follow "Unveiling the Potential of
Quantization with MXFP4: Strategies for Quantization Error Reduction"
([arXiv 2603.08713](https://arxiv.org/abs/2603.08713)).

## Installation

Both pip and uv use the package metadata in `pyproject.toml`. The distribution
name and Python import name are both `quant_suite_core`. The package requires
Python 3.11 or newer and PyTorch 2.13 or newer.

Install with pip:

```bash
python -m pip install .
python -m pip install ".[test]"
```

Or install with uv:

```bash
uv pip install .
uv pip install ".[test]"
```

For a uv-managed `.venv`, install the package and its test dependencies with:

```bash
uv sync --extra test
```

## Quick start

```python
import torch
from quant_suite_core import fake_quantize, native_dequantize, native_quantize, QuantConfig
from quant_suite_core.fake_presets import FAKE_PRESETS

# MXFP4 with 1x16 blocks, from a preset; any mapping of QuantConfig fields works.
config = QuantConfig.from_dict(FAKE_PRESETS["mxfp4"])
x = torch.randn(128, 256)

# Eager reference backend: runs on CPU.
y = fake_quantize(x, "x", config, use_triton=False)

# Triton backend (the default): needs a CUDA device.
if torch.cuda.is_available():
    y_gpu = fake_quantize(x.cuda(), "x", config)

# The dict form round-trips.
assert QuantConfig.from_dict(config.to_dict()) == config

# Native packed MXFP4 (1x32 OCP blocks): packed uint8 data and E8M0 scales.
native_config = QuantConfig.from_dict(FAKE_PRESETS["mxfp4_ocp"])
packed = native_quantize(x, "x", native_config, use_triton=False)
x_hat = native_dequantize(packed, output_dtype=torch.float32)
```

`fake_quantize(..., use_triton=True)` and
`native_quantize(..., use_triton=True)`, the defaults, need a CUDA GPU. Requires
SM90+; validated on B200 (SM100); native FP4 packing requires SM100.
`use_triton=False` runs the same quantization with PyTorch ops and also works on
CPU. Triton must be importable either way, since the Triton backend is imported
with the package; on Linux, PyTorch installs it. Results were recorded with
Triton 3.8.

The package root exports three entry points: `fake_quantize`,
`native_quantize` and `native_dequantize`.
Lower-level APIs are imported from their modules: the QDQ backends from
`quant_suite_core.qdq_pytorch` and `quant_suite_core.qdq_triton`, the preset
tables from `quant_suite_core.fake_presets` and `quant_suite_core.native_presets`,
and the hook protocols for extension authors from `quant_suite_core.extension`.

To add a quantization rule without changing this package, see
[Extending QuantConfig](#extending-quantconfig) below. Algorithms that use those
extension points but are not part of the supported packages live under
[`../experimental/`](../experimental/README.md).

## Native quantization in core and in the recipe package

QuantSuite has two native (packed) quantization paths.

`native_quantize` and `native_dequantize` in this package take a
`QuantConfig`, so the scale rounding comes from its `scale_round_mode`
(`ScaleRoundMode`). They support MXFP4 and MXFP8 (E2M1 or E4M3 elements with
E8M0 scales) and NVFP4 (E2M1 elements, FP8 scales and a tensorwise or rowwise
global scale), with symmetric 1xN blocks. Quantization and packing run in this
package's own Triton kernel or, with `use_triton=False`, in PyTorch ops.
Scales are row-major, one per block; `is_sf_swizzled_layout=True` returns E8M0
scales in the 128x4 blocked layout instead, while NVFP4 scales stay row-major.
No dependency beyond this package is needed.

The `native` tensor recipe of
[`quant_suite_recipe`](../recipe/README.md#native-quantization-paths)
(`NativeQuantRecipe`) does not call `native_quantize`. It calls the MSLK
kernels, names the format with `PrecisionType` (`nvfp4`, `mx4-16`, `mx4-32`)
and the scale rounding with `NativeRoundingMode`, and keeps MSLK's
blocked scale layout. Both enums are defined in
`quant_suite_core.native_presets`.

Use `native_quantize` for standalone tensor quantization, MXFP8, a CPU
fallback, the `ScaleRoundMode` choices, or row-major scales. Use the recipe
`native` path when a model recipe should quantize through the recipe runtime
and its consumer expects MSLK's scale layout.

## Extending QuantConfig

An extension adds one optional rule to `QuantConfig` without a subclass. Declare
it as a frozen dataclass subclass of
`quant_suite_core.extension.QuantConfigExtension`: the class attribute `name`
identifies it, and the dataclass fields are its typed options. Override the hooks
it needs (`amax_rule`, `post_process` or `finalize`) and, optionally, `validate`:

```python
from dataclasses import dataclass
from typing import ClassVar

from quant_suite_core.extension import QuantConfigExtension


@dataclass(frozen=True)
class ExtraBits(QuantConfigExtension):
    name: ClassVar[str] = "extra_bits"
    bits: int = 2

    def validate(self, config): ...

    def post_process(self, config, phase): ...
```

A config enables extensions through its `extensions` field, with at most one per
hook. Two instances are equal when their options are, so `ExtraBits(2)` and
`ExtraBits(4)` give different configs.

A `QuantConfigBuilder` parses config dicts that enable extensions. Start from
`DEFAULT_QUANT_CONFIG_BUILDER`, which knows the core presets, and add extension
classes and presets with `extend`. A config dict carries the base fields plus an
`"extensions"` mapping from extension name to options; omitted options take their
defaults:

```python
from quant_suite_core.quant_config import DEFAULT_QUANT_CONFIG_BUILDER

builder = DEFAULT_QUANT_CONFIG_BUILDER.extend((ExtraBits,), presets=MY_PRESETS)
config = builder.build(
    {"elem_dtype": "e2m1", "block_size": "1x32", "extensions": {"extra_bits": {"bits": 2}}}
)
assert builder.build(config.to_dict()) == config
```

An unknown extension name, an unknown option or an option of the wrong type
raises `ValueError`, and a spec that is not a mapping raises `TypeError`.
`to_dict` writes the `"extensions"` mapping only when the config has extensions,
so a config without extensions serializes with core tokens only.

`QuantConfig` itself cannot be subclassed; optional behaviour is always added as
an extension.

## Errors

The exception type says what is wrong:

- `ValueError`: the input is wrong, such as an unknown key, preset or enum name,
  a bad value, a combination `QuantConfig` rejects, or a tensor dtype other than
  bfloat16, float16 or float32.
- `NotImplementedError`: the input is valid, but the selected path does not
  support it, such as a config `native_quantize` cannot pack, a config the
  Triton backend does not implement, or a CPU tensor passed to a CUDA-only
  kernel.
- `RuntimeError`: the environment cannot run it, such as a GPU older than the
  kernel requires.

An argument of the wrong Python type, such as a spec that is not a mapping,
raises `TypeError`. `NotImplementedError` is a subclass of `RuntimeError`, so
`except RuntimeError` catches both.

## Package contents

- `quant_config.py`: `QuantConfig`, its format enums and `QuantConfigBuilder`
- `fake_quant.py`: `fake_quantize`, the dispatch over the two QDQ backends
- `qdq_pytorch.py`: `qdq`, the eager reference backend (CPU or CUDA)
- `qdq_triton.py`: `qdq_triton_mega`, the fused Triton backend (CUDA)
- `quant_primitives.py`: element and scale conversions shared by the backends
- `extension.py`: `QuantConfigExtension` and the hook protocols for extension authors
- `native_quant.py`: native quantization entry points
- `mx_packing.py`: packed MX representations
- `hadamard_utils.py`: Sylvester Hadamard transforms, power-of-two orders only
- `rht_generator.py`: randomized Hadamard transform generator
- `triton_utils.py`: Triton kernels and helpers
- `fake_presets.py`: named fake-quantization presets
- `native_presets.py`: native precision and rounding enums, and named native-quantization presets
- `metrics.py`: quantization error metrics

Packaging metadata lives entirely in `pyproject.toml`. The package version has
a single source in `quant_suite_core/__init__.py`.

## Repository layout

```text
quant-suite/
├── core/           # quant_suite_core (this package)
│   ├── pyproject.toml
│   ├── quant_suite_core/
│   └── tests/
├── recipe/         # quant_suite_recipe
├── vllm/           # quant_suite_vllm
└── experimental/   # experimental algorithm packages
```

## Source repository

The source is hosted at https://github.com/meta-pytorch/quant-suite.

## License

BSD-3-Clause; see [`LICENSE`](LICENSE). Portions adapted from Microsoft
microxcaling (MIT) are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
