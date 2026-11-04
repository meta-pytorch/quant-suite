# mx_plus

> Experimental. No API, numerics or serialization guarantees; see
> [`../README.md`](../README.md).

`mx_plus` (distribution `quant_suite_mx_plus`) adds MX+ to QuantSuite as an
extension package. It is also the template for adding your own experimental
quantization: everything it does goes through the public extension points of
`quant_suite_core` and `quant_suite_recipe`, neither of them imports it, and
once it is installed its runtime is the environment's recipe runtime, so the
QuantSuite vLLM plugin serves its recipes with no further setup.

## What MX+ computes

An MX block stores `k` elements in a narrow element format (for MXFP4, E2M1)
that share one power-of-two scale (E8M0). The scale is chosen from the block's
largest-magnitude element, so that element's exponent is effectively known from
the scale, and its own exponent bits carry little information.

MX+ spends extra precision on exactly that element. For each block:

1. Find the element with the largest magnitude, `x_max`.
2. Scale it into the element domain with the block scale `s`: `y = x_max * s`.
3. Take its private exponent `e = max(floor(log2|y|), 2 - 2^(ebits-1))`.
4. Round `y` to `m + n` mantissa bits at exponent `e`, round-half-to-even,
   where `m` is the element format's mantissa width and `n` the number of
   extra bits.
5. Clamp to the element format's range and write it back in place of the
   normally quantized value.

Every other element and the block scale are quantized exactly as in plain MX,
so the shared scale byte does not change. `n` is the extension's
`extra_mantissa_bits` option, from 1 to 8. The preset `mxfp4_plus2` is
MXFP4 (E2M1 elements, E8M0 scale, block 1x32, `pow2max` scale with `floor`
rounding) with `n = 2`.

## Adding your own experimental quantization

An experimental package contributes four things, each in its own module. MX+
is the worked example; the sections below walk through it in order.

| step | file | defines | depends on |
| --- | --- | --- | --- |
| 1. the rule | `mx_plus/mx_plus.py` | `apply_mx_plus` | `quant_suite_core` |
| 2. the config extension | `mx_plus/config.py` | `MxPlus` | `quant_suite_core` |
| 3. presets | `mx_plus/presets.py` | `MX_PLUS_PRESETS` | nothing |
| 4. the runtime | `mx_plus/runtime.py` | `RUNTIME` | the three above, `quant_suite_recipe` |

The runtime is published as an entry point (step 5), so installing the package
(step 6) is all vLLM needs (step 7).

### 1. The rule

`apply_mx_plus(config, phase, extra_mantissa_bits)` receives a `PostProcessPhase` from whichever
QDQ backend is running. The phase owns what differs between backends: the
block view of the input (`block_view()`), the per-block scale
(`block_scale()`), and writing elements back (`scatter_block_elements()`). The
rule only does the arithmetic above, so one implementation serves both the
eager and the Triton backend.

### 2. The config extension

`MxPlus` is a frozen dataclass subclass of
`quant_suite_core.extension.QuantConfigExtension`:

```python
@dataclass(frozen=True)
class MxPlus(QuantConfigExtension):
    name: ClassVar[str] = "mx_plus"
    extra_mantissa_bits: int = 2
```

- `name = "mx_plus"`: the key a config dict enables it under, as
  `"extensions": {"mx_plus": {"extra_mantissa_bits": 2}}`; `to_dict` writes
  it back the same way.
- `extra_mantissa_bits`: its one typed option. Two `MxPlus` instances are
  equal exactly when their options are, so the option is part of the config's
  identity.

It overrides `validate` (an E8M0 scale, `global_scale_granularity` and
`global_scale_dtype` both `none`, which also makes it exclusive with MBS, and
`1 <= extra_mantissa_bits <= 8`) and one hook, `post_process`, which calls
`apply_mx_plus` with the option. The hooks an extension can override and how a
config enables extensions are described in
[Extending QuantConfig](../../core/README.md#extending-quantconfig).

MX+ needs no new recipe type, because a rule on `QuantConfig` reaches every
`type: fake` stage. An algorithm that needs its own recipe type adds its
registration to the runtime in step 4.

### 3. Presets

`MX_PLUS_PRESETS` maps preset names to plain specs, the same shape as
`quant_suite_core.fake_presets.FAKE_PRESETS`. A preset `p` is usable as
`fake_p` and as `quant_config: p`.

### 4. The runtime

A `QuantRecipeRuntime` accepts the `mx_plus` extension and the `mxfp4_plus2`
preset once its config builder knows them. `mx_plus/runtime.py` builds one
from the built-in defaults:

```python
from quant_suite_core.quant_config import DEFAULT_QUANT_CONFIG_BUILDER
from quant_suite_recipe.runtime import QuantRecipeRuntime

RUNTIME = QuantRecipeRuntime(
    config_builder=DEFAULT_QUANT_CONFIG_BUILDER.extend((MxPlus,), MX_PLUS_PRESETS)
)
```

Start from the defaults: an installed runtime is only accepted when it still
accepts every built-in recipe type, config extension and preset. An algorithm
with its own recipe type also passes
`(*QUANT_RECIPE_REGISTRATIONS, (MyRecipe, my_quantize_tensor))` as the first
argument, with `QUANT_RECIPE_REGISTRATIONS` from `quant_suite_recipe.runtime`.
`MyRecipe` extends `QuantTensorRecipe` and sets the class attribute
`discriminator` to its payload's `type` string.

The runtime works on its own too:

```python
from mx_plus.runtime import RUNTIME

recipe = RUNTIME.load_quant_recipe_from_yaml(
    "experimental/mx_plus/examples/llama_3_1_8b_instruct/mxfp4_plus2_core.yaml"
)
```

### 5. Publish the entry point

`pyproject.toml` publishes the runtime in the `quant_suite.recipe_runtimes`
entry-point group:

```toml
[project.entry-points."quant_suite.recipe_runtimes"]
mx_plus = "mx_plus.runtime:RUNTIME"
```

Once the package is installed,
`quant_suite_recipe.runtime_provider.QUANT_RECIPE_RUNTIME` resolves to
`RUNTIME` in every process of that environment, not just in vLLM: any
integration that uses the environment's runtime accepts `mx_plus` and
`mxfp4_plus2`. Code that needs exactly this runtime, such as a test, imports
`mx_plus.runtime.RUNTIME` directly.

### 6. Install

The package is not published to PyPI. Install it from a checkout, next to the
QuantSuite packages it extends:

```bash
git clone https://github.com/meta-pytorch/quant-suite && cd quant-suite
pip install -e core -e recipe -e vllm -e experimental/mx_plus
```

or straight from git:

```bash
pip install "quant_suite_mx_plus @ git+https://github.com/meta-pytorch/quant-suite#subdirectory=experimental/mx_plus"
```

Add the `test` extra (`-e "experimental/mx_plus[test]"`) to run the tests.

### 7. Serve with vLLM

With the package installed, the QuantSuite vLLM plugin uses `RUNTIME` as
`QUANT_RECIPE_RUNTIME` in every vLLM process, so a recipe can name `mx_plus`
and `mxfp4_plus2` directly:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --quantization quant_suite \
  --hf-overrides '{"quantization_config_file":"experimental/mx_plus/examples/llama_3_1_8b_instruct/mxfp4_plus2_core.yaml"}' \
  --attention-backend FLASH_ATTN
```

The example recipes apply MXFP4 + MX+ to the linear layers, the configuration
the MX+ paper reports; the same config can be used for any tensor, including
MoE and Attention tensors. [`examples/`](examples/) has one for Llama 3.1 8B
Instruct and one for Qwen3-8B.

If more than one package publishes a runtime, resolution refuses to choose:
set `QUANT_SUITE_RECIPE_RUNTIME` to the entry-point name of the one to use
(here, `QUANT_SUITE_RECIPE_RUNTIME=mx_plus`). See
[`../../recipe/README.md`](../../recipe/README.md#choosing-a-runtime) for the
full resolution rule.

## Config forms

A fake-quant config can be written in three forms. The first two name a
preset and are equivalent; the shorthand is the usual choice.
[`examples/llama_3_1_8b_instruct/mxfp4_plus2_core.yaml`](examples/llama_3_1_8b_instruct/mxfp4_plus2_core.yaml)
uses the first two side by side:

```yaml
linear_recipe:
  '.*':
    weight: fake_mxfp4_plus2              # 1. preset shorthand
    activation:
      type: fake
      quant_config: mxfp4_plus2           # 2. preset name
    bias: none
```

The third is an inline mapping, needed only for a config no preset names;
fields left out take their defaults. Written out, `mxfp4_plus2` is:

```yaml
linear_recipe:
  '.*':
    weight:
      type: fake
      quant_config:                       # 3. inline mapping
        elem_dtype: e2m1
        block_size: 1x32
        scale_dtype: e8m0
        scale_norm_mode: pow2max
        scale_round_mode: floor
        extensions:
          mx_plus:
            extra_mantissa_bits: 2        # two extra mantissa bits
```

A runtime whose builder does not know `MxPlus`, such as
`QuantRecipeRuntime()`, rejects all three: `fake_mxfp4_plus2` is an unknown
preset, and a config that enables `mx_plus` fails to parse with an error that
lists the extensions the builder knows.

Without a recipe, build a config directly:

```python
from mx_plus.presets import MX_PLUS_PRESETS
from mx_plus.runtime import RUNTIME
from quant_suite_core import fake_quantize

builder = RUNTIME.config_builder
config = builder.build_preset("mxfp4_plus2")
y = fake_quantize(x, "activation", config)

# Or with a different number of extra bits:
config = builder.build(
    {
        **MX_PLUS_PRESETS["mxfp4_plus2"],
        "extensions": {"mx_plus": {"extra_mantissa_bits": 4}},
    }
)
```

## Write your own

1. Copy this directory to `experimental/<your_algorithm>/`, rename the inner
   `mx_plus/` package to `<your_algorithm>/`, and rename the dist name
   (`quant_suite_<your_algorithm>`) and the entry-point name.
2. Implement the rule against the phase protocols in
   `quant_suite_core.extension`.
3. Subclass `QuantConfigExtension` as a frozen dataclass: a unique `name`,
   typed options as fields, one hook, and `validate` for the configs it cannot
   handle.
4. Add presets if a named config helps users.
5. Build `RUNTIME` from `DEFAULT_QUANT_CONFIG_BUILDER` (and
   `QUANT_RECIPE_REGISTRATIONS`, if you add a recipe type) and publish it in
   the `quant_suite.recipe_runtimes` entry-point group.
6. Adapt the tests, the README and its citation, and the license files to
   your algorithm. [Adding an algorithm](../README.md#adding-an-algorithm)
   lists what every experimental package must ship.

## Tests

```bash
pip install -e "experimental/mx_plus[test]"
pytest experimental/mx_plus/tests
```

- `test_mx_plus_config.py`: options, mapping round-trips, validation,
  and the amax-element rounding (just below a tie, negative zero).
- `test_mx_plus_recipe.py`: the three YAML forms through a
  `QuantRecipeRuntime` whose builder knows `MxPlus`, and the example recipe.
- `test_mx_plus_runtime.py`: `RUNTIME` resolves `fake_mxfp4_plus2`, and
  runtime resolution selects it from its entry point.
- `test_mx_plus_qdq.py` (GPU): both backends, error reduction, the
  subnormal-scale case, the unchanged E8M0 scale byte.
- `test_import_boundary.py`: the package imports only public QuantSuite
  modules.

`mx_plus_golden_vectors.py` is not a test but the hand-run golden-vector
refactor check over `MX_PLUS_PRESETS`, on the harness in `recipe/tests`; its
docstring says how to run it. A red check means the quantization output moved.

## Limitations

- E8M0 block scales only.
- Exclusive with MBS and with a global scale.
- No fused kernel: both backends run the rule as a PyTorch pass after
  quantization, so expect it to be slower than plain MX.

## Citation

MX+ was introduced in:

```bibtex
@inproceedings{lee2025mxplus,
  title     = {{MX+}: Pushing the Limits of Microscaling Formats for Efficient Large Language Model Serving},
  author    = {Lee, Jungi and Park, Junyong and Cha, Soohyun and Cho, Jaehoon and Sim, Jaewoong},
  booktitle = {Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture (MICRO '25)},
  year      = {2025}
}
```

This package is an independent implementation; the element rounding is
adapted from Microsoft microxcaling (see `THIRD_PARTY_NOTICES.md`). MX+ is used
as a baseline in:

```bibtex
@misc{chhugani2026mxfp4,
  title         = {Unveiling the Potential of Quantization with {MXFP4}: Strategies for Quantization Error Reduction},
  author        = {Chhugani, Jatin and Jeong, Geonhwa and Su, Bor-Yiing and Pan, Yunjie and Yang, Hanmei and Ankit, Aayush and Yu, Jiecao and Deng, Summer and Chen, Yunqing and Satish, Nadathur and Kim, Changkyu},
  year          = {2026},
  eprint        = {2603.08713},
  archivePrefix = {arXiv}
}
```
