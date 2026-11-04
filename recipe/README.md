# QuantSuite Recipe

`quant_suite_recipe` provides configuration and execution APIs for
QuantSuite. It defines typed, serializable recipes for tensor quantization,
transform pipelines, Linear, MoE, and attention modules, and model-wide regex
selection. Its runtime deserializes complete model recipes, builds per-tensor
pipelines, and applies those pipelines to tensors for the built-in `none`, `fake`,
and `native` tensor recipes.

A runtime built with additional recipe types, config extensions or presets
accepts those too (see [Custom runtimes](#custom-runtimes)); this package does
not import them. Low-level MX formats and transforms are provided by
[`quant_suite_core`](../core/README.md).

## Install

The package is installed from source. From a checkout of the repository:

```bash
python -m pip install ./core ./recipe
```

The package requires Python 3.11 or newer and depends on `quant_suite_core`,
PyYAML, and PyTorch 2.13 or newer. Results were recorded with Triton 3.8.
Install the
`native` extra when native MXFP4 or NVFP4 execution through MSLK is required:

```bash
python -m pip install './recipe[native]'
```

## Recipe model

The package is organized by dependency direction:

```text
quant_suite_recipe/
├── api/
│   ├── definition.py
│   └── implementation.py
├── definition/
│   └── quant_recipe.py
├── implementation/
│   ├── attention.py
│   ├── linear.py
│   ├── moe.py
│   ├── quant_pipeline.py
│   └── quantize_tensor.py
├── runtime.py
├── runtime_provider.py
└── yaml_to_quant_recipe.py
```

Concrete recipe dataclasses live under `quant_suite_recipe.definition`, concrete
handlers and executable pipeline modules live under
`quant_suite_recipe.implementation`, `runtime.py` defines `QuantRecipeRuntime`
and the built-in `QUANT_RECIPE_REGISTRATIONS`, and `runtime_provider.py`
resolves the environment's runtime that framework integrations import (see
[Choosing a runtime](#choosing-a-runtime)). `QuantRecipeRuntime` owns
recipe-to-stage construction because it joins registered recipe types to
their handlers.

The package root exports no names. Import recipe types from
`quant_suite_recipe.definition` and the contracts for new recipe types from
`quant_suite_recipe.api`.

The schema is hierarchical:

- `QuantStageRecipe` represents one pipeline stage.
  - `QuantTensorRecipe` represents a quantization stage.
    - `NoQuantRecipe` is the identity operation.
    - `FakeQuantRecipe` describes fake (QDQ) quantization with a `QuantConfig`.
    - `NativeQuantRecipe` describes native MXFP4 or NVFP4 quantization.
  - `RHTRecipe` describes a randomized Hadamard transform.
- `PipelineRecipe` is an ordered sequence of stages for one tensor.
- `QuantLinearRecipe`, `QuantMoERecipe`, and `QuantAttentionRecipe` group the
  tensor pipelines used by a module.
- `QuantRecipe` maps module-name regexes to module recipes.

An omitted tensor field produces an empty `PipelineRecipe`. An explicit `none`
node is retained as a `NoQuantRecipe`; runtime pipeline construction skips all
no-op stages.

## Deserializing recipes

Use the selected runtime when the concrete recipe type is selected by serialized
data. Runtime deserialization accepts complete model-level payloads and returns
a `QuantRecipe`; extract the module recipe and tensor pipeline you need from
that object:

```python
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME

model_recipe = QUANT_RECIPE_RUNTIME.deserialize(
    {
        "linear_recipe": {
            "model\\.layers\\..*\\.mlp": {
                "activation": "none",
                # MXFP4 with 1x16 blocks and floor scale rounding: no preset
                # names this config, so it is written as a mapping.
                "weight": {
                    "type": "fake",
                    "quant_config": {
                        "elem_dtype": "e2m1",
                        "elem_round_mode": "nearest",
                        "block_size": "1x16",
                        "scale_dtype": "e8m0",
                        "scale_norm_mode": "pow2max",
                        "scale_round_mode": "floor",
                    },
                },
                "bias": "none",
            },
        },
    }
)

linear_recipe = model_recipe.get_linear_recipe("model.layers.3.mlp")
weight_recipe = linear_recipe.weight
```

Inside a tensor field, supported loose-config forms are:

- `"none"`
- `"fake_<preset>"`, such as `"fake_nvfp4"`
- `"native_<preset>"`, such as `"native_nvfp4"` or `"native_mxfp4_ocp"`;
  the `_oas` presets (`mxfp4_32_oas`, `mxfp4_16_oas`) use `even` scale rounding
- a mapping with a `type` discriminator and that type's fields; a mapping
  without `type` raises a `ValueError`

The `quant_config` value of a `type: fake` mapping is either a preset name,
such as `quant_config: nvfp4`, or a mapping of `QuantConfig` fields. Any other
string raises a `ValueError`.

Unknown keys raise a `ValueError` that names them and lists the allowed keys,
at every level: the model-level recipe, a module recipe, a pipeline, a stage,
and a stage's config. A misspelled key, such as `weigth` or `use_trition`, is
therefore an error rather than a silently ignored setting.

Concrete classes can also be constructed directly:

```python
from quant_suite_core import QuantConfig
from quant_suite_core.native_presets import NativeRoundingMode, PrecisionType
from quant_suite_recipe.definition import FakeQuantRecipe, NativeQuantRecipe

fake = FakeQuantRecipe(
    quant_config=QuantConfig.from_dict(
        {
            "elem_dtype": "e2m1",
            "elem_round_mode": "nearest",
            "block_size": "1x16",
            "scale_dtype": "fp8",
            "scale_norm_mode": "amax",
            "scale_round_mode": "none",
            "global_scale_granularity": "tensorwise",
            "global_scale_dtype": "fp32",
        }
    )
)
native = NativeQuantRecipe(
    precision=PrecisionType.NVFP4,
    scale_round_mode=NativeRoundingMode.NEAREST,
)
```

`NativeQuantRecipe` supports `nvfp4`, `mx4-16`, and `mx4-32`.
Its scale rounding mode is a `NativeRoundingMode`, serialized by name
(`nearest`, `floor`, `even`, `stochastic`, `ceil`) and parsed
case-insensitively; any other value raises a `ValueError`. `nvfp4` always rounds its FP8 block scales to nearest:
`floor`, `even` and `ceil` have no effect on it, and `stochastic` raises a
`ValueError`.

### Native quantization paths

The `native` recipe and `quant_suite_core.native_quantize` are separate
implementations. The `native` recipe calls the MSLK Triton kernels, which the
`native` extra installs:

- `mx4-16` and `mx4-32` quantize MXFP4 with 1x16 or 1x32 blocks and pass
  `scale_round_mode` to the kernel as the MSLK rounding mode of the same name;
- `nvfp4` uses a tensorwise global scale of `448 * 6 / amax(tensor)`, or 1.0
  for an all-zero tensor; its kernel takes no rounding mode, so
  `scale_round_mode` does not apply to it.

`QuantizedTensor.scale` then holds the scales in MSLK's blocked layout for both
formats. `dequantize_tensor` converts them back to one scale per block before
it applies them. On the `fake` path, by contrast, `scale` holds the
`QuantInfo.local_unscale` of `fake_quantize`.

Core `native_quantize` takes a `QuantConfig` and rounds scales by
`ScaleRoundMode`. It runs its own Triton kernel or PyTorch ops, also supports
MXFP8, and returns row-major scales unless asked for the 128x4 blocked E8M0
layout. For MXFP4, the `NativeRoundingMode` values `floor`, `ceil`, and `even`
correspond to the `ScaleRoundMode` values `floor`, `ceil`, and `oas`; with
those pairs and blocked scales, the core tests check that both paths produce
identical bytes. `nearest` and `stochastic` have no `ScaleRoundMode`
counterpart.

Use the `native` recipe inside a model recipe whose consumer expects MSLK's
scale layout. Use core `native_quantize` for standalone tensors, MXFP8, the
other `ScaleRoundMode` choices, or a path without MSLK.

## Composable pipelines

Each tensor field is a `PipelineRecipe`, so transforms and quantization can be
ordered explicitly:

```python
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME

model_recipe = QUANT_RECIPE_RUNTIME.deserialize(
    {
        "linear_recipe": {
            "model\\.layers\\..*\\.mlp": {
                "weight": [
                    {
                        "rht": {
                            "type": "right",
                            "block_size": 128,
                            "seed": 0,
                        }
                    },
                    {"quantize": "fake_nvfp4"},
                ],
            },
        },
    }
)

weight_recipe = model_recipe.get_linear_recipe("model.layers.3.mlp").weight
```

The canonical serialized form is a `stages` list containing one-key stage
nodes:

```yaml
stages:
  - rht:
      type: right
      block_size: 128
      seed: 0
  - quantize: fake_nvfp4
```

For a single quantization stage, a tensor field also accepts a shorthand string
or a bare quantization mapping. A transform-only pipeline must
use list syntax. `RHTRecipe` accepts `right`, `right_inv`, `left`, and `left_inv` (case-insensitive);
a non-identity transform requires a power-of-two `block_size >= 2`.

## Executing recipes

Import the runtime from `quant_suite_recipe.runtime_provider`. Importing the
package root remains definition-only.

```python
import torch
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME

model_recipe = QUANT_RECIPE_RUNTIME.deserialize(
    {
        "linear_recipe": {
            "model\\.layers\\..*\\.mlp": {
                "weight": "fake_nvfp4",
            },
        },
    }
)

linear_recipe = model_recipe.get_linear_recipe("model.layers.3.mlp")
weight_pipeline = QUANT_RECIPE_RUNTIME.create_pipeline(linear_recipe.weight)
quantized_weight = QUANT_RECIPE_RUNTIME.apply_pipeline(
    weight_pipeline,
    torch.randn(32, 128, dtype=torch.bfloat16),
)
```

For one-shot execution of a single tensor recipe, no pipeline is needed:

```python
quantized_weight = QUANT_RECIPE_RUNTIME.quantize_tensor(
    torch.randn(32, 128, dtype=torch.bfloat16),
    "weight",
    linear_recipe.weight[0],
)
```

Framework integrations use the environment's runtime,
`quant_suite_recipe.runtime_provider.QUANT_RECIPE_RUNTIME`; without an
installed runtime package it is the built-in runtime (`none`, `fake`, and
`native` tensor recipes). The caller owns
parameter allocation and stores one prepared Linear executor on its module:

```python
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME

quantized_linear = QUANT_RECIPE_RUNTIME.prepare_linear(linear_recipe)
weight = quantized_linear.quantize_weight(weight)
output = quantized_linear(
    activation,
    weight,
    bias,
)
```

The prepared executor owns the activation, weight, and bias pipelines and
selects the registered matrix-multiplication implementation. It supports
unquantized, fake-quantized, and native-quantized execution. Native kernels load
MSLK only when a native precision is executed. The package does not allocate
weights or define a framework-specific Linear layer.

The runtime API exposes eight methods:

- `deserialize(data)`: parse a complete model payload into a `QuantRecipe`
- `load_quant_recipe_from_yaml(path)`: load and deserialize a complete model
  recipe with this runtime
- `quantize_tensor(tensor, tensor_name, recipe)`: directly execute one
  `QuantTensorRecipe` without constructing a `Pipeline`
- `create_pipeline(recipe)`: build a `Pipeline` from a `QuantStageRecipe` or
  `PipelineRecipe`
- `prepare_linear(recipe)`: build a module-owned Linear executor that caches its
  pipelines and applies the runtime's registered matrix multiplication
- `prepare_moe(recipe)`: build a module-owned MoE executor containing its
  activation and weight pipelines
- `prepare_attention(recipe)`: build a module-owned Attention executor containing
  its Q, K, V, attention-weight, and recurrent-state pipelines and the runtime's
  selected batched matrix multiplication
- `apply_pipeline(pipeline, tensor)`: execute a `Pipeline` and return a
  `QuantizedTensor`

Low-level callers can select a runtime explicitly and pass extracted tensor or
module recipes to it. Integrations should import the canonical runtime from
`quant_suite_recipe.runtime_provider`. Use `quantize_tensor()` for one-shot,
single-stage quantization. Framework integrations should prepare an
executor during module initialization and store it on the owning module as
`quantized_linear`, `quantized_moe`, or `quantized_attention`. The prepared
executor owns and registers its pipelines, so callers do not manage pipeline
dictionaries or rebuild pipelines in `forward()`.

The built-in Attention executor uses `torch.bmm`. BMM consumers must
reject unsupported native packed Q, K, V, and attention-weight pipelines during
their setup. `prepare_attention()` itself remains available to consumers such
as KDA that use only the independent recurrent-state pipeline. A runtime built
with another `quantized_attention_type` selects another Attention executor,
which must keep the same contract.

## Linear, MoE, and attention recipes

Every tensor field below is a `PipelineRecipe`:

| Recipe | Tensor fields |
| --- | --- |
| `QuantLinearRecipe` | `activation`, `weight`, `bias` |
| `QuantMoERecipe` | `a1`, `a2`, `w13`, `w2` |
| `QuantAttentionRecipe` | `q`, `k`, `v`, `attn_weights`, `recurrent_state` |

Linear and MoE recipes also expose `init_quantized_weight`.

For an MoE layer, `a1 @ w13` is the gate/up projection and `a2 @ w2` is the down
projection. A common incoherence pipeline applies the same right Hadamard to
`a2` and `w2`, then quantizes the rotated weight. The two rotations cancel in
the GEMM while the weight is quantized in a more uniform basis:

```python
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME

model_recipe = QUANT_RECIPE_RUNTIME.deserialize(
    {
        "moe_recipe": {
            ".*experts.*": {
                "a1": "none",
                "w13": "fake_nvfp4",
                "a2": [
                    {
                        "rht": {
                            "type": "right",
                            "block_size": 128,
                            "seed": 0,
                        }
                    }
                ],
                "w2": [
                    {
                        "rht": {
                            "type": "right",
                            "block_size": 128,
                            "seed": 0,
                        }
                    },
                    {"quantize": "fake_nvfp4"},
                ],
            },
        },
    }
)

moe_recipe = model_recipe.get_moe_recipe("model.layers.3.experts")
```

Construct `PipelineRecipe` directly when code already has concrete stage
objects. Use runtime `deserialize()` when the recipe comes from serialized
configuration.

## Model-level regex recipes

`QuantRecipe` maps regular-expression patterns to module recipes. The first
matching pattern is returned by `get_linear_recipe()`, `get_moe_recipe()`, or
`get_attention_recipe()`. If no pattern matches, the getter returns an identity
module recipe. A section omitted from `QuantRecipe(...)` or from deserialized
input is an empty mapping and leaves that module category unaffected.

Patterns are tried in insertion order with Python's `re.match`: a pattern is
anchored at the start of the module name but not at the end, so
`model\.layers\.1` also matches `model.layers.12.mlp`. Add `$` (or a trailing
`\.`) to pin the end. A pattern that is not a valid regular expression
raises a `ValueError` when the recipe is built.

```python
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME

recipe = QUANT_RECIPE_RUNTIME.deserialize(
    {
        "linear_recipe": {
            "model\\.layers\\..*\\.mlp.*": {"weight": "fake_nvfp4"},
        },
        "moe_recipe": {
            "model\\.layers\\..*\\.experts.*": {"w13": "fake_nvfp4"},
        },
        "attention_recipe": {
            "model\\.layers\\..*\\.self_attn": {"q": "fake_nvfp4"},
        },
    }
)

linear_for_layer = recipe.get_linear_recipe("model.layers.3.mlp.down_proj")
```

For a uniform Linear and MoE configuration, use `.*` entries directly:

```python
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME

recipe = QUANT_RECIPE_RUNTIME.deserialize(
    {
        "linear_recipe": {
            ".*": {"weight": "fake_nvfp4"},
        },
        "moe_recipe": {
            ".*": {
                "w13": "fake_nvfp4",
                "w2": "fake_nvfp4",
            },
        },
    }
)
```

For multiple regexes or attention recipes, include those sections in the same
complete payload and deserialize once.

## Serialization and provider runtimes

Every recipe class implements `to_dict()`, and concrete leaf `from_dict()`
methods parse only that concrete payload. `QuantRecipe.to_dict()` uses flattened
keys such as `linear_recipe::<regex>::weight`; `QuantRecipeRuntime.deserialize()`
accepts both that round-trip form and nested mappings.

`quant_suite_recipe` does not expose a mutable registration hook; a runtime
receives everything it accepts when it is constructed (see
[Custom runtimes](#custom-runtimes)). A runtime parses every `QuantConfig`
through its config builder, so the `fake_<preset>` shorthand,
`quant_config: <preset>` and an inline mapping all accept the presets and
config extensions that builder knows. `FakeQuantRecipe.to_dict()` uses the `fake_<preset>` shorthand only
for a built-in preset and an explicit `quant_config` mapping otherwise, which
the same runtime reads back.

The selected runtime validates each concrete tensor recipe and dispatches to its
registered module-level handler while constructing the pipeline. Pipeline stages
store that handler directly; they do not retain the runtime object. A pipeline
should therefore be created by the same provider runtime that deserialized the
recipe.

## Loading YAML

`QuantRecipeRuntime.load_quant_recipe_from_yaml()` reads the same nested schema.
A present section must map at least one regex pattern to a module recipe; omit a
section to leave that module category unaffected. The whole YAML mapping is
deserialized, so an unknown top-level key raises a `ValueError` like any other
unknown key.

```python
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME

recipe = QUANT_RECIPE_RUNTIME.load_quant_recipe_from_yaml("quant_recipe.yaml")
```

Using the runtime method ensures YAML parsing and execution use the same recipe
definitions and implementations. Low-level tests and tools can invoke the same
method on an explicitly constructed runtime:

```python
from quant_suite_recipe.runtime import QuantRecipeRuntime

recipe = QuantRecipeRuntime().load_quant_recipe_from_yaml("quant_recipe.yaml")
```

A representative configuration is:

```yaml
linear_recipe:
  '.*self_attn.*':
    activation: none
    weight: fake_nvfp4
    bias: none

moe_recipe:
  '.*experts.*':
    a1: none
    w13: fake_nvfp4
    a2:
      - rht:
          type: right
          block_size: 128
          seed: 0
    w2:
      - rht:
          type: right
          block_size: 128
          seed: 0
      - quantize: fake_nvfp4

attention_recipe:
  '.*self_attn.*':
    q: fake_nvfp4
    k: fake_nvfp4
    v: fake_nvfp4
    attn_weights: none
```

Only `none`, `fake`, and `native` are accepted by the built-in runtime.
Additional types are accepted by a runtime built with their registrations.

## Custom runtimes

`QuantRecipeRuntime` takes everything it accepts as constructor arguments, each
defaulting to the built-in value, so `QuantRecipeRuntime()` is the built-in
runtime:

- `registrations`: every tensor recipe type the runtime accepts, each as a
  `(recipe_class, quantize_tensor_handler)` pair, by default
  `quant_suite_recipe.runtime.QUANT_RECIPE_REGISTRATIONS` (`none`, `fake`,
  `native`). A recipe class extends `QuantTensorRecipe` and sets the class
  attribute `discriminator` to the `type` string its payload carries; its
  `from_dict` receives the payload, which may hold `type`, and should raise
  `ValueError` for any other key it does not define;
- `config_builder`: the `QuantConfigBuilder` that parses every `QuantConfig`,
  by default `DEFAULT_QUANT_CONFIG_BUILDER`. Extend it with
  `QuantConfigExtension` classes that recipes may enable under a
  `quant_config` dict's `"extensions"` key, and with presets usable as
  `fake_<name>` and `quant_config: <name>`;
- `module_recipe_types`: the module recipe classes used for deserialization;
- `quantized_linear_type`, `quantized_attention_type`: the executor classes
  `prepare_linear` and `prepare_attention` build.

```python
from quant_suite_core.quant_config import DEFAULT_QUANT_CONFIG_BUILDER
from quant_suite_recipe.runtime import QUANT_RECIPE_REGISTRATIONS, QuantRecipeRuntime

runtime = QuantRecipeRuntime(
    (*QUANT_RECIPE_REGISTRATIONS, (MyRecipe, my_quantize_tensor)),
    config_builder=DEFAULT_QUANT_CONFIG_BUILDER.extend(
        (MyRule,),  # a quant_suite_core.extension.QuantConfigExtension
        {"mxfp4_my_rule": {"elem_dtype": "e2m1", "extensions": {"my_rule": {}}}},
    ),
)
```

Two registrations that share a discriminator are rejected, and so are an
extension or preset name the builder already knows.

The `registrations` and `config_builder` properties return what a runtime was
built with, so a framework integration can check what a runtime accepts.

A recipe type whose payload needs the runtime's config builder overrides
`QuantTensorRecipe.from_dict_with(data, config_builder)`. The default calls
`from_dict(data)`.

## Choosing a runtime

Code picks a runtime one of two ways:

- **The environment's runtime.** Integrations and applications import
  `QUANT_RECIPE_RUNTIME` from `quant_suite_recipe.runtime_provider`, so every
  integration in a process parses and executes recipes the same way.
- **An exact runtime.** Tests, golden checks and benchmarks construct the
  runtime they mean, such as `QuantRecipeRuntime()` for the built-ins only or a
  package's own runtime object, so the result does not depend on what else is
  installed.

`QUANT_RECIPE_RUNTIME` is resolved once, when `runtime_provider` is first
imported. It starts from the built-in runtime as the base. A package publishes
one complete runtime in the `quant_suite.recipe_runtimes` entry-point group:

```toml
[project.entry-points."quant_suite.recipe_runtimes"]
my_algorithm = "my_algorithm.runtime:RUNTIME"
```

| Installed runtimes | Selected |
| --- | --- |
| none | the base |
| one | that runtime |
| several | the one whose entry-point name `QUANT_SUITE_RECIPE_RUNTIME` gives; unset raises and lists them |

The base and the selected runtime are then compared by the recipe types, config
extensions and presets they accept. If the base already accepts everything the
selected runtime does, the base is kept. Otherwise the selected runtime must
accept everything the base does.

Importing `runtime_provider` raises `RuntimeError`, naming the entry point and
its package, when `QUANT_SUITE_RECIPE_RUNTIME` names no installed runtime, when
the entry point fails to import, or when neither runtime accepts everything the
other does, and `TypeError` when the entry point is not a `QuantRecipeRuntime`.

## Errors

Recipes use the same exception types as `quant_suite_core`:

- `ValueError`: the recipe or input is wrong, such as an unknown key, an unknown
  or unregistered recipe `type` (while deserializing or when a runtime
  dispatches the recipe), a missing `type`, or a tensor dtype other than
  bfloat16, float16 or float32.
- `NotImplementedError`: the recipe and input are valid, but the selected
  implementation does not support them, such as a float32 tensor for native
  `mx4-32` or `mx4-16`, `transpose=True` with a native recipe, or a native
  Linear `bias`.
- `RuntimeError`: the environment cannot run it.

An argument of the wrong Python type, such as a recipe instance that does not
match the class its `type` is bound to, raises `TypeError`. So does a recipe
field of the wrong type: `use_triton` and `init_quantized_weight` must be
bools, and an RHT `block_size` and `seed` must be ints (`seed` may also be
omitted).

## Backend details

Recipes describe quantization but do not duplicate the low-level format and
kernel documentation. See
[`quant_suite_core`](../core/README.md) for supported MX formats,
presets, scale behavior, and Hadamard primitives.
