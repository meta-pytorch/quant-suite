# QuantSuite vLLM

`quant_suite_vllm` is the vLLM integration package for QuantSuite. Its
`QuantSuiteConfig` delegates layer dispatch to one method provider. Recipe
deserialization and Linear, MoE, Attention, MLA, and KDA setup use one recipe
runtime, selected once per process (see [Recipe runtime](#recipe-runtime));
callers do not inject runtimes into method constructors.

The package targets `vllm>=0.30,<0.31` and was validated with vLLM 0.30.0.

For a complete installation, serving, smoke-test, and lm-eval workflow, see
[`examples/README.md`](examples/README.md). The recorded limit-100 validation
matrix is in [`examples/results/QUICK_VALIDATION_RESULTS.md`](examples/results/QUICK_VALIDATION_RESULTS.md).

```python
from quant_suite_vllm.plugin import register
from quant_suite_vllm.quant_config import QuantSuiteConfig

register()
config = QuantSuiteConfig.from_config(
    {"quant_recipe": {"linear_recipe": {".*": {}}}}
)
```

The MoE method additionally receives the `FusedMoEConfig` owned by its vLLM
layer. It prepares and stores the selected activation and weight pipelines when
checkpoint loading finishes, before inference begins.

The Attention method similarly prepares its Q, K, V, and attention-weight
pipelines after loading and binds them to `QuantSuiteAttentionImpl`. The forward
path then executes causal paged attention without inspecting recipes or choosing
handlers. The backend class is available at
`quant_suite_vllm.attention.backend.QuantSuiteAttentionBackend`. Application
code selects vLLM's `CUSTOM` attention backend, and the package plugin registers
its implementation in every vLLM process. The backend supports FP16, BF16, and
FP32 with no-quant and fake-quant pipelines. Native attention quantization is
not supported. The standard server therefore needs no application-owned
registration:

```bash
vllm serve MODEL \
  --quantization quant_suite \
  --hf-overrides '{"quantization_config_file":"recipe.yaml"}' \
  --attention-backend CUSTOM
```

MHA and MLA use the batched matrix multiplication selected by the prepared
`QuantizedAttention`. When that executor does not support multi-request BMM,
MLA processes one request per chunk and MHA rejects a multi-request chunk. MLA
extracts its projection weights lazily from `kv_b_proj` and reuses them for
prefill and decode. Select it with the package selector while continuing to pass vLLM's
`CUSTOM` backend value (vLLM does not define a `CUSTOM_MLA` enum value):

```bash
QUANT_SUITE_ATTN_BACKEND=CUSTOM_MLA \
  vllm serve MODEL \
  --quantization quant_suite \
  --hf-overrides '{"quantization_config_file":"recipe.yaml"}' \
  --attention-backend CUSTOM
```

The package plugin also installs idempotent KDA adapters. When the framework
provides `KimiK3DeltaAttention`, it uses the `PluggableLayer.register_oot()`
extension point for both
`KimiGatedDeltaNetAttention` and NVIDIA `KimiK3DeltaAttention`. The adapters
preserve the upstream constructors and attach `QuantSuiteKDAMethod` only when
the engine selects `QuantSuiteConfig`. Implementations without that extension
point use an isolated constructor-patching adapter.

KDA registration is independent of the `CUSTOM` attention-backend slot. A
model that combines KDA with full-attention MLA still selects `CUSTOM_MLA` as
shown above; its KDA layers are registered separately by the package plugin.

`QuantSuiteKDAMethod` validates the layer's attention recipe when the layer is
built. When the recipe quantizes the recurrent state, the adapter wraps the
layer's forward once at that point. After checkpoint loading, the method
prepares the selected recurrent-state pipeline, which the wrapped forward reads
on each call; loading weights again replaces the prepared pipeline. No-quant
and fake quantization are supported for ordinary prefill and decode. Kimi-K3
prefill checkpoint states are also quantized when their cache indices are
present. Cache block zero remains the framework-owned null block and is never
modified. Native recurrent-state quantization and non-noop Q/K/V/attention
weight recipes are rejected when the layer is built, and speculative decoding
and RecoverSSM after checkpoint loading, rather than interpreted against an
incompatible KDA tensor contract.

## Install

The package is installed from source. From a checkout of the repository,
install the three packages:

```bash
python -m pip install ./core ./recipe ./vllm
```

Installing the package registers the `quant_suite_vllm.plugin:register` entry
point in the `vllm.general_plugins` group. vLLM discovers that metadata and
registers the `quant_suite` quantization method without copying files into
vLLM.

Applications that do not use entry-point discovery can call
`quant_suite_vllm.plugin.register()` before constructing vLLM engine arguments.

The examples install vLLM 0.30.0 and lm-evaluation-harness 0.4.13 in an
isolated environment, then install the local QuantSuite packages in editable
mode. Both pip and uv workflows are provided.

## Recipe runtime

The plugin deserializes and executes every recipe with
`quant_suite_recipe.runtime_provider.QUANT_RECIPE_RUNTIME`, the environment's
recipe runtime. It is resolved once per process, when the module is first
imported; `register()` imports it, so a misconfigured environment fails at
registration. A package adds quantization rules, presets or recipe types to
vLLM by publishing one complete runtime in the `quant_suite.recipe_runtimes`
entry-point group:

```toml
[project.entry-points."quant_suite.recipe_runtimes"]
my_algorithm = "my_algorithm.runtime:RUNTIME"
```

The selection rules, including `QUANT_SUITE_RECIPE_RUNTIME` and the errors, are
in [`../recipe/README.md`](../recipe/README.md#choosing-a-runtime). Build a
runtime from the defaults, as
[`experimental/mx_plus`](../experimental/mx_plus/README.md) does, to keep the
built-ins.

`QuantSuiteConfig.from_config` takes a mapping with the serialized recipe
under `quant_recipe` and deserializes it with the same `QUANT_RECIPE_RUNTIME`.
The mapping may also hold the `quant_method` key that vLLM sets on a
checkpoint's quantization config. Any other key raises `ValueError`, and a
`quant_recipe` that is not a mapping raises `TypeError`.

## Writing a recipe for vLLM

A recipe is a YAML mapping with up to three sections. Each section maps
module-name regexes to the recipe for the matching modules; the full schema,
shorthands and pipelines are documented in
[`../recipe/README.md`](../recipe/README.md).

| Section | Applies to | Tensor fields |
| --- | --- | --- |
| `linear_recipe` | vLLM Linear layers | `activation`, `weight`, `bias` |
| `moe_recipe` | fused MoE expert layers | `a1`, `w13`, `a2`, `w2` |
| `attention_recipe` | attention and KDA layers | `q`, `k`, `v`, `attn_weights`, `recurrent_state` |

Each tensor field takes `none`, a preset shorthand such as `fake_mxfp8` or
`native_nvfp4`, a `type:` mapping, or a list of pipeline stages.

The regexes are matched against the module prefix vLLM passes when it builds
each layer, for example:

- Linear: `model.layers.3.self_attn.qkv_proj`, `model.layers.3.mlp.down_proj`;
- MoE experts: `model.layers.3.mlp.experts`;
- attention: `model.layers.3.self_attn.attn`;
- KDA: `model.layers.3.self_attn`.

Exact names depend on the model's vLLM implementation. Patterns are tried in
order with `re.match` (anchored at the start only) and the first match wins; a
module that matches nothing gets an identity recipe. A pattern that is not a
valid regular expression raises a `ValueError` when the recipe is built. For
example, to quantize K and V of the first 16 layers only:

```yaml
linear_recipe:
  '.*':
    weight: fake_mxfp8
attention_recipe:
  '^model\.layers\.(?:[0-9]|1[0-5])\.self_attn\.attn$':
    k: fake_mxfp4
    v: fake_mxfp4
```

## Recipe transport

vLLM reads an online quantization configuration from `hf_overrides`. QuantSuite
implements vLLM's file-based configuration hook and loads the YAML recipe with
the canonical QuantSuite runtime. This preserves regex selector insertion
order. The example launcher accepts the recipe path directly:

```bash
examples/serve.sh \
  "$MODEL_ID" \
  examples/recipes/fake_mxfp8.yaml \
  mha \
  --revision "$MODEL_REVISION"
```

vLLM gives `quantization_config_file` precedence when both file and JSON
overrides are present. `quantization_config_dict_json` is also accepted; it
must preserve the insertion order of nested selector mappings.

MHA and GQA use the default QuantSuite `CUSTOM` backend. MLA uses the same vLLM
`CUSTOM` enum value, but selects QuantSuite's MLA implementation before plugin
registration:

```bash
examples/serve.sh \
  "$MODEL_ID" \
  examples/recipes/fake_mxfp8.yaml \
  mla \
  --revision "$MODEL_REVISION"
```

The KDA recipes under `examples/recipes/` quantize Linear and MoE tensors with
the selected fake-MX preset. Within Attention, they quantize only
`recurrent_state`; Q, K, V, and attention weights remain no-ops so full
attention is not changed by the KDA-specific setting.

Native MXFP4/NVFP4 execution requires the optional
`quant_suite_recipe[native]` dependency and `mslk>=1.0.0`. No-quant and fake
MXFP8/MXFP6/MXFP4 do not require MSLK. Native recurrent-state quantization is
not supported.

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `QUANT_SUITE_RECIPE_RUNTIME` | unset | Entry-point name of the recipe runtime to use when several are installed; see [Recipe runtime](#recipe-runtime). |
| `QUANT_SUITE_ATTN_BACKEND` | unset | `CUSTOM_MLA` registers the MLA implementation as vLLM's `CUSTOM` backend; otherwise the MHA implementation is registered. |
| `QUANT_SUITE_MHA_MICRO_BATCH_SIZE` | `8` | Maximum requests per MHA batched-matmul chunk. |
| `QUANT_SUITE_MLA_MHA_MICRO_BATCH_SIZE` | `32` | Requests per chunk in MLA prefill. |
| `QUANT_SUITE_MLA_MQA_MICRO_BATCH_SIZE` | `32` | Requests per chunk in MLA decode. |

Each micro-batch is one batched matrix multiplication over that many
requests, so a larger value uses more memory per launch and needs fewer
launches.

## Limitations

The MHA and MLA backends do not support sliding-window attention, ALiBi slopes
or logits soft-capping. A model that configures any of them fails with
`NotImplementedError` when the attention layer is built.

## Dependency boundary

The published package depends only on QuantSuite's public core and recipe APIs,
PyTorch, and vLLM. Additional recipe types, config extensions and presets are
provided by separate packages that publish a recipe runtime (see
[Recipe runtime](#recipe-runtime)).
