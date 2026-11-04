# QuantSuite vLLM examples

These examples reproduce the QuantSuite validation with vLLM 0.30.0
and lm-evaluation-harness 0.4.13. Llama 3.1 8B Instruct and Qwen3-8B use one
16-configuration GSM8K protocol. OLMoE, DeepSeek, and Kimi use their
model-appropriate protocols.

The low-bit formats follow [*Unveiling the Potential of Quantization with
MXFP4: Strategies for Quantization Error
Reduction*](https://arxiv.org/abs/2603.08713). The benchmark is the test split
of OpenAI's [Grade School Math (GSM8K)](https://github.com/openai/grade-school-math),
which is released under the MIT license.

## Environment

Create an isolated environment with either uv or pip. Both scripts install the
released `vllm==0.30.0` wheel, `lm-eval[api]==0.4.13`, and editable local copies
of `quant_suite_core`, `quant_suite_recipe`, and `quant_suite_vllm`.

```bash
# uv
ENV_DIR=/path/to/quant-suite-vllm-env bash examples/setup_uv.sh

# pip
ENV_DIR=/path/to/quant-suite-vllm-pip-env bash examples/setup_pip.sh
```

The scripts infer the QuantSuite source root from their own location. Override
`QUANT_SUITE_ROOT`, `PYTHON_BIN`, `VLLM_SOURCE`, or `LM_EVAL_SOURCE` for a
different checkout layout. Activate the environment and verify plugin
registration:

```bash
source /path/to/quant-suite-vllm-env/bin/activate
python examples/verify_plugin.py
```

The output should identify the `quant_suite` vLLM entry point and
`quant_suite_vllm.quant_config.QuantSuiteConfig`. A plain environment has no
entry point in the `quant_suite.recipe_runtimes` group, so it prints
`recipe runtime: built-in`.

### Experimental packages

An experimental package that publishes a recipe runtime becomes the
environment's `QUANT_RECIPE_RUNTIME`, and so the plugin's, as soon as it is
installed in the same environment as the server (see
[Recipe runtime](../README.md#recipe-runtime)). For example,
[`experimental/mx_plus`](../../experimental/mx_plus/) adds the `mx_plus` config
extension and the `mxfp4_plus2` preset:

```bash
uv pip install -e ../experimental/mx_plus  # or pip install -e in a pip env
python examples/verify_plugin.py
```

`verify_plugin.py` should now list the `mx_plus` entry point and print
`recipe runtime: mx_plus`.
[`llama_3_1_8b_instruct/mxfp4_plus2_core.yaml`](../../experimental/mx_plus/examples/llama_3_1_8b_instruct/mxfp4_plus2_core.yaml)
is a recipe that uses it:

```bash
ATTENTION_BACKEND=FLASH_ATTN examples/serve.sh \
  meta-llama/Llama-3.1-8B-Instruct \
  ../experimental/mx_plus/examples/llama_3_1_8b_instruct/mxfp4_plus2_core.yaml gqa
```

It and its Qwen3-8B counterpart, `qwen3_8b/mxfp4_plus2_core.yaml`, apply
MXFP4 + MX+ to the Linear layers. Their GSM8K scores are in
[`results/GSM8K_RESULTS.md`](results/GSM8K_RESULTS.md#experimental-mx-rows),
and the steps to reproduce them are in
[`GSM8K_REPRODUCTION.md`](GSM8K_REPRODUCTION.md#6-experimental-mx-rows).

Experimental packages carry no stability guarantee; see
[`experimental/README.md`](../../experimental/README.md).

## Models and results

[`models.yaml`](models.yaml) records immutable model revisions, server settings,
the Attention backend, operational client concurrency, and the recipe for every
published row. The five validation models are, in order:

- Llama 3.1 8B Instruct: Linear and 32-layer GQA.
- Qwen3-8B: Linear and 36-layer GQA; thinking is disabled.
- OLMoE 1B-7B Instruct: Linear, routed MoE, and MHA.
- DeepSeek V2 Lite Chat: Linear, routed MoE, and MLA.
- Kimi Linear 48B A3B Base: Linear, routed MoE, KDA, and full-attention MLA.

Each model's recipes are in `recipes/<model>/`. `models.yaml` maps every
configuration to its recipe; use that mapping rather than inferring a recipe
from a similarly named file.

- [`GSM8K_REPRODUCTION.md`](GSM8K_REPRODUCTION.md) has the exact protocol,
  the 16 Llama/Qwen configurations, and the serve and evaluate commands.
- [`results/GSM8K_RESULTS.md`](results/GSM8K_RESULTS.md) has the measured
  tables for all five models, the backend controls, and the comparison with the
  paper's values.

## Start a server

`serve.sh MODEL RECIPE {mha|gqa|mla} [vLLM arguments...]` starts vLLM with a
QuantSuite recipe. For example:

```bash
ATTENTION_BACKEND=FLASH_ATTN SERVER_SEED=0 MAX_MODEL_LEN=16384 \
PORT=8000 GPU_MEMORY_UTILIZATION=0.5 \
  bash examples/serve.sh \
    meta-llama/Llama-3.1-8B-Instruct \
    examples/recipes/llama_3_1_8b_instruct/mxfp4_mbs128_wdyn_core.yaml \
    gqa \
    --revision 0e9e39f249a16976918f6564b8830bc894c89659
```

`ATTENTION_BACKEND` defaults to `CUSTOM`, which recipes that quantize Attention
tensors need; recipes that quantize only Linear or MoE tensors can use
`FLASH_ATTN`. MLA selects QuantSuite's `CUSTOM_MLA` implementation.

The launcher passes the recipe through vLLM's `quantization_config_file`
override, preserving YAML selector order. It also accepts `HOST`, `PORT`,
`SERVED_MODEL_NAME`, `MAX_MODEL_LEN`, `TENSOR_PARALLEL_SIZE`,
`GPU_MEMORY_UTILIZATION`, `SERVER_SEED`, `ENFORCE_EAGER`, `TRUST_REMOTE_CODE`,
`PYTHON_BIN`, and `VLLM_BIN`.

## Evaluate

Follow [`GSM8K_REPRODUCTION.md`](GSM8K_REPRODUCTION.md). Llama and Qwen use
lm-eval's `gsm8k_llama` task directly. OLMoE and Kimi use `eval_gsm8k.sh`, and
DeepSeek uses `eval_deepseek_gsm8k.py` with its published fixed eight-shot
prompt.

## Quick integration check

A shorter check exercises every integration path without the full protocol.
`smoke_server.py` checks a running server's health endpoint, model list, and
one short chat completion. `eval_server.sh MODEL_LABEL RECIPE_LABEL pilot` runs
100 examples each of GSM8K and the MMLU-Pro biology subset; `full` runs the
complete GSM8K and MMLU-Pro tasks. Set `API_MODE=completions` and `TOKENIZER`
for a base checkpoint.

The generic recipes in `recipes/` apply one preset to every Linear, MoE, and
Attention tensor: `no_quant.yaml`, `fake_mxfp8.yaml`, `fake_mxfp6.yaml`, and
`fake_mxfp4.yaml`. The `kda_fake_mxfp8.yaml`, `kda_fake_mxfp6.yaml`, and
`kda_fake_mxfp4.yaml` recipes leave Q, K, V, and attention weights unquantized
and quantize the KDA recurrent state. The recorded limit-100 results for four
models are in
[`results/QUICK_VALIDATION_RESULTS.md`](results/QUICK_VALIDATION_RESULTS.md).
They are integration evidence, not accuracy results.

## Tensor-parallel fallback

The published matrix uses TP1. If a separate TP>1 experiment stalls while
initializing optional CUDA multicast or custom all-reduce backends, opt into
the Torch/distributed fallback:

```bash
VLLM_TP_TORCH_FALLBACK=1 TENSOR_PARALLEL_SIZE=2 \
  bash examples/serve.sh "$MODEL_ID" "$RECIPE" gqa \
    --revision "$MODEL_REVISION"
```

This compatibility switch can reduce communication performance and is not
part of the unified matrix.
