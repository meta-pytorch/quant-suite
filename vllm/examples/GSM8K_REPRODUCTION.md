# Reproducing the GSM8K results

These steps reproduce the tables in
[`results/GSM8K_RESULTS.md`](results/GSM8K_RESULTS.md) with vLLM 0.30.0 and
`lm-eval[api]==0.4.13`. [`README.md`](README.md) describes the benchmark, the
paper the formats follow, and the validated models.

## 1. Install

From the QuantSuite checkout, create and activate an isolated environment:

```bash
ENV_DIR=/path/to/quant-suite-vllm-env bash examples/setup_uv.sh
# or
ENV_DIR=/path/to/quant-suite-vllm-env bash examples/setup_pip.sh

source /path/to/quant-suite-vllm-env/bin/activate
```

The launchers install vLLM 0.30.0, `lm-eval[api]==0.4.13`, and editable
copies of QuantSuite core, recipe, and vLLM integration packages.

## 2. Unified protocol

Llama 3.1 8B Instruct and Qwen3-8B use the same evaluation contract:

- lm-eval task `gsm8k_llama`, task version 3;
- fixed 8-shot chain-of-thought examples;
- `local-chat-completions`, the checkpoint chat template,
  `--apply_chat_template`, and `--fewshot_as_multiturn`;
- task defaults `max_gen_toks=1024` and `temperature=0`;
- context length 16,384, BF16 checkpoint weights, eager execution, and TP1;
- server and request seed 0, with the lm-eval harness's default seeds;
- Qwen `enable_thinking=false` on every server invocation.

The pinned checkpoints are:

| Model | Revision | Compatible Attention layers |
| --- | --- | ---: |
| `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | 0–31 |
| `Qwen/Qwen3-8B` | `2069b3fae1114555f3c020c81410e51fa0f656f2` | 0–35 |

The 16 configurations below are present for each model in
[`models.yaml`](models.yaml). The native BF16 configuration, `bf16`, maps to
`null` for every model, including OLMoE, DeepSeek, and Kimi.

| Configuration | Linear activation/weight | Attention | Backend | Client concurrency |
| --- | --- | --- | --- | ---: |
| `bf16` | native BF16 | native | `FLASH_ATTN` | 256 |
| `bf16_custom_noop` | BF16 | explicit Q/K/V/P `none` | `CUSTOM` | 8 |
| `mxfp8_core` | `fake_mxfp8` | none | `FLASH_ATTN` | 256 |
| `mxfp8_qkvp_all` | `fake_mxfp8` | MXFP8 Q/K/V/P, all layers | `CUSTOM` | 8 |
| `mxfp6_core` | `fake_mxfp6` | none | `FLASH_ATTN` | 256 |
| `mxfp6_qkvp_all` | `fake_mxfp6` | MXFP6 Q/K/V/P, all layers | `CUSTOM` | 8 |
| `mxfp4_ocp_core` | `fake_mxfp4_ocp` | none | `FLASH_ATTN` | 256 |
| `mxfp4_16_floor_core` | explicit fake E2M1, 1x16, E8M0, `pow2max`, floor | none | `FLASH_ATTN` | 256 |
| `mxfp4_core` | `fake_mxfp4` | none | `FLASH_ATTN` | 256 |
| `mxfp4_kv_all` | `fake_mxfp4` | MXFP4 K/V, all layers | `CUSTOM` | 8 |
| `mxfp4_qkvp_first` | `fake_mxfp4` | MXFP4 Q/K/V/P, layer 0 | `CUSTOM` | 8 |
| `mxfp4_qkvp_all` | `fake_mxfp4` | MXFP4 Q/K/V/P, all layers | `CUSTOM` | 8 |
| `mxfp4_mbs128_core` | static `fake_mxfp4_mbs128` | none | `FLASH_ATTN` | 256 |
| `mxfp4_mbs128_wdyn_core` | static MBS128 activation, dynamic MBS128 weight | none | `FLASH_ATTN` | 256 |
| `nvfp4_core` | `fake_nvfp4` | none | `FLASH_ATTN` | 256 |
| `nvfp4_qkvp_all` | `fake_nvfp4` | NVFP4 Q/K/V/P, all layers | `CUSTOM` | 8 |

All Linear biases are `none`. `mxfp4_16_floor_core` is fake quantization with
an explicit config, not native MXFP4. `mxfp4_mbs128_core` and
`mxfp4_mbs128_wdyn_core` are the paper's MBS-S and MBS-H: both use E0M8
macro scaling, and `mxfp4_mbs128_wdyn_core` sets `mbs_dynamic: true` only for
the weight. Recipe mappings are sparse and
order-sensitive; do not place a catch-all selector before an anchored
Attention selector.

Custom rows also set request timeout 1,800 seconds. The lower concurrency and
longer timeout are operational controls for the slower custom backend; they do
not change prompts, chat rendering, generation, or scoring.

## 3. Start vLLM

`serve.sh` defaults to `ATTENTION_BACKEND=CUSTOM`. Set
`ATTENTION_BACKEND=FLASH_ATTN` for a core-only QuantSuite row.

Llama core-only example:

```bash
ATTENTION_BACKEND=FLASH_ATTN SERVER_SEED=0 MAX_MODEL_LEN=16384 \
PORT=8000 GPU_MEMORY_UTILIZATION=0.5 \
  bash examples/serve.sh \
    meta-llama/Llama-3.1-8B-Instruct \
    examples/recipes/llama_3_1_8b_instruct/mxfp4_mbs128_core.yaml \
    gqa \
    --revision 0e9e39f249a16976918f6564b8830bc894c89659
```

Qwen custom-Attention example:

```bash
SERVER_SEED=0 MAX_MODEL_LEN=16384 PORT=8000 GPU_MEMORY_UTILIZATION=0.5 \
  bash examples/serve.sh \
    Qwen/Qwen3-8B \
    examples/recipes/qwen3_8b/mxfp4_qkvp_all.yaml \
    gqa \
    --revision 2069b3fae1114555f3c020c81410e51fa0f656f2 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

For `bf16_custom_noop`, use the same custom command shape and the model's
`bf16_custom_noop.yaml`. For `bf16`, omit QuantSuite and its recipe:

```bash
VLLM_PLUGINS= vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --served-model-name quant-suite-validation \
  --attention-backend FLASH_ATTN \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.5 \
  --seed 0 \
  --host 127.0.0.1 \
  --port 8000 \
  --enforce-eager
```

For Qwen's native control, substitute its pinned model/revision and append
`--default-chat-template-kwargs '{"enable_thinking": false}'`.

## 4. Evaluate

Use this command for a Flash row:

```bash
lm_eval run \
  --model local-chat-completions \
  --model_args model=quant-suite-validation,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=256,max_retries=3,seed=0 \
  --tasks gsm8k_llama \
  --num_fewshot 8 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size 1 \
  --output_path /path/to/output \
  --log_samples
```

Use the identical task command for a Custom row, changing only the operational
API settings:

```bash
lm_eval run \
  --model local-chat-completions \
  --model_args model=quant-suite-validation,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=8,max_retries=3,timeout=1800,seed=0 \
  --tasks gsm8k_llama \
  --num_fewshot 8 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size 1 \
  --output_path /path/to/output \
  --log_samples
```

Do not add a 256-token override: `gsm8k_llama` version 3 supplies the
1,024-token task default. Repeat the serve/evaluate sequence for all 16 entries
under the selected model in `models.yaml`.

### DeepSeek published eight-shot protocol

DeepSeek does not use lm-eval's standard five-shot prompt. The demonstrations
are GSM8K training problems (Cobbe et al., arXiv:2110.14168; MIT License) as
formatted in the DeepSeek-V2 report (arXiv:2405.04434, Appendix G); see
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). The bundled
evaluator contains the fixed eight demonstrations published with the model,
adds the test question, renders the entire prompt as one user message through
the tokenizer's checkpoint chat template, and sends the result to
`/v1/completions`. It uses greedy decoding, seed 1234, and a 2,048-token output
allowance:

```bash
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite-Chat \
  --revision 85864749cd611b4353ce1decdb286193298f64c7 \
  --local-dir /path/to/DeepSeek-V2-Lite-Chat

python examples/eval_deepseek_gsm8k.py \
  --tokenizer /path/to/DeepSeek-V2-Lite-Chat \
  --base-url http://127.0.0.1:8000 \
  --model quant-suite-validation \
  --num-concurrent 8
```

The primary metric is the last-number flexible numeric exact match used by
lm-eval's GSM8K task. The strict `####` extractor is not applicable because
this published prompt elicits `The answer is ...` rather than `#### ...`.

## 5. Paper comparison

[`results/GSM8K_RESULTS.md`](results/GSM8K_RESULTS.md#paper-references)
compares the results of this protocol with the paper's values. They are not
exact-reproduction claims: the Llama rows use this protocol, which differs from
the one used for the paper's Llama values. The paper's format comparison still
holds: MXFP4-OCP is the least accurate of the paper's low-bit formats, and MBS-S
and MBS-H stay close to NVFP4.

## 6. Experimental MX+ rows

The [MX+ rows](results/GSM8K_RESULTS.md#experimental-mx-rows) use the
experimental [`mx_plus`](../../experimental/mx_plus/) package and its
`mxfp4_plus2` preset. Install it into the environment from step 1, from the
same checkout as the other QuantSuite packages:

```bash
uv pip install --editable ../experimental/mx_plus
# or, in a pip environment
pip install --editable ../experimental/mx_plus
```

Check that the plugin selects its recipe runtime:

```bash
python examples/verify_plugin.py
```

The output should list the
`mx_plus mx_plus.runtime:RUNTIME quant_suite_mx_plus` entry point and end with
`recipe runtime: mx_plus`. With `mx_plus` as the only installed runtime
package, the plugin uses it without further configuration. Set
`QUANT_SUITE_RECIPE_RUNTIME=mx_plus` on the server only when more than one
runtime package is installed.

The recipes quantize Linear layers only, so the servers use the core-only
settings from step 3, with `FLASH_ATTN`:

```bash
ATTENTION_BACKEND=FLASH_ATTN SERVER_SEED=0 MAX_MODEL_LEN=16384 \
PORT=8000 GPU_MEMORY_UTILIZATION=0.5 \
  bash examples/serve.sh \
    meta-llama/Llama-3.1-8B-Instruct \
    ../experimental/mx_plus/examples/llama_3_1_8b_instruct/mxfp4_plus2_core.yaml \
    gqa \
    --revision 0e9e39f249a16976918f6564b8830bc894c89659

ATTENTION_BACKEND=FLASH_ATTN SERVER_SEED=0 MAX_MODEL_LEN=16384 \
PORT=8000 GPU_MEMORY_UTILIZATION=0.5 \
  bash examples/serve.sh \
    Qwen/Qwen3-8B \
    ../experimental/mx_plus/examples/qwen3_8b/mxfp4_plus2_core.yaml \
    gqa \
    --revision 2069b3fae1114555f3c020c81410e51fa0f656f2 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

Evaluate each with the Flash row command from step 4.

## Model-specific protocols

OLMoE, DeepSeek, and Kimi use their model-appropriate protocols and are not
part of the unified 32-row Llama/Qwen matrix.

- OLMoE uses its pinned revision, context 4096, its chat task, and
  its model-specific sparse MHA recipes.
- DeepSeek uses its pinned revision, context 8192, and the bundled evaluator's
  fixed eight-shot content in one checkpoint-templated user message; its
  commands are in step 4.
- Kimi uses its pinned local tokenizer, context 8192, plain completions, and
  its KDA/full-attention MLA recipes. Its server needs `TRUST_REMOTE_CODE=1`
  and GPU memory utilization 0.85.

Take the server settings from their entries in `models.yaml`. OLMoE and Kimi
are evaluated with lm-eval's `gsm8k` task through the bundled wrapper:

```bash
# OLMoE: chat completions
BASE_URL=http://127.0.0.1:8000 \
  bash examples/eval_gsm8k.sh chat /path/to/output

# Kimi: plain completions with the checkpoint's tokenizer
BASE_URL=http://127.0.0.1:8000 \
  bash examples/eval_gsm8k.sh completions /path/to/output \
    /path/to/Kimi-Linear-48B-A3B-Base
```

Their scores must not be compared as though they used the unified
`gsm8k_llama` protocol.

## Tensor parallelism

The unified matrix uses TP1.
