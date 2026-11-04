# QuantSuite vLLM GSM8K results

The primary matrix measures the test split of OpenAI's
[Grade School Math (GSM8K)](https://github.com/openai/grade-school-math),
released under the MIT license. Values are percentages reported as
`mean ± sample standard error`. The evaluated low-bit formats follow
[*Unveiling the Potential of Quantization with MXFP4: Strategies for
Quantization Error Reduction*](https://arxiv.org/abs/2603.08713).

## Configurations and presets

A configuration name ends with its scope. `core` quantizes every Linear
activation and weight in Llama and Qwen, and the routed-MoE `a1`, `w13`, `a2`,
and `w2` in OLMoE, DeepSeek, and Kimi. Every other scope includes `core`:
`kv_all` adds Attention K and V in every softmax-Attention layer, `qkvp_first`
adds Q, K, V, and P in the first softmax-Attention layer, `qkvp_all` adds Q, K,
V, and P in every softmax-Attention layer, and `kda_state_all` adds the
recurrent state of every KDA layer. Attention and KDA tensors use the
configuration's core preset. `bf16` and `bf16_custom_noop` quantize nothing.

The Core preset columns below name the fake-quantization format of each
configuration. The presets are entries of `FAKE_PRESETS` in
[`core/quant_suite_core/fake_presets.py`](../../../core/quant_suite_core/fake_presets.py),
except `mxfp4_plus2`, which the experimental
[`mx_plus`](../../../experimental/mx_plus/) package defines. Two Llama/Qwen
configurations write an inline config instead of a preset; they are listed by
recipe.

| Preset | Element | Block | Scale dtype | Scale normalization | Scale rounding | Second-level scale | Element rounding | Hadamard | Extension |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mxfp8` | E4M3 | 1x16 | E8M0 | `pow2max` | OAS_S | none | nearest | none | none |
| `mxfp6` | E2M3 | 1x16 | E8M0 | `pow2max` | OAS | none | nearest | none | none |
| `mxfp4` | E2M1 | 1x16 | E8M0 | `pow2max` | OAS | none | nearest | none | none |
| `mxfp4_ocp` | E2M1 | 1x32 | E8M0 | `pow2max` | floor | none | nearest | none | none |
| inline, `mxfp4_16_floor_core` | E2M1 | 1x16 | E8M0 | `pow2max` | floor | none | nearest | none | none |
| `mxfp4_mbs128` | E2M1 | 1x16 | E8M0 | `pow2max` | OAS | MBS128, E0M8, static | nearest | none | none |
| inline, `mxfp4_mbs128_wdyn_core` weight | E2M1 | 1x16 | E8M0 | `pow2max` | OAS | MBS128, E0M8, dynamic | nearest | none | none |
| `nvfp4` | E2M1 | 1x16 | FP8 (E4M3) | `amax` | none | tensorwise, FP32 | nearest | none | none |
| `mxfp4_plus2` | E2M1 | 1x32 | E8M0 | `pow2max` | floor | none | nearest | none | MX+, 2 extra mantissa bits |

- Every preset is symmetric. `mxfp4_mbs128_wdyn_core` uses `mxfp4_mbs128` for
  activations and the inline dynamic config for weights.
- Scale normalization: `pow2max` divides the block amax by the largest power of
  two in the element range; `amax` divides it by the element format's largest
  value.
- Scale rounding: floor rounds the E8M0 block scale down to a power of two. OAS
  (overflow-aware scaling) rounds down unless the block amax would then round
  past the element format's largest value, in which case it rounds up. OAS_S
  is OAS for element formats that reserve the all-ones code, such as E4M3, so
  the largest value is skipped. For `nvfp4`, the FP8 block scale is a plain
  round-to-nearest cast.
- Second-level scale: MBS128 is a macro block scale shared by the quant blocks
  in each run of 128 elements along a row. Its E0M8 factor lies in `[1, 2)`.
  Static derives it from the macro block's amax; dynamic picks the candidate
  with the smallest squared reconstruction error over the macro block.
  `nvfp4` instead has one FP32 scale per tensor.
- Element rounding: nearest rounds to nearest, ties to even. No configuration
  uses stochastic rounding (SR), and no recipe applies a Hadamard transform.
- Extension: MX+ gives each block's largest element two extra mantissa bits.

## Unified Llama/Qwen protocol

The Llama and Qwen tables use vLLM 0.30.0, lm-evaluation-harness 0.4.13, and
the `gsm8k_llama` version 3 task with fixed 8-shot CoT through local chat
completions. The full protocol and the commands are in
[`GSM8K_REPRODUCTION.md`](../GSM8K_REPRODUCTION.md#2-unified-protocol). Flash
and Custom rows differ only in client concurrency and request timeout, which do
not alter the task, prompt construction, chat rendering, generation, or
scoring.

Strict and flexible are the task's independent `strict-match` and
`flexible-extract` filters; flexible extraction is not mathematically required
to be higher.

Q, K, and V name the corresponding Attention tensors. P is `attn_weights`, the
post-softmax weights before multiplication by V. All Linear biases are
unquantized, and omitted recipe fields remain unquantized.

## Llama 3.1 8B Instruct

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Revision: `0e9e39f249a16976918f6564b8830bc894c89659`
- Compatible Attention layers: 0–31

| Configuration | Core preset | Attention | Backend | Strict match | Flexible extract |
| --- | --- | --- | --- | ---: | ---: |
| `bf16` | native BF16 | native | Flash | 86.125853 ± 0.952165 | 86.504928 ± 0.941132 |
| `bf16_custom_noop` | BF16 | explicit Q/K/V/P none | Custom GQA | 86.125853 ± 0.952165 | 86.732373 ± 0.934393 |
| `mxfp8_core` | `mxfp8` | none | Flash | 84.836998 ± 0.987933 | 85.291888 ± 0.975606 |
| `mxfp8_qkvp_all` | `mxfp8` | `mxfp8` Q/K/V/P, 32 layers | Custom GQA | 84.988628 ± 0.983859 | 85.291888 ± 0.975606 |
| `mxfp6_core` | `mxfp6` | none | Flash | 83.775588 ± 1.015513 | 84.154663 ± 1.005847 |
| `mxfp6_qkvp_all` | `mxfp6` | `mxfp6` Q/K/V/P, 32 layers | Custom GQA | 84.609553 ± 0.993980 | 84.988628 ± 0.983859 |
| `mxfp4_ocp_core` | `mxfp4_ocp` | none | Flash | 57.467779 ± 1.361801 | 58.074299 ± 1.359172 |
| `mxfp4_16_floor_core` | inline, 1x16 floor | none | Flash | 54.890068 ± 1.370646 | 55.572403 ± 1.368669 |
| `mxfp4_core` | `mxfp4` | none | Flash | 74.071266 ± 1.207141 | 74.829416 ± 1.195433 |
| `mxfp4_kv_all` | `mxfp4` | `mxfp4` K/V, 32 layers | Custom GQA | 56.557998 ± 1.365351 | 57.012889 ± 1.363634 |
| `mxfp4_qkvp_first` | `mxfp4` | `mxfp4` Q/K/V/P, layer 0 | Custom GQA | 71.038666 ± 1.249393 | 71.796816 ± 1.239493 |
| `mxfp4_qkvp_all` | `mxfp4` | `mxfp4` Q/K/V/P, 32 layers | Custom GQA | 28.506444 ± 1.243504 | 29.037149 ± 1.250359 |
| `mxfp4_mbs128_core` | `mxfp4_mbs128` | none | Flash | 78.695982 ± 1.127845 | 80.439727 ± 1.092610 |
| `mxfp4_mbs128_wdyn_core` | `mxfp4_mbs128`, dynamic weight | none | Flash | 79.226687 ± 1.117457 | 79.833207 ± 1.105230 |
| `nvfp4_core` | `nvfp4` | none | Flash | 79.605762 ± 1.109860 | 80.591357 ± 1.089392 |
| `nvfp4_qkvp_all` | `nvfp4` | `nvfp4` Q/K/V/P, 32 layers | Custom GQA | 69.143290 ± 1.272308 | 70.962851 ± 1.250359 |

## Qwen3-8B

- Model: `Qwen/Qwen3-8B`
- Revision: `2069b3fae1114555f3c020c81410e51fa0f656f2`
- Compatible Attention layers: 0–35
- Thinking: disabled

| Configuration | Core preset | Attention | Backend | Strict match | Flexible extract |
| --- | --- | --- | --- | ---: | ---: |
| `bf16` | native BF16 | native | Flash | 90.978014 ± 0.789154 | 90.978014 ± 0.789154 |
| `bf16_custom_noop` | BF16 | explicit Q/K/V/P none | Custom GQA | 91.205459 ± 0.780116 | 91.205459 ± 0.780116 |
| `mxfp8_core` | `mxfp8` | none | Flash | 91.660349 ± 0.761565 | 91.584534 ± 0.764702 |
| `mxfp8_qkvp_all` | `mxfp8` | `mxfp8` Q/K/V/P, 36 layers | Custom GQA | 90.523124 ± 0.806779 | 90.447309 ± 0.809661 |
| `mxfp6_core` | `mxfp6` | none | Flash | 91.357089 ± 0.774004 | 91.281274 ± 0.777069 |
| `mxfp6_qkvp_all` | `mxfp6` | `mxfp6` Q/K/V/P, 36 layers | Custom GQA | 91.887794 ± 0.752040 | 91.963609 ± 0.748826 |
| `mxfp4_ocp_core` | `mxfp4_ocp` | none | Flash | 73.237301 ± 1.219476 | 84.685368 ± 0.991973 |
| `mxfp4_16_floor_core` | inline, 1x16 floor | none | Flash | 75.663381 ± 1.181994 | 85.140258 ± 0.979750 |
| `mxfp4_core` | `mxfp4` | none | Flash | 86.884003 ± 0.929850 | 87.869598 ± 0.899289 |
| `mxfp4_kv_all` | `mxfp4` | `mxfp4` K/V, 36 layers | Custom GQA | 78.771797 ± 1.126378 | 82.486732 ± 1.046931 |
| `mxfp4_qkvp_first` | `mxfp4` | `mxfp4` Q/K/V/P, layer 0 | Custom GQA | 86.201668 ± 0.949978 | 89.234268 ± 0.853748 |
| `mxfp4_qkvp_all` | `mxfp4` | `mxfp4` Q/K/V/P, 36 layers | Custom GQA | 73.388931 ± 1.217275 | 75.360121 ± 1.186950 |
| `mxfp4_mbs128_core` | `mxfp4_mbs128` | none | Flash | 87.187263 ± 0.920640 | 87.642153 ± 0.906505 |
| `mxfp4_mbs128_wdyn_core` | `mxfp4_mbs128`, dynamic weight | none | Flash | 86.656558 ± 0.936649 | 87.869598 ± 0.899289 |
| `nvfp4_core` | `nvfp4` | none | Flash | 88.097043 ± 0.891970 | 88.400303 ± 0.882049 |
| `nvfp4_qkvp_all` | `nvfp4` | `nvfp4` Q/K/V/P, 36 layers | Custom GQA | 86.353298 ± 0.945574 | 86.429113 ± 0.943358 |

## Experimental MX+ rows

These rows use the experimental
[`mx_plus`](../../../experimental/mx_plus/) package and its `mxfp4_plus2`
preset: the `mxfp4_ocp` format (E2M1, 1x32 blocks, E8M0 `pow2max` scale, floor
rounding) with two extra mantissa bits on each block's largest element. They
are not part of the 16-configuration matrix above. They use the same unified
protocol, vLLM 0.30.0 and lm-evaluation-harness 0.4.13, with
`quant_suite_mx_plus` installed so the plugin selects the `mx_plus` recipe
runtime. The `bf16` and `mxfp4_ocp_core` rows are repeated from the tables
above for comparison; `mxfp4_ocp_core` differs from `mxfp4_plus2_core` only by
the MX+ extension. Steps to reproduce are in
[`GSM8K_REPRODUCTION.md`](../GSM8K_REPRODUCTION.md#6-experimental-mx-rows).

| Model | Configuration | Core preset | Attention | Backend | Strict match | Flexible extract |
| --- | --- | --- | --- | --- | ---: | ---: |
| Llama 3.1 8B Instruct | `bf16` | native BF16 | native | Flash | 86.125853 ± 0.952165 | 86.504928 ± 0.941132 |
| Llama 3.1 8B Instruct | `mxfp4_ocp_core` | `mxfp4_ocp` | none | Flash | 57.467779 ± 1.361801 | 58.074299 ± 1.359172 |
| Llama 3.1 8B Instruct | `mxfp4_plus2_core` | `mxfp4_plus2` | none | Flash | 62.168309 ± 1.335841 | 62.471569 ± 1.333717 |
| Qwen3-8B | `bf16` | native BF16 | native | Flash | 90.978014 ± 0.789154 | 90.978014 ± 0.789154 |
| Qwen3-8B | `mxfp4_ocp_core` | `mxfp4_ocp` | none | Flash | 73.237301 ± 1.219476 | 84.685368 ± 0.991973 |
| Qwen3-8B | `mxfp4_plus2_core` | `mxfp4_plus2` | none | Flash | 77.786202 ± 1.144999 | 86.201668 ± 0.949978 |

The recipes are `llama_3_1_8b_instruct/mxfp4_plus2_core.yaml` and
`qwen3_8b/mxfp4_plus2_core.yaml` in
[`experimental/mx_plus/examples/`](../../../experimental/mx_plus/examples/).
They apply MXFP4 + MX+ to every Linear activation and weight, the
configuration the MX+ paper reports; the same config can be used for any
tensor.

## Backend controls

The control comparison is native BF16 with `FLASH_ATTN` (`bf16`) versus BF16
on the `CUSTOM` GQA path with Q, K, V, and P explicitly set to `none`
(`bf16_custom_noop`). Paired differences are Custom minus Flash in percentage
points; the SE is from per-sample differences.

| Model | Metric | Paired difference ± SE | Custom-only / Flash-only | Exact McNemar p |
| --- | --- | ---: | ---: | ---: |
| Llama 3.1 8B Instruct | Strict | +0.000000 ± 0.525461 | 24 / 24 | 1.000000 |
| Llama 3.1 8B Instruct | Flexible | +0.227445 ± 0.473603 | 21 / 18 | 0.749259 |
| Qwen3-8B | Strict | +0.227445 ± 0.251467 | 7 / 4 | 0.548828 |
| Qwen3-8B | Flexible | +0.227445 ± 0.251467 | 7 / 4 | 0.548828 |

No backend-control shift is detectable at this sample size.

## Paper references

The Paper format column uses the paper's names; Configuration is the
corresponding row above.

| Model | Paper format | Configuration | Paper GSM8K | Current strict | Current flexible |
| --- | --- | --- | ---: | ---: | ---: |
| Llama | BF16 | `bf16` | 83.18 | 86.125853 | 86.504928 |
| Llama | MXFP4-OCP | `mxfp4_ocp_core` | 65.37 | 57.467779 | 58.074299 |
| Llama | MBS-S | `mxfp4_mbs128_core` | 74.37 | 78.695982 | 80.439727 |
| Llama | MBS-H | `mxfp4_mbs128_wdyn_core` | 78.52 | 79.226687 | 79.833207 |
| Llama | NVFP4 | `nvfp4_core` | 77.12 | 79.605762 | 80.591357 |
| Qwen | BF16 | `bf16` | 90.38 | 90.978014 | 90.978014 |
| Qwen | MXFP4-OCP | `mxfp4_ocp_core` | 83.83 | 73.237301 | 84.685368 |
| Qwen | MBS-S | `mxfp4_mbs128_core` | 87.84 | 87.187263 | 87.642153 |
| Qwen | MBS-H | `mxfp4_mbs128_wdyn_core` | 87.92 | 86.656558 | 87.869598 |
| Qwen | NVFP4 | `nvfp4_core` | 88.78 | 88.097043 | 88.400303 |

These columns compare paper values with results from the unified protocol; they
are not exact-reproduction claims. The Llama rows use a different evaluation
protocol from the paper's Llama values, so absolute values differ. The paper's
format comparison holds in every column: MXFP4-OCP is the least accurate of
the paper's low-bit formats, and MBS-S and MBS-H are within 1.5 points of NVFP4
under the unified protocol (within 2.75 points in the paper's values). The official
[Llama 3.1 8B Instruct model
card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) reports 84.5 for
GSM8K 8-shot CoT.

## Model-specific protocols

OLMoE, DeepSeek, and Kimi use their model-appropriate protocols, described in
[`GSM8K_REPRODUCTION.md`](../GSM8K_REPRODUCTION.md#model-specific-protocols).
They are not part of the unified 32-row Llama/Qwen matrix.

## OLMoE 1B-7B Instruct

- Model: `allenai/OLMoE-1B-7B-0924-Instruct`
- Revision: `7f1c97f440f06ce36705e4f2b843edb5925f4498`
- Protocol: chat completions with the checkpoint chat template
- Server: MHA, context 4096, GPU memory utilization 0.50
- Core: routed-MoE `a1`, `w13`, `a2`, and `w2`
- Compatible Attention layers: all 16 MHA layers

| Configuration | Core preset | Attention quantization | Strict match | Flexible extract |
| --- | --- | --- | ---: | ---: |
| `bf16` | native BF16 | none | 45.186 ± 1.371 | 44.276 ± 1.368 |
| `mxfp8_core` | `mxfp8` | none | 47.384 ± 1.375 | 46.929 ± 1.375 |
| `mxfp8_qkvp_all` | `mxfp8` | `mxfp8` Q/K/V/P, 16 layers | 45.337 ± 1.371 | 44.807 ± 1.370 |
| `mxfp6_core` | `mxfp6` | none | 46.475 ± 1.374 | 45.868 ± 1.373 |
| `mxfp6_qkvp_all` | `mxfp6` | `mxfp6` Q/K/V/P, 16 layers | 46.020 ± 1.373 | 45.262 ± 1.371 |
| `mxfp4_core` | `mxfp4` | none | 41.243 ± 1.356 | 41.243 ± 1.356 |
| `mxfp4_kv_all` | `mxfp4` | `mxfp4` K/V, 16 layers | 28.203 ± 1.239 | 28.279 ± 1.241 |
| `mxfp4_qkvp_first` | `mxfp4` | `mxfp4` Q/K/V/P, layer 0 | 35.178 ± 1.315 | 35.330 ± 1.317 |
| `mxfp4_qkvp_all` | `mxfp4` | `mxfp4` Q/K/V/P, 16 layers | 25.701 ± 1.204 | 26.232 ± 1.212 |
| `nvfp4_core` | `nvfp4` | none | 44.276 ± 1.368 | 44.049 ± 1.367 |
| `nvfp4_qkvp_all` | `nvfp4` | `nvfp4` Q/K/V/P, 16 layers | 40.106 ± 1.350 | 39.121 ± 1.344 |

## DeepSeek V2 Lite Chat

- Model: `deepseek-ai/DeepSeek-V2-Lite-Chat`
- Revision: `85864749cd611b4353ce1decdb286193298f64c7`
- Protocol: the published fixed eight-shot chain-of-thought content in one
  checkpoint-templated user message, sent to the completions endpoint
- Server: MLA, context 8192, GPU memory utilization 0.50
- Core: routed-MoE `a1`, `w13`, `a2`, and `w2`
- Compatible Attention layers: all 27 MLA layers
- Output allowance: 2,048 tokens

The published prompt elicits answers in `The answer is ...` form. Flexible
numeric exact match is therefore the primary metric. Strict match is marked
N/A because lm-eval's strict GSM8K extractor requires a `####` answer marker
that this protocol does not request.

| Configuration | Core preset | Attention quantization | Strict match | Flexible extract | Responses truncated at 2,048 tokens |
| --- | --- | --- | ---: | ---: | ---: |
| `bf16` | native BF16 | none | N/A | 69.371 ± 1.269 | 1 |
| `mxfp8_core` | `mxfp8` | none | N/A | 69.674 ± 1.266 | 2 |
| `mxfp8_qkvp_all` | `mxfp8` | `mxfp8` Q/K/V/P, 27 layers | N/A | 68.385 ± 1.280 | 1 |
| `mxfp6_core` | `mxfp6` | none | N/A | 69.295 ± 1.270 | 0 |
| `mxfp6_qkvp_all` | `mxfp6` | `mxfp6` Q/K/V/P, 27 layers | N/A | 68.309 ± 1.281 | 1 |
| `mxfp4_core` | `mxfp4` | none | N/A | 67.324 ± 1.291 | 0 |
| `mxfp4_kv_all` | `mxfp4` | `mxfp4` K/V, 27 layers | N/A | 29.416 ± 1.255 | 9 |
| `mxfp4_qkvp_first` | `mxfp4` | `mxfp4` Q/K/V/P, layer 0 | N/A | 65.807 ± 1.306 | 0 |
| `mxfp4_qkvp_all` | `mxfp4` | `mxfp4` Q/K/V/P, 27 layers | N/A | 12.737 ± 0.918 | 32 |
| `nvfp4_core` | `nvfp4` | none | N/A | 67.475 ± 1.290 | 3 |
| `nvfp4_qkvp_all` | `nvfp4` | `nvfp4` Q/K/V/P, 27 layers | N/A | 27.672 ± 1.232 | 22 |

The low MXFP4 Attention scores are not core-only evaluations. Each value in the `Responses truncated at
2,048 tokens` column is the number of the 1,319 responses that hit the output
limit and were truncated; those partial responses were still scored.

## Kimi-Linear 48B-A3B Base

- Model: `moonshotai/Kimi-Linear-48B-A3B-Base`
- Revision: `3b171c17bfc4ee348599b6781a2ca8715c21c8dc`
- Protocol: plain completions with the tokenizer from the same pinned checkpoint
- Server: KDA plus full-attention MLA, context 8192, GPU memory utilization 0.85
- Core: routed-MoE `a1`, `w13`, `a2`, and `w2`
- Compatible softmax-Attention layers: 3, 7, 11, 15, 19, 23, and 26
- KDA layers: 0–2, 4–6, 8–10, 12–14, 16–18, 20–22, 24, and 25

Kimi is a base checkpoint. The completions protocol sends prompts without a
chat template. All twelve Kimi rows use this protocol.

| Configuration | Core preset | Attention or KDA quantization | Strict match | Flexible extract |
| --- | --- | --- | ---: | ---: |
| `bf16` | native BF16 | none | 83.927 ± 1.012 | 83.927 ± 1.012 |
| `mxfp8_core` | `mxfp8` | none | 84.685 ± 0.992 | 84.610 ± 0.994 |
| `mxfp8_qkvp_all` | `mxfp8` | `mxfp8` Q/K/V/P, 7 MLA layers | 83.851 ± 1.014 | 83.851 ± 1.014 |
| `mxfp6_core` | `mxfp6` | none | 83.776 ± 1.016 | 83.700 ± 1.017 |
| `mxfp6_qkvp_all` | `mxfp6` | `mxfp6` Q/K/V/P, 7 MLA layers | 83.548 ± 1.021 | 83.472 ± 1.023 |
| `mxfp4_core` | `mxfp4` | none | 81.425 ± 1.071 | 81.425 ± 1.071 |
| `mxfp4_kv_all` | `mxfp4` | `mxfp4` K/V, 7 MLA layers | 80.364 ± 1.094 | 80.440 ± 1.093 |
| `mxfp4_qkvp_first` | `mxfp4` | `mxfp4` Q/K/V/P, MLA layer 3 | 80.440 ± 1.093 | 80.440 ± 1.093 |
| `mxfp4_qkvp_all` | `mxfp4` | `mxfp4` Q/K/V/P, 7 MLA layers | 79.757 ± 1.107 | 79.757 ± 1.107 |
| `mxfp4_kda_state_all` | `mxfp4` | `mxfp4` recurrent state, 20 KDA layers | 81.653 ± 1.066 | 81.729 ± 1.064 |
| `nvfp4_core` | `nvfp4` | none | 82.411 ± 1.049 | 82.411 ± 1.049 |
| `nvfp4_qkvp_all` | `nvfp4` | `nvfp4` Q/K/V/P, 7 MLA layers | 80.591 ± 1.089 | 80.516 ± 1.091 |

## Scientific scope

These scores use fake quantization. Llama and Qwen share the unified protocol;
OLMoE, DeepSeek, and Kimi use their model-appropriate protocols. Cross-protocol
and cross-model comparisons should therefore be interpreted cautiously.
