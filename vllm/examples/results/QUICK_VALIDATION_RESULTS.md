# QuantSuite vLLM 0.30 limit-100 validation

These measurements validate integration behavior; they are not full-benchmark
quality claims. Each row uses 100 GSM8K examples and 100 examples from the
MMLU-Pro biology subset. Values are `metric ± standard error`.

## Common configuration

- vLLM 0.30.0
- lm-evaluation-harness 0.4.13
- QuantSuite packages installed from an editable local checkout
- BF16 checkpoint weights on one NVIDIA B200
- vLLM eager execution and tensor parallelism 1
- `local-chat-completions` with the checkpoint chat template
- Five-shot task defaults and deterministic greedy generation
- Server/request seed `1234`
- Harness seeds `0,1234,1234,1234`
- `--limit 100` for both `gsm8k` and `mmlu_pro_biology`

| Model | Hugging Face revision | Coverage | Backend | Context |
| --- | --- | --- | --- | ---: |
| `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | Linear, GQA | `CUSTOM` | 4096 |
| `allenai/OLMoE-1B-7B-0924-Instruct` | `7f1c97f440f06ce36705e4f2b843edb5925f4498` | Linear, routed MoE, MHA | `CUSTOM` | 4096 |
| `deepseek-ai/DeepSeek-V2-Lite-Chat` | `85864749cd611b4353ce1decdb286193298f64c7` | Linear, routed MoE, MLA | `CUSTOM_MLA` through vLLM `CUSTOM` | 8192 |
| `moonshotai/Kimi-Linear-48B-A3B-Base` | `3b171c17bfc4ee348599b6781a2ca8715c21c8dc` | Linear, routed MoE, KDA, full-attention MLA | `CUSTOM_MLA` through vLLM `CUSTOM` | 8192 |

The listed Hugging Face revisions are the reproduction pins; the measured
checkpoints were not verified byte-for-byte against them.
QuantSuite itself was editable and not commit-pinned across every preset, so
these measurements are integration evidence rather than a controlled
cross-preset accuracy comparison.

## Llama 3.1 8B Instruct

| Preset | Smoke | GSM8K strict | GSM8K flexible | MMLU-Pro biology | Evaluation time |
| --- | --- | ---: | ---: | ---: | ---: |
| `no_quant` | `OK`, 0.19 s | 0.80 ± 0.0402 | 0.81 ± 0.0394 | 0.63 ± 0.0485 | 456.06 s |
| `fake_mxfp8` | `OK`, 0.27 s | 0.74 ± 0.0441 | 0.78 ± 0.0416 | 0.65 ± 0.0479 | 598.37 s |
| `fake_mxfp6` | `OK`, 0.26 s | 0.81 ± 0.0394 | 0.81 ± 0.0394 | 0.57 ± 0.0498 | 613.73 s |
| `fake_mxfp4` | `OK`, 0.27 s | 0.29 ± 0.0456 | 0.36 ± 0.0482 | 0.24 ± 0.0429 | 1,270.98 s |

## OLMoE 1B-7B Instruct

| Preset | Smoke | GSM8K strict | GSM8K flexible | MMLU-Pro biology | Startup | Evaluation time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `no_quant` | `OK.` | 0.5000 ± 0.0503 | 0.5000 ± 0.0503 | 0.2000 ± 0.0402 | 44 s | 257.46 s |
| `fake_mxfp8` | `OK. How can I assist you today` | 0.5000 ± 0.0503 | 0.4900 ± 0.0502 | 0.3000 ± 0.0461 | 51 s | 370.33 s |
| `fake_mxfp6` | `OK.` | 0.4500 ± 0.0500 | 0.4400 ± 0.0499 | 0.2600 ± 0.0441 | 49 s | 380.56 s |
| `fake_mxfp4` | `OK, I understand. How can I` | 0.1800 ± 0.0386 | 0.1700 ± 0.0378 | 0.1900 ± 0.0394 | 40 s | 481.57 s |

## DeepSeek V2 Lite Chat

| Preset | Smoke | GSM8K strict | GSM8K flexible | MMLU-Pro biology | Startup | Evaluation time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `no_quant` | `OK` | 0.02 ± 0.0141 | 0.54 ± 0.0501 | 0.25 ± 0.0435 | 34 s | 452.04 s |
| `fake_mxfp8` | `OK` | 0.02 ± 0.0141 | 0.45 ± 0.0500 | 0.22 ± 0.0416 | 34 s | 749.28 s |
| `fake_mxfp6` | `OK` | 0.02 ± 0.0141 | 0.42 ± 0.0496 | 0.29 ± 0.0456 | 35 s | 709.70 s |
| `fake_mxfp4` | `I understand that you want me to answer` | 0.00 ± 0.0000 | 0.01 ± 0.0100 | 0.01 ± 0.0100 | 35 s | 1,686.36 s |

DeepSeek uses an 8192-token context: an MMLU-Pro request can have at least
2,049 input tokens and request 2,048 output tokens, which exceeds 4096.

## Kimi Linear 48B A3B Base

The fake KDA recipes quantize Linear and MoE tensors with the selected preset.
Within Attention, they quantize only `recurrent_state`; Q, K, V, and attention
weights remain no-ops.

| Preset | Smoke | GSM8K strict | GSM8K flexible | MMLU-Pro biology | Startup | Evaluation time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `no_quant` | `nOK:OK:OK:OK` | 0.7900 ± 0.0409 | 0.7600 ± 0.0429 | 0.6100 ± 0.0490 | 128 s | 898.21 s |
| `kda_fake_mxfp8` | `Reply with exactly: OK: 1` | 0.8100 ± 0.0394 | 0.8000 ± 0.0402 | 0.5800 ± 0.0496 | 48 s | 1,120.10 s |
| `kda_fake_mxfp6` | `Reply with exactly: OK: 1` | 0.7900 ± 0.0409 | 0.7900 ± 0.0409 | 0.6300 ± 0.0485 | 38 s | 702.70 s |
| `kda_fake_mxfp4` | `Reply with exactly: OK}Reply with` | 0.6800 ± 0.0469 | 0.6500 ± 0.0479 | 0.6000 ± 0.0492 | 39 s | 1,287.99 s |

Kimi runs at tensor parallelism 1; the checkpoint is approximately 92 GiB and
fits on one B200.

## Setup

lm-eval's API models need `lm-eval[api]==0.4.13`, which supplies `tenacity`,
`openai`, and `tiktoken`. The provided setup scripts install this extra.

## Interpretation

The smoke responses demonstrate a healthy generation path, not exact
instruction following; some models continued after the requested `OK`. The
scores cover only 100 examples per task and one MMLU-Pro subject, so apparent
differences within the displayed standard errors should not be interpreted as
full-benchmark conclusions. Use `eval_server.sh ... full` for complete GSM8K
and aggregate MMLU-Pro evaluation.
