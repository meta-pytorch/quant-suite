# tensor_dump

> Experimental. No API, numerics or serialization guarantees; see
> [`../README.md`](../README.md).

`tensor_dump` (distribution `quant_suite_tensor_dump`) records the operands and
results of the matrix multiplications in a QuantSuite model's linear and
attention layers while vLLM runs it, so they can be analysed offline. The
attention records hold everything needed to recompute the attention output,
so the recorded Q, K and V can be replayed through other attention
implementations and their outputs compared against the recorded one.

Like every experimental package it goes through the public extension points of
`quant_suite_recipe` only: a tensor recipe type, `dump`, and the executor
classes of the package's recipe runtime.

## What is recorded

| layer | tensors | taken from |
| --- | --- | --- |
| linear | `input`, `weight`, `output` | the executor's `forward`: the raw activation, the registered weight (and bias), and the layer output including the bias |
| attention | `q`, `k`, `v`, `output` | the two attention products: `Q @ K^T` gives Q and K, `P @ V` gives V and the attention output |

A recipe that only marks layers for dumping changes no value: the `dump` stage
returns its input unchanged and the executors compute exactly what the
built-in ones compute. With a `fake` stage beside it, the recorded attention
operands are the fake-quantized ones the products see.

- Linear rows cover the tokens of the step.
- Attention is recorded per request, unpadded. K and V have one head per KV
  head, not one per query head, and hold the request's whole context: the
  cached tokens and the step's new ones.
- Q and K are taken after the rotary embedding.
- Each tensor is the shard of the rank that writes it. A row-parallel linear
  layer's output is the partial sum before the all-reduce.

## The dump recipe

The `dump` tensor recipe type marks a module for dumping. Its `tensors` option
selects what the module writes; leave it out to write every tensor of the
module's kind. Recipe patterns select the layers, as for any recipe:

```yaml
linear_recipe:
  '^model\.layers\.0\.':
    activation:
      type: dump
      tensors: [input, weight, output]
  '.*':
    activation:
      type: dump
      tensors: [input, output]

attention_recipe:
  '.*':
    q:
      type: dump
      tensors: [q, k, v, output]
```

This is
[`examples/llama_3_1_8b_instruct/dump_bf16.yaml`](examples/llama_3_1_8b_instruct/dump_bf16.yaml).
The stage can go in any pipeline slot of the module; `activation` and `q` are
the convention. A module recipe holds at most one `dump` stage, and its
`tensors` must belong to the module's kind (`input`, `weight`, `output` for
linear; `q`, `k`, `v`, `output` for attention), which is checked when the
layer's executor is built.

Both sections are needed even for a model with no quantization: without
`linear_recipe` or `attention_recipe`, the QuantSuite vLLM plugin does not
attach to those layers. `lm_head` and the embedding are not QuantSuite layers
and are never recorded.

The weight of a layer is written once, at the first step the layer writes.
Writing every weight of a model is as large as its checkpoint, so the example
selects the weights of one decoder layer only.

## Environment variables

| variable | default | meaning |
| --- | --- | --- |
| `QUANT_SUITE_TENSOR_DUMP_DIR` | unset | Output root. Unset or empty disables dumping: the executors only compute, and the other variables are ignored. |
| `QUANT_SUITE_TENSOR_DUMP_STEPS` | `0:` | Step ids that may be written: a comma list of `N`, `A:B` (half-open) and `A:`, e.g. `0,5:8`. |
| `QUANT_SUITE_TENSOR_DUMP_PHASE` | `all` | Step phases that may be written: a comma list of `prefill`, `decode`, `mixed`, or `all`. |
| `QUANT_SUITE_TENSOR_DUMP_MAX_STEPS` | `1` | At most this many steps are written, counting steps that pass the other filters; `0` means no cap. |
| `QUANT_SUITE_TENSOR_DUMP_RANKS` | `all` | Comma list of the global ranks that write. |
| `VLLM_USE_V2_MODEL_RUNNER` | set by vLLM | Must be `0` while dumping: the V2 model runner's warmup forwards cannot be told apart from requests, so the dump refuses to start under it. |

A step is one model forward. Step ids count the forwards of a process from 0;
forwards without attention metadata, such as the profiling and warmup runs of
vLLM's V1 model runner, get no step id and are never written. The V2 model
runner warms up with real forwards that carry attention metadata and cannot be
told apart from requests, so the dump refuses to start under it: launch vLLM
with `VLLM_USE_V2_MODEL_RUNNER=0`, which selects the V1 runner. All ranks of a tensor- or pipeline-parallel group run the same
forwards, so their step ids agree; ranks that do not write still count steps.

The requests of a step come from the attention metadata's `query_start_loc`
and `seq_lens`, which every vLLM attention backend provides, so linear layers
can be recorded with any attention backend.

A request is `decode` when it has one query token after position 0, and
`prefill` otherwise, including chunked-prefill chunks. A step is `prefill` or
`decode` when all its requests are, and `mixed` otherwise.

A malformed value fails at startup with an error that names the variable.

## Output

```
$QUANT_SUITE_TENSOR_DUMP_DIR/
  rank_0/
    manifest.jsonl
    weights/<layer>/weight.pt               # and bias.pt
    step_000000/
      <linear layer>/input.pt
      <linear layer>/output.pt
      <attention layer>/req000.q.pt
      <attention layer>/req000.k.pt
      <attention layer>/req000.v.pt
      <attention layer>/req000.output.pt
```

Each `.pt` file holds one tensor, saved with `torch.save`. Layer directories
use the layer name verbatim. A manifest that already exists is an error, so
two runs never mix: use a new directory for each run.

`manifest.jsonl` indexes the files, one JSON object per line; read it rather
than parsing paths. Every path is relative to `rank_<R>/`.

- `run`: written first. `format_version`, `rank`, `world_size`, `tp_rank`,
  `tp_size`, `options` and `torch_version`.
- `step`: one per written step. `step`, `phase`, `num_actual_tokens`, and
  `requests`, each with `request` (its index in the step's batch), `q_len`,
  `kv_len`, `q_start_pos`, `token_start` (its first row in the linear tensors)
  and `phase`.
- `linear`: `step`, `layer` and `tensors`, which maps each written tensor to
  its `file`, `shape` and `dtype`.
- `weight`: as `linear`, for the weight and bias; `step` is the step it was
  written at.
- `attention`: `step`, `layer`, `request`, `phase`, `tensors`, and `replay`:

| `replay` field | value |
| --- | --- |
| `q_len`, `kv_len` | query tokens, and keys and values attended |
| `q_start_pos` | absolute position of the first query; always `kv_len - q_len` |
| `causal`, `causal_alignment` | `true`, `"bottom_right"`: query `i` attends to keys `0 .. q_start_pos + i` |
| `softmax_scale` | the scale applied to `Q @ K^T` |
| `num_q_heads`, `num_kv_heads`, `head_dim`, `gqa_group` | per-rank head counts; query head `h` uses KV head `h // gqa_group` |
| `sliding_window`, `logits_soft_cap`, `alibi` | `null` |
| `layout` | `"NHD"`: Q and the output are `[q_len, num_q_heads, head_dim]`, K and V are `[kv_len, num_kv_heads, head_dim]` |
| `dtype` | dtype of the tensors |
| `rope_applied` | `true` |
| `reference` | how the recorded output was computed: `Q @ K^T` accumulated and softmax taken in `float32`, `P @ V` with P in `dtype` |

`tensors` lists only the tensors that were written.

### Replaying an attention record

```python
import json
from pathlib import Path

import torch
import torch.nn.functional as F

root = Path("/tmp/llama31_8b_dump/rank_0")
records = [json.loads(line) for line in (root / "manifest.jsonl").open()]
record = next(r for r in records if r["record"] == "attention")
replay = record["replay"]
q, k, v, expected = (
    torch.load(root / record["tensors"][name]["file"]).float()
    for name in ("q", "k", "v", "output")
)

group = replay["gqa_group"]
k = k.repeat_interleave(group, dim=1)  # one KV head per query head
v = v.repeat_interleave(group, dim=1)
causal = (
    torch.arange(replay["q_len"])[:, None] + replay["q_start_pos"]
    >= torch.arange(replay["kv_len"])
)
output = F.scaled_dot_product_attention(
    q.transpose(0, 1),
    k.transpose(0, 1),
    v.transpose(0, 1),
    attn_mask=causal,
    scale=replay["softmax_scale"],
).transpose(0, 1)
print((output - expected).abs().max())
```

## Install

The package is not published to PyPI. Install it from a checkout, next to the
QuantSuite packages it extends:

```bash
git clone https://github.com/meta-pytorch/quant-suite && cd quant-suite
pip install -e core -e recipe -e vllm -e experimental/tensor_dump
```

The `vllm` extra (`-e "experimental/tensor_dump[vllm]"`) installs
`quant_suite_vllm`, which writing dumps needs; the package itself loads
without vLLM. Add the `test` extra to run the tests.

Once installed, the package's runtime is the environment's recipe runtime.
Its executors compute exactly what the built-in ones compute, so recipes
without `dump` stages give the same results; with `QUANT_SUITE_TENSOR_DUMP_DIR`
unset, nothing is written. If another package also publishes a runtime, set
`QUANT_SUITE_RECIPE_RUNTIME=tensor_dump`. A single runtime cannot combine the
`dump` type with another package's recipe types; `DumpRecipe`, `dump_tensor`,
`DumpLinear` and `DumpAttention` are public so that such a runtime can be built
by hand.

## Run with vLLM

```bash
QUANT_SUITE_TENSOR_DUMP_DIR=/tmp/llama31_8b_dump \
QUANT_SUITE_TENSOR_DUMP_MAX_STEPS=4 \
QUANT_SUITE_MHA_MICRO_BATCH_SIZE=1 \
VLLM_USE_V2_MODEL_RUNNER=0 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --quantization quant_suite \
  --hf-overrides '{"quantization_config_file":"experimental/tensor_dump/examples/llama_3_1_8b_instruct/dump_bf16.yaml"}' \
  --attention-backend CUSTOM \
  --enforce-eager \
  --dtype bfloat16 \
  --max-num-seqs 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching
```

then send one request:

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "The capital of France is", "max_tokens": 3, "temperature": 0}'
```

- `VLLM_USE_V2_MODEL_RUNNER=0` selects vLLM's V1 model runner, whose warmup
  forwards carry no attention metadata and so are never recorded.
- `--attention-backend CUSTOM` selects the QuantSuite attention backend,
  the only one whose products the executor sees.
- `--enforce-eager` runs every forward in Python; the dump raises under CUDA
  graphs or `torch.compile`.
- `QUANT_SUITE_MHA_MICRO_BATCH_SIZE=1` and `--max-num-seqs 1` each make every
  attention product hold one request, which attention dumps require. Without
  either, an attention layer with a `dump` stage fails at startup.
- `--max-num-batched-tokens` at least the prompt length makes the prefill one
  step, and `--no-enable-prefix-caching` makes it cover the whole prompt.
- `max_tokens: 3` runs three steps (a prefill and two decodes), and
  `QUANT_SUITE_TENSOR_DUMP_MAX_STEPS=4` bounds the dump if more requests
  arrive.

## Limitations

- Attention is recorded with the QuantSuite attention backend
  (`--attention-backend CUSTOM`) only. Its attention is computed with explicit products, so it is close to, but not
  bitwise equal to, fused attention kernels, and its cost grows with
  `q_len * kv_len` per request.
- MHA and GQA attention only; MLA, linear attention, sliding window and soft
  capping are not supported.
- MoE expert GEMMs, `lm_head` and the embedding are not recorded.
- Eager execution only.
- One request per attention product.
- Weights are recorded as registered: unquantized or fake-quantized. Natively
  quantized weights raise `NotImplementedError`.
- Step ids agree across tensor- and pipeline-parallel ranks, not across
  data-parallel ranks, which schedule independently. Requests cannot be linked
  across steps: `request` is an index within one step.
- vLLM's V1 model runner only (`VLLM_USE_V2_MODEL_RUNNER=0`); the V2 model
  runner raises at startup.
- Speculative decoding raises at startup, and dual-batch overlap at the first
  step.
- Every written tensor is copied to the host and saved synchronously, so
  written steps are slow.

## Tests

```bash
pip install -e "experimental/tensor_dump[test]"
pytest experimental/tensor_dump/tests
```

- `test_dump_recipe.py`: the `dump` forms, validation, round trips, the
  example recipe, and the checks made when an executor is built.
- `test_options.py`: the environment variables and their errors.
- `test_dump_linear.py`: a linear layer's files and manifest, its unchanged
  output, the step filters, dummy and nested forwards, prefill/decode labels,
  metadata without backend-specific fields, and the unsupported vLLM settings.
- `test_dump_attention.py`: the attention files at the product level: shapes,
  unexpanded K and V, several requests in one step, a prefill then a decode,
  unchanged results and the single-request checks.
- `test_dump_attention_gpu.py` (GPU and vLLM): prefill, chunked prefill,
  decode and a two-request step through the QuantSuite attention backend, one
  of them inside vLLM's own forward context; the output is unchanged and the
  dumped tensors replay to the dumped output.
- `test_runtime.py`: runtime resolution selects `RUNTIME` from its entry point,
  which builds the dump executors and still parses built-in presets.
- `test_import_boundary.py`: the package imports only public QuantSuite
  modules, and vLLM only inside functions.

## Source

`tensor_dump` is a debugging and analysis tool, not an algorithm, so there is
no paper to cite.
