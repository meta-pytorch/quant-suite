# tensor_dump

> Experimental. No API, numerics or serialization guarantees; see
> [`../README.md`](../README.md).

`tensor_dump` (distribution `quant_suite_tensor_dump`) records the operands and
results of a QuantSuite model's linear layers, and the inputs and output of its
attention layers, while vLLM runs it, so they can be analysed offline. The
attention records hold everything needed to recompute the attention output,
so the recorded Q, K and V can be replayed through other attention
implementations and their outputs compared against the recorded one.

Linear layers go through the public extension points of `quant_suite_recipe`
only: a tensor recipe type, `dump`, and the linear executor of the package's
recipe runtime. Attention layers are recorded by forward hooks on vLLM's
`Attention` layer, which capture the same tensors under every attention
backend; recording needs a backend whose metadata describes the step's
requests (see [Environment variables](#environment-variables)).

## What is recorded

| layer | tensors | taken from |
| --- | --- | --- |
| linear | `input`, `weight`, `output` | the executor's `forward`: the raw activation, the registered weight (and bias), and the layer output including the bias |
| attention | `q`, `k`, `v`, `output` | a forward hook on vLLM's `Attention` layer: its `query`, `key` and `value` arguments, and the output it returns |

A recipe that only marks layers for dumping changes no value: the `dump` stage
returns its input unchanged and the executors compute exactly what the
built-in ones compute. The attention hooks only read.

- Linear rows cover the tokens of the step.
- Attention is recorded per request, unpadded. K and V have one head per KV
  head, not one per query head, and hold **only the step's new tokens**: rows
  `[q_start_pos, kv_len)` of the request's keys and values. A request's full
  keys and values at a step are the concatenation of its rows from every step
  up to that one.
- Q and K are what the model passes to the attention layer: after any rotary
  embedding, QK norm and query pre-scale. The output is the attention layer's,
  before anything the model applies after it, such as an output gate.
- Q, K and V are taken before any quantization inside the attention backend,
  such as an FP8 KV cache; `kv_cache_dtype` in the record says which.
- Each tensor is the shard of the rank that writes it. A row-parallel linear
  layer's output is the partial sum before the all-reduce.

## The dump recipe

The `dump` tensor recipe type marks a linear layer for dumping. Its `tensors`
option selects what the layer writes, from `input`, `weight` and `output`;
leave it out to write all three. Recipe patterns select the layers, as for any
recipe:

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
```

This is
[`examples/llama_3_1_8b_instruct/dump_bf16.yaml`](examples/llama_3_1_8b_instruct/dump_bf16.yaml).
The stage can go in any pipeline slot of the layer; `activation` is the
convention. A linear recipe holds at most one `dump` stage, which is checked
when the layer's executor is built.

Attention layers are not selected by the recipe: a `dump` stage in
`attention_recipe` is an error that names the environment variables below.

The recipe needs a `linear_recipe` section, even when no linear layer is
dumped (`'.*': {}` then): without it the QuantSuite vLLM plugin does not attach
to the linear layers, and building the first linear executor is what starts
the dump session, attention hooks included. `attention_recipe` is not needed.
`lm_head` and the embedding are not QuantSuite layers and are never recorded.

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
| `QUANT_SUITE_TENSOR_DUMP_ATTENTION` | unset | Regular expression over vLLM's attention layer names (`Attention.layer_name`, e.g. `model.layers.0.self_attn.attn`), matched with `re.match` like recipe patterns. Unset or empty records no attention layer. |
| `QUANT_SUITE_TENSOR_DUMP_ATTENTION_TENSORS` | `q,k,v,output` | Comma list of the tensors each selected attention layer writes. |
| `VLLM_USE_V2_MODEL_RUNNER` | set by vLLM | Must be `0` while dumping: the V2 model runner's warmup forwards cannot be told apart from requests, so the dump refuses to start under it. |

A step is one model forward. Step ids count the forwards of a process from 0;
forwards without attention metadata, such as the profiling and warmup runs of
vLLM's V1 model runner, get no step id and are never written. The V2 model
runner warms up with real forwards that carry attention metadata and cannot be
told apart from requests, so the dump refuses to start under it: launch vLLM
with `VLLM_USE_V2_MODEL_RUNNER=0`, which selects the V1 runner. All ranks of a tensor- or pipeline-parallel group run the same
forwards, so their step ids agree; ranks that do not write still count steps.

The requests of a step come from the attention metadata's `query_start_loc`
and `seq_lens`, so recording either kind of layer needs an attention backend
whose metadata has both: vLLM's `FLASH_ATTN`, `TRITON_ATTN`, `FLEX_ATTENTION`,
`ROCM_ATTN` and `CPU_ATTN`. `FLASHINFER` is not supported: the dump raises at
the first step under it. vLLM selects `FLASHINFER` by default on GPUs of
compute capability 10.x; choose another backend with
`--attention-backend FLASH_ATTN`. An attention layer reads its own metadata
entry; `q_start_pos` is `kv_len - q_len`. Linear and attention records share
the step ids, and attention can be recorded without any linear `dump` stage.

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
use the layer name verbatim: the linear layer's prefix, and vLLM's
`layer_name` for attention. A manifest that already exists is an error, so
two runs never mix: use a new directory for each run.

`manifest.jsonl` indexes the files, one JSON object per line; read it rather
than parsing paths. Every path is relative to `rank_<R>/`.

- `run`: written first. `format_version` (2), `rank`, `world_size`,
  `tp_rank`, `tp_size`, `options` and `torch_version`.
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
| `q_start_pos` | absolute position of the first query, which is also the first of the new rows in K and V; always `kv_len - q_len` |
| `q_len`, `kv_len` | the request's query tokens in this step, and the keys and values it attends to, cached and new |
| `kv_rows` | `[q_start_pos, kv_len]`: the half-open range of the request's full keys and values that `k` and `v` hold |
| `causal` | `true`: query `i` attends to keys `0 .. q_start_pos + i` |
| `scale` | the softmax scale applied to `Q @ K^T` (`impl.scale` of the layer) |
| `sliding_window` | vLLM's window `W` of the layer (`Attention.sliding_window`), or `null` for full attention. Query `i` attends to keys `q_start_pos + i - W + 1 .. q_start_pos + i`, `W` keys including its own; FlashAttention's `window_size` is `(W - 1, 0)` |
| `logits_soft_cap` | the layer's soft cap on `Q @ K^T`, or `null` for none |
| `num_q_heads`, `num_kv_heads` | per-rank head counts; query head `h` uses KV head `h // (num_q_heads // num_kv_heads)` |
| `head_size`, `head_size_v` | head size of Q and K, and of V and the output |
| `layout` | `"NHD"`: Q and the output are `[q_len, num_q_heads, head_size]`, K and V are `[q_len, num_kv_heads, head_size]` |
| `dtype` | dtype of Q as passed to the layer |
| `kv_cache_dtype` | vLLM's KV-cache dtype setting of the layer (`auto` means unquantized) |
| `backend` | vLLM's name for the attention backend that computed the output, e.g. `FLASH_ATTN` |

`tensors` lists only the tensors that were written.

### Replaying an attention record

`k` and `v` of one record hold only that step's new rows, so the full keys and
values of a request are rebuilt from its records at this step and the steps
before. With one request per step, they are the records of the same layer
from the request's first step (`q_start_pos == 0`) on:

```python
import json
from pathlib import Path

import torch
import torch.nn.functional as F

root = Path("/tmp/llama31_8b_dump/rank_0")
records = [json.loads(line) for line in (root / "manifest.jsonl").open()]
layer = "model.layers.0.self_attn.attn"
history = [r for r in records if r["record"] == "attention" and r["layer"] == layer]
record = history[-1]
replay = record["replay"]
first = max(i for i, r in enumerate(history) if r["replay"]["q_start_pos"] == 0)


def load(r, name):
    return torch.load(root / r["tensors"][name]["file"]).float()


k = torch.cat([load(r, "k") for r in history[first:]])
v = torch.cat([load(r, "v") for r in history[first:]])
q, expected = load(record, "q"), load(record, "output")

group = replay["num_q_heads"] // replay["num_kv_heads"]
query_pos = torch.arange(replay["q_len"])[:, None] + replay["q_start_pos"]
key_pos = torch.arange(replay["kv_len"])
mask = key_pos <= query_pos
if replay["sliding_window"] is not None:
    mask &= key_pos > query_pos - replay["sliding_window"]
output = F.scaled_dot_product_attention(
    q.transpose(0, 1),
    k.repeat_interleave(group, dim=1).transpose(0, 1),
    v.repeat_interleave(group, dim=1).transpose(0, 1),
    attn_mask=mask,
    scale=replay["scale"],
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
`dump` type with another package's recipe types; `DumpRecipe`, `dump_tensor`
and `DumpLinear` are public so that such a runtime can be built by hand.

## Run with vLLM

```bash
QUANT_SUITE_TENSOR_DUMP_DIR=/tmp/llama31_8b_dump \
QUANT_SUITE_TENSOR_DUMP_MAX_STEPS=4 \
QUANT_SUITE_TENSOR_DUMP_ATTENTION='.*' \
VLLM_USE_V2_MODEL_RUNNER=0 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --quantization quant_suite \
  --hf-overrides '{"quantization_config_file":"experimental/tensor_dump/examples/llama_3_1_8b_instruct/dump_bf16.yaml"}' \
  --enforce-eager \
  --dtype bfloat16 \
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
- `--quantization quant_suite` with a recipe that has a `linear_recipe`
  section starts the dump session; nothing is recorded without it.
- `--enforce-eager` runs every forward in Python, so the attention layers'
  forward hooks run; the dump raises under CUDA graphs or `torch.compile`.
- `--no-enable-prefix-caching` is required while attention layers are
  selected: rebuilding a request's keys and values needs every token's to have
  been computed, and prefix caching skips cached prompt tokens. The dump
  raises at startup if vLLM's config has prefix caching on.
- The attention backend must not be `FLASHINFER`, vLLM's default on GPUs of
  compute capability 10.x; add `--attention-backend FLASH_ATTN` there. Several
  requests per step are recorded.
- `--max-num-batched-tokens` at least the prompt length makes the prefill one
  step; a smaller value splits it into chunks, which are recorded as separate
  steps.
- `max_tokens: 3` runs three steps (a prefill and two decodes), and
  `QUANT_SUITE_TENSOR_DUMP_MAX_STEPS=4` bounds the dump if more requests
  arrive.

## Limitations

- Attention records describe causal MHA/GQA decoder attention, with or without
  a sliding window and a soft cap. A selected layer that is a subclass of
  vLLM's `Attention` (such as chunked-local or cross attention), uses ALiBi or
  attention sinks, shares another layer's KV cache, or is not a decoder layer
  raises `NotImplementedError` at the first forward; exclude it with
  `QUANT_SUITE_TENSOR_DUMP_ATTENTION`. MLA layers are not `Attention` layers
  and are never matched.
- The bidirectional ranges of multimodal prefix-LM models are not recorded;
  records of text-only requests replay as causal.
- MoE expert GEMMs, `lm_head` and the embedding are not recorded.
- Eager execution only.
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
  example recipe, the checks made when an executor is built, and the
  rejection of `dump` stages in attention recipes.
- `test_options.py`: the environment variables and their errors.
- `test_dump_linear.py`: a linear layer's files and manifest, its unchanged
  output, the step filters, dummy and nested forwards, prefill/decode labels,
  metadata without backend-specific fields, and the unsupported vLLM settings.
- `test_dump_attention.py`: the attention hooks on a stand-in for vLLM's
  `Attention` layer: files and replay fields (window and soft cap set and
  unset), unchanged outputs, several requests in one step, a chunked prefill
  then a decode holding only new rows, layer and tensor selection, dummy
  forwards, unsupported layers, and the prefix-caching check.
- `test_dump_attention_gpu.py` (GPU and vLLM): two requests through vLLM's
  FlashAttention kernel over a chunked prefill and decodes, with and without a
  sliding window, once inside vLLM's own forward context; the outputs are
  unchanged, the dumped rows rebuild each request's keys and values, and every
  record replays to its dumped output.
- `test_runtime.py`: runtime resolution selects `RUNTIME` from its entry point,
  which builds the dump executors and still parses built-in presets.
- `test_import_boundary.py`: the package imports only public QuantSuite
  modules, and vLLM only inside functions.

## Source

`tensor_dump` is a debugging and analysis tool, not an algorithm, so there is
no paper to cite.
