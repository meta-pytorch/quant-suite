# tensor_dump

> Experimental. No API, numerics or serialization guarantees; see
> [`../README.md`](../README.md).

For a step-by-step guide to collecting dumps, see [`INSTRUCTIONS.md`](INSTRUCTIONS.md).

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

The weight of a layer is written once per dump directory, at the first step
the layer writes; later runs appended to the directory check it instead (see
[Appending runs](#appending-runs)). Writing every weight of a model is as large
as its checkpoint, so the example selects the weights of one decoder layer
only.

## Environment variables

| variable | default | meaning |
| --- | --- | --- |
| `QUANT_SUITE_TENSOR_DUMP_DIR` | unset | Output root. Unset or empty disables dumping: the executors only compute, and the other variables are ignored. |
| `QUANT_SUITE_TENSOR_DUMP_STEPS` | `0:` | Step ids that may be written: a comma list of `N`, `A:B` (half-open) and `A:`, e.g. `0,5:8`. |
| `QUANT_SUITE_TENSOR_DUMP_PHASE` | `all` | Step phases that may be written: a comma list of `prefill`, `decode`, `mixed`, or `all`. |
| `QUANT_SUITE_TENSOR_DUMP_MAX_STEPS` | `1` | At most this many steps are written, counting steps that pass the other filters; `0` means no cap. |
| `QUANT_SUITE_TENSOR_DUMP_RANKS` | `all` | Comma list of the global ranks that write. |
| `QUANT_SUITE_TENSOR_DUMP_MAX_BYTES` | unset | Byte budget of each rank's directory: an integer with an optional suffix `K`, `M`, `G` or `T` (case-insensitive, powers of 1000, so `500G` is 500 * 10^9 bytes). Unset or `0` means no budget. See [Byte budget](#byte-budget). |
| `QUANT_SUITE_TENSOR_DUMP_ATTENTION` | unset | Regular expression over vLLM's attention layer names (`Attention.layer_name`, e.g. `model.layers.0.self_attn.attn`), matched with `re.match` like recipe patterns. Unset or empty records no attention layer. |
| `QUANT_SUITE_TENSOR_DUMP_ATTENTION_TENSORS` | `q,k,v,output` | Comma list of the tensors each selected attention layer writes. |
| `VLLM_USE_V2_MODEL_RUNNER` | set by vLLM | Must be `0` while dumping: the V2 model runner's warmup forwards cannot be told apart from requests, so the dump refuses to start under it. |

A step is one model forward. `QUANT_SUITE_TENSOR_DUMP_STEPS` and
`QUANT_SUITE_TENSOR_DUMP_MAX_STEPS` count the forwards of the current process
from 0; the step ids written to the manifest continue after those of earlier
runs in the same directory (see [Appending runs](#appending-runs)). The
profiling and warmup runs of vLLM's V1 model runner get no step id and are
never written: they carry no attention metadata, or, like the FlashInfer
attention warmup that vLLM runs when every attention layer uses `FLASHINFER`,
attention metadata whose `slot_mapping` gives no token a KV-cache slot (every
entry `-1`). Under decode or prefill context parallelism a rank may hold the
slots of none of a request step's tokens, so there only forwards without
attention metadata are skipped. The V2 model runner warms up with real forwards
that carry attention metadata and cannot be told apart from requests, so the
dump refuses to start under it: launch vLLM with `VLLM_USE_V2_MODEL_RUNNER=0`,
which selects the V1 runner. All ranks of a
tensor- or pipeline-parallel group run the same forwards, so their step ids
agree; ranks that do not write still count steps.

The requests of a step come from the attention metadata, so recording either
kind of layer needs an attention backend whose metadata describes them:

- `FLASH_ATTN`, `TRITON_ATTN`, `FLEX_ATTENTION`, `ROCM_ATTN` and `CPU_ATTN`
  provide `query_start_loc` and `seq_lens`.
- `FLASHINFER`, vLLM's default on GPUs of compute capability 10.x, puts a
  step's decodes before its prefills and runs each group either on a TRTLLM
  or XQA kernel, whose metadata holds the requests' lengths, or on a native
  FlashInfer wrapper, which keeps them to itself. The dump reads the first
  and raises `RuntimeError` naming the second:

  | requests | read on | raise on |
  | --- | --- | --- |
  | prefills | the TRTLLM prefill: compute capability 10.x, chosen by default for an unquantized KV cache or a quantized query, and by `--attention-config.use_trtllm_attention=1` | the native wrapper: other GPUs, decode context parallelism, `--attention-config.use_trtllm_attention=0` |
  | decodes | the TRTLLM or XQA decode: compute capability 9.0, 10.x and 12.x | the native wrapper: `--attention-config.use_trtllm_attention=0`, head counts or sizes those kernels do not support; and decode context parallelism, whose `seq_lens` are local to the rank |

  With vLLM's defaults on compute capability 10.x every step is read. On
  other GPUs only steps without prefills are, so use
  `--attention-backend FLASH_ATTN` there.
- `TRITON_FLASHINFER` gives each step either `TRITON_ATTN`'s or
  `FLASHINFER`'s metadata, read as above.

An attention layer reads its own metadata entry; `q_start_pos` is
`kv_len - q_len`. Linear and attention records share the step ids, and
attention can be recorded without any linear `dump` stage.

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
`layer_name` for attention. No file is ever overwritten.

`manifest.jsonl` indexes the files, one JSON object per line; read it rather
than parsing paths. Every path is relative to `rank_<R>/`.

- `run`: written at the first written step of each run. `format_version`
  (2), `run` (0 for the first run in the directory, then 1, 2, ...),
  `first_step` (the manifest step id of the run's process step 0), `rank`,
  `world_size`, `tp_rank`, `tp_size`, `options` and `torch_version`.
- `step`: one per written step. `step`, `phase`, `num_actual_tokens`, and
  `requests`, each with `request` (its index in the step's batch), `q_len`,
  `kv_len`, `q_start_pos`, `token_start` (its first row in the linear tensors)
  and `phase`.
- `linear`: `step`, `layer` and `tensors`, which maps each written tensor to
  its `file`, `shape`, `dtype` and `bytes` (the file size).
- `weight`: as `linear`, for the weight and bias, whose entries also hold
  `sha256`, the SHA-256 of the tensor's bytes; `step` is the step it was
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

### Appending runs

A run whose rank directory already holds a `manifest.jsonl` appends to it:

- it writes a new `run` record, and numbers its steps from `first_step`, one
  after the largest step id recorded so far, so step ids are unique across
  runs and no step directory is reused;
- a weight that an earlier run recorded is not written again. At the layer's
  first written step, its current weight and bias are compared with the
  record by shape, dtype and SHA-256; a difference means a different model or
  recipe and raises `RuntimeError`. A weight file that no record lists also
  raises;
- a manifest in another format, or with a line an interrupted run left
  incomplete, raises `RuntimeError`.

Only one process may write a rank directory at a time.

Checksums cost host time: each checked weight is copied to the host and hashed
with SHA-256 on one CPU core, so the check adds time proportional to the
selected weight bytes (roughly a second per GB). A first run hashes the host
copy it saves; a later run hashes each weight it checks and writes nothing. The
cost falls on the first written step of every run.

### Request boundaries

Requests carry no ids. A reader segments them with the per-request fields of
the `step` records (and `replay` of the `attention` records), in step-id order:

- `q_start_pos == 0` marks a request's first step;
- its next step, if written, has `q_start_pos` equal to this step's `kv_len`.

Runs do not break this: step ids keep increasing across them, and every
request of a run starts in that run. With one request per step (for example
one client request at a time), consecutive steps chain into requests by these
two rules. With several requests per step, `request` is only an index within
the step's batch and vLLM may reorder the batch between steps, so requests
cannot be followed across steps; requests that start at `q_start_pos == 0` in
a step can still be told apart within it. The rules need every token to be
computed: with prefix caching on (allowed when no attention layer is
selected), a request's first step can have `q_start_pos > 0`. Steps left out
by the step filters or the byte budget break the chains they fall in.

### Byte budget

With `QUANT_SUITE_TENSOR_DUMP_MAX_BYTES` set, each rank counts the bytes its
directory holds: the manifest and every file it lists, earlier runs included
(from the manifest's `bytes`, read once when the rank first needs it). The
count is checked when a step that would be written starts: once it has
reached the budget, no further step is written, and one INFO message says so.
A step that starts below the budget is written whole, so the directory can
end up over the budget by up to one step; a long prefill step of a large
model can be tens of GB. Ranks count only their own directory.

### Merging a dump

A dump holds one file per tensor, layer and step. `tensor_dump.merge`
rewrites it as one file per request, phase and layer:

```bash
python -m tensor_dump.merge /tmp/llama31_8b_dump /tmp/llama31_8b_merged [--delete-source] [--threads N]
```

or `tensor_dump.merge.merge(src, dst, delete_source=False)`. Merge is
I/O-bound; the command line caps torch's intra-op threads to avoid
oversubscription, at the smaller of 4 and the CPU count unless `--threads`
sets it. `merge()` called from Python leaves the caller's thread setting
alone. For each rank:

```
<dst>/rank_<R>/
  manifest.json
  weights/<layer>.pt                 # {"weight": W, "bias": b}
  req_<NNNN>/prefill/<layer>.pt      # linear: {"input", "output"}
  req_<NNNN>/decode/<layer>.pt       # attention: {"q", "k", "v", "output"}
```

- `<NNNN>` is the request's `id` (see [Reading a dump](#reading-a-dump)).
  A request's rows of a step go to the phase it has in that step, so the
  requests of a `mixed` step are split between `prefill` and `decode`.
- Each tensor stacks the request's rows of the phase along dim 0 in step
  order, which is position order: every prefill chunk, then one row per
  decode step. Linear `input` is `[T, in]` and `output` `[T, out]`;
  attention `q` and `output` are `[T, num_q_heads, head_size]`, `k` and `v`
  `[T, num_kv_heads, head_size]`. Dtypes are kept.
- `k` and `v` hold each step's new rows, so for a request whose prefill rows
  come before its decode rows, the full keys of decode row `i` are
  `cat([prefill["k"], decode["k"][: i + 1]])`.
- `manifest.json` keeps everything the per-step manifest held: the `run`
  and `step` records; per weight, its file, shapes, dtypes and SHA-256; per
  request and phase, the `segments` that map its rows to steps (`row_start`,
  `q_len`, `step`, `run`, `request`, `token_start`, `q_start_pos`,
  `kv_len`; row `row_start + j` is at position `q_start_pos + j`), each
  layer's file, shapes and dtypes, and each attention layer's `replay`
  fields other than the positions the segments give. Fields that change
  between steps are stored per row range (`replay_ranges`) instead.
- Per request it also records `prompt_len`, `generated_len`,
  `starts_at_zero` (its first step is in the dump), `may_continue` (its run
  did not write the next step, and a step filter, `MAX_STEPS` or the byte
  budget would have left that step out) and `complete` (`starts_at_zero` and
  not `may_continue`: the dump holds the request from position 0, and
  nothing the dump options left out could have continued it). The reader's
  `Request.complete` is `starts_at_zero` only. Incomplete requests are
  merged like the others.
- Each merged file is read back and compared bit for bit with the tensors it
  was made from before the manifest lists it; a difference raises
  `RuntimeError`.
- With `--delete-source`, the dump's files are deleted once everything made
  from them is verified: a step directory once every request with rows in
  it is, a weight once its merged copy is, and each rank's `manifest.jsonl`
  last. Files no manifest lists are never deleted.
- Only bytes verified against the merged copy are ever deleted. When
  `--delete-source` resumes a merge, the merged files of requests and weights
  an earlier run wrote are read again and compared bit for bit with the
  dump's files still there before any of those is deleted; a difference
  raises `RuntimeError` and deletes nothing of that request or weight.
- A merge that stops part way resumes when run again with the same
  arguments: requests and weights the destination manifest lists are checked
  against the dump and skipped. A run appended to the dump after a merge
  adds its requests on the next merge. Memory holds one request's tensors
  of one layer at a time, plus an index of the dump's manifest that grows
  with its number of records.

### Reading a dump

`tensor_dump.reader.TensorDump` reads a dump directory, all runs and ranks
included, as written or merged: a directory with any
`rank_<R>/manifest.jsonl` is read as written, otherwise its
`rank_<R>/manifest.json` as merged, and both give the same steps, requests
and tensors. It follows requests across steps by the rules of
[Request boundaries](#request-boundaries), pairs each linear layer's
activation and output with its weight, and rebuilds a request's full keys and
values at any step from the new rows of its earlier steps:

```python
from tensor_dump.reader import TensorDump

dump = TensorDump("/tmp/llama31_8b_dump")
for step in dump.steps():
    print(step.step, step.phase, [(r.q_start_pos, r.q_len) for r in step.requests])

request = dump.requests()[0]
print(request.prompt_len, request.generated_len, request.steps)

layer = "model.layers.0.self_attn.attn"
for step in request.steps:
    attention = dump.attention(layer, request, step)
    # attention.q: [q_len, Hq, D]; attention.k, attention.v: [kv_len, Hkv, D]
    print(step, attention.replay.q_start_pos, attention.k.shape)

for pair in dump.linear_pairs(phase="decode", layer=r"model\.layers\.0\."):
    tensors = dump.load_linear(pair)  # activation, weight, bias, output

print(dump.attention_layers())  # by name, layers.2 before layers.10
prefill = dump.phase_tensors(layer, request, "prefill")  # or the request's id
# {"q": [prompt_len, Hq, D], "k": ..., "v": ..., "output": ...}
```

- Tensors load on demand, and each is checked against the shape and dtype its
  manifest entry records.
- `dump.attention` raises `ValueError` when part of the history is missing:
  the request's first steps were not written (it starts after position 0), or
  one of its steps has no `k` and `v` for the layer.
- The rebuilt history is cached per layer and request until the request's last
  step is read, so reading a request's steps in order costs time linear in its
  length.
- `dump.load_linear` keeps the weight of the last layer it loaded;
  `linear_pairs` yields pairs ordered by rank, layer and step, so each weight
  is loaded once.
- `dump.requests()` raises `ValueError` when two requests cannot be told apart
  (see [Request boundaries](#request-boundaries)).
- `dump.phase_tensors` returns a merged file's tensors as they are, and
  builds the same tensors from the per-step files of a dump as written.
- In a merged dump, a step's linear tensors are assembled from the files of
  its requests, and exist only for steps whose every request was merged.
- A file that cannot be read as a torch file raises `ValueError` naming it,
  and so does a malformed manifest, with its line for `manifest.jsonl`.

The record's `replay` fields give the scale, window, soft cap and causal
offset to recompute the output with another attention implementation. A
plain fp32 recomputation:

```python
import torch

a = dump.attention(layer, request, request.steps[-1])
r = a.replay
group = r.num_q_heads // r.num_kv_heads
query_pos = torch.arange(r.q_len)[:, None] + r.q_start_pos
key_pos = torch.arange(r.kv_len)
mask = key_pos <= query_pos
if r.sliding_window is not None:
    mask &= key_pos > query_pos - r.sliding_window
output = torch.nn.functional.scaled_dot_product_attention(
    a.q.float().transpose(0, 1),
    a.k.float().repeat_interleave(group, dim=1).transpose(0, 1),
    a.v.float().repeat_interleave(group, dim=1).transpose(0, 1),
    attn_mask=mask,
    scale=r.scale,
).transpose(0, 1)
print((output - a.output.float()).abs().max())
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
  forwards carry no attention metadata or no KV-cache slots and so are never
  recorded.
- `--quantization quant_suite` with a recipe that has a `linear_recipe`
  section starts the dump session; nothing is recorded without it.
- `--enforce-eager` runs every forward in Python, so the attention layers'
  forward hooks run; the dump raises under CUDA graphs or `torch.compile`.
- `--no-enable-prefix-caching` is required while attention layers are
  selected: rebuilding a request's keys and values needs every token's to have
  been computed, and prefix caching skips cached prompt tokens. The dump
  raises at startup if vLLM's config has prefix caching on.
- The default attention backend works: `FLASH_ATTN`, or `FLASHINFER` on GPUs
  of compute capability 10.x. On other GPUs, `FLASHINFER` runs prefills on a
  path the dump cannot read (see
  [Environment variables](#environment-variables)). Several requests per step
  are recorded.
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
- Under `FLASHINFER`, a step whose prefills or decodes run on FlashInfer's
  native wrappers raises `RuntimeError` (see
  [Environment variables](#environment-variables)).
- Weights are recorded as registered: unquantized or fake-quantized. Natively
  quantized weights raise `NotImplementedError`.
- Step ids agree across tensor- and pipeline-parallel ranks, not across
  data-parallel ranks, which schedule independently.
- Requests carry no ids. They are linked across steps when each step holds
  one request (see [Request boundaries](#request-boundaries)); when several
  requests share a step, `request` is only an index within that step and
  identifies no request across steps.
- vLLM's V1 model runner only (`VLLM_USE_V2_MODEL_RUNNER=0`); the V2 model
  runner raises at startup.
- Speculative decoding raises at startup, and dual-batch overlap at the first
  step.
- Every written tensor is copied to the host and saved synchronously, so
  written steps are slow.
- One process per rank directory: concurrent runs into the same directory
  are not supported.
- A layer is recorded once per step: a module that runs twice in one forward,
  such as a shared layer, raises `RuntimeError` at its second call in a
  written step.

## Tests

```bash
pip install -e "experimental/tensor_dump[test]"
pytest experimental/tensor_dump/tests
```

- `test_dump_recipe.py`: the `dump` forms, validation, round trips, the
  example recipe, the checks made when an executor is built, and the
  rejection of `dump` stages in attention recipes.
- `test_options.py`: the environment variables and their errors.
- `test_dump_append.py`: runs appended to one directory (step ids, unchanged
  files, weights checked instead of rewritten, and the errors for a different
  weight, an unlisted weight file, another format and a truncated manifest),
  and the byte budget, earlier runs included.
- `test_dump_linear.py`: a linear layer's files and manifest, its unchanged
  output, the step filters, dummy forwards (without attention metadata or
  without KV-cache slots) and nested forwards, prefill/decode labels,
  metadata without backend-specific fields, FlashInfer metadata (decodes,
  prefills, a mixed batch, a chunked prefill, and the paths it cannot read),
  and the unsupported vLLM settings.
- `test_dump_attention.py`: the attention hooks on a stand-in for vLLM's
  `Attention` layer: files and replay fields (window and soft cap set and
  unset), unchanged outputs, several requests in one step, a chunked prefill
  then a decode holding only new rows, FlashInfer metadata mapped to rows,
  layer and tensor selection, dummy forwards (without attention metadata or
  without KV-cache slots), unsupported layers, and the prefix-caching check.
- `test_dump_attention_gpu.py` (GPU and vLLM): two requests through vLLM's
  FlashAttention kernel over a chunked prefill and decodes, with and without a
  sliding window, once inside vLLM's own forward context; the outputs are
  unchanged, the dumped rows rebuild each request's keys and values, and every
  record replays to its dumped output.
- `test_reader.py`: the reader on dumps the writer wrote over two appended
  runs, read as written and merged: runs and steps, request segmentation (a
  chunked prefill, requests in sequence, two requests in one step, and
  ambiguous requests), linear pairs, rebuilt keys and values at every step,
  phase tensors, attention layers, missing history, unreadable files and
  malformed manifests, and an fp32 recomputation of a windowed and a full
  layer.
- `test_merge.py`: merged files and manifest against the written tensors
  and the reader (a chunked prefill, requests in sequence, two requests in
  one step and a mixed step), the keys of a decode row as a slice, replay
  fields that change between steps, deleting the source, a failed
  verification deleting nothing unverified, re-verifying merged files before
  a resumed merge deletes their source, resuming, an incomplete request, and
  the command line and its thread cap.
- `test_runtime.py`: runtime resolution selects `RUNTIME` from its entry point,
  which builds the dump executors and still parses built-in presets.
- `test_import_boundary.py`: the package imports only public QuantSuite
  modules, and vLLM only inside functions.

## Source

`tensor_dump` is a debugging and analysis tool, not an algorithm, so there is
no paper to cite.
