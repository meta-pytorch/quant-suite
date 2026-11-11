# Collecting tensor dumps: a step-by-step guide

This guide walks through one complete dump with `tensor_dump`: install, write
a recipe, launch vLLM, send requests, then merge and read the result. The
worked example is `meta-llama/Llama-3.1-8B-Instruct` in BF16. Other decoder
models work the same way, including models with sliding-window attention.
The [README](README.md) is the reference for every option and file format.

## 1. What gets dumped

| layer | tensors | rows |
| --- | --- | --- |
| linear (selected by the recipe) | `input`, `output`, and `weight` (with `bias` if the layer has one) | `input` and `output`: one row per token of the step. The weight and bias are written once per dump directory. |
| attention (selected by an environment variable) | `q`, `k`, `v`, `output`, plus a `replay` record | one row per new token of each request. `k` and `v` hold only the step's new tokens. |

- Dumping changes no values. The `dump` recipe stage is an identity, the
  linear executor computes what the built-in one computes, and the attention
  hooks only read.
- Each attention record also holds `replay` metadata: `q_start_pos`, `q_len`,
  `kv_len`, `scale`, `sliding_window`, `logits_soft_cap`, the head counts and
  sizes, the dtype, and the KV-cache dtype. That is enough to recompute the
  output, and the reader rebuilds each request's full K and V for you.
- Q and K are what the model passes to the attention layer, after rotary
  embedding, QK norm and any query pre-scale. The output is taken before
  anything the model applies afterwards, such as an output gate.
- A **step** is one model forward. A request has a `prefill` step for its
  prompt (several steps if chunked prefill splits the prompt), then one
  `decode` step per generated token that is fed back. A step that holds both
  kinds of request is `mixed`.
- With tensor parallelism, each rank writes its own shard. A row-parallel
  layer's output is the partial sum before the all-reduce.

**Not covered:**

- `lm_head` and the embedding, which are not QuantSuite layers;
- MoE expert GEMMs;
- attention variants other than causal MHA/GQA decoder attention: MLA,
  cross-attention, chunked-local attention, ALiBi, attention sinks, and
  layers that share another layer's KV cache. A selected layer of one of
  these kinds raises; exclude it with the regular expression in step 4;
- vLLM's V2 model runner, CUDA graphs and `torch.compile`, which all raise;
- speculative decoding (raises at startup) and dual-batch overlap (raises at
  the first step);
- natively quantized weights, which raise when `weight` is selected.

## 2. Install

The QuantSuite vLLM plugin requires `vllm>=0.30,<0.31` (see
[`../../vllm/pyproject.toml`](../../vllm/pyproject.toml)). Install that first,
then the QuantSuite packages from a checkout:

```bash
python -m venv dump-env && source dump-env/bin/activate
pip install "vllm>=0.30,<0.31"

git clone https://github.com/meta-pytorch/quant-suite
cd quant-suite
pip install -e core -e recipe -e vllm -e experimental/tensor_dump
```

Check that the environment's recipe runtime is `tensor_dump`'s. Run this from
a directory outside the checkout (see the last row of
[Troubleshooting](#11-troubleshooting)):

```bash
cd ~ && python -c "
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME
import tensor_dump.runtime, vllm
print('vllm', vllm.__version__, vllm.__file__)
print('tensor_dump runtime:', QUANT_RECIPE_RUNTIME is tensor_dump.runtime.RUNTIME)
"
```

It must print `tensor_dump runtime: True`. If another installed package also
publishes a recipe runtime, the import raises and lists both. Then set
`QUANT_SUITE_RECIPE_RUNTIME=tensor_dump` here and on the server.

## 3. Write the recipe

The recipe selects the **linear** layers and the tensors each one writes. A
`dump` stage marks a layer. Its `tensors` option takes any of `input`,
`weight` and `output`; leave it out to write all three. Patterns are regular
expressions over the layer name, matched with `re.match` and tried in
order. The first match wins.

`dump_recipe.yaml`, which writes the input and output of every decoder-layer
linear layer and the weights of layer 0 only:

```yaml
linear_recipe:
  '^model\.layers\.0\.':
    activation:
      type: dump
      tensors: [input, weight, output]
  '^model\.layers\.[0-9]+\.':
    activation:
      type: dump
      tensors: [input, output]
  '.*': {}
```

- **The `linear_recipe` section is required, even for an attention-only
  dump.** Without it, the QuantSuite plugin leaves the linear layers to vLLM,
  and nothing starts the dump session. Building the first linear executor is
  what starts it, attention hooks included. For attention only, use:

  ```yaml
  linear_recipe:
    '.*': {}
  ```

- Leave out `attention_recipe`. Attention layers are not selected by the
  recipe, and a `dump` stage in `attention_recipe` is an error.
- A layer holds at most one `dump` stage. The weights are as large as the
  checkpoint, so select `weight` only for the layers you need.
- [`examples/llama_3_1_8b_instruct/dump_bf16.yaml`](examples/llama_3_1_8b_instruct/dump_bf16.yaml)
  is a ready-made variant.

## 4. Select attention layers

```bash
export QUANT_SUITE_TENSOR_DUMP_ATTENTION='model\.layers\.[0-9]+\.self_attn\.attn$'
export QUANT_SUITE_TENSOR_DUMP_ATTENTION_TENSORS=q,k,v,output   # the default
```

- `QUANT_SUITE_TENSOR_DUMP_ATTENTION` is a regular expression over vLLM's
  attention layer names (for example `model.layers.0.self_attn.attn`),
  matched with `re.match`. Unset or empty records no attention layer; `.*`
  records all of them.
- `QUANT_SUITE_TENSOR_DUMP_ATTENTION_TENSORS` is a comma list from `q`, `k`,
  `v` and `output`. Rebuilding K and V needs `k` and `v` at every step of the
  request.

## 5. Launch vLLM

```bash
export QUANT_SUITE_TENSOR_DUMP_DIR=$HOME/dumps/llama31_8b
export QUANT_SUITE_TENSOR_DUMP_MAX_STEPS=0        # no step cap (the default is 1)
export QUANT_SUITE_TENSOR_DUMP_MAX_BYTES=50G      # backstop; see step 6
export QUANT_SUITE_TENSOR_DUMP_ATTENTION='model\.layers\.[0-9]+\.self_attn\.attn$'
export VLLM_USE_V2_MODEL_RUNNER=0

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --quantization quant_suite \
  --hf-overrides "{\"quantization_config_file\": \"$PWD/dump_recipe.yaml\"}" \
  --enforce-eager \
  --no-enable-prefix-caching \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1
```

Required:

| setting | why |
| --- | --- |
| `--quantization quant_suite` with the recipe in `--hf-overrides` | Linear layers are recorded by QuantSuite's linear executor, and building it starts the dump session. Without it nothing is written. |
| `--enforce-eager` | The dump code must run in Python at every forward. Under CUDA graphs or `torch.compile` the dump raises. |
| `--no-enable-prefix-caching` | Rebuilding a request's K and V needs every token to be computed in some step. Prefix caching skips cached prompt tokens, so the dump refuses to start while attention layers are selected. |
| `VLLM_USE_V2_MODEL_RUNNER=0` | vLLM 0.30 defaults to the V2 model runner, whose warmup runs real forwards that look like requests. The dump refuses to start under it. |
| `QUANT_SUITE_TENSOR_DUMP_DIR` | The output root. Unset or empty disables dumping, and the other variables are ignored. |

Recommended:

- `--max-num-seqs 1`: one request per step, so the reader can follow each
  request from step to step (see step 7).
- `--max-num-batched-tokens` at least the longest prompt: each prompt is then
  one prefill step. A smaller value splits a prompt into chunks, each recorded
  as its own prefill step, which is also fine.
- Use a relative `--hf-overrides` path only if you start the server from the
  directory it is relative to.

The attention hooks capture the same tensors under every attention backend,
and the record's `backend` field says which one ran. The dump reads each
step's requests from the attention metadata's `query_start_loc` and
`seq_lens`, which vLLM's FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION, ROCM_ATTN
and CPU_ATTN backends provide and FLASHINFER does not. FLASHINFER is vLLM's
default on GPUs of compute capability 10.x, so on those GPUs add
`--attention-backend FLASH_ATTN` to the launch command. Elsewhere the default
is FLASH_ATTN and no flag is needed. vLLM's profiling and warmup forwards
carry no attention metadata, so they are never written. Nothing exists under
the dump directory until the first real step.

## 6. Control the volume

| variable | default | meaning |
| --- | --- | --- |
| `QUANT_SUITE_TENSOR_DUMP_STEPS` | `0:` | Step ids that may be written: a comma list of `N`, `A:B` (half-open) and `A:`, e.g. `0,5:8`. |
| `QUANT_SUITE_TENSOR_DUMP_PHASE` | `all` | Comma list from `prefill`, `decode` and `mixed`, or `all`. |
| `QUANT_SUITE_TENSOR_DUMP_MAX_STEPS` | `1` | At most this many steps are written, counting only steps that pass the other filters. `0` means no cap. |
| `QUANT_SUITE_TENSOR_DUMP_MAX_BYTES` | unset | Byte budget per rank directory: an integer with an optional `K`, `M`, `G` or `T` suffix (powers of 1000). Unset or `0` means no budget. |
| `QUANT_SUITE_TENSOR_DUMP_RANKS` | `all` | Comma list of the global ranks that write. Other ranks still count steps. |

Step ids count the real forwards of the server process from 0, across all
requests. With one request at a time, a later request's step ids depend on how
long the earlier ones were, so the step filters are most useful for the first
request.

Recipes for common goals (with one request in flight):

| goal | settings |
| --- | --- |
| prefill only | `PHASE=prefill`, `MAX_STEPS=0` (or `MAX_STEPS=N` for the first N prefill steps) |
| the first N decode steps | `PHASE=decode`, `MAX_STEPS=N`. Decode-only records cannot be replayed, because the prompt's K and V are missing. To replay them, also write the prefill with `PHASE=all`, `MAX_STEPS=N+1` (single-step prefill). |
| a full prompt plus N decode steps | `MAX_STEPS=0`, and send the request with `"max_tokens": N + 1`. The prefill produces the first token, and each decode step feeds one back. |
| one example at a time | `MAX_STEPS=0` and a `MAX_BYTES` backstop. Send one request, wait for it to finish, check the size, then send the next. You can also restart the server between examples (step 8). |

`MAX_STEPS` is the number of steps written, not the number of tokens: a
1,000-token prefill is one step.

**The byte budget** counts the manifest and every file it lists, earlier runs
included. It is checked when a step that would be written **starts**. Once the
count has reached the budget, no further step is written, and one INFO log line
says so. A step that starts below the budget is written whole, so the
directory can end up over the budget by up to one step. Set the budget to your
limit minus the largest step you expect, which is usually the longest prefill.

**Estimate the size before running.** With `b` bytes per element (2 for BF16):

```text
per token  = b * ( sum over dumped linear layers of (in_features + out_features)
                 + sum over dumped attention layers of (Hq*D + Hkv*D + Hkv*Dv + Hq*Dv) )
per request = (prompt tokens + decode steps) * per token
weights     = b * sum over layers with `weight` of (out_features * in_features + bias)
```

Count only the tensors you select. `Hq` and `Hkv` are the query and KV heads,
`D` the Q/K head size and `Dv` the V head size. With tensor parallelism, use
per-rank shard sizes: a column-parallel layer's input and a row-parallel
layer's (partial-sum) output are full-size on every rank.

Worked example, Llama-3.1-8B in BF16 with the recipe above and all 32
attention layers:

| part | per decoder layer, per token | 32 layers |
| --- | --- | --- |
| `qkv_proj` 4096 -> 6144, `o_proj` 4096 -> 4096, `gate_up_proj` 4096 -> 28672, `down_proj` 14336 -> 4096: 69,632 elements | 139,264 B | 4.46 MB |
| attention, Hq 32, Hkv 8, D = Dv = 128: 10,240 elements | 20,480 B | 0.66 MB |
| **total** | | **about 5.1 MB per token** |

So a 1,000-token prompt with 255 decode steps is about 1,255 x 5.1 MB = 6.4 GB.
Layer 0's weights add 0.44 GB once; all 32 layers' weights would add 14 GB.
Every step also writes `2 x (dumped linear layers) + 4 x (attention layers) x
(requests)` files (384 per step here). Each file adds container overhead, and
the file system rounds small files up to its block size. Decode steps,
whose files are small, therefore take noticeably more disk than the formula
says. Measure the first example and scale from it.

## 7. Send requests

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "The capital of France is",
  "max_tokens": 4,
  "temperature": 0
}'
```

This writes 4 steps: one prefill and three decodes. The response's
`usage.prompt_tokens` and `usage.completion_tokens` give the token counts.
Record them from the dumping server itself, since a server with other
settings can generate a different length.

Every written tensor is copied to the host and saved synchronously, so written
steps are slow. Expect a long prefill step of a large model to take seconds to
minutes.

Tips for evaluation harnesses:

- **One request at a time.** Set the harness's concurrency to 1, and pass
  `--max-num-seqs 1` to the server as well. With several requests per step,
  the reader cannot follow requests across steps.
- **No retries, long timeouts.** A retry sends the request again, so it is
  dumped twice. A timeout shorter than a slow dumped generation causes
  exactly that.
- **Selecting single examples.** If the harness samples few-shot examples in
  a way that depends on example order (for example, one seeded random stream
  that advances per example), running example `k` alone can build a
  different prompt than a full run does. Make the harness build the same
  prompt (for example, build examples `0..k` and send only `k`), and compare
  it with a full run's prompt before dumping.
- **One extra decode step.** With vLLM's asynchronous scheduling, a request
  that ends on a stop token can get one more decode step than one that ends
  at `max_tokens`. That step is a real forward that feeds back the last
  generated token, and its records are valid. Count decode steps from the
  dump, or pass `--no-async-scheduling` to avoid the extra step.

## 8. Append runs

Point a new server at the same `QUANT_SUITE_TENSOR_DUMP_DIR` to add to an
existing dump, for example after a restart or to add examples later:

- The new run writes a `run` record whose `first_step` is one past the largest
  step id recorded so far. Step ids continue and are unique across runs.
- Weights are written once. At a layer's first written step, the new run
  compares its weight and bias with the recorded shape, dtype and SHA-256
  instead of writing them again. A mismatch means a different model or recipe,
  and the run raises. The check copies and hashes each selected weight, so it
  adds time proportional to the selected weight bytes at the run's first
  written step.
- The byte budget includes earlier runs.
- Only one process may write a rank directory at a time.

Use a new directory for a different model, recipe or dtype.

## 9. Merge

A dump holds one file per tensor, layer and step. Merging rewrites it as one
file per request, phase and layer:

```bash
python -m tensor_dump.merge SRC DST [--delete-source] [--threads N]
```

```text
DST/rank_<R>/
  manifest.json
  weights/<layer>.pt                 # {"weight": W, "bias": b}
  req_<NNNN>/prefill/<layer>.pt      # linear: {"input", "output"}
  req_<NNNN>/decode/<layer>.pt       # attention: {"q", "k", "v", "output"}
```

- Each tensor stacks the request's rows of that phase along dim 0, in position
  order. The full keys of decode row `i` are
  `cat([prefill["k"], decode["k"][: i + 1]])`.
- `manifest.json` keeps the runs, steps, weight checksums and replay fields,
  and records per request `prompt_len`, `generated_len` and `complete`
  (its first step is in the dump, and no filter or budget cut it short).
- **Verification:** each merged file is read back and compared bit for bit
  with the tensors it was made from before the manifest lists it. A difference
  raises `RuntimeError`.
- **`--delete-source`** deletes a source file only once everything made from
  it is verified, and each rank's `manifest.jsonl` last. Files that no
  manifest lists are never deleted. Steps with rows that belong to no request
  are kept, with a warning.
- **Resuming:** run the same command again. Requests and weights already in
  `DST` are checked against the dump and skipped. With `--delete-source`,
  they are first read back and compared bit for bit with the source files
  still there. A run appended to the dump after a merge is added by the next
  merge.
- `--threads` caps torch's intra-op threads (default: the smaller of 4 and
  the CPU count). Merging is I/O-bound.
- Merge only a dump that no server is writing to. The merge holds an index of
  every manifest record in memory, so a dump with millions of files can need
  several GB of RAM.

A safe sequence is to merge without `--delete-source`, read a few requests
from both directories, then run the same command with `--delete-source`.

## 10. Read

`TensorDump` reads a dump as written or as merged, through the same API:

```python
from tensor_dump.reader import TensorDump

dump = TensorDump("/path/to/dump")          # or the merged directory
print(dump.ranks, dump.runs())

requests = dump.requests()                  # requests followed across steps
r = requests[0]
print(r.id, r.prompt_len, r.generated_len, r.steps)

# Linear: activation, weight, bias and output of each layer and step.
for pair in dump.linear_pairs(phase="prefill", layer=r"model\.layers\.0\."):
    t = dump.load_linear(pair)              # t.activation, t.weight, t.bias, t.output

# Attention at one step, with the request's full K/V rebuilt.
layers = dump.attention_layers()            # natural order: layers.2 before layers.10
a = dump.attention(layers[0], r, r.steps[-1])
# a.q, a.output: [q_len, Hq, D]; a.k, a.v: [kv_len, Hkv, D]; a.replay: metadata

# A request's rows of one phase, stacked: the merged file's layout.
prefill = dump.phase_tensors(layers[0], r, "prefill")   # {"q", "k", "v", "output"}
```

- `requests()` raises `ValueError` if two requests cannot be told apart,
  which can happen when several requests share steps.
- `attention()` raises `ValueError` if part of the history is missing, for
  example when the prefill step was not written.
- `linear_pairs()` takes `rank`, `steps`, `phase` and `layer` (a regular
  expression). It yields pairs by rank, layer and step, so each weight is
  loaded once.

**Replay.** `replay` gives the scale, window, soft cap and causal offset.
Query `i` sits at position `q_start_pos + i` and attends to keys up to its
own position. With a sliding window `W`, it attends to the last `W` keys,
its own included: positions `q_start_pos + i - W + 1 .. q_start_pos + i`
(FlashAttention's `window_size` is `(W - 1, 0)`).

```python
import torch

def replay(a):
    r = a.replay
    group = r.num_q_heads // r.num_kv_heads
    q = a.q.float()
    k = a.k.float().repeat_interleave(group, dim=1)
    v = a.v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) * r.scale
    if r.logits_soft_cap is not None:
        scores = r.logits_soft_cap * torch.tanh(scores / r.logits_soft_cap)
    query_pos = r.q_start_pos + torch.arange(r.q_len)[:, None]
    key_pos = torch.arange(r.kv_len)[None, :]
    mask = key_pos <= query_pos
    if r.sliding_window is not None:
        mask &= key_pos > query_pos - r.sliding_window
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)

out = replay(a)
print((out - a.output.float()).abs().max())
```

## 11. Troubleshooting

| symptom | cause | fix |
| --- | --- | --- |
| `tensor_dump does not support vLLM's V2 model runner ...` at model load | vLLM 0.30 uses the V2 runner by default | Set `VLLM_USE_V2_MODEL_RUNNER=0`. |
| `QUANT_SUITE_TENSOR_DUMP_ATTENTION is set ... Start vLLM with --no-enable-prefix-caching` | Prefix caching is on while attention layers are selected | Add `--no-enable-prefix-caching`. |
| `tensor_dump requires eager execution ...` at the first request | CUDA graphs or `torch.compile` are on | Add `--enforce-eager`. |
| `tensor_dump needs query_start_loc and seq_lens in the attention metadata ...` at the first request | The attention backend is FLASHINFER (vLLM's default on compute capability 10.x), whose metadata lacks these fields | Add `--attention-backend FLASH_ATTN`. |
| The server dies during warmup while JIT-compiling a sampling kernel (a missing CUDA header or build tool) | The system CUDA toolkit does not match the one PyTorch was built with | Set `VLLM_USE_FLASHINFER_SAMPLER=0` to use the PyTorch sampler, or install a matching toolkit. Greedy decoding picks the same tokens either way. |
| `Invalid config type: 'dump'` | The server's environment does not resolve to `tensor_dump`'s runtime: the package is missing there, or another runtime was selected | Install `tensor_dump` in the server's environment, run the check in step 2, and set `QUANT_SUITE_RECIPE_RUNTIME=tensor_dump` if several runtimes are installed. |
| `Several QuantSuite recipe runtimes are installed ...` | Another package also publishes a runtime | Set `QUANT_SUITE_RECIPE_RUNTIME=tensor_dump`. |
| No files written, and the log warns `recipe has dump stages but QUANT_SUITE_TENSOR_DUMP_DIR is unset` | The variable is not in the server's environment | Export it in the shell that starts `vllm serve`. |
| No files written, no warning | Missing `--quantization quant_suite`, or the recipe has no `linear_recipe` section | See steps 3 and 5. |
| Only one step written | `QUANT_SUITE_TENSOR_DUMP_MAX_STEPS` defaults to `1` | Set it to `0` or the number of steps you want. |
| Writing stops early, and an INFO line names `QUANT_SUITE_TENSOR_DUMP_MAX_BYTES` | The byte budget was reached | Raise the budget, or append to the dump in a new run. |
| No attention files | `QUANT_SUITE_TENSOR_DUMP_ATTENTION` is unset or matches no layer name | Check the names in the model's module tree, e.g. `model.layers.0.self_attn.attn`. |
| `tensor_dump cannot record attention layer ...` | The layer is a variant the record cannot describe (step 1) | Exclude it from `QUANT_SUITE_TENSOR_DUMP_ATTENTION`. |
| `tensor_dump would overwrite ...` | A layer ran twice in one step (a module called twice per forward), or the directory holds files no manifest lists | Leave shared layers out of the selection, or use a new directory. |
| `... in the weights an earlier run recorded ... runs a different model or recipe`, `... exists but no weight record lists it`, or `... holds a run in format ...` | Appending to a directory written by another model, recipe or format | Use a new directory. |
| `... does not end with a newline` or `line N of ... is not JSON` | An earlier run stopped while writing the manifest | Use a new directory for further runs. |
| A multi-GPU run fails with an execute-model timeout on a written step | Written steps are slower than vLLM's timeout for tensor-parallel workers | Raise `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`. |
| `import vllm` fails or loads the wrong code, or `vllm.__file__` points into your working directory | The current directory holds a `vllm/` directory that shadows the installed package | Run the server and scripts from another directory. |
