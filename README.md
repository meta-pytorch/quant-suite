# QuantSuite: A Toolkit for Low-Precision Quantization Research

QuantSuite is a PyTorch toolkit for low-precision quantization research. It
focuses on bit-accurate emulation (fake quantization, or QDQ) of block-scaled
formats such as MXFP4/6/8 and NVFP4, and has limited support for native
quantization to packed formats. Serializable model recipes describe how each
tensor of Linear, MoE and attention layers is transformed and quantized, an
extension mechanism adds quantization rules without changing the core
packages, and a vLLM plugin runs quantized-model evaluations.

QuantSuite is meant for numerics research and evaluation, not for deploying
quantized models.

## What QuantSuite provides

- **Fake quantization.** `fake_quantize` quantizes and dequantizes a tensor as
  set by a `QuantConfig`, whose element format, scale format, block shape (1xN
  or 2D), scale rounding, element rounding (nearest or stochastic) and global
  scale (tensorwise, rowwise or macro block scale) are chosen independently.
  Named presets cover MXFP4/6/8, NVFP4 and their block-size, rounding and
  macro-block-scale variants. There are two backends: `qdq`, the eager PyTorch
  reference, which runs on CPU or GPU, and `qdq_triton_mega`, a fused Triton
  kernel.
- **Native quantization (limited).** `native_quantize` and `native_dequantize`
  produce packed MXFP4, MXFP8 and NVFP4 data and scales, and Linear layers can
  run native FP4 matrix multiplications through the optional MSLK kernels.
  Triton FP4 packing and the MSLK kernels need an SM100 GPU; see the
  [core](core/README.md) and [recipe](recipe/README.md) READMEs for details.
- **Recipe-level transformations.** Each tensor's recipe is an ordered
  pipeline of stages, so a transformation can run before or after
  quantization. The available transformation is currently the randomized
  Hadamard transform (RHT), an `rht` stage with right, left and inverse modes,
  a power-of-two block size and an optional seed. For example, the same right
  RHT on an MoE layer's `a2` and `w2` cancels in their matrix multiplication
  while `w2` is quantized in the rotated basis.
- **Layer coverage.** Recipes map module-name regexes to per-tensor pipelines
  for:
  - Linear layers: activation, weight and bias;
  - MoE layers: the activations and weights of both expert matrix
    multiplications (`a1`, `w13`, `a2`, `w2`);
  - attention layers: Q, K, V, the attention weights (softmax output), and the
    recurrent state of KDA linear-attention layers.
- **Extensions.** A `QuantConfigExtension` adds an optional quantization rule
  with typed options to `QuantConfig`, through hooks on the scale computation,
  the post-processing and the final result. A package can also add tensor
  recipe types and presets, and publish a recipe runtime through an entry
  point that the vLLM plugin then uses. The experimental MX+ package is built
  this way.
- **vLLM plugin.** `quant_suite_vllm` registers the `quant_suite` quantization
  method, which applies recipes to Linear layers and fused MoE experts. It
  provides attention backends for multi-head attention (including grouped-query
  attention) and multi-head latent attention (MLA), and quantizes the recurrent
  state of Kimi Delta Attention (KDA) layers. Attention and KDA layers accept
  no-quant and fake-quant recipes.

## Packages

| Package | Directory | Purpose |
| --- | --- | --- |
| `quant_suite_core` | [`core/`](core/README.md) | Tensor-level building blocks: fake QDQ for MX, NVFP4 and related formats, native packed MXFP4/MXFP8/NVFP4, Hadamard and Triton utilities |
| `quant_suite_recipe` | [`recipe/`](recipe/README.md) | Typed, serializable recipes for tensors, transform pipelines, Linear, MoE and attention modules, and the runtime that executes them |
| `quant_suite_vllm` | [`vllm/`](vllm/README.md) | vLLM integration: the `quant_suite` quantization method, MHA and MLA attention backends, and KDA recurrent-state quantization |
| `quant_suite_mx_plus` | [`experimental/mx_plus/`](experimental/mx_plus/README.md) | Experimental MX+ extension and the template for experimental packages; see [`experimental/`](experimental/README.md) |

`quant_suite_recipe` depends on `quant_suite_core`, and `quant_suite_vllm`
depends on both. Experimental packages have no stability guarantee.

## Install

All packages require Python 3.11 or newer and PyTorch 2.13 or newer.
`quant_suite_vllm` targets `vllm>=0.30,<0.31`.

The packages are installed from source. From a checkout of this repository:

```bash
python -m pip install ./core ./recipe ./vllm
```

To work on the code, including the experimental MX+ package, install in
editable mode:

```bash
git clone https://github.com/meta-pytorch/quant-suite && cd quant-suite
pip install -e core -e recipe -e vllm -e experimental/mx_plus
```

## Quick start

```python
import torch
from quant_suite_core import fake_quantize, native_dequantize, native_quantize, QuantConfig
from quant_suite_core.fake_presets import FAKE_PRESETS

# MXFP4 with 1x16 blocks, from a preset; any mapping of QuantConfig fields works.
config = QuantConfig.from_dict(FAKE_PRESETS["mxfp4"])
x = torch.randn(128, 256)

# Eager reference backend: runs on CPU.
y = fake_quantize(x, "x", config, use_triton=False)

# Triton backend (the default): needs a CUDA device.
if torch.cuda.is_available():
    y_gpu = fake_quantize(x.cuda(), "x", config)

# The dict form round-trips.
assert QuantConfig.from_dict(config.to_dict()) == config

# Native packed MXFP4 (1x32 OCP blocks): packed uint8 data and E8M0 scales.
native_config = QuantConfig.from_dict(FAKE_PRESETS["mxfp4_ocp"])
packed = native_quantize(x, "x", native_config, use_triton=False)
x_hat = native_dequantize(packed, output_dtype=torch.float32)
```

The Triton backend needs an SM90 or newer GPU, and native FP4 packing needs
SM100. See the [core README](core/README.md) for details, the
[recipe README](recipe/README.md) for model recipes, and the
[vLLM README](vllm/README.md) for running and evaluating quantized models in
vLLM.

## Status

QuantSuite is in early development, so the API may change before 1.0. See
[CHANGELOG.md](CHANGELOG.md) for releases.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

QuantSuite is BSD-3-Clause licensed, as found in the [LICENSE](LICENSE) file.
Some packages include code adapted from third-party projects; see each
package's `THIRD_PARTY_NOTICES.md`.
