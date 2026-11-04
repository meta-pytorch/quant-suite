# Changelog

All notable changes to QuantSuite are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/). The API may
change before 1.0.

## [Unreleased]

## 0.1.0

Initial public release.

- `quant_suite_core`: tensor-level building blocks: fake QDQ for MX, NVFP4 and
  related formats, native packed MXFP4/MXFP8/NVFP4, Hadamard and Triton
  utilities.
- `quant_suite_recipe`: typed, serializable recipes for tensors, transform
  pipelines, Linear, MoE and attention modules, and the runtime that executes
  them.
- `quant_suite_vllm`: vLLM integration: the `quant_suite` quantization method,
  MHA and MLA attention backends, and KDA recurrent-state quantization.
- `quant_suite_mx_plus` (experimental): MX+ extension and the template for
  experimental packages. Experimental packages have no stability guarantee.
