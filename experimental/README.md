# Experimental algorithms

This directory holds quantization algorithms that plug into QuantSuite through
its public extension points but are not part of the supported packages.

| package | algorithm | source |
| --- | --- | --- |
| [`mx_plus/`](mx_plus/) (`mx_plus`, dist `quant_suite_mx_plus`) | MX+: extra mantissa bits on each block's largest element | Lee et al., MICRO '25 |

## Policy

**No stability guarantee.** An experimental package has no API, numerics or
serialization guarantees. Each one is versioned on its own schedule and may
change, be removed, or graduate into a supported package.

**No frozen identities.** Experimental code adds no enum members, tokens or
presets to `quant_suite_core`. It contributes
`quant_suite_core.extension.QuantConfigExtension` classes and presets, which it
adds to a `QuantConfigBuilder`, and, if it needs one, a tensor recipe
registration. It combines them with the built-in ones in one
`quant_suite_recipe.runtime.QuantRecipeRuntime`, published in the
`quant_suite.recipe_runtimes` entry-point group so it becomes the
environment's `QUANT_RECIPE_RUNTIME`, which the QuantSuite vLLM plugin uses,
once the package is installed. Its tokens are not frozen: a config that
names one may stop parsing when the package changes.

**One-way imports.** `quant_suite_core`, `quant_suite_recipe` and
`quant_suite_vllm` never import experimental code. An experimental package
imports only documented public modules of those packages: no private names, and
nothing from another experimental package.

**One package per algorithm.** Each algorithm is its own directory and its own
distribution, with its own `pyproject.toml`, tests and README. Experimental
packages are never published to PyPI; install them from a checkout
(`pip install -e experimental/<algo>`) or from a git subdirectory URL.

**Cite and test.** Each package's README cites the source of its algorithm,
and the package ships tests that run on CPU, with GPU tests skipped when no GPU
is present.

## Adding an algorithm

This section lists what every experimental package must have. For how to write
one, start from [`mx_plus/`](mx_plus/), which is written as the template: its
README walks through the rule, the `QuantConfigExtension`, the presets, the
runtime and its entry point, and serving with vLLM, step by step.

A new package needs:

- the layout `experimental/<algo>/<algo>/`: the outer directory holds the
  packaging files, tests and README, and the inner one is the importable
  package, with a short import name (`import <algo>`);
- the distribution name `quant_suite_<algo>` in `pyproject.toml`. The import
  name needs no `quant_suite_` prefix, so pick one distinctive enough not to
  clash with other installed packages;
- `LICENSE` (BSD-3, as here) and `THIRD_PARTY_NOTICES.md` if it adapts
  third-party code;
- a README with what the algorithm computes, install and usage, and the
  citation;
- a `pyproject.toml` that publishes its runtime in the
  `quant_suite.recipe_runtimes` entry-point group;
- tests: CPU tests for parsing and validation; a test that the rule changes
  the output, since a hook that is never reached passes every other test; a
  test that runtime resolution selects the package's runtime; GPU tests that skip
  without a GPU; and an import-boundary test.
