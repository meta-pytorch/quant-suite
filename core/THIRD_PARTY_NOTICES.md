# Third-Party Notices

`quant_suite_core` includes code adapted from the following third-party project.

## Microsoft microxcaling

- Project: https://github.com/microsoft/microxcaling
- License: MIT License
- Copyright: Copyright (c) Microsoft Corporation.
- Adapted code: `_safe_lshift`, `_safe_rshift`, and the exponent, scaling and
  rounding steps in `fp32_to_fpx` in `quant_suite_core/quant_primitives.py`, and
  their Triton ports (`_safe_lshift`, `_safe_rshift`, and the default
  (non-`ranking_only`) path of `_fp32_to_fpx_helper`) in
  `quant_suite_core/triton_utils.py`, are adapted from `mx/elemwise_ops.py`,
  and were modified by Meta Platforms, Inc. and affiliates.

### License text

```text
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
```
