# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recorded quantization outputs for ``mx_plus_golden_vectors.py``. DATA ONLY.

The ``mx_plus`` suite: the same format, inputs and seeding protocol as
``recipe/tests/golden_vectors_data.py``, over ``MX_PLUS_PRESETS``. See
``golden_vectors.py`` there for the column definitions, the reproduction inputs
and the regeneration command.

Regenerating is an explicit act, never a reflex. A red golden check means the
quantization output moved; that is the signal this harness exists to produce.
"""

# GPU the table was recorded on. Reported on failure: a mismatch on a different
# architecture is a different question from a mismatch on this one.
GENERATED_ON: str = "NVIDIA B200"

GOLDEN_VECTOR_TABLE: str = """
mxfp4_plus2_l_had16|qdq_triton_mega|bfloat16 32x384 bfloat16 4681252a2d19f3b39a8891c8e084a0b689bca3aac83e8922f603d572ccd7d0a1 -3 4.5 0.00455729 0 0
mxfp4_plus2_l_had16|qdq_triton_mega|float32 32x384 float32 c73c066eeed4adb93ba7a9499b391376809c29347abbc97c3bc3fa296b0bf8f5 -3 4.5 0.00431315 0 0
mxfp4_plus2_l_had16|qdq|bfloat16 32x384 bfloat16 4681252a2d19f3b39a8891c8e084a0b689bca3aac83e8922f603d572ccd7d0a1 -3 4.5 0.00455729 0 0
mxfp4_plus2_l_had16|qdq|float32 32x384 float32 c73c066eeed4adb93ba7a9499b391376809c29347abbc97c3bc3fa296b0bf8f5 -3 4.5 0.00431315 0 0
mxfp4_plus2_r_had16|qdq_triton_mega|bfloat16 32x384 bfloat16 6c1ada172e5722e4aad9730c6916c010a9691a4a30c663f44689f9c01bed7c3d -3 4.5 0.0103861 0 0
mxfp4_plus2_r_had16|qdq_triton_mega|float32 32x384 float32 c5e02690a4a7c1d81e4511a4adf3804219617f14b3e7f866d8f348563ca1c098 -3 4.5 0.0101827 0 0
mxfp4_plus2_r_had16|qdq|bfloat16 32x384 bfloat16 6c1ada172e5722e4aad9730c6916c010a9691a4a30c663f44689f9c01bed7c3d -3 4.5 0.0103861 0 0
mxfp4_plus2_r_had16|qdq|float32 32x384 float32 c5e02690a4a7c1d81e4511a4adf3804219617f14b3e7f866d8f348563ca1c098 -3 4.5 0.0101827 0 0
mxfp4_plus2|qdq_triton_mega|bfloat16 32x384 bfloat16 fab6886b873b79aee708ab3c570e42ff905b8c66d0b8d717363a121f2cad1471 -5 4 -0.0150757 0 0
mxfp4_plus2|qdq_triton_mega|float32 32x384 float32 56dc0d108c232323d250aebba234cf44aaab8a35bf75472017546c1d0288ea11 -5 4 -0.0150859 0 0
mxfp4_plus2|qdq|bfloat16 32x384 bfloat16 fab6886b873b79aee708ab3c570e42ff905b8c66d0b8d717363a121f2cad1471 -5 4 -0.0150757 0 0
mxfp4_plus2|qdq|float32 32x384 float32 56dc0d108c232323d250aebba234cf44aaab8a35bf75472017546c1d0288ea11 -5 4 -0.0150859 0 0
"""
