# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Self-contained checks on the Sylvester Hadamard transforms in hadamard_utils.

The checks cover orthonormality, uniform entry magnitude, the Sylvester
recursion, involution, and seeded draw compatibility.
"""

import math
import unittest

import torch
from quant_suite_core.hadamard_utils import (
    fast_hadamard_transform,
    is_pow2,
    random_hadamard_matrix,
    sylvester_matrix,
)

# Orders covered by these tests.
SIZES: tuple[int, ...] = (16, 32, 64, 128, 256, 1024, 2048, 4096)

SEED: int = 20260918

FLOAT64_EPS: float = 2.0**-52

# What "orthogonal in float64" means for ``max|Q Q^T - I|`` at these orders.
DOUBLE_ORTHOGONALITY_ATOL: float = 1e-12


def devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


class TestDeviceCoverage(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA device available")
    def test_cuda_is_covered(self) -> None:
        """Every other case loops over ``devices()``; without this marker a
        CPU-only host would halve the coverage without reporting a skip."""
        self.assertIn("cuda", devices())


class TestSylvesterProperties(unittest.TestCase):
    def test_orthonormal(self) -> None:
        for device in devices():
            for size in SIZES:
                with self.subTest(device=device, size=size):
                    q = random_hadamard_matrix(size, device, seed=SEED)
                    identity = torch.eye(size, device=device, dtype=torch.float64)
                    torch.testing.assert_close(
                        q @ q.T, identity, atol=DOUBLE_ORTHOGONALITY_ATOL, rtol=0
                    )

    def test_entries_are_plus_minus_one_over_sqrt_n(self) -> None:
        for device in devices():
            for size in SIZES:
                with self.subTest(device=device, size=size):
                    q = random_hadamard_matrix(size, device, seed=SEED)
                    self.assertEqual(q.shape, (size, size))
                    self.assertEqual(q.dtype, torch.float64)
                    magnitude = q.abs()
                    entry = float(magnitude.flatten()[0])
                    self.assertTrue(torch.equal(magnitude, torch.full_like(q, entry)))
                    # One float64 rounding of 1/sqrt(n) and nothing else. A
                    # float32 divisor would miss this by ~1.7e-08 relative.
                    self.assertLess(abs(entry * math.sqrt(size) - 1.0), 4 * FLOAT64_EPS)

    def test_sylvester_recursion_holds(self) -> None:
        for size in (1, 2, 4, 16, 256):
            with self.subTest(size=size):
                h = sylvester_matrix(size)
                self.assertEqual(h.shape, (size, size))
                self.assertTrue(torch.equal(h, h.T))
                self.assertTrue(torch.equal(h @ h, size * torch.eye(size).double()))
                if size > 1:
                    half = size // 2
                    inner = sylvester_matrix(half)
                    self.assertTrue(torch.equal(h[:half, :half], inner))
                    self.assertTrue(torch.equal(h[:half, half:], inner))
                    self.assertTrue(torch.equal(h[half:, :half], inner))
                    self.assertTrue(torch.equal(h[half:, half:], -inner))

    def test_fast_hadamard_transform_is_an_involution(self) -> None:
        for device in devices():
            for size in (16, 64, 2048):
                with self.subTest(device=device, size=size):
                    generator = torch.Generator(device=device).manual_seed(SEED)
                    x = torch.randn(
                        (4, size),
                        generator=generator,
                        device=device,
                        dtype=torch.float64,
                    )
                    torch.testing.assert_close(
                        fast_hadamard_transform(fast_hadamard_transform(x)),
                        x,
                        atol=1e-12,
                        rtol=0.0,
                    )

    def test_seed_is_honoured_and_varies(self) -> None:
        for device in devices():
            with self.subTest(device=device):
                a = random_hadamard_matrix(256, device, seed=SEED)
                b = random_hadamard_matrix(256, device, seed=SEED)
                c = random_hadamard_matrix(256, device, seed=SEED + 1)
                self.assertTrue(torch.equal(a, b))
                self.assertFalse(torch.equal(a, c))

    def test_seeded_draw_preserves_process_rng(self) -> None:
        for device in devices():
            with self.subTest(device=device):
                cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
                cpu_state = torch.random.get_rng_state()
                random_hadamard_matrix(32, device, seed=42)
                self.assertTrue(torch.equal(cpu_state, torch.random.get_rng_state()))
                if cuda_state is not None:
                    self.assertTrue(torch.equal(cuda_state, torch.cuda.get_rng_state()))

    def test_seeded_signs_match_global_device_draw(self) -> None:
        for device in devices():
            with self.subTest(device=device):
                cuda_devices = [torch.cuda.current_device()] if device == "cuda" else []
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(42)
                    signs = torch.randint(0, 2, (32,), device=device).double() * 2 - 1
                    expected = signs[:, None] * sylvester_matrix(32, device)
                    expected /= math.sqrt(32)
                    torch.manual_seed(SEED)
                    actual = random_hadamard_matrix(32, device, seed=42)
                    self.assertTrue(torch.equal(actual, expected))

    def test_unseeded_draw_consumes_existing_rng(self) -> None:
        for device in devices():
            with self.subTest(device=device):
                cuda_devices = [torch.cuda.current_device()] if device == "cuda" else []
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(SEED)
                    signs = torch.randint(0, 2, (32,), device=device).double() * 2 - 1
                    expected = signs[:, None] * sylvester_matrix(32, device)
                    expected /= math.sqrt(32)
                    expected_next = torch.rand(8, device=device)
                    torch.manual_seed(SEED)
                    actual = random_hadamard_matrix(32, device)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(
                        torch.equal(torch.rand(8, device=device), expected_next)
                    )


class TestPowerOfTwoRestriction(unittest.TestCase):
    def test_is_pow2(self) -> None:
        for n in (1, 2, 4, 8, 16, 1024):
            self.assertTrue(is_pow2(n))
        for n in (-8, 0, 3, 12, 60, 172, 1000, 4095):
            self.assertFalse(is_pow2(n))

    def test_random_hadamard_matrix_rejects_non_pow2(self) -> None:
        for size in (0, -8, 3, 12, 60, 172, 1000, 4095):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    random_hadamard_matrix(size, "cpu", seed=SEED)

    def test_fast_hadamard_transform_rejects_non_pow2(self) -> None:
        for size in (3, 12, 172, 1000):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    fast_hadamard_transform(torch.randn(2, size))

    def test_sylvester_matrix_rejects_non_pow2(self) -> None:
        with self.assertRaises(ValueError):
            sylvester_matrix(12)
