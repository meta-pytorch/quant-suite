# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from parameterized import parameterized
from quant_suite_core.rht_generator import apply_block_hadamard, RHTGenerator


def _apply_hadamard(
    tensor: torch.Tensor,
    generator: RHTGenerator,
    *,
    is_left: bool,
    inverse: bool = False,
) -> torch.Tensor:
    matrix = generator.get_matrix(device=tensor.device, dtype=tensor.dtype)
    return apply_block_hadamard(tensor, matrix, is_left=is_left, inverse=inverse)


class TestApplyHadamard(unittest.TestCase):
    """Block-diagonal RHT primitive tests.

    Pure-torch (no quant kernels), so these run on CPU as well as GPU.
    """

    def setUp(self) -> None:
        RHTGenerator.clear_cache()

    @parameterized.expand(
        [
            ("left_forward_then_inverse", True, False),
            ("left_inverse_then_forward", True, True),
            ("right_forward_then_inverse", False, False),
            ("right_inverse_then_forward", False, True),
        ]
    )
    def test_inverse_round_trips(self, name: str, is_left: bool, inverse: bool) -> None:
        """forward then inverse returns the original tensor.

        ``R`` is ``diag(+-1) @ H``, which is asymmetric whatever the block size,
        so an inverse that forgets to transpose R would fail to round-trip.
        """
        block_size = 16
        gen = RHTGenerator(block_size=block_size, seed=0)
        R = gen.get_matrix(device="cpu").float()
        self.assertFalse(
            torch.allclose(R, R.T),
            "the random sign vector should yield an asymmetric R (R.T != R)",
        )

        # 32x48 -> multiple blocks along both axes
        x = torch.randn(32, 48, dtype=torch.float32)
        first = _apply_hadamard(x, gen, is_left=is_left, inverse=inverse)
        round_trip = _apply_hadamard(first, gen, is_left=is_left, inverse=not inverse)
        torch.testing.assert_close(
            round_trip,
            x,
            rtol=1e-4,
            atol=1e-4,
            msg=(
                "forward+inverse should be identity "
                f"(is_left={is_left}, inverse={inverse})"
            ),
        )

    def test_transform_is_block_diagonal(self) -> None:
        """R is applied independently per block (no cross-block mixing)."""
        block_size = 16
        gen = RHTGenerator(block_size=block_size, seed=0)
        R = gen.get_matrix(device="cpu").float()

        # LEFT: block-diagonal R.T applied to each block of rows.
        x = torch.randn(2 * block_size, 5, dtype=torch.float32)
        out_left = _apply_hadamard(x, gen, is_left=True)
        expected_left = torch.cat([R.T @ x[:block_size], R.T @ x[block_size:]], dim=0)
        torch.testing.assert_close(out_left, expected_left, rtol=1e-4, atol=1e-4)

        # LEFT inverse: block-diagonal R applied to each block of rows.
        out_left_inv = _apply_hadamard(x, gen, is_left=True, inverse=True)
        expected_left_inv = torch.cat([R @ x[:block_size], R @ x[block_size:]], dim=0)
        torch.testing.assert_close(
            out_left_inv, expected_left_inv, rtol=1e-4, atol=1e-4
        )

        # RIGHT: block-diagonal R applied to each block of cols.
        y = torch.randn(5, 2 * block_size, dtype=torch.float32)
        out_right = _apply_hadamard(y, gen, is_left=False)
        expected_right = torch.cat(
            [y[:, :block_size] @ R, y[:, block_size:] @ R], dim=1
        )
        torch.testing.assert_close(out_right, expected_right, rtol=1e-4, atol=1e-4)


class TestApplyBlockHadamard(unittest.TestCase):
    """The pure block-Hadamard core and the cached generator matrix.

    Pure-torch/CPU: no quant kernels, so these run without CUDA.
    """

    def setUp(self) -> None:
        RHTGenerator.clear_cache()

    def test_apply_block_hadamard_is_fullgraph_traceable(self) -> None:
        # torch.compile(fullgraph=True) RAISES on any graph break, so this test
        # FAILS if the pure core stops being traceable. Eager backend asserts
        # traceability without an inductor toolchain.
        matrix = RHTGenerator(block_size=16, seed=7).get_matrix(
            device=torch.device("cpu"), dtype=torch.float32
        )

        def fn(x: torch.Tensor) -> torch.Tensor:
            return apply_block_hadamard(x, matrix, is_left=False, inverse=False)

        x = torch.randn(4, 16, dtype=torch.float32)
        eager = fn(x)
        compiled = torch.compile(fn, fullgraph=True, backend="eager")(x)
        self.assertTrue(torch.equal(compiled, eager))

    def test_generator_is_cached_per_config_and_disables_grad(self) -> None:
        first = RHTGenerator(block_size=16, seed=7)
        self.assertIs(RHTGenerator(block_size=16, seed=7), first)
        self.assertIsNot(RHTGenerator(block_size=16, seed=8), first)
        matrix = first.get_matrix(device=torch.device("cpu"))
        self.assertTrue(
            torch.equal(matrix, first.get_matrix(device=torch.device("cpu")))
        )
        self.assertFalse(matrix.requires_grad)


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class TestHadamard(unittest.TestCase):
    """Test Hadamard cancellation property in GEMM operations.

    Tests verify the mathematical property: when both operands have
    matching Hadamard transforms, they cancel out (H @ H.T = I).
    """

    def setUp(self) -> None:
        RHTGenerator.clear_cache()
        torch.manual_seed(42)
        self.device = "cuda"
        self.block_size = 16
        self.rht_gen = RHTGenerator(block_size=self.block_size)

    def test_fwd_hadamard_cancellation(self) -> None:
        """Test that RIGHT Hadamard cancels for FWD gemm.

        FWD: A @ W.T (A: mxk, W: nxk)
        Config: fwd: {act: "right", weight: "right"}
        Both operands use RIGHT on k dimension, so R cancels.
        """
        m, k, n = 32, 32, 32
        A = torch.randn(m, k, dtype=torch.float32, device=self.device)
        W = torch.randn(n, k, dtype=torch.float32, device=self.device)

        # Apply RIGHT Hadamard
        A_transformed = _apply_hadamard(A, self.rht_gen, is_left=False)
        W_transformed = _apply_hadamard(W, self.rht_gen, is_left=False)

        # (A @ R) @ (W @ R).T = (A @ R) @ (R.T @ W.T) = A @ (R @ R.T) @ W.T = A @ W.T
        result_with_hadamard = A_transformed @ W_transformed.T
        result_original = A @ W.T

        torch.testing.assert_close(
            result_with_hadamard,
            result_original,
            rtol=1e-4,
            atol=1e-4,
            msg="RIGHT Hadamard should cancel: (A @ R) @ (W @ R).T = A @ W.T",
        )

    def test_dgrad_hadamard_cancellation(self) -> None:
        """Test that LEFT/RIGHT Hadamard cancels for DGRAD gemm.

        DGRAD: G @ W (G: mxn, W: nxk)
        Config: dgrad: {grad: "right", weight: "left"}
        - G uses RIGHT on n dimension (columns)
        - W uses LEFT on n dimension (rows)
        When G's columns and W's rows both have Hadamard, R cancels.
        """
        m, n, k = 32, 32, 32
        G = torch.randn(m, n, dtype=torch.float32, device=self.device)
        W = torch.randn(n, k, dtype=torch.float32, device=self.device)

        # G uses RIGHT, W uses LEFT
        G_transformed = _apply_hadamard(G, self.rht_gen, is_left=False)
        W_transformed = _apply_hadamard(W, self.rht_gen, is_left=True)

        # (G @ R) @ (R.T @ W) = G @ (R @ R.T) @ W = G @ W
        result_with_hadamard = G_transformed @ W_transformed
        result_original = G @ W

        torch.testing.assert_close(
            result_with_hadamard,
            result_original,
            rtol=1e-4,
            atol=1e-4,
            msg="DGRAD Hadamard should cancel: (G @ R) @ (R.T @ W) = G @ W",
        )

    def test_wgrad_hadamard_cancellation(self) -> None:
        """Test that LEFT Hadamard cancels for WGRAD gemm.

        WGRAD: G.T @ A (G: mxn, A: mxk)
        Config: wgrad: {grad: "left", act: "left"}
        Both operands use LEFT on m dimension (rows), so R cancels.
        """
        m, n, k = 32, 32, 32
        G = torch.randn(m, n, dtype=torch.float32, device=self.device)
        A = torch.randn(m, k, dtype=torch.float32, device=self.device)

        # Both use LEFT
        G_transformed = _apply_hadamard(G, self.rht_gen, is_left=True)
        A_transformed = _apply_hadamard(A, self.rht_gen, is_left=True)

        # (R.T @ G).T @ (R.T @ A) = G.T @ R @ R.T @ A = G.T @ A
        result_with_hadamard = G_transformed.T @ A_transformed
        result_original = G.T @ A

        torch.testing.assert_close(
            result_with_hadamard,
            result_original,
            rtol=1e-4,
            atol=1e-4,
            msg="WGRAD Hadamard should cancel: (R.T @ G).T @ (R.T @ A) = G.T @ A",
        )

    def test_halo2_dgrad_hadamard_cancellation(self) -> None:
        """Test HALO2 DGRAD: both operands use INV (inverse after forward).

        DGRAD: G @ W (G: mxn, W: nxk)
        Config: dgrad: {grad: "left_inv", weight: "right_inv"}
        - G uses LEFT_INV: forward then post-quant inverse on rows.
        - W uses RIGHT_INV: forward then post-quant inverse on cols.

        Without quantization, INV is identity, so result = G @ W.
        """
        m, n, k = 32, 32, 32
        G = torch.randn(m, n, dtype=torch.float32, device=self.device)
        W = torch.randn(n, k, dtype=torch.float32, device=self.device)

        # G uses LEFT_INV
        G_pre = _apply_hadamard(G, self.rht_gen, is_left=True)
        G_transformed = _apply_hadamard(G_pre, self.rht_gen, is_left=True, inverse=True)

        # W uses RIGHT_INV
        W_pre = _apply_hadamard(W, self.rht_gen, is_left=False)
        W_transformed = _apply_hadamard(
            W_pre, self.rht_gen, is_left=False, inverse=True
        )

        # Without quantization, INV transforms are identity
        torch.testing.assert_close(
            G_transformed,
            G,
            rtol=1e-4,
            atol=1e-4,
            msg="LEFT_INV should be identity without quantization",
        )
        torch.testing.assert_close(
            W_transformed,
            W,
            rtol=1e-4,
            atol=1e-4,
            msg="RIGHT_INV should be identity without quantization",
        )

        result_with_hadamard = G_transformed @ W_transformed
        result_original = G @ W

        torch.testing.assert_close(
            result_with_hadamard,
            result_original,
            rtol=1e-4,
            atol=1e-4,
            msg="HALO2 DGRAD: INV transforms should be identity without quant",
        )

    def test_halo2_wgrad_hadamard_cancellation(self) -> None:
        """Test HALO2 WGRAD: act uses INV, grad uses none.

        WGRAD: G.T @ A (G: mxn, A: mxk)
        Config: wgrad: {grad: "none", act: "right_inv"}
        - G uses NONE: unchanged
        - A uses RIGHT_INV: forward then post-quant inverse on cols.

        Without quantization, RIGHT_INV is identity, so result = G.T @ A.
        """
        m, n, k = 32, 32, 32
        G = torch.randn(m, n, dtype=torch.float32, device=self.device)
        A = torch.randn(m, k, dtype=torch.float32, device=self.device)

        # G uses NONE
        G_transformed = G

        # A uses RIGHT_INV
        A_pre = _apply_hadamard(A, self.rht_gen, is_left=False)
        A_transformed = _apply_hadamard(
            A_pre, self.rht_gen, is_left=False, inverse=True
        )

        # Without quantization, RIGHT_INV is identity
        torch.testing.assert_close(
            A_transformed,
            A,
            rtol=1e-4,
            atol=1e-4,
            msg="RIGHT_INV should be identity without quantization",
        )

        result_with_hadamard = G_transformed.T @ A_transformed
        result_original = G.T @ A

        torch.testing.assert_close(
            result_with_hadamard,
            result_original,
            rtol=1e-4,
            atol=1e-4,
            msg="HALO2 WGRAD: RIGHT_INV should be identity without quant",
        )


if __name__ == "__main__":
    unittest.main()
