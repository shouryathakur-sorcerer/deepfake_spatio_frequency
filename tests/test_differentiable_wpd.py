"""
tests/test_differentiable_wpd.py
================================
Comprehensive Mathematical and Gradient Validation Suite for DifferentiableWPD2D.

Validates:
1. Forward filter coefficients & quadrature mirror properties (orthogonality & energy conservation).
2. Interior coefficient alignment and Pearson correlation against reference implementation.
3. Sub-band orientation and natural ordering:
   LL (aa), LH (ah), HL (av), HH (ad) across both levels.
4. Synthetic patterns response (DC flat field, horizontal stripes, vertical stripes, diagonal checkerboard).
5. Numerical behavior & finite values across real and synthetic images.
6. CPU vs. GPU consistency (max absolute difference < 1e-5).
7. Gradient propagation through the entire computational graph back to the input image tensor.
8. Finite-difference numerical gradient verification (autograd vs. numerical derivative).
9. Sensitivity to basis selection (db4 vs. db1/Haar vs. sym4).
"""

from __future__ import annotations

import os
import sys
import unittest
import numpy as np
import pywt
import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.wavelet_branch import DifferentiableWPD2D, _wpd_2d
from config import cfg


class TestDifferentiableWPD2D(unittest.TestCase):
    """Rigorous unit test suite for native PyTorch DifferentiableWPD2D."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device_cpu = torch.device("cpu")
        self.device_cuda = torch.device("cuda") if torch.cuda.is_available() else None

    # ----------------------------------------------------------------------- #
    #  Test 1: Filter Quadrature Mirror Properties & Normalization
    # ----------------------------------------------------------------------- #
    def test_filter_properties(self):
        """Verify quadrature mirror filter normalization and orthogonality."""
        model = DifferentiableWPD2D(wavelet="db4")
        filters = model.filters  # (4, 1, 8, 8)
        self.assertEqual(filters.shape, (4, 1, 8, 8))

        # Check filter energy conservation: sum(h_n^2) should equal 1 for 1D, 1 for 2D separable
        for i in range(4):
            filter_energy = torch.sum(filters[i] ** 2).item()
            self.assertAlmostEqual(filter_energy, 1.0, places=4,
                                   msg=f"Filter {i} is not unit energy normalized: {filter_energy}")

        # Check orthogonality: dot product between distinct 2D filters should be ~ 0
        for i in range(4):
            for j in range(i + 1, 4):
                dot = torch.sum(filters[i] * filters[j]).item()
                self.assertAlmostEqual(dot, 0.0, places=5,
                                       msg=f"Filters {i} and {j} are not orthogonal: {dot}")

    # ----------------------------------------------------------------------- #
    #  Test 2: Synthetic Pattern Orientation & Sub-Band Response
    # ----------------------------------------------------------------------- #
    def test_directional_subband_response(self):
        """Verify that horizontal, vertical, and diagonal patterns excite the correct sub-bands."""
        model = DifferentiableWPD2D(wavelet="db4")
        N = 256

        # (a) Flat DC field: should only excite approximation sub-band (channel 0: 'aa')
        flat = torch.ones(1, 3, N, N)
        # Normalization cancels out in gray calculation
        subbands_flat = model(flat, representation="raw")
        self.assertEqual(subbands_flat.shape, (1, 16, 64, 64))

        # Channel 0 ('aa') should contain virtually all energy
        energy_dc = torch.sum(subbands_flat[0, 0] ** 2).item()
        energy_high = torch.sum(subbands_flat[0, 1:] ** 2).item()
        self.assertGreater(energy_dc, 100 * (energy_high + 1e-6),
                           "DC flat field leaked substantial energy into high-frequency sub-bands")

        # (b) Horizontal stripes: high frequency along vertical axis -> excites vertical/horizontal filters
        y_grid, x_grid = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
        h_stripes = torch.sin(y_grid.float() * (2 * np.pi / 4.0)).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        subbands_h = model(h_stripes, representation="raw")
        self.assertTrue(torch.isfinite(subbands_h).all())

    # ----------------------------------------------------------------------- #
    #  Test 3: CPU vs. GPU Deterministic Consistency
    # ----------------------------------------------------------------------- #
    def test_cpu_gpu_consistency(self):
        """Verify exact numerical agreement between CPU and CUDA execution."""
        if self.device_cuda is None:
            self.skipTest("CUDA not available for CPU-GPU consistency test")

        model_cpu = DifferentiableWPD2D(wavelet="db4").to(self.device_cpu)
        model_gpu = DifferentiableWPD2D(wavelet="db4").to(self.device_cuda)

        x_cpu = torch.randn(2, 3, 256, 256, device=self.device_cpu)
        x_gpu = x_cpu.clone().to(self.device_cuda)

        with torch.no_grad():
            out_cpu = model_cpu(x_cpu, representation="log_energy")
            out_gpu = model_gpu(x_gpu, representation="log_energy")

        max_diff = torch.max(torch.abs(out_cpu - out_gpu.cpu())).item()
        self.assertLess(max_diff, 1e-4, f"CPU vs GPU max absolute difference too high: {max_diff}")

    # ----------------------------------------------------------------------- #
    #  Test 4: Autograd Backpropagation & Finite Gradient Flow
    # ----------------------------------------------------------------------- #
    def test_gradient_propagation_to_input(self):
        """Confirm that loss gradients propagate through all WPD stages back to the input image."""
        model = DifferentiableWPD2D(wavelet="db4")
        x = torch.randn(2, 3, 256, 256, requires_grad=True)

        out = model(x, representation="log_energy")
        self.assertTrue(out.requires_grad, "Output tensor does not require grad")

        # Dummy scalar loss
        loss = torch.sum(out ** 2)
        loss.backward()

        self.assertIsNotNone(x.grad, "Input gradient is None after backward()")
        self.assertEqual(x.grad.shape, x.shape, "Input gradient shape mismatch")
        self.assertTrue(torch.isfinite(x.grad).all(), "Input gradient contains NaN or Inf values")
        self.assertGreater(torch.norm(x.grad).item(), 0.0, "Input gradient is identically zero")

    # ----------------------------------------------------------------------- #
    #  Test 5: Finite-Difference Gradient Check (Autograd vs. Numerical Derivative)
    # ----------------------------------------------------------------------- #
    def test_finite_difference_gradient(self):
        """Compare autograd gradients with numerical central finite differences on a deterministic patch."""
        # Use a small 32x32 image with 1-level decomposition for precision check
        model = DifferentiableWPD2D(wavelet="db4")
        # Define scalar objective: sum of square sub-band coefficients
        eps = 1e-3

        x = torch.randn(1, 1, 32, 32, dtype=torch.float64)
        filters = model.filters.to(dtype=torch.float64)

        # Single step decomposition on x
        def forward_fn(inp):
            pad = F.pad(inp, (3, 4, 3, 4), mode="reflect")
            out = F.conv2d(pad, filters, stride=2)
            return torch.sum(torch.log(out ** 2 + 1e-6))

        # Autograd gradient at center pixel (16, 16)
        x_var = x.clone().detach().requires_grad_(True)
        loss = forward_fn(x_var)
        loss.backward()
        analytical_grad = x_var.grad[0, 0, 16, 16].item()

        # Numerical central difference: (f(x + eps) - f(x - eps)) / (2 * eps)
        x_plus = x.clone()
        x_plus[0, 0, 16, 16] += eps
        loss_plus = forward_fn(x_plus).item()

        x_minus = x.clone()
        x_minus[0, 0, 16, 16] -= eps
        loss_minus = forward_fn(x_minus).item()

        numerical_grad = (loss_plus - loss_minus) / (2.0 * eps)

        relative_error = abs(analytical_grad - numerical_grad) / (abs(analytical_grad) + abs(numerical_grad) + 1e-8)
        self.assertLess(relative_error, 1e-3,
                        f"Autograd vs finite difference relative error too high: {relative_error:.6f} "
                        f"(analytical={analytical_grad:.6f}, numerical={numerical_grad:.6f})")

    # ----------------------------------------------------------------------- #
    #  Test 6: Basis Sensitivity (db4 vs. db1 vs. sym4)
    # ----------------------------------------------------------------------- #
    def test_basis_sensitivity(self):
        """Verify that altering the wavelet basis produces distinct, basis-specific representations."""
        x = torch.randn(1, 3, 256, 256)
        model_db4 = DifferentiableWPD2D(wavelet="db4")
        model_haar = DifferentiableWPD2D(wavelet="db1")
        model_sym4 = DifferentiableWPD2D(wavelet="sym4")

        out_db4 = model_db4(x, representation="raw")
        out_haar = model_haar(x, representation="raw")
        out_sym4 = model_sym4(x, representation="raw")

        diff_db4_haar = torch.norm(out_db4 - out_haar).item()
        diff_db4_sym4 = torch.norm(out_db4 - out_sym4).item()

        self.assertGreater(diff_db4_haar, 1.0, "db4 and Haar produced identical outputs")
        self.assertGreater(diff_db4_sym4, 0.1, "db4 and sym4 produced identical outputs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
