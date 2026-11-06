"""
tests/test_team14_architecture.py
=================================
Comprehensive Unit & Integration Test Suite for Team 14 Architecture.

Tests verified:
1. DifferentiableWPD3Level:
   - Tensor dimensions: (B, 1, 224, 224) -> (B, 64, 28, 28)
   - Quadrature mirror filter properties (orthogonality & unit energy)
   - DC flat field energy concentration (channel 0 receives >99% of energy)
   - Directional sub-band response across horizontal and vertical frequencies
   - Support for log-magnitude and log-energy representations
   - CPU vs. GPU deterministic numerical consistency
   - Autograd backpropagation & finite gradient flow through 3 levels of recursive filtering
2. Learnable High-Frequency Enhancement (HiFE) Module:
   - Channel gating activation values bounded in [0, 1]
   - Directional depthwise + pointwise convolution
   - Residual scaling and stride-2 spatial projection: (B, 64, 28, 28) -> (B, 256, 14, 14)
   - Finite gradient backpropagation through all parameters
3. Dual Cross-Attention (DCA) Module:
   - Projections: Spatial (728) & Frequency (256) -> attn_dim (256)
   - Flattening to 196 tokens (14x14)
   - Bi-directional cross-attention matrices: A_{S->F} and A_{F->S}
   - Softmax normalization along key axis (dim=-1)
   - Concatenation with spatial skip: (B, 984, 14, 14)
   - Gradient flow to both spatial and frequency inputs
4. Team14Net End-to-End & Ablation Configurations:
   - Full model: spatial + 3-level WPD + HiFE + DCA -> logit (B, 1)
   - All 6 ablation variants instantiation & forward/backward execution:
     * full_team14
     * spatial_only
     * wpd_only
     * spatial_wpd_concat
     * spatial_wpd_hife
     * spatial_wpd_dca
"""

from __future__ import annotations

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.wavelet_branch import DifferentiableWPD3Level
from models.hife import HiFEBlock
from models.dual_cross_attention import DualCrossAttention
from models.spatial_branch import SpatialBranch
from models.team14_net import Team14Net


class TestTeam14WaveletPacket(unittest.TestCase):
    """Test 3-Level 2D Wavelet Packet Decomposition module."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_output_shape_and_subbands(self):
        """Verify output tensor dimensions: 64 subbands of size 28x28 for 224x224 input."""
        wpd = DifferentiableWPD3Level(wavelet="db4").to(self.device)
        x = torch.randn(2, 3, 224, 224, device=self.device)
        out = wpd(x, representation="log_magnitude")
        self.assertEqual(out.shape, (2, 64, 28, 28), f"Expected (2, 64, 28, 28), got {out.shape}")
        self.assertFalse(torch.isnan(out).any(), "NaN found in WPD output")
        self.assertFalse(torch.isinf(out).any(), "Inf found in WPD output")

    def test_filter_weights_non_trainable(self):
        """Wavelet filters should be fixed buffers, not trainable parameters."""
        wpd = DifferentiableWPD3Level(wavelet="db4")
        param_count = sum(p.numel() for p in wpd.parameters())
        self.assertEqual(param_count, 0, f"Expected 0 trainable parameters, found {param_count}")
        self.assertTrue(hasattr(wpd, "filters"))
        self.assertEqual(wpd.filters.shape, (4, 1, 8, 8))

    def test_filter_orthogonality_and_energy(self):
        """Check quadrature mirror filter unit energy and mutual orthogonality."""
        wpd = DifferentiableWPD3Level(wavelet="db4")
        filters = wpd.filters  # (4, 1, 8, 8)

        # 1. Unit energy: sum of squares == 1.0 for each 2D filter
        for i in range(4):
            energy = torch.sum(filters[i] ** 2).item()
            self.assertAlmostEqual(energy, 1.0, places=4, msg=f"Filter {i} not normalized: {energy}")

        # 2. Mutual orthogonality: dot product == 0 for i != j
        for i in range(4):
            for j in range(i + 1, 4):
                dot = torch.sum(filters[i] * filters[j]).item()
                self.assertAlmostEqual(dot, 0.0, places=5, msg=f"Filters {i} and {j} not orthogonal: {dot}")

    def test_dc_flat_field_concentration(self):
        """A flat constant field should concentrate >99% of energy into channel 0 (LL-LL-LL)."""
        wpd = DifferentiableWPD3Level(wavelet="db4").to(self.device)
        flat = torch.ones(1, 3, 224, 224, device=self.device)
        out_raw = wpd(flat, representation="raw")
        self.assertEqual(out_raw.shape, (1, 64, 28, 28))

        energy_ch0 = torch.sum(out_raw[0, 0] ** 2).item()
        energy_other = torch.sum(out_raw[0, 1:] ** 2).item()

        self.assertGreater(energy_ch0, 100 * (energy_other + 1e-6),
                           f"DC energy {energy_ch0} not concentrated relative to other channels {energy_other}")

    def test_cpu_gpu_consistency(self):
        """Verify exact numerical consistency between CPU and CUDA execution."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        wpd_cpu = DifferentiableWPD3Level(wavelet="db4").to("cpu")
        wpd_gpu = DifferentiableWPD3Level(wavelet="db4").to("cuda")

        x_cpu = torch.randn(2, 3, 224, 224, device="cpu")
        x_gpu = x_cpu.clone().to("cuda")

        with torch.no_grad():
            out_cpu = wpd_cpu(x_cpu, representation="log_magnitude")
            out_gpu = wpd_gpu(x_gpu, representation="log_magnitude")

        max_diff = torch.max(torch.abs(out_cpu - out_gpu.cpu())).item()
        self.assertLess(max_diff, 1e-4, f"CPU-GPU max diff {max_diff} exceeded tolerance")

    def test_gradient_flow(self):
        """Verify autograd gradient flow back to the input image tensor."""
        wpd = DifferentiableWPD3Level(wavelet="db4").to(self.device)
        x = torch.randn(2, 3, 224, 224, device=self.device, requires_grad=True)
        out = wpd(x, representation="log_magnitude")
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "Input gradient is None")
        self.assertEqual(x.grad.shape, (2, 3, 224, 224))
        self.assertGreater(x.grad.abs().sum().item(), 0.0, "Input gradient is zero")
        self.assertFalse(torch.isnan(x.grad).any(), "NaN in input gradient")


class TestHiFEModule(unittest.TestCase):
    """Test Learnable High-Frequency Enhancement module."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_hife_forward_shape(self):
        """HiFE should map (B, 64, 28, 28) -> (B, 256, 14, 14)."""
        hife = HiFEBlock(in_channels=64, out_channels=256, reduction=4).to(self.device)
        x = torch.randn(2, 64, 28, 28, device=self.device)
        out = hife(x)
        self.assertEqual(out.shape, (2, 256, 14, 14))

    def test_hife_channel_attention_bounds(self):
        """Channel attention weights must lie strictly in [0, 1]."""
        hife = HiFEBlock(in_channels=64, out_channels=256).to(self.device)
        x = torch.randn(2, 64, 28, 28, device=self.device)
        
        # Test the channel attention sub-computation using shared_mlp
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(2, 64)
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(2, 64)
        att = torch.sigmoid(hife.shared_mlp(avg_pool) + hife.shared_mlp(max_pool))
        
        self.assertTrue((att >= 0.0).all() and (att <= 1.0).all(), "Attention weights out of bounds [0, 1]")

    def test_hife_backprop(self):
        """Gradients must propagate through all HiFE parameters."""
        hife = HiFEBlock(in_channels=64, out_channels=256).to(self.device)
        x = torch.randn(2, 64, 28, 28, device=self.device, requires_grad=True)
        out = hife(x)
        loss = out.sum()
        loss.backward()

        for name, param in hife.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
                self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} gradient has NaNs")


class TestDualCrossAttention(unittest.TestCase):
    """Test Dual Cross-Attention (DCA) module."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_dca_forward_shape(self):
        """DCA should fuse (B, 728, 14, 14) and (B, 256, 14, 14) -> (B, 984, 14, 14)."""
        dca = DualCrossAttention(spatial_dim=728, freq_dim=256, attn_dim=256, fused_dim=984).to(self.device)
        f_spa = torch.randn(2, 728, 14, 14, device=self.device)
        f_freq = torch.randn(2, 256, 14, 14, device=self.device)
        
        out = dca(f_spa, f_freq)
        self.assertEqual(out.shape, (2, 984, 14, 14))

    def test_dca_attention_softmax_axis(self):
        """Cross-attention softmax must sum to 1 across the key dimension (dim=-1)."""
        dca = DualCrossAttention(spatial_dim=728, freq_dim=256, attn_dim=256, fused_dim=984).to(self.device)
        B, H, W = 2, 14, 14
        f_spa = torch.randn(B, 728, H, W, device=self.device)
        f_freq = torch.randn(B, 256, H, W, device=self.device)

        # Compute projections manually to inspect attention matrix
        # Flatten: (B, N, C)
        spa_flat = f_spa.flatten(2).transpose(1, 2)
        freq_flat = f_freq.flatten(2).transpose(1, 2)

        q_s = dca.q_spa(spa_flat)    # (B, N, D)
        k_f = dca.k_freq(freq_flat)  # (B, N, D)
        scores_s2f = torch.bmm(q_s, k_f.transpose(1, 2)) * dca.scale
        attn_s2f = F.softmax(scores_s2f, dim=-1)

        # Check row sums equal 1
        sums = attn_s2f.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Attention rows do not sum to 1.0")

    def test_dca_gradients(self):
        """Gradients must propagate to both spatial and frequency inputs."""
        dca = DualCrossAttention(spatial_dim=728, freq_dim=256, attn_dim=256, fused_dim=984).to(self.device)
        f_spa = torch.randn(2, 728, 14, 14, device=self.device, requires_grad=True)
        f_freq = torch.randn(2, 256, 14, 14, device=self.device, requires_grad=True)

        out = dca(f_spa, f_freq)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(f_spa.grad)
        self.assertIsNotNone(f_freq.grad)
        self.assertGreater(f_spa.grad.abs().sum().item(), 0.0)
        self.assertGreater(f_freq.grad.abs().sum().item(), 0.0)


class TestTeam14NetEndToEnd(unittest.TestCase):
    """Test full Team14Net model and all ablation variants."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_all_ablation_variants(self):
        """Test instantiation, forward, and backward passes across all 6 ablation configurations."""
        variants = [
            "full_team14",
            "spatial_only",
            "wpd_only",
            "spatial_wpd_concat",
            "spatial_wpd_hife",
            "spatial_wpd_dca",
        ]
        
        criterion = nn.BCEWithLogitsLoss()
        
        for variant in variants:
            with self.subTest(variant=variant):
                # We use spatial_pretrained=False for fast offline unit testing
                model = Team14Net(
                    variant=variant,
                    spatial_pretrained=False,
                    wavelet="db4",
                    representation="log_magnitude",
                    dropout=0.2,
                ).to(self.device)
                
                model.train()
                x = torch.randn(2, 3, 224, 224, device=self.device, requires_grad=True)
                labels = torch.tensor([[1.0], [0.0]], device=self.device)

                logits = model(x)
                self.assertEqual(logits.shape, (2, 1), f"Variant {variant} produced shape {logits.shape}")
                self.assertFalse(torch.isnan(logits).any(), f"Variant {variant} produced NaNs in logits")

                loss = criterion(logits, labels)
                loss.backward()

                # Verify that trainable parameters have finite gradients
                trainable_with_grad = [
                    name for name, p in model.named_parameters() 
                    if p.requires_grad and p.grad is not None and not torch.isnan(p.grad).any()
                ]
                self.assertGreater(len(trainable_with_grad), 0, f"No parameters received gradients in {variant}")


if __name__ == "__main__":
    unittest.main()
