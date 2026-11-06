"""
models/wavelet_branch.py
========================
Multi-Scale Wavelet Frequency Branch.

Features:
1. **Three-Level 2D Daubechies-4 WPD (Team 14 ITR Specification)**:
   Decomposes luminance channel Y_0 = 0.299R + 0.587G + 0.114B across 3 levels:
   Level 1: 1 -> 4 sub-bands (112x112)
   Level 2: 4 -> 16 sub-bands (56x56)
   Level 3: 16 -> 64 sub-bands (28x28)
   Produces W_Q in R^{B x 64 x 28 x 28} directly on GPU.

2. **Differentiable PyTorch Filter Bank**:
   Uses separable 2D convolution with orthogonal db4 scaling (h_L) and wavelet (h_H) filters.
   Preserves autograd computation graph, enables GPU acceleration, and avoids CPU sync.

3. **Mathematical Representation Choices**:
   - 'log_magnitude': log(|W| + eps)  [Standard dynamic range compression]
   - 'log_energy'   : log(W^2 + eps)  [True logarithmic energy formulation]
   - 'abs_magnitude': |W|
   - 'raw'          : W

4. **Backward Compatibility**:
   Retains 2-level WPD (16 sub-bands) and standard DWT (4 sub-bands) for ablation benchmarking.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from models.hife import HiFEBlock

_EPS: float = 1e-8


def _conv_bn_relu(
    in_c: int,
    out_c: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> nn.Sequential:
    """Helper module: Conv2d -> BatchNorm2d -> ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


# --------------------------------------------------------------------------- #
#  Wavelet Filter Bank Construction (Orthogonal Daubechies-4)
# --------------------------------------------------------------------------- #

def get_2d_wavelet_filters(wavelet: str = "db4") -> Tuple[torch.Tensor, int]:
    """Build the 4 separable 2D quadrature mirror filters (LL, LH, HL, HH)."""
    w = pywt.Wavelet(wavelet)
    dec_lo = torch.tensor(w.dec_lo, dtype=torch.float32)
    dec_hi = torch.tensor(w.dec_hi, dtype=torch.float32)

    ll = torch.outer(dec_lo, dec_lo).unsqueeze(0).unsqueeze(0)  # Low-Low (Approx)
    lh = torch.outer(dec_lo, dec_hi).unsqueeze(0).unsqueeze(0)  # Low-High (Horizontal)
    hl = torch.outer(dec_hi, dec_lo).unsqueeze(0).unsqueeze(0)  # High-Low (Vertical)
    hh = torch.outer(dec_hi, dec_hi).unsqueeze(0).unsqueeze(0)  # High-High (Diagonal)

    filters = torch.cat([ll, lh, hl, hh], dim=0)  # (4, 1, K, K)
    filter_len = len(w.dec_lo)
    return filters, filter_len


# --------------------------------------------------------------------------- #
#  PyWavelets CPU Reference Implementations (for numerical validation)
# --------------------------------------------------------------------------- #

def pywt_wpd_3level_cpu(
    luminance_np: np.ndarray,
    wavelet: str = "db4",
) -> np.ndarray:
    """Full 3-level 2D Wavelet Packet Decomposition via PyWavelets on CPU.
    Returns array of shape (64, H/8, W/8).
    """
    wp = pywt.WaveletPacket2D(data=luminance_np, wavelet=wavelet, maxlevel=3, mode="reflect")
    terminal_nodes = [node for node in wp.get_level(3, order="natural")]
    subbands = np.stack(
        [node.data.astype(np.float32) for node in terminal_nodes],
        axis=0,
    )  # (64, H_sub, W_sub)
    return subbands


def batch_wpd_3level_cpu(
    images_np: np.ndarray,
    wavelet: str = "db4",
    representation: str = "raw",
) -> np.ndarray:
    """Batch 3-level WPD on CPU for RGB images.
    Parameters:
        images_np: (B, 3, H, W) normalized float array.
    Returns:
        (B, 64, H/8, W/8)
    """
    B, C, H, W = images_np.shape
    # Unnormalize to [0, 1] using standard cfg
    mean = np.array(cfg.img_mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(cfg.img_std, dtype=np.float32).reshape(1, 3, 1, 1)
    unnorm = np.clip(images_np * std + mean, 0.0, 1.0)
    # Luminance Y = 0.299R + 0.587G + 0.114B
    lum = (
        0.299 * unnorm[:, 0]
        + 0.587 * unnorm[:, 1]
        + 0.114 * unnorm[:, 2]
    )  # (B, H, W)

    batch_out = []
    for b in range(B):
        subbands = pywt_wpd_3level_cpu(lum[b], wavelet=wavelet)
        batch_out.append(subbands)
    arr = np.stack(batch_out, axis=0)  # (B, 64, H/8, W/8)

    if representation == "log_magnitude":
        return np.log(np.abs(arr) + _EPS)
    elif representation == "log_energy":
        return np.log(arr ** 2 + _EPS)
    elif representation == "abs_magnitude":
        return np.abs(arr)
    return arr


def batch_wavelet_cpu(
    images_np: np.ndarray,
    wavelet: str = "db4",
    level: int = 2,
    representation: str = "log_magnitude",
) -> np.ndarray:
    """Backward compatibility CPU function for 2-level WPD."""
    B, C, H, W = images_np.shape
    mean = np.array(cfg.img_mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(cfg.img_std, dtype=np.float32).reshape(1, 3, 1, 1)
    unnorm = np.clip(images_np * std + mean, 0.0, 1.0)
    lum = 0.299 * unnorm[:, 0] + 0.587 * unnorm[:, 1] + 0.114 * unnorm[:, 2]

    batch_out = []
    for b in range(B):
        wp = pywt.WaveletPacket2D(data=lum[b], wavelet=wavelet, maxlevel=level, mode="reflect")
        nodes = [node for node in wp.get_level(level, order="natural")]
        subbands = np.stack([node.data.astype(np.float32) for node in nodes], axis=0)
        batch_out.append(subbands)
    arr = np.stack(batch_out, axis=0)
    if representation in ("log_magnitude", "log_energy"):
        return np.log(np.abs(arr) + _EPS)
    return arr


batch_wpd = batch_wavelet_cpu


# --------------------------------------------------------------------------- #
#  Differentiable 3-Level GPU/CPU Wavelet Packet Decomposition (Team 14 ITR)
# --------------------------------------------------------------------------- #

class DifferentiableWPD3Level(nn.Module):
    """Native PyTorch implementation of 3-Level 2D Daubechies-4 WPD.
    Recursively decomposes all low- and high-frequency nodes across 3 levels:
        Input:  (B, 1, 224, 224) Luminance
        Level 1: 1 -> 4 sub-bands (112x112)
        Level 2: 4 -> 16 sub-bands (56x56)
        Level 3: 16 -> 64 sub-bands (28x28)
    Fully differentiable, batch-vectorized, and supports CPU and CUDA.
    """

    def __init__(self, wavelet: str = "db4") -> None:
        super().__init__()
        self.wavelet = wavelet
        filters, self.filter_len = get_2d_wavelet_filters(wavelet)
        self.register_buffer("filters", filters)  # (4, 1, K, K)
        # Pad for stride-2 dyadic downsampling: K=8 -> pad=(3, 4, 3, 4)
        self.pad = (self.filter_len // 2 - 1, self.filter_len // 2,
                     self.filter_len // 2 - 1, self.filter_len // 2)

    def _decompose_step(self, x: torch.Tensor) -> torch.Tensor:
        """Decompose tensor of shape (B, C, H, W) into (B, C*4, H/2, W/2)."""
        B, C, H, W = x.shape
        x_reshaped = x.view(B * C, 1, H, W)
        x_pad = F.pad(x_reshaped, self.pad, mode="reflect")
        out = F.conv2d(x_pad, self.filters, stride=2)
        H_out, W_out = out.shape[-2], out.shape[-1]
        return out.view(B, C * 4, H_out, W_out)

    def forward(
        self,
        images: torch.Tensor,
        representation: str = "log_magnitude",
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        images : torch.Tensor
            Normalized RGB tensor, shape (B, 3, 224, 224).
        representation : str
            'log_magnitude': log(|W| + eps)
            'log_energy'   : log(W^2 + eps)
            'abs_magnitude': |W|
            'raw'          : W

        Returns
        -------
        torch.Tensor
            Sub-band tensor W_Q, shape (B, 64, 28, 28).
        """
        device = images.device
        mean = torch.tensor(cfg.img_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(cfg.img_std, device=device).view(1, 3, 1, 1)
        imgs_01 = (images * std + mean).clamp(0.0, 1.0)

        # Convert to luminance Y_0 = 0.299R + 0.587G + 0.114B (ITR Section III-C)
        luminance = (
            0.299 * imgs_01[:, 0:1]
            + 0.587 * imgs_01[:, 1:2]
            + 0.114 * imgs_01[:, 2:3]
        )  # (B, 1, 224, 224)

        # Level 1: 1 -> 4 sub-bands (112x112)
        l1 = self._decompose_step(luminance)
        # Level 2: 4 -> 16 sub-bands (56x56)
        l2 = self._decompose_step(l1)
        # Level 3: 16 -> 64 sub-bands (28x28)
        l3 = self._decompose_step(l2)

        # Apply representation transformation
        if representation in ("log_magnitude", "log_energy_magnitude"):
            return torch.log(torch.abs(l3) + _EPS)
        elif representation == "log_energy":
            return torch.log(l3 ** 2 + _EPS)
        elif representation == "abs_magnitude":
            return torch.abs(l3)
        elif representation == "raw":
            return l3
        else:
            raise ValueError(f"Unknown representation '{representation}'")


# --------------------------------------------------------------------------- #
#  Differentiable 2-Level WPD (for Backward Compatibility with Existing Checkpoints)
# --------------------------------------------------------------------------- #

class DifferentiableWPD2D(nn.Module):
    """Native PyTorch GPU implementation of 2-Level 2D Daubechies-4 WPD (16 sub-bands).
    Retained for backward compatibility with 256x256 checkpoints.
    """

    def __init__(self, wavelet: str = "db4") -> None:
        super().__init__()
        filters, self.filter_len = get_2d_wavelet_filters(wavelet)
        self.register_buffer("filters", filters)
        self.pad = (self.filter_len // 2 - 1, self.filter_len // 2,
                     self.filter_len // 2 - 1, self.filter_len // 2)

    def _decompose_step(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_reshaped = x.view(B * C, 1, H, W)
        x_pad = F.pad(x_reshaped, self.pad, mode="reflect")
        out = F.conv2d(x_pad, self.filters, stride=2)
        H_out, W_out = out.shape[-2], out.shape[-1]
        return out.view(B, C * 4, H_out, W_out)

    def forward(
        self,
        images: torch.Tensor,
        representation: str = "log_magnitude",
    ) -> torch.Tensor:
        device = images.device
        mean = torch.tensor(cfg.img_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(cfg.img_std, device=device).view(1, 3, 1, 1)
        imgs_01 = (images * std + mean).clamp(0.0, 1.0)
        gray = (
            0.299 * imgs_01[:, 0:1]
            + 0.587 * imgs_01[:, 1:2]
            + 0.114 * imgs_01[:, 2:3]
        )
        l1 = self._decompose_step(gray)
        l2 = self._decompose_step(l1)

        if representation in ("log_energy", "log_magnitude"):
            return torch.log(torch.abs(l2) + _EPS)
        elif representation == "abs_magnitude":
            return torch.abs(l2)
        return l2


_wpd_2d = DifferentiableWPD2D


# --------------------------------------------------------------------------- #
#  Wavelet Frequency Branch Module
# --------------------------------------------------------------------------- #

class WaveletBranch(nn.Module):
    """Complete Frequency Branch with WPD and optional HiFE enhancement.

    Parameters
    ----------
    wavelet : str
        Wavelet basis name (default ``"db4"``).
    decomp_level : int
        Wavelet decomposition depth (3 for Team 14 ITR = 64 sub-bands; 2 = 16 sub-bands).
    use_hife : bool
        Whether to attach the learnable High-Frequency Enhancement module.
    return_grid : bool
        If True, returns (B, 256, 14, 14) token grid for DCA fusion.
        If False, pools to (B, embed_dim) for flat fusion.
    embed_dim : int
        Output embedding dimensionality when return_grid=False.
    """

    def __init__(
        self,
        wavelet: str = cfg.wavelet_name,
        decomp_level: int = 3,
        representation: str = "log_magnitude",
        use_hife: bool = True,
        return_grid: bool = True,
        embed_dim: int = cfg.embed_dim,
    ) -> None:
        super().__init__()
        self.decomp_level = decomp_level
        self.representation = representation
        self.use_hife = use_hife
        self.return_grid = return_grid

        # ── 1. Wavelet Decomposition Module ───────────────────────────
        if decomp_level == 3:
            self.wpd = DifferentiableWPD3Level(wavelet=wavelet)
            num_subbands = 64
        else:
            self.wpd = DifferentiableWPD2D(wavelet=wavelet)
            num_subbands = 16

        # ── 2. HiFE Module ────────────────────────────────────────────
        if use_hife and decomp_level == 3:
            self.hife = HiFEBlock(in_channels=num_subbands, out_channels=256)
        else:
            self.hife = None

        # ── 3. Projection / Pooling for Non-Grid Mode ─────────────────
        if not return_grid:
            self.pool = nn.AdaptiveAvgPool2d(1)
            in_ch = 256 if (use_hife and decomp_level == 3) else num_subbands
            self.proj = nn.Sequential(
                nn.Linear(in_ch, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.pool = None
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Normalized RGB tensor, shape (B, 3, 224, 224).

        Returns
        -------
        torch.Tensor
            If return_grid=True: shape (B, 256, 14, 14).
            If return_grid=False: shape (B, embed_dim).
        """
        w_subbands = self.wpd(x, representation=self.representation)  # (B, 64, 28, 28)

        if self.hife is not None:
            f_freq = self.hife(w_subbands)  # (B, 256, 14, 14)
        else:
            f_freq = w_subbands

        if self.return_grid:
            return f_freq

        # Vector embedding mode
        pooled = self.pool(f_freq).flatten(1)
        return self.proj(pooled)
