"""
models/baselines.py
===================
Comparative Baseline Architectures for Deepfake Detection Benchmarks.

Provides standard baselines to substantiate architectural claims:
- **SpatialOnlyNet (B1)**: Spatial Xception branch alone (no frequency features).
- **FrequencyOnlyNet (B2)**: 16-subband WPD-CNN branch alone (no spatial features).
- **NaiveConcatNet (B3)**: Dual-branch spatial + frequency with direct concatenation (no gating).
- **FFTBaselineNet (B4)**: Fourier Analysis baseline using 2D log-magnitude power spectra.
"""

from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn
import torch.fft

from config import cfg
from models.spatial_branch import SpatialBranch
from models.wavelet_branch import WaveletBranch, _conv_bn_relu
from models.dual_branch_net import DualBranchNet


# --------------------------------------------------------------------------- #
#  Baseline B1: Spatial Branch Alone (Xception)
# --------------------------------------------------------------------------- #

class SpatialOnlyNet(nn.Module):
    """Spatial-only detector using ImageNet-pretrained Xception backbone."""

    def __init__(
        self,
        spatial_backbone: str = cfg.spatial_backbone,
        pretrained: bool = cfg.spatial_pretrained,
        embed_dim: int = cfg.embed_dim,
        head_dropout: float = cfg.head_dropout,
    ) -> None:
        super().__init__()
        self.spatial_branch = SpatialBranch(
            backbone_name=spatial_backbone,
            pretrained=pretrained,
            embed_dim=embed_dim,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_spatial = self.spatial_branch(x)  # (B, D)
        return self.classifier(f_spatial)   # (B, 1)


# --------------------------------------------------------------------------- #
#  Baseline B2: Frequency Branch Alone (WPD-CNN)
# --------------------------------------------------------------------------- #

class FrequencyOnlyNet(nn.Module):
    """Pure frequency-domain detector using 16-subband Wavelet Packet Decomposition."""

    def __init__(
        self,
        wavelet: str = cfg.wavelet_name,
        wavelet_level: int = cfg.wavelet_decomp_level,
        embed_dim: int = cfg.embed_dim,
        head_dropout: float = cfg.head_dropout,
    ) -> None:
        super().__init__()
        self.wavelet_branch = WaveletBranch(
            wavelet=wavelet,
            decomp_level=wavelet_level,
            embed_dim=embed_dim,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_freq = self.wavelet_branch(x)  # (B, D)
        return self.classifier(f_freq)   # (B, 1)


# --------------------------------------------------------------------------- #
#  Baseline B3: Naive Feature Concatenation (Ablation of SE / Gating)
# --------------------------------------------------------------------------- #

class NaiveConcatNet(DualBranchNet):
    """Dual-branch detector using direct concatenation without attention or gating."""

    def __init__(
        self,
        spatial_backbone: str = cfg.spatial_backbone,
        spatial_pretrained: bool = cfg.spatial_pretrained,
        wavelet: str = cfg.wavelet_name,
        wavelet_level: int = cfg.wavelet_decomp_level,
        embed_dim: int = cfg.embed_dim,
        head_dropout: float = cfg.head_dropout,
    ) -> None:
        super().__init__(
            spatial_backbone=spatial_backbone,
            spatial_pretrained=spatial_pretrained,
            wavelet=wavelet,
            wavelet_level=wavelet_level,
            embed_dim=embed_dim,
            head_dropout=head_dropout,
            fusion_type="concat",
        )


# --------------------------------------------------------------------------- #
#  Baseline B4: Fast Fourier Transform (FFT) Power Spectrum Detector
# --------------------------------------------------------------------------- #

class FFTBaselineNet(nn.Module):
    """Established Fourier analysis baseline (Frank et al. / Durall et al.).
    Computes centered 2D log-magnitude power spectrum and passes through a CNN stem.
    """

    def __init__(
        self,
        embed_dim: int = cfg.embed_dim,
        head_dropout: float = cfg.head_dropout,
    ) -> None:
        super().__init__()
        # 4-stage convolutional stem operating on 1-channel log-power spectrum
        self.stem = nn.Sequential(
            _conv_bn_relu(1, 64, stride=2),    # (B, 64, 128, 128)
            _conv_bn_relu(64, 128, stride=2),  # (B, 128, 64, 64)
            _conv_bn_relu(128, 256, stride=2), # (B, 256, 32, 32)
            _conv_bn_relu(256, 512, stride=2), # (B, 512, 16, 16)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1) De-normalize to [0, 1]
        device = x.device
        mean = torch.tensor(cfg.img_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(cfg.img_std, device=device).view(1, 3, 1, 1)
        x_01 = (x * std + mean).clamp(0.0, 1.0)

        # (2) Convert to grayscale
        gray = 0.299 * x_01[:, 0:1] + 0.587 * x_01[:, 1:2] + 0.114 * x_01[:, 2:3]

        # (3) 2D FFT & shift DC to center
        fft2 = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft2)
        magnitude = torch.abs(fft_shift)
        log_spectrum = torch.log(magnitude + 1e-6)  # (B, 1, 256, 256)

        # (4) Stem & Classifier
        emb = self.stem(log_spectrum)
        return self.classifier(emb)


# --------------------------------------------------------------------------- #
#  Model Factory
# --------------------------------------------------------------------------- #

def get_model(model_name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name.

    Supported names:
    - ``'full_team14'``                    : Team 14 Dual-Branch (Spatial + 3-Lvl WPD + HiFE + DCA).
    - ``'spatial_wpd_concat'``             : Team 14 Ablation: Spatial + 3-Lvl WPD with concatenation.
    - ``'spatial_wpd_hife'``               : Team 14 Ablation: Spatial + 3-Lvl WPD + HiFE with concat.
    - ``'spatial_wpd_dca'``                : Team 14 Ablation: Spatial + 3-Lvl WPD + DCA (without HiFE).
    - ``'wpd_only'``                       : Team 14 Ablation: 3-Lvl WPD + HiFE frequency branch alone.
    - ``'dual_branch'`` / ``'parallel_se'`` : Legacy dual-branch with parallel SE.
    - ``'cross_gated'``                    : Legacy dual-branch with true cross-modal gating.
    - ``'spatial_only'`` / ``'b1'``         : Spatial branch alone.
    - ``'frequency_only'`` / ``'b2'``       : Frequency branch alone.
    - ``'naive_concat'`` / ``'b3'``         : Naive concatenation without gating.
    - ``'fft_baseline'`` / ``'b4'``         : 2D FFT power spectrum detector.
    """
    name = model_name.lower().strip()
    if name in ("full_team14", "team14", "team14_net"):
        from models.team14_net import Team14Net
        return Team14Net(variant="full_team14", **kwargs)
    elif name in ("spatial_wpd_concat", "team14_concat"):
        from models.team14_net import Team14Net
        return Team14Net(variant="spatial_wpd_concat", **kwargs)
    elif name in ("spatial_wpd_hife", "team14_hife"):
        from models.team14_net import Team14Net
        return Team14Net(variant="spatial_wpd_hife", **kwargs)
    elif name in ("spatial_wpd_dca", "team14_dca"):
        from models.team14_net import Team14Net
        return Team14Net(variant="spatial_wpd_dca", **kwargs)
    elif name in ("wpd_only", "team14_wpd"):
        from models.team14_net import Team14Net
        return Team14Net(variant="wpd_only", **kwargs)
    elif name in ("dual_branch", "parallel_se", "proposed"):
        return DualBranchNet(fusion_type="parallel_se", **kwargs)
    elif name in ("cross_gated", "cross_attention"):
        return DualBranchNet(fusion_type="cross_gated", **kwargs)
    elif name in ("spatial_only", "b1", "spatial"):
        return SpatialOnlyNet(**kwargs)
    elif name in ("frequency_only", "b2", "frequency", "wavelet"):
        return FrequencyOnlyNet(**kwargs)
    elif name in ("naive_concat", "b3", "concat"):
        return NaiveConcatNet(**kwargs)
    elif name in ("fft_baseline", "b4", "fft"):
        return FFTBaselineNet(**kwargs)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Choose from: ['full_team14', 'spatial_wpd_concat', 'spatial_wpd_hife', 'spatial_wpd_dca', 'wpd_only', 'dual_branch', 'cross_gated', 'spatial_only', 'frequency_only', 'naive_concat', 'fft_baseline']"
        )


__all__ = [
    "SpatialOnlyNet",
    "FrequencyOnlyNet",
    "NaiveConcatNet",
    "FFTBaselineNet",
    "get_model",
]
