"""
models/dual_branch_net.py
=========================
End-to-end Dual-Branch Spatio-Frequency Deepfake Detector.

Integrates:
- **SpatialBranch**     : Pre-trained CNN/ViT backbone for visual features.
- **WaveletBranch**     : Multi-scale WPD frequency feature extractor.
- **CrossDomainFusion** : Dual-attention feature fusion module.
- **Classification Head**: Binary linear classifier with dropout.

Architecture
------------
::

    Input: (B, 3, 256, 256)
         ┌──────────────────┬───────────────────────┐
         ▼                  ▼                        │
    SpatialBranch      WaveletBranch                 │
    F_spatial (B, D)   F_frequency (B, D)            │
         │                  │                        │
         └──── CrossDomainFusion ────┘               │
                    │                                │
               F_fused (B, D)                        │
                    │                                │
            Classification Head                      │
                    │                                │
               logit (B, 1)   ← raw score (no sigmoid; use BCEWithLogitsLoss)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import cfg
from models.fusion import CrossDomainFusion
from models.spatial_branch import SpatialBranch
from models.wavelet_branch import WaveletBranch


class DualBranchNet(nn.Module):
    """Dual-Branch Spatio-Frequency deepfake detector.

    Parameters
    ----------
    spatial_backbone : str
        ``timm`` model name for the spatial branch.
    spatial_pretrained : bool
        Use ImageNet pre-trained weights.
    wavelet : str
        Wavelet basis name for the frequency branch.
    wavelet_level : int
        Wavelet Packet decomposition depth.
    embed_dim : int
        Unified embedding dimensionality ``D`` used across both branches
        and the fusion module.
    head_dropout : float
        Dropout before the final classification linear layer.
    """

    def __init__(
        self,
        spatial_backbone: str = cfg.spatial_backbone,
        spatial_pretrained: bool = cfg.spatial_pretrained,
        wavelet: str = cfg.wavelet_name,
        wavelet_level: int = cfg.wavelet_decomp_level,
        decomposition_type: str = "wpd",
        representation: str = "log_energy",
        use_gpu_wavelet: bool = False,
        embed_dim: int = cfg.embed_dim,
        head_dropout: float = cfg.head_dropout,
        fusion_type: str = "parallel_se",
    ) -> None:
        super().__init__()

        # ── Branch 1: Spatial ────────────────────────────────────────
        self.spatial_branch = SpatialBranch(
            backbone_name=spatial_backbone,
            pretrained=spatial_pretrained,
            embed_dim=embed_dim,
        )

        # ── Branch 2: Frequency ──────────────────────────────────────
        self.wavelet_branch = WaveletBranch(
            wavelet=wavelet,
            decomp_level=wavelet_level,
            decomposition_type=decomposition_type,
            representation=representation,
            use_gpu_wavelet=use_gpu_wavelet,
            embed_dim=embed_dim,
        )

        # ── Fusion ───────────────────────────────────────────────────
        self.fusion = CrossDomainFusion(embed_dim=embed_dim, fusion_type=fusion_type)

        # ── Classification Head ──────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(embed_dim, 1),  # single logit for binary classification
        )

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input images, shape ``(B, 3, 256, 256)``.

        Returns
        -------
        Tensor, shape ``(B, 1)``
            Raw logits (apply ``torch.sigmoid`` or use
            ``BCEWithLogitsLoss`` for training).
        """
        # (1) Extract domain-specific embeddings
        f_spatial: torch.Tensor = self.spatial_branch(x)    # (B, D)
        f_freq: torch.Tensor = self.wavelet_branch(x)       # (B, D)

        # (2) Fuse embeddings with dual-attention
        f_fused: torch.Tensor = self.fusion(f_spatial, f_freq)  # (B, D)

        # (3) Binary classification
        logits: torch.Tensor = self.classifier(f_fused)     # (B, 1)
        return logits
