"""
models/spatial_branch.py
========================
Spatial Context Branch – extracts deep visual features from RGB inputs
using a pre-trained Separable Xception backbone via the timm library.

Supports two operational modes:
1. Grid Output Mode (`return_grid=True`, default for Team 14 ITR):
   Extracts Middle Flow exit feature map:
   Input:  (B, 3, 224, 224)
   Output: F_spa ∈ ℝ^{B × 728 × 14 × 14}
   (Preserves spatial token coordinates for Dual Cross-Attention).

2. Vector Embedding Mode (`return_grid=False`, for backward compatibility):
   AdaptiveAvgPool2d(1) → Flatten → Linear projection → ℝ^{B × D}.
"""

from __future__ import annotations

from typing import List
import timm
import torch
import torch.nn as nn

from config import cfg


class SpatialBranch(nn.Module):
    """Pre-trained spatial feature extractor using Xception.

    Parameters
    ----------
    backbone_name : str
        Model architecture accepted by ``timm.create_model``
        (default ``"legacy_xception"``).
    pretrained : bool
        Whether to load ImageNet-pretrained weights.
    embed_dim : int
        Output embedding dimensionality when ``return_grid=False``.
    return_grid : bool
        If True, returns the 2D feature grid (B, 728, 14, 14).
        If False, applies GAP and projects to (B, embed_dim).
    """

    def __init__(
        self,
        backbone_name: str = cfg.spatial_backbone,
        pretrained: bool = cfg.spatial_pretrained,
        embed_dim: int = cfg.embed_dim,
        return_grid: bool = True,
    ) -> None:
        super().__init__()
        self.return_grid = return_grid
        self.backbone_name = backbone_name

        # Extract features up to Middle Flow exit (Stage 3 = 728 channels)
        # For legacy_xception:
        # Stage 0: 64 ch, Stage 1: 128 ch, Stage 2: 256 ch, Stage 3: 728 ch, Stage 4: 2048 ch
        out_indices = (3,) if return_grid else (-1,)
        self.backbone: nn.Module = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        channels = self.backbone.feature_info.channels()
        self.out_channels: int = channels[0] if return_grid else channels[-1]

        if not return_grid:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Sequential(
                nn.Linear(self.out_channels, embed_dim),
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
            Input RGB images, shape (B, 3, 224, 224) [or (B, 3, 256, 256)].

        Returns
        -------
        torch.Tensor
            If return_grid=True: shape (B, 728, 14, 14) [for 224x224 input].
            If return_grid=False: shape (B, embed_dim).
        """
        # features_only=True returns a list containing the requested stage
        feats = self.backbone(x)
        feat = feats[0]  # (B, 728, H', W')

        if self.return_grid:
            return feat

        # Flat vector mode
        pooled = self.pool(feat).flatten(1)
        return self.proj(pooled)
