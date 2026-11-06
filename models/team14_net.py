"""
models/team14_net.py
====================
Complete Team 14 Dual-Branch Spatio-Frequency Deepfake Detection Architecture.
Implements the exact end-to-end framework specified in:
Team 14 Interim Technical Report (ITR) Sections III & IV, Figures 1 & Eq. (1)-(13).

Components:
-----------
1. Spatial Context Branch:
   Pretrained Separable Xception extracting F_spa ∈ ℝ^{B × 728 × 14 × 14}.
2. Multi-Scale Wavelet Packet Frequency Branch:
   3-level 2D orthogonal Daubechies-4 (db4) WPD on luminance yielding W_Q ∈ ℝ^{B × 64 × 28 × 28}.
3. Learnable High-Frequency Enhancement (HiFE) Module:
   Channel-attention gating + directional conv + stride-2 downsampling yielding F_freq ∈ ℝ^{B × 256 × 14 × 14}.
4. Dual Cross-Attention (DCA) Module:
   Bi-directional cross-attention (A_{S->F} and A_{F->S}) with spatial skip connection yielding F_fused ∈ ℝ^{B × 984 × 14 × 14}.
5. Classification Head:
   GAP(F_fused) → z ∈ ℝ^{B × 984} → MLP(984 → 512 → 1) → logit y_hat.
   Optimized via BCEWithLogitsLoss with label smoothing (ε = 0.05).
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from models.spatial_branch import SpatialBranch
from models.wavelet_branch import DifferentiableWPD3Level
from models.hife import HiFEBlock
from models.dual_cross_attention import DualCrossAttention


class Team14Net(nn.Module):
    """End-to-end Team 14 Dual-Branch Spatio-Frequency Network with support for ablations.

    Parameters
    ----------
    variant : str, default='full_team14'
        Architecture configuration:
        - 'full_team14'        : Spatial + 3-Level WPD + HiFE + DCA (Headline Model)
        - 'spatial_only'       : Spatial Xception branch alone (Ablation B1)
        - 'wpd_only'           : 3-Level WPD + HiFE frequency branch alone (Ablation B2)
        - 'spatial_wpd_concat' : Spatial + Frequency with simple concatenation (Ablation B3)
        - 'spatial_wpd_hife'   : Spatial + Frequency + HiFE with simple concatenation
        - 'spatial_wpd_dca'    : Spatial + Frequency (without HiFE) with DCA
    spatial_pretrained : bool
        Whether to load ImageNet-pretrained weights for the Xception backbone.
    wavelet : str
        Wavelet basis family (default 'db4').
    representation : str
        Wavelet coefficient transformation ('log_magnitude', 'log_energy', 'abs_magnitude', 'raw').
    dropout : float
        Classification head dropout (default 0.5 as per ITR Section III-F).
    """

    def __init__(
        self,
        variant: str = "full_team14",
        spatial_pretrained: bool = cfg.spatial_pretrained,
        wavelet: str = cfg.wavelet_name,
        representation: str = "log_magnitude",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.variant = variant.lower()
        self.wavelet = wavelet
        self.representation = representation

        # ── 1. Spatial Branch ─────────────────────────────────────────
        if self.variant != "wpd_only":
            self.spatial_branch = SpatialBranch(
                backbone_name=cfg.spatial_backbone,
                pretrained=spatial_pretrained,
                return_grid=True,
            )
            spatial_ch = 728
        else:
            self.spatial_branch = None
            spatial_ch = 0

        # ── 2. Frequency Branch (3-Level WPD) ─────────────────────────
        if self.variant != "spatial_only":
            self.wpd = DifferentiableWPD3Level(wavelet=wavelet)
            num_subbands = 64
        else:
            self.wpd = None
            num_subbands = 0

        # ── 3. High-Frequency Enhancement (HiFE) ──────────────────────
        use_hife = self.variant in ("full_team14", "wpd_only", "spatial_wpd_hife")
        if use_hife and self.wpd is not None:
            self.hife = HiFEBlock(in_channels=64, out_channels=256)
            freq_ch = 256
        elif self.wpd is not None:
            # Without HiFE, project 64 sub-bands (28x28) to 256 channels (14x14) via plain conv
            self.hife = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            freq_ch = 256
        else:
            self.hife = None
            freq_ch = 0

        # ── 4. Fusion Module ──────────────────────────────────────────
        if self.variant in ("full_team14", "spatial_wpd_dca"):
            self.fusion = DualCrossAttention(
                spatial_dim=728,
                freq_dim=256,
                attn_dim=256,
                fused_dim=984,
                num_heads=4,
                dropout=0.1,
            )
            fused_dim = 984
        elif self.variant in ("spatial_wpd_concat", "spatial_wpd_hife"):
            # Simple vector/grid concatenation without attention
            fused_dim = 728 + 256  # 984
            self.fusion = nn.Sequential(
                nn.Conv2d(fused_dim, fused_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fused_dim),
                nn.ReLU(inplace=True),
            )
        elif self.variant == "spatial_only":
            self.fusion = None
            fused_dim = 728
        elif self.variant == "wpd_only":
            self.fusion = None
            fused_dim = 256
        else:
            raise ValueError(f"Unknown model variant '{variant}'")

        self.fused_dim = fused_dim

        # ── 5. Classification Head (ITR Section III-F, Eq. 12) ────────
        # GAP(F_fused) -> z in R^{fused_dim} -> MLP(fused_dim -> 512 -> 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),  # outputs single raw logit (use with BCEWithLogitsLoss)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input facial image tensor, shape (B, 3, 224, 224).

        Returns
        -------
        torch.Tensor
            Scalar logit per sample, shape (B, 1).
        """
        # ── 1. Spatial Forward ────────────────────────────────────────
        if self.spatial_branch is not None:
            f_spa = self.spatial_branch(x)  # (B, 728, 14, 14)
        else:
            f_spa = None

        # ── 2. Frequency Forward ──────────────────────────────────────
        if self.wpd is not None:
            w_q = self.wpd(x, representation=self.representation)  # (B, 64, 28, 28)
            f_freq = self.hife(w_q)                                # (B, 256, 14, 14)
        else:
            f_freq = None

        # ── 3. Fusion Forward ─────────────────────────────────────────
        if self.variant in ("full_team14", "spatial_wpd_dca"):
            f_fused = self.fusion(f_spa, f_freq)                   # (B, 984, 14, 14)
        elif self.variant in ("spatial_wpd_concat", "spatial_wpd_hife"):
            cat = torch.cat([f_spa, f_freq], dim=1)                 # (B, 984, 14, 14)
            f_fused = self.fusion(cat)
        elif self.variant == "spatial_only":
            f_fused = f_spa                                         # (B, 728, 14, 14)
        elif self.variant == "wpd_only":
            f_fused = f_freq                                        # (B, 256, 14, 14)
        else:
            raise RuntimeError(f"Invalid state for variant '{self.variant}'")

        # ── 4. Classification Forward ─────────────────────────────────
        z = self.gap(f_fused).flatten(1)  # (B, fused_dim)
        logit = self.classifier(z)        # (B, 1)
        return logit
