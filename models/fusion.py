"""
models/fusion.py
================
Cross-Domain and Dual-Branch Feature Fusion Modules.

Architectures Supported
-----------------------
1. **ParallelSEFusion (`fusion_type='parallel_se'`)** [Default / Checkpoint-Compatible]:
   Applies independent Squeeze-and-Excitation style channel recalibration to each
   branch prior to concatenation and projection:
   ::
       α_s = σ( W_s2 · ReLU( W_s1 · F_spatial ) )
       α_f = σ( W_f2 · ReLU( W_f1 · F_frequency ) )
       F_fused = Linear_BN_ReLU([ α_s * F_spatial ; α_f * F_frequency ])

   Note: In previous drafts, this was informally called "cross-domain attention".
   Scientifically, this is *Parallel Self-Gated Branch Recalibration* because α_s and α_f
   are computed independently without cross-branch information exchange.

2. **CrossGatedFusion (`fusion_type='cross_gated'`)** [Recommended Novel Cross-Modal Head]:
   True cross-modal gating where the excitation weights of each domain are explicitly
   conditioned on the joint spatial-frequency representation:
   ::
       z_joint = [ F_spatial ; F_frequency ] ∈ ℝ^{B × 2D}
       latent  = ReLU( W_joint · z_joint )    ∈ ℝ^{B × 2D/r}
       α_s     = σ( W_spatial_gate · latent ) ∈ (0, 1)^{B × D}
       α_f     = σ( W_freq_gate · latent )    ∈ (0, 1)^{B × D}
       F_fused = Linear_BN_ReLU([ α_s * F_spatial ; α_f * F_frequency ])

3. **NaiveConcatFusion (`fusion_type='concat'`)** [Ablation / Baseline B3]:
   Direct vector concatenation without attention or gating:
   ::
       F_fused = Linear_BN_ReLU([ F_spatial ; F_frequency ])
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn

from config import cfg


# --------------------------------------------------------------------------- #
#  Squeeze-and-Excitation style Channel Attention (Branch-Independent)
# --------------------------------------------------------------------------- #

class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention applied to a 1-D embedding.

    Produces per-channel attention weights ``α ∈ (0, 1)^{B × D}`` via:
        α = σ( W₂ · ReLU( W₁ · x ) )
    """

    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        bottleneck = max(dim // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(dim, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return channel-wise attention weights ``α ∈ (0, 1)^{B × D}``."""
        return self.fc(x)


# --------------------------------------------------------------------------- #
#  Baseline B3: Naive Feature Concatenation (No Attention)
# --------------------------------------------------------------------------- #

class NaiveConcatFusion(nn.Module):
    """Direct concatenation and linear projection baseline (no gating)."""

    def __init__(
        self,
        embed_dim: int = cfg.embed_dim,
        dropout: float = cfg.fusion_dropout,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(
        self,
        f_spatial: torch.Tensor,
        f_freq: torch.Tensor,
    ) -> torch.Tensor:
        concat = torch.cat([f_spatial, f_freq], dim=1)
        return self.projection(concat)


# --------------------------------------------------------------------------- #
#  True Cross-Modal Gated Fusion (P0.1 Remediation / Novelty Path B)
# --------------------------------------------------------------------------- #

class CrossGatedFusion(nn.Module):
    """True cross-domain feature gating.

    Computes channel weights for both spatial and frequency domains from a
    shared joint embedding:
        z_joint = [f_spatial; f_freq]
        latent  = ReLU(W_shared · z_joint)
        α_s     = Sigmoid(W_s · latent)
        α_f     = Sigmoid(W_f · latent)
    """

    def __init__(
        self,
        embed_dim: int = cfg.embed_dim,
        reduction: int = cfg.fusion_reduction_ratio,
        dropout: float = cfg.fusion_dropout,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        joint_dim = embed_dim * 2
        bottleneck = max(joint_dim // reduction, 1)

        self.joint_encoder = nn.Sequential(
            nn.Linear(joint_dim, bottleneck, bias=False),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(inplace=True),
        )

        self.spatial_gate = nn.Sequential(
            nn.Linear(bottleneck, embed_dim, bias=False),
            nn.Sigmoid(),
        )

        self.freq_gate = nn.Sequential(
            nn.Linear(bottleneck, embed_dim, bias=False),
            nn.Sigmoid(),
        )

        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(
        self,
        f_spatial: torch.Tensor,
        f_freq: torch.Tensor,
    ) -> torch.Tensor:
        # Joint descriptor
        z_joint = torch.cat([f_spatial, f_freq], dim=1)  # (B, 2D)
        latent = self.joint_encoder(z_joint)              # (B, bottleneck)

        # Cross-conditioned channel attention
        alpha_s = self.spatial_gate(latent)               # (B, D)
        alpha_f = self.freq_gate(latent)                  # (B, D)

        # Modulate
        f_spatial_gated = f_spatial * alpha_s
        f_freq_gated = f_freq * alpha_f

        # Project
        concat = torch.cat([f_spatial_gated, f_freq_gated], dim=1)
        return self.projection(concat)


# --------------------------------------------------------------------------- #
#  Public Interface / Backward-Compatible CrossDomainFusion
# --------------------------------------------------------------------------- #

class CrossDomainFusion(nn.Module):
    """Configurable feature fusion supporting:
    - ``fusion_type='parallel_se'`` (exact match with existing checkpoints)
    - ``fusion_type='cross_gated'`` (true cross-modal joint gating)
    - ``fusion_type='concat'``      (naive concatenation ablation)

    Parameters
    ----------
    embed_dim : int
        Dimensionality ``D`` of both input embeddings.
    reduction : int
        SE / joint bottleneck reduction ratio.
    dropout : float
        Dropout probability before the final output.
    fusion_type : str
        ``'parallel_se'``, ``'cross_gated'``, or ``'concat'``.
    """

    def __init__(
        self,
        embed_dim: int = cfg.embed_dim,
        reduction: int = cfg.fusion_reduction_ratio,
        dropout: float = cfg.fusion_dropout,
        fusion_type: str = "parallel_se",
    ) -> None:
        super().__init__()
        self.fusion_type = fusion_type.lower()
        self.embed_dim = embed_dim

        if self.fusion_type in ("parallel_se", "se"):
            # Original architecture: exactly matches weights in checkpoints/best_model.pth
            self.attn_spatial = _SEBlock(embed_dim, reduction)
            self.attn_freq = _SEBlock(embed_dim, reduction)
            self.projection = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
        elif self.fusion_type in ("cross_gated", "cross_attention"):
            self.impl = CrossGatedFusion(embed_dim=embed_dim, reduction=reduction, dropout=dropout)
        elif self.fusion_type in ("concat", "naive"):
            self.impl = NaiveConcatFusion(embed_dim=embed_dim, dropout=dropout)
        else:
            raise ValueError(
                f"Unknown fusion_type '{fusion_type}'. "
                f"Supported options: 'parallel_se', 'cross_gated', 'concat'."
            )

    def forward(
        self,
        f_spatial: torch.Tensor,
        f_freq: torch.Tensor,
    ) -> torch.Tensor:
        if self.fusion_type in ("parallel_se", "se"):
            # (1) Channel attention
            alpha_s: torch.Tensor = self.attn_spatial(f_spatial)  # (B, D)
            alpha_f: torch.Tensor = self.attn_freq(f_freq)        # (B, D)

            # (2) Element-wise re-weighting
            f_spatial_att = f_spatial * alpha_s
            f_freq_att = f_freq * alpha_f

            # (3) Concatenate & project
            concat = torch.cat([f_spatial_att, f_freq_att], dim=1)
            return self.projection(concat)
        else:
            return self.impl(f_spatial, f_freq)


# Alias for explicit scientific naming
ParallelSEFusion = CrossDomainFusion

__all__ = [
    "CrossDomainFusion",
    "ParallelSEFusion",
    "CrossGatedFusion",
    "NaiveConcatFusion",
]
