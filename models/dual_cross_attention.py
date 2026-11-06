"""
models/dual_cross_attention.py
==============================
Dual Cross-Attention (DCA) Spatio-Frequency Fusion Module.
Specified in Team 14 Interim Technical Report (ITR) Section III-E, Eq. (7)-(11).

Combines:
1. Spatial-to-Frequency Cross-Attention (A_{S->F}): Spatial queries attend to frequency keys/values.
2. Frequency-to-Spatial Cross-Attention (A_{F->S}): Frequency queries attend to spatial keys/values.
3. Residual fusion with spatial skip-connection and LayerNorm producing F_fused in R^{B x 984 x 14 x 14}.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualCrossAttention(nn.Module):
    """Dual Cross-Attention module fusing spatial and spectral token grids.

    Parameters
    ----------
    spatial_dim : int, default=728
        Channel dimension of the spatial feature grid F_spa.
    freq_dim : int, default=256
        Channel dimension of the enhanced frequency feature grid F_freq.
    attn_dim : int, default=256
        Shared projection dimensionality for queries and keys (d_k).
    fused_dim : int, default=984
        Final fused channel dimension (spatial_dim + freq_dim = 728 + 256 = 984).
    num_heads : int, default=4
        Number of attention heads.
    dropout : float, default=0.1
        Attention and projection dropout.
    """

    def __init__(
        self,
        spatial_dim: int = 728,
        freq_dim: int = 256,
        attn_dim: int = 256,
        fused_dim: int = 984,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.freq_dim = freq_dim
        self.attn_dim = attn_dim
        self.fused_dim = fused_dim
        self.num_heads = num_heads
        self.scale = 1.0 / math.sqrt(attn_dim // num_heads)

        # ── 1. Spatial -> Frequency Projections (ITR Eq. 7) ───────────
        # Queries from spatial, Keys & Values from frequency
        self.q_spa = nn.Linear(spatial_dim, attn_dim, bias=False)
        self.k_freq = nn.Linear(freq_dim, attn_dim, bias=False)
        self.v_freq = nn.Linear(freq_dim, freq_dim, bias=False)

        # ── 2. Frequency -> Spatial Projections (ITR Eq. 8) ───────────
        # Queries from frequency, Keys & Values from spatial
        self.q_freq = nn.Linear(freq_dim, attn_dim, bias=False)
        self.k_spa = nn.Linear(spatial_dim, attn_dim, bias=False)
        self.v_spa = nn.Linear(spatial_dim, spatial_dim, bias=False)

        # ── 3. Attention Dropout ──────────────────────────────────────
        self.attn_drop = nn.Dropout(p=dropout)

        # ── 4. Fusion Bottleneck Projection & Residual (ITR Eq. 11) ───
        cat_dim = freq_dim + spatial_dim  # 256 + 728 = 984
        self.proj = nn.Linear(cat_dim, fused_dim)
        self.spatial_res_proj = (
            nn.Linear(spatial_dim, fused_dim)
            if spatial_dim != fused_dim
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(fused_dim)
        self.proj_drop = nn.Dropout(p=dropout)

    def _cross_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        v_dim: int,
    ) -> torch.Tensor:
        """Multi-head cross-attention mechanism.
        q: (B, N, attn_dim)
        k: (B, N, attn_dim)
        v: (B, N, v_dim)
        Returns: (B, N, v_dim)
        """
        B, N, _ = q.shape
        H = self.num_heads
        d_k = self.attn_dim // H
        d_v = v_dim // H

        # Reshape to (B, H, N, d)
        q_h = q.view(B, N, H, d_k).transpose(1, 2)
        k_h = k.view(B, N, H, d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # Reshape values
        v_h = v.view(B, N, H, d_v).transpose(1, 2)
        out = torch.matmul(attn, v_h)  # (B, H, N, d_v)
        out = out.transpose(1, 2).contiguous().view(B, N, v_dim)
        return out

    def forward(
        self,
        f_spa: torch.Tensor,
        f_freq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        f_spa : torch.Tensor
            Spatial feature grid, shape (B, 728, 14, 14).
        f_freq : torch.Tensor
            Enhanced frequency feature grid, shape (B, 256, 14, 14).

        Returns
        -------
        torch.Tensor
            Fused spatial-frequency grid, shape (B, 984, 14, 14).
        """
        B, C_s, H, W = f_spa.shape
        _, C_f, H_f, W_f = f_freq.shape
        assert H == H_f and W == W_f, f"Grid mismatch: {H}x{W} vs {H_f}x{W_f}"

        N = H * W  # 14x14 = 196 tokens

        # Reshape to token sequences: (B, 196, channels)
        x_spa = f_spa.view(B, C_s, N).transpose(1, 2)    # (B, 196, 728)
        x_freq = f_freq.view(B, C_f, N).transpose(1, 2)  # (B, 196, 256)

        # ── Compute Spatial -> Frequency Attention (ITR Eq. 9) ────────
        q1 = self.q_spa(x_spa)
        k1 = self.k_freq(x_freq)
        v1 = self.v_freq(x_freq)
        a_s_to_f = self._cross_attention(q1, k1, v1, self.freq_dim)  # (B, 196, 256)

        # ── Compute Frequency -> Spatial Attention (ITR Eq. 10) ───────
        q2 = self.q_freq(x_freq)
        k2 = self.k_spa(x_spa)
        v2 = self.v_spa(x_spa)
        a_f_to_s = self._cross_attention(q2, k2, v2, self.spatial_dim)  # (B, 196, 728)

        # ── Concatenation and Residual Skip Connection (ITR Eq. 11) ───
        cat_attn = torch.cat([a_s_to_f, a_f_to_s], dim=-1)  # (B, 196, 984)
        proj_attn = self.proj(cat_attn)

        # Residual connection from spatial features
        res_spa = self.spatial_res_proj(x_spa)
        f_fused_tokens = self.norm(proj_attn + res_spa)
        f_fused_tokens = self.proj_drop(f_fused_tokens)

        # Reshape back to 2D feature grid: (B, 984, 14, 14)
        f_fused = f_fused_tokens.transpose(1, 2).view(B, self.fused_dim, H, W)
        return f_fused
