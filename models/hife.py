"""
models/hife.py
==============
Learnable High-Frequency Enhancement (HiFE) Module.
Specified in Team 14 Interim Technical Report (ITR) Section III-D, Eq. (5)-(6).

Under lossy video compression (e.g., H.264 c40) or aggressive JPEG quantization,
the high-frequency directional wavelet sub-bands suffer coefficient attenuation.
HiFE adaptively re-weights and enhances high-frequency detail features via:
1. Dual-pooling channel-attention gating:
       W_gate = σ( MLP(AvgPool(W_Q)) + MLP(MaxPool(W_Q)) )
2. Directional convolutional feature transformation F_conv(W_Q).
3. Residual fusion:
       W_enh = W_Q ⊙ W_gate + F_conv(W_Q)   ∈ ℝ^{B × 64 × 28 × 28}
4. Spatial grid projection (stride-2 separable conv) to align with spatial branch:
       F_freq = ConvProject(W_enh)           ∈ ℝ^{B × 256 × 14 × 14}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HiFEBlock(nn.Module):
    """Learnable High-Frequency Enhancement (HiFE) residual module.

    Parameters
    ----------
    in_channels : int, default=64
        Number of sub-band channels produced by 3-level WPD.
    out_channels : int, default=256
        Output channel dimensionality matching the frequency token projection.
    reduction : int, default=4
        Channel attention reduction ratio for the shared MLP.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 256,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ── 1. Shared Channel Attention MLP ───────────────────────────
        bottleneck = max(in_channels // reduction, 8)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, in_channels, bias=False),
        )

        # ── 2. Directional Convolution Block ──────────────────────────
        # Depthwise-separable 3x3 conv preserving spatial resolution (28x28)
        self.f_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # ── 3. Spatial Matching Projection (28x28 -> 14x14) ───────────
        # Aligns frequency tokens with the spatial branch grid (14x14)
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, w_q: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        w_q : torch.Tensor
            Wavelet packet sub-band tensor, shape (B, 64, 28, 28).

        Returns
        -------
        torch.Tensor
            Enhanced frequency feature map, shape (B, 256, 14, 14).
        """
        B, C, H, W = w_q.shape

        # Spatial average and max pooling across (28x28)
        avg_pool = F.adaptive_avg_pool2d(w_q, (1, 1)).view(B, C)
        max_pool = F.adaptive_max_pool2d(w_q, (1, 1)).view(B, C)

        # Channel gating weights
        mlp_avg = self.shared_mlp(avg_pool)
        mlp_max = self.shared_mlp(max_pool)
        w_gate = torch.sigmoid(mlp_avg + mlp_max).view(B, C, 1, 1)

        # Directional convolution transformation
        conv_out = self.f_conv(w_q)

        # Residual enhancement (ITR Eq. 6)
        w_enh = w_q * w_gate + conv_out

        # Stride-2 projection: (B, 64, 28, 28) -> (B, 256, 14, 14)
        f_freq = self.proj(w_enh)
        return f_freq
