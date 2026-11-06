"""
models
======
Dual-Branch Spatio-Frequency deepfake detection model components and baselines.

Public API
----------
- SpatialBranch       : CNN/Transformer backbone for spatial features.
- WaveletBranch       : Multi-scale wavelet-packet frequency extractor.
- DifferentiableWPD2D : Native PyTorch GPU 2D wavelet packet decomposition filter bank.
- CrossDomainFusion   : Configurable feature fusion module.
- ParallelSEFusion    : Branch-independent channel attention fusion (checkpoint-compatible).
- CrossGatedFusion    : True cross-modal joint gated attention fusion.
- NaiveConcatFusion   : Direct concatenation baseline (Baseline B3).
- DualBranchNet       : End-to-end detector combining all branches.
- SpatialOnlyNet      : Baseline B1 (Spatial branch alone).
- FrequencyOnlyNet    : Baseline B2 (Frequency branch alone).
- NaiveConcatNet      : Baseline B3 (Naive concatenation).
- FFTBaselineNet      : Baseline B4 (FFT power spectrum).
- get_model           : Unified factory function to instantiate any model or baseline.
"""

from .spatial_branch import SpatialBranch
from .wavelet_branch import (
    WaveletBranch,
    DifferentiableWPD2D,
    DifferentiableWPD3Level,
    batch_wavelet_cpu,
    batch_wpd,
    batch_wpd_3level_cpu,
)
from .hife import HiFEBlock
from .dual_cross_attention import DualCrossAttention
from .team14_net import Team14Net
from .fusion import (
    CrossDomainFusion,
    ParallelSEFusion,
    CrossGatedFusion,
    NaiveConcatFusion,
)
from .dual_branch_net import DualBranchNet
from .baselines import (
    SpatialOnlyNet,
    FrequencyOnlyNet,
    NaiveConcatNet,
    FFTBaselineNet,
    get_model,
)

__all__ = [
    "SpatialBranch",
    "WaveletBranch",
    "DifferentiableWPD2D",
    "DifferentiableWPD3Level",
    "batch_wavelet_cpu",
    "batch_wpd",
    "batch_wpd_3level_cpu",
    "HiFEBlock",
    "DualCrossAttention",
    "Team14Net",
    "CrossDomainFusion",
    "ParallelSEFusion",
    "CrossGatedFusion",
    "NaiveConcatFusion",
    "DualBranchNet",
    "SpatialOnlyNet",
    "FrequencyOnlyNet",
    "NaiveConcatNet",
    "FFTBaselineNet",
    "get_model",
]
