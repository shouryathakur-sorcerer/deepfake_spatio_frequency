"""
config.py
=========
Central configuration for the Dual-Branch Spatio-Frequency Deepfake Detector.

All hyperparameters, filesystem paths, compression quality factors, and
training / evaluation settings are collected here so that every other module
imports a single `Config` instance instead of scattering magic numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Config:
    """Immutable (by convention) experiment configuration."""

    # ------------------------------------------------------------------
    # Paths  (override via environment or CLI wrapper)
    # ------------------------------------------------------------------
    data_root: str = os.getenv(
        "DATA_ROOT",
        "./data/_raw/real_vs_fake/real-vs-fake"
        if os.path.exists("./data/_raw/real_vs_fake/real-vs-fake")
        else "./data/faceforensics"
    )
    checkpoint_dir: str = os.getenv("CKPT_DIR", "./checkpoints")
    log_dir: str = os.getenv("LOG_DIR", "./logs")

    # ------------------------------------------------------------------
    # Image / Dataset
    # ------------------------------------------------------------------
    image_size: int = 256                       # H = W of face-aligned crops
    num_workers: int = 4                        # DataLoader workers
    pin_memory: bool = True

    # ImageNet normalisation constants
    img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ------------------------------------------------------------------
    # JPEG compression augmentation
    # ------------------------------------------------------------------
    # Mapping from compression tier tags to JPEG quality ranges (Q_min, Q_max).
    # NOTE ON TERMINOLOGY: Unlike H.264 video CRF (where higher CRF = heavier compression),
    # in JPEG Quality Factor (1-100), LOWER quality factor = HEAVIER degradation.
    # Q=23 is heavy compression, Q=40/50 is moderate, and Q=100 is uncompressed.
    # Both standardized keys (q100, q50, q40, q23) and legacy keys (c0, c23, c40) are supported.
    compression_quality_map: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "q100": (100, 100),  # Clean / Lossless baseline
            "q50":  (50, 50),    # Moderate compression
            "q40":  (40, 40),    # Medium compression
            "q23":  (23, 23),    # Heavy compression
            # Legacy aliases maintained for backward compatibility:
            "c0":   (100, 100),  # Legacy alias for Q=100
            "c40":  (40, 40),    # Legacy alias for Q=40 (medium JPEG)
            "c23":  (23, 23),    # Legacy alias for Q=23 (heavy JPEG)
        }
    )
    # When training, randomly pick one of these compression levels per sample
    train_compression_levels: List[str] = field(
        default_factory=lambda: ["q100", "q50", "q23"]
    )
    jpeg_augment_prob: float = 0.5              # prob. of applying JPEG aug

    # ------------------------------------------------------------------
    # Model – Spatial Branch
    # ------------------------------------------------------------------
    spatial_backbone: str = "legacy_xception"    # any `timm` model name
    spatial_pretrained: bool = True
    embed_dim: int = 512                        # D – unified embedding dim

    # ------------------------------------------------------------------
    # Model – Wavelet Branch
    # ------------------------------------------------------------------
    wavelet_name: str = "db4"                   # PyWavelets wavelet family
    wavelet_decomp_level: int = 2               # WPD decomposition depth
    wavelet_stem_channels: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )

    # ------------------------------------------------------------------
    # Model – Fusion
    # ------------------------------------------------------------------
    fusion_reduction_ratio: int = 8             # SE-block channel squeeze
    fusion_dropout: float = 0.3

    # ------------------------------------------------------------------
    # Classification Head
    # ------------------------------------------------------------------
    head_dropout: float = 0.5

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    use_amp: bool = True                        # automatic mixed precision
    grad_clip_norm: float = 1.0

    # Cosine Annealing LR schedule
    lr_min: float = 1e-6
    lr_warmup_epochs: int = 2

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    eval_batch_size: int = 64

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed: int = 42

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def ensure_dirs(self) -> None:
        """Create checkpoint and log directories if they do not exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


# Singleton default config – importable everywhere
cfg = Config()
