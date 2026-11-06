"""
dataset.py
==========
PyTorch Dataset & DataLoader factory for face-aligned deepfake crops.

Key Features
------------
* Loads face-aligned **256 × 256** RGB crops from a directory tree
  organised as ``<split>/<label>/``.
* **On-the-fly JPEG compression augmentation** via ``albumentations`` –
  simulates c0 (quality 100), c23, and c40 degradation so the model
  learns compression-robust representations during training.
* Standard ImageNet normalisation (mean & std).

Expected directory layout::

    data_root/
    ├── train/
    │   ├── real/
    │   │   ├── 00001.png
    │   │   └── ...
    │   └── fake/
    │       ├── 00001.png
    │       └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from config import Config, cfg


# --------------------------------------------------------------------------- #
#  Augmentation pipelines
# --------------------------------------------------------------------------- #

def _build_jpeg_augmentation(
    compression_map: Dict[str, Tuple[int, int]],
    compression_levels: List[str],
    prob: float,
) -> A.ImageCompression:
    """Return an ``albumentations.ImageCompression`` transform that randomly
    samples a JPEG quality factor from the configured compression tiers.

    Parameters
    ----------
    compression_map : dict
        ``{ "c0": (100, 100), "c23": (23, 23), "c40": (40, 40) }``.
    compression_levels : list[str]
        Subset of keys to sample from during training.
    prob : float
        Probability of applying the compression transform per sample.

    Returns
    -------
    A.ImageCompression
        Configured augmentation object.
    """
    # Pick a random tier on each call (Albumentations handles quality_lower /
    # quality_upper internally; we override them dynamically via a
    # ``ReplayCompose`` wrapper later if needed).
    # For simplicity we set a wide range covering all configured tiers.
    all_qs = [q for lvl in compression_levels for q in compression_map[lvl]]
    q_lo, q_hi = min(all_qs), max(all_qs)
    try:
        return A.ImageCompression(
            quality_range=(q_lo, q_hi),
            compression_type="jpeg",
            p=prob,
        )
    except TypeError:
        return A.ImageCompression(
            quality_lower=q_lo,
            quality_upper=q_hi,
            p=prob,
        )


def build_train_transforms(cfg: Config) -> A.Compose:
    """Full training augmentation pipeline.

    Steps
    -----
    1. Resize to ``cfg.image_size`` (safety net).
    2. Random horizontal flip.
    3. Random brightness / contrast jitter.
    4. **JPEG compression augmentation** (core contribution).
    5. ImageNet normalisation.
    6. Convert to ``CHW`` PyTorch tensor.
    """
    try:
        noise_layer = A.GaussNoise(p=0.2)
    except Exception:
        noise_layer = A.NoOp()

    return A.Compose([
        A.Resize(cfg.image_size, cfg.image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3,
        ),
        noise_layer,
        _build_jpeg_augmentation(
            cfg.compression_quality_map,
            cfg.train_compression_levels,
            cfg.jpeg_augment_prob,
        ),
        A.Normalize(mean=cfg.img_mean, std=cfg.img_std),
        ToTensorV2(),
    ])


def build_eval_transforms(
    cfg: Config,
    jpeg_quality: Optional[int] = None,
) -> A.Compose:
    """Evaluation pipeline.

    Parameters
    ----------
    jpeg_quality : int | None
        If given, applies deterministic JPEG compression at this quality
        factor **before** normalisation – used for per-compression-tier
        robustness benchmarking (``eval.py``).
    """
    transforms: list = [A.Resize(cfg.image_size, cfg.image_size)]
    if jpeg_quality is not None:
        try:
            transforms.append(
                A.ImageCompression(
                    quality_range=(jpeg_quality, jpeg_quality),
                    compression_type="jpeg",
                    p=1.0,
                )
            )
        except TypeError:
            transforms.append(
                A.ImageCompression(
                    quality_lower=jpeg_quality,
                    quality_upper=jpeg_quality,
                    p=1.0,
                )
            )
    transforms += [
        A.Normalize(mean=cfg.img_mean, std=cfg.img_std),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


# --------------------------------------------------------------------------- #
#  PyTorch Dataset
# --------------------------------------------------------------------------- #

class DeepfakeDataset(Dataset):
    """Binary deepfake detection dataset.

    Attributes
    ----------
    samples : list[tuple[str, int]]
        ``(image_path, label)`` pairs.  ``label = 0`` → real, ``1`` → fake.
    transform : A.Compose
        Albumentations pipeline applied to each sample.

    Shape contract
    ~~~~~~~~~~~~~~
    ``__getitem__`` returns ``(image, label)`` where:
    - ``image`` : ``torch.Tensor`` of shape ``(3, H, W)`` (float32, normalised).
    - ``label`` : ``torch.Tensor`` scalar ``{0, 1}`` (float32 for BCE loss).
    """

    LABEL_MAP: Dict[str, int] = {"real": 0, "fake": 1}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
    ) -> None:
        super().__init__()
        split_path = Path(root) / split
        if not split_path.exists() and split == "val":
            alt_path = Path(root) / "valid"
            if alt_path.exists():
                split_path = alt_path
        self.root = split_path
        self.transform = transform
        self.samples: List[Tuple[str, int]] = self._discover_samples()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _discover_samples(self) -> List[Tuple[str, int]]:
        """Walk ``<root>/<split>/<label>/`` and collect image paths."""
        samples: List[Tuple[str, int]] = []
        for label_name, label_id in self.LABEL_MAP.items():
            label_dir = self.root / label_name
            if not label_dir.is_dir():
                continue
            for fname in sorted(os.listdir(label_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    fpath = label_dir / fname
                    try:
                        if fpath.stat().st_size > 0:
                            samples.append((str(fpath), label_id))
                    except OSError:
                        pass
        if not samples:
            raise FileNotFoundError(
                f"No images found under {self.root}.  "
                f"Expected sub-folders: {list(self.LABEL_MAP.keys())}"
            )
        return samples

    # ------------------------------------------------------------------ #
    #  Dataset protocol
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]

        # Read as BGR -> convert to RGB (OpenCV convention)
        image: Optional[np.ndarray] = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            # Defensive fallback to next valid sample if a file is corrupted
            alt_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(alt_idx)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]  # (3, H, W) float32 tensor

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image, label_tensor


# --------------------------------------------------------------------------- #
#  DataLoader factory
# --------------------------------------------------------------------------- #

def build_dataloaders(
    cfg: Config,
    jpeg_quality: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """Construct train / val / test ``DataLoader`` instances.

    Parameters
    ----------
    cfg : Config
        Global experiment configuration.
    jpeg_quality : int | None
        If given, all splits use this deterministic JPEG quality
        (useful for compression-specific evaluation).

    Returns
    -------
    dict[str, DataLoader]
        ``{"train": ..., "val": ..., "test": ...}``
    """
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        split_dir = Path(cfg.data_root) / split
        if not split_dir.exists() and split == "val":
            alt_dir = Path(cfg.data_root) / "valid"
            if alt_dir.exists():
                split_dir = alt_dir
        if not split_dir.exists():
            continue  # silently skip missing splits

        is_train = split == "train"
        if is_train and jpeg_quality is None:
            transform = build_train_transforms(cfg)
        else:
            transform = build_eval_transforms(cfg, jpeg_quality=jpeg_quality)

        ds = DeepfakeDataset(
            root=cfg.data_root,
            split=split,
            transform=transform,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size if is_train else cfg.eval_batch_size,
            shuffle=is_train,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=is_train,
        )
    return loaders
