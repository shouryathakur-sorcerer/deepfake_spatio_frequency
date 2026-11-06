"""
prepare_data.py
===============
Downloads and organises face images for deepfake detection training.

Data Source
-----------
**Kaggle – "140k Real and Fake Faces"** by xhlulu
(https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

- 70,000 **real** faces (Flickr-Faces-HQ resized)
- 70,000 **fake** faces (StyleGAN-generated)
- Already aligned and cropped – perfect for our 256×256 pipeline.

Setup (one-time)
~~~~~~~~~~~~~~~~
1. Go to https://www.kaggle.com/settings -> "Create New Token"
2. Save the downloaded ``kaggle.json`` to:
   - Windows: ``C:\\Users\\<you>\\.kaggle\\kaggle.json``
   - Linux/Mac: ``~/.kaggle/kaggle.json``
3. Run this script:  ``python prepare_data.py``

Usage
-----
::

    python prepare_data.py                          # default: 10k images, 70/15/15 split
    python prepare_data.py --max-per-class 5000     # use 5k real + 5k fake
    python prepare_data.py --max-per-class 0        # use ALL 70k per class
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple


# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

DATASET_SLUG = "xhlulu/140k-real-and-fake-faces"
RAW_DIR_NAME = "140k-real-and-fake-faces"
DEFAULT_OUT = os.path.join("data", "faceforensics")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# --------------------------------------------------------------------------- #
#  Download helpers
# --------------------------------------------------------------------------- #

def download_from_kaggle(dest_dir: str) -> str:
    """Download the dataset ZIP via the Kaggle CLI.

    Returns
    -------
    str
        Path to the extracted dataset root directory.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERROR] `kaggle` package not installed.  Run:  pip install kaggle")
        sys.exit(1)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(
            "[ERROR] Kaggle authentication failed.\n"
            "  1. Go to https://www.kaggle.com/settings -> Create New Token\n"
            "  2. Save kaggle.json to:  "
            f"{Path.home() / '.kaggle' / 'kaggle.json'}\n"
            f"  Error: {e}"
        )
        sys.exit(1)

    print(f"[INFO] Downloading '{DATASET_SLUG}' from Kaggle …")
    os.makedirs(dest_dir, exist_ok=True)
    api.dataset_download_files(DATASET_SLUG, path=dest_dir, unzip=True)

    extracted = os.path.join(dest_dir, RAW_DIR_NAME)
    if not os.path.isdir(extracted):
        # Some Kaggle datasets extract directly into dest_dir
        extracted = dest_dir
    print(f"[INFO] Download complete -> {extracted}")
    return extracted


# --------------------------------------------------------------------------- #
#  File discovery
# --------------------------------------------------------------------------- #

def _find_images(directory: str) -> List[str]:
    """Recursively find all image files under ``directory``."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images: List[str] = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if Path(f).suffix.lower() in exts:
                images.append(os.path.join(root, f))
    return images


def discover_raw_images(raw_root: str) -> Tuple[List[str], List[str]]:
    """Locate real and fake image lists inside the downloaded dataset.

    The Kaggle "140k Real and Fake Faces" dataset has this layout::

        real_vs_fake/
        └── real-vs-fake/
            ├── train/
            │   ├── real/
            │   └── fake/
            └── test/    (optional, sometimes present)
                ├── real/
                └── fake/

    We merge train+test from the raw download and re-split ourselves.
    """
    # Walk common directory patterns to find real/ and fake/ folders
    real_imgs: List[str] = []
    fake_imgs: List[str] = []

    for root, dirs, _ in os.walk(raw_root):
        basename = os.path.basename(root).lower()
        if basename == "real":
            real_imgs.extend(_find_images(root))
        elif basename == "fake":
            fake_imgs.extend(_find_images(root))

    if not real_imgs or not fake_imgs:
        # Fallback: just look for any two large image directories
        print(f"[WARN] Could not auto-detect real/fake folders under {raw_root}")
        print("       Listing top-level directories for manual inspection:")
        for item in sorted(os.listdir(raw_root)):
            full = os.path.join(raw_root, item)
            if os.path.isdir(full):
                count = len(_find_images(full))
                print(f"         {item}/  ({count} images)")
        sys.exit(1)

    return real_imgs, fake_imgs


# --------------------------------------------------------------------------- #
#  Split and copy
# --------------------------------------------------------------------------- #

def split_and_copy(
    images: List[str],
    label: str,
    out_root: str,
    seed: int = 42,
) -> None:
    """Shuffle and distribute images into train/val/test folders.

    Parameters
    ----------
    images : list[str]
        Source image paths.
    label : str
        ``"real"`` or ``"fake"`` – target sub-folder name.
    out_root : str
        Destination root (e.g. ``data/faceforensics``).
    seed : int
        Random seed for reproducible shuffling.
    """
    rng = random.Random(seed)
    imgs = list(images)
    rng.shuffle(imgs)

    n = len(imgs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": imgs[:n_train],
        "val": imgs[n_train : n_train + n_val],
        "test": imgs[n_train + n_val :],
    }

    for split_name, split_imgs in splits.items():
        dest_dir = os.path.join(out_root, split_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        for i, src_path in enumerate(split_imgs):
            ext = Path(src_path).suffix
            dst_path = os.path.join(dest_dir, f"{i:06d}{ext}")
            shutil.copy2(src_path, dst_path)
        print(f"  {split_name}/{label}: {len(split_imgs):,} images")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare real/fake face dataset for training."
    )
    parser.add_argument(
        "--out-dir", type=str, default=DEFAULT_OUT,
        help="Output directory for train/val/test splits.",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=10_000,
        help="Max images per class (real/fake). 0 = use all available.",
    )
    parser.add_argument(
        "--download-dir", type=str, default="data/_raw",
        help="Directory to download and extract the raw dataset into.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Step 1: Download ──────────────────────────────────────────────
    raw_root = download_from_kaggle(args.download_dir)

    # ── Step 2: Discover images ───────────────────────────────────────
    real_imgs, fake_imgs = discover_raw_images(raw_root)
    print(f"\n[INFO] Found {len(real_imgs):,} real images")
    print(f"[INFO] Found {len(fake_imgs):,} fake images")

    # ── Step 3: Subsample (optional) ──────────────────────────────────
    if args.max_per_class > 0:
        rng = random.Random(args.seed)
        if len(real_imgs) > args.max_per_class:
            real_imgs = rng.sample(real_imgs, args.max_per_class)
        if len(fake_imgs) > args.max_per_class:
            fake_imgs = rng.sample(fake_imgs, args.max_per_class)
        print(f"[INFO] Subsampled to {len(real_imgs):,} real + {len(fake_imgs):,} fake")

    # ── Step 4: Split and copy ────────────────────────────────────────
    print(f"\n[INFO] Splitting into {args.out_dir} (train/val/test) …\n")
    split_and_copy(real_imgs, "real", args.out_dir, seed=args.seed)
    split_and_copy(fake_imgs, "fake", args.out_dir, seed=args.seed)

    # ── Step 5: Summary ──────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  DATA PREPARATION COMPLETE")
    print(f"{'='*50}")
    for split in ("train", "val", "test"):
        split_dir = os.path.join(args.out_dir, split)
        real_count = len(_find_images(os.path.join(split_dir, "real")))
        fake_count = len(_find_images(os.path.join(split_dir, "fake")))
        print(f"  {split:>5s}:  {real_count:>6,} real  |  {fake_count:>6,} fake  |  {real_count+fake_count:>6,} total")
    print(f"{'='*50}")
    print(f"\nReady! Run training with:")
    print(f"  python train.py --data-root {args.out_dir}")


if __name__ == "__main__":
    main()
