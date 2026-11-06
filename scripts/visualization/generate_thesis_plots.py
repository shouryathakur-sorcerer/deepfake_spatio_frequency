"""
generate_thesis_plots.py
========================
Automated publication-quality figure generator for the thesis project:
"Deepfake Image Detection Using Multi-Level Frequency Analysis"

Generated Figures (saved to ./figures in both high-res PNG and vector PDF):
--------------------------------------------------------------------------
1. roc_curves.png / .pdf
   Multi-tier Receiver Operating Characteristic (ROC) curves comparing c0, c23, and c40 tiers.
2. confusion_matrices.png / .pdf
   Side-by-side normalized confusion matrices for c0, c23, and c40 showing raw counts and percentages.
3. score_distributions.png / .pdf
   Bimodal probability score histograms for Real vs Fake images illustrating decision separation.
4. metrics_comparison_bar.png / .pdf
   Grouped bar chart comparing Accuracy, Precision, Recall, and F1 across all three compression tiers.

Usage
-----
::

    # Generate full figures using all 20,000 test images:
    py -3.12 generate_thesis_plots.py

    # Quick run on a subset (e.g., 2,000 images per tier) for fast preview:
    py -3.12 generate_thesis_plots.py --max-samples 2000

    # Specify custom checkpoint or output directory:
    py -3.12 generate_thesis_plots.py --checkpoint "./checkpoints/best_model.pth" --output-dir "./figures"
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import Dict, List, Tuple

# Set matplotlib cache directory to avoid temp/permission warnings
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib_deepfake_cache")
)

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for clean script rendering
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc as calc_auc,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import Config, cfg
from dataset import DeepfakeDataset, build_eval_transforms
from models.dual_branch_net import DualBranchNet
from utils import compute_metrics, load_checkpoint, seed_everything


# --------------------------------------------------------------------------- #
#  Styling & Publication Aesthetics
# --------------------------------------------------------------------------- #

def apply_plot_style() -> None:
    """Set academic publication aesthetics for Matplotlib figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })


# --------------------------------------------------------------------------- #
#  Inference across Compression Tiers
# --------------------------------------------------------------------------- #

@torch.no_grad()
def collect_tier_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tier_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect ground-truth labels and predicted probabilities for a loader."""
    model.eval()
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    pbar = tqdm(loader, desc=f"Evaluating [{tier_name}]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
            logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        all_probs.append(probs)
        all_labels.append(labels.numpy().ravel())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    return y_true, y_prob


# --------------------------------------------------------------------------- #
#  Plotting Functions
# --------------------------------------------------------------------------- #

def plot_roc_curves(
    tier_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    dpi: int = 300,
) -> None:
    """Plot multi-tier ROC curves comparing c0, c23, and c40."""
    plt.figure(figsize=(7.5, 6.0))
    colors = {"c0": "#1f77b4", "c23": "#ff7f0e", "c40": "#2ca02c"}
    labels_map = {
        "c0": "c0 (Raw / Uncompressed, Q=100)",
        "c23": "c23 (Medium JPEG, Q=23)",
        "c40": "c40 (Heavy JPEG, Q=40)",
    }

    for tier, (y_true, y_prob) in tier_results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = calc_auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=colors.get(tier, "blue"),
            lw=2.2,
            label=f"{labels_map.get(tier, tier)} (AUC = {roc_auc:.4f})",
        )

    # Diagonal chance line
    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.2, linestyle="--", label="Random Chance (AUC = 0.5000)")

    plt.xlim([-0.01, 1.01])
    plt.ylim([0.80, 1.01])  # Focused view on the upper region where high-performing models reside
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("ROC Curves Across JPEG Compression Tiers")
    plt.legend(loc="lower right", framealpha=0.92)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        filepath = os.path.join(output_dir, f"roc_curves.{ext}")
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {os.path.join(output_dir, 'roc_curves.png')}")


def plot_confusion_matrices(
    tier_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    dpi: int = 300,
) -> None:
    """Plot side-by-side normalized confusion matrices for c0, c23, c40."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    tiers = ["c0", "c23", "c40"]
    titles = {
        "c0": "c0 (Raw, Q=100)",
        "c23": "c23 (Medium, Q=23)",
        "c40": "c40 (Heavy, Q=40)",
    }

    for i, tier in enumerate(tiers):
        ax = axes[i]
        y_true, y_prob = tier_results[tier]
        y_pred = (y_prob >= 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.85, vmax=1.0)

        ax.set_title(titles.get(tier, tier), pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Real", "Fake"])
        ax.set_yticklabels(["Real", "Fake"])
        ax.set_xlabel("Predicted Label")
        if i == 0:
            ax.set_ylabel("True Label")

        # Overlay text counts and percentages inside each cell
        thresh = cm_norm.max() / 2.0
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                pct = cm_norm[r, c] * 100
                text_color = "white" if cm_norm[r, c] > 0.94 else "black"
                ax.text(
                    c, r, f"{val:,}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color=text_color,
                    fontweight="bold" if r == c else "normal",
                    fontsize=10,
                )

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.20, 0.015, 0.60])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Normalized Accuracy Rate", rotation=270, labelpad=15)

    for ext in ("png", "pdf"):
        filepath = os.path.join(output_dir, f"confusion_matrices.{ext}")
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {os.path.join(output_dir, 'confusion_matrices.png')}")


def plot_score_distributions(
    tier_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    dpi: int = 300,
) -> None:
    """Plot bimodal probability score histograms for Real vs Fake classes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    tiers = ["c0", "c23", "c40"]
    titles = {
        "c0": "c0 (Raw, Q=100)",
        "c23": "c23 (Medium, Q=23)",
        "c40": "c40 (Heavy, Q=40)",
    }

    bins = np.linspace(0.0, 1.0, 40)

    for i, tier in enumerate(tiers):
        ax = axes[i]
        y_true, y_prob = tier_results[tier]

        real_probs = y_prob[y_true == 0]
        fake_probs = y_prob[y_true == 1]

        ax.hist(real_probs, bins=bins, alpha=0.65, color="#1f77b4", label="Real Faces", density=True)
        ax.hist(fake_probs, bins=bins, alpha=0.65, color="#d62728", label="Fake Faces", density=True)
        ax.axvline(0.5, color="black", linestyle="--", lw=1.2, label="Decision Threshold (0.5)")

        ax.set_title(titles.get(tier, tier))
        ax.set_xlabel("Predicted Deepfake Probability P(Fake)")
        if i == 0:
            ax.set_ylabel("Probability Density")
        ax.legend(loc="upper center", framealpha=0.9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        filepath = os.path.join(output_dir, f"score_distributions.{ext}")
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {os.path.join(output_dir, 'score_distributions.png')}")


def plot_metrics_comparison_bar(
    tier_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    dpi: int = 300,
) -> None:
    """Plot grouped bar chart of Accuracy, Precision, Recall, and F1 across tiers."""
    metrics_by_tier: Dict[str, Dict[str, float]] = {}
    for tier, (y_true, y_prob) in tier_results.items():
        metrics_by_tier[tier] = compute_metrics(y_true, y_prob)

    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
    tiers = ["c0", "c23", "c40"]
    tier_labels = ["c0 (Raw)", "c23 (Medium)", "c40 (Heavy)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    x = np.arange(len(categories))
    width = 0.24

    plt.figure(figsize=(8.5, 5.0))
    for i, tier in enumerate(tiers):
        m = metrics_by_tier[tier]
        vals = [
            m["accuracy"] * 100,
            m["precision"] * 100,
            m["recall"] * 100,
            m["f1"] * 100,
        ]
        offset = (i - 1) * width
        bars = plt.bar(x + offset, vals, width, label=tier_labels[i], color=colors[i], alpha=0.9)

        # Value annotations above each bar
        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    plt.ylabel("Score (%)")
    plt.title("Performance Comparison Across JPEG Compression Tiers")
    plt.xticks(x, categories)
    plt.ylim([85, 103])  # Set axis to emphasize subtle differences above 85%
    plt.legend(loc="lower left", framealpha=0.95)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        filepath = os.path.join(output_dir, f"metrics_comparison_bar.{ext}")
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {os.path.join(output_dir, 'metrics_comparison_bar.png')}")


# --------------------------------------------------------------------------- #
#  Main Orchestrator
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-grade figures (ROC, Confusion Matrix, Distributions) for thesis."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints/best_model.pth",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--data-root", type=str, default=cfg.data_root,
        help="Path to root dataset folder containing test/ split.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./figures",
        help="Directory where output PNG and PDF figures will be saved.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for test evaluation.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Subsample N test images per tier for rapid preview (0 = use all test images).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Figure DPI resolution (default: 300 DPI for publication).",
    )
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()

    seed_everything(args.seed)
    apply_plot_style()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")

    # ── Load Model ────────────────────────────────────────────────────
    model = DualBranchNet(
        spatial_backbone=cfg.spatial_backbone,
        spatial_pretrained=False,
        wavelet=cfg.wavelet_name,
        wavelet_level=cfg.wavelet_decomp_level,
        embed_dim=cfg.embed_dim,
    ).to(device)

    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # ── Evaluate across Tiers ─────────────────────────────────────────
    tiers: Dict[str, int] = {
        "c0": 100,
        "c23": 23,
        "c40": 40,
    }

    tier_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    print(f"\n[INFO] Evaluating test split across compression tiers (c0, c23, c40) ...")
    for tier_name, jpeg_quality in tiers.items():
        transform = build_eval_transforms(cfg, jpeg_quality=jpeg_quality)
        dataset = DeepfakeDataset(root=args.data_root, split="test", transform=transform)

        if args.max_samples > 0 and args.max_samples < len(dataset):
            indices = np.random.choice(len(dataset), size=args.max_samples, replace=False)
            dataset = Subset(dataset, indices)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

        y_true, y_prob = collect_tier_predictions(model, loader, device, tier_name)
        tier_results[tier_name] = (y_true, y_prob)

    # ── Generate Figures ──────────────────────────────────────────────
    print(f"\n[INFO] Rendering and saving publication figures to '{args.output_dir}' ...")
    plot_roc_curves(tier_results, args.output_dir, dpi=args.dpi)
    plot_confusion_matrices(tier_results, args.output_dir, dpi=args.dpi)
    plot_score_distributions(tier_results, args.output_dir, dpi=args.dpi)
    plot_metrics_comparison_bar(tier_results, args.output_dir, dpi=args.dpi)

    print(f"\n{'='*60}")
    print("  THESIS PLOT GENERATION COMPLETE!")
    print(f"  All figures saved in PNG (300 DPI) and vector PDF to:")
    print(f"  {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
