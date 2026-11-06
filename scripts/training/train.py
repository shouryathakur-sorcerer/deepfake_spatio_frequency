"""
train.py
========
Complete training engine for the Dual-Branch Spatio-Frequency Deepfake
Detector with:

* ``BCEWithLogitsLoss`` criterion.
* ``AdamW`` optimiser with Cosine Annealing LR schedule (optional warm-up).
* PyTorch Automatic Mixed Precision (``torch.amp``).
* Per-epoch validation with AUC-ROC & accuracy logging.
* Best-model checkpointing (by validation AUC).
* ``tqdm`` progress bars.

Usage
-----
::

    python train.py                        # uses defaults from config.py
    python train.py --epochs 50 --lr 3e-4  # override via CLI
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────
from config import Config, cfg
from dataset import build_dataloaders
from models import DualBranchNet, get_model
from utils import (
    compute_metrics,
    count_parameters,
    save_checkpoint,
    seed_everything,
)


# --------------------------------------------------------------------------- #
#  Training & Validation step helpers
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    cfg: Config,
) -> float:
    """Run one training epoch.

    Returns
    -------
    float
        Mean training loss for the epoch.
    """
    model.train()
    running_loss: float = 0.0
    num_batches: int = 0

    pbar = tqdm(loader, desc=f"Train  Epoch {epoch+1:>3d}", leave=False)
    for images, labels in pbar:
        # images: (B, 3, 256, 256), labels: (B,)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad(set_to_none=True)

        # ── Forward with AMP ─────────────────────────────────────
        with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
            logits: torch.Tensor = model(images)  # (B, 1)
            loss: torch.Tensor = criterion(logits, labels)

        # ── Backward with gradient scaling ───────────────────────
        scaler.scale(loss).backward()
        if cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    cfg: Config,
) -> Tuple[float, Dict[str, float]]:
    """Run validation and compute metrics.

    Returns
    -------
    tuple[float, dict]
        ``(val_loss, metrics_dict)`` where ``metrics_dict`` contains
        accuracy, AUC-ROC, precision, recall, and F1.
    """
    model.eval()
    running_loss: float = 0.0
    num_batches: int = 0

    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    pbar = tqdm(loader, desc=f"Val    Epoch {epoch+1:>3d}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
            logits: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(logits, labels)

        running_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy().ravel())

    val_loss = running_loss / max(num_batches, 1)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_prob)
    return val_loss, metrics


# --------------------------------------------------------------------------- #
#  Main training loop
# --------------------------------------------------------------------------- #

def main(cfg: Config, model_type: str = "dual_branch") -> None:
    """Orchestrate training end-to-end."""
    seed_everything(cfg.seed)
    cfg.ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model Architecture: {model_type.upper()}")

    # ── Data ──────────────────────────────────────────────────────────
    loaders: Dict[str, DataLoader] = build_dataloaders(cfg)
    assert "train" in loaders, "Training split not found under data_root."
    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    print(f"[INFO] Train batches: {len(train_loader)}")
    if val_loader:
        print(f"[INFO] Val   batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────
    model = get_model(
        model_name=model_type,
        spatial_backbone=cfg.spatial_backbone,
        spatial_pretrained=cfg.spatial_pretrained,
        wavelet=cfg.wavelet_name,
        wavelet_level=cfg.wavelet_decomp_level,
        embed_dim=cfg.embed_dim,
        head_dropout=cfg.head_dropout,
    ).to(device)
    print(f"[INFO] Trainable parameters: {count_parameters(model):,}")

    # ── Loss, Optimiser, Scheduler ────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs - cfg.lr_warmup_epochs,
        eta_min=cfg.lr_min,
    )
    # Simple linear warm-up scheduler for the first N epochs.
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=cfg.lr_warmup_epochs,
    )
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, scheduler],
        milestones=[cfg.lr_warmup_epochs],
    )

    # ── AMP scaler ────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler(enabled=cfg.use_amp)

    # ── Training loop ─────────────────────────────────────────────────
    best_auc: float = 0.0
    for epoch in range(cfg.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, cfg,
        )

        # Validation (if split exists)
        val_loss, val_metrics = 0.0, {}
        if val_loader is not None:
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, epoch, cfg,
            )

        combined_scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # ── Logging ──────────────────────────────────────────────
        auc = val_metrics.get("auc_roc", 0.0)
        acc = val_metrics.get("accuracy", 0.0)
        print(
            f"Epoch {epoch+1:>3d}/{cfg.epochs} │ "
            f"train_loss={train_loss:.4f} │ "
            f"val_loss={val_loss:.4f} │ "
            f"val_acc={acc:.4f} │ "
            f"val_auc={auc:.4f} │ "
            f"lr={lr_now:.2e} │ "
            f"{elapsed:.1f}s"
        )

        # ── Checkpoint best model ────────────────────────────────
        if auc > best_auc:
            best_auc = auc
            ckpt_name = "best_model.pth" if model_type in ("dual_branch", "proposed") else f"{model_type}_best.pth"
            ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_name)
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                combined_scheduler,
                epoch=epoch,
                best_auc=best_auc,
            )
            print(f"  ↳ Saved best checkpoint (AUC={best_auc:.4f}) → {ckpt_path}")

    # Save final checkpoint regardless
    final_name = "final_model.pth" if model_type in ("dual_branch", "proposed") else f"{model_type}_final.pth"
    final_path = os.path.join(cfg.checkpoint_dir, final_name)
    save_checkpoint(
        final_path,
        model,
        optimizer,
        combined_scheduler,
        epoch=cfg.epochs - 1,
        best_auc=best_auc,
    )
    print(f"\n[INFO] Training complete. Best val AUC: {best_auc:.4f}")
    print(f"[INFO] Final checkpoint → {final_path}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args() -> Tuple[Config, str]:
    """Override ``Config`` defaults from the command line."""
    parser = argparse.ArgumentParser(
        description="Train Dual-Branch Spatio-Frequency Deepfake Detector or Baselines",
    )
    parser.add_argument(
        "--model-type", type=str, default="full_team14",
        choices=[
            "full_team14",
            "spatial_wpd_concat",
            "spatial_wpd_hife",
            "spatial_wpd_dca",
            "wpd_only",
            "dual_branch",
            "cross_gated",
            "spatial_only",
            "frequency_only",
            "naive_concat",
            "fft_baseline",
        ],
        help="Model architecture or baseline to train.",
    )
    parser.add_argument("--data-root", type=str, default=cfg.data_root)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.learning_rate)
    parser.add_argument("--backbone", type=str, default=cfg.spatial_backbone)
    parser.add_argument("--embed-dim", type=int, default=cfg.embed_dim)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--checkpoint-dir", type=str, default=cfg.checkpoint_dir)
    args = parser.parse_args()

    config = Config(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        spatial_backbone=args.backbone,
        embed_dim=args.embed_dim,
        use_amp=not args.no_amp,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )
    return config, args.model_type


if __name__ == "__main__":
    _cfg, _mtype = parse_args()
    main(_cfg, model_type=_mtype)
