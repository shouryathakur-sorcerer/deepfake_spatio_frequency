"""
run_controlled_experiments.py
==============================
Controlled Baseline & Architectural Ablation Experiment Engine.

Executes a standardized, paired benchmark across:
1. B1: spatial_only (ImageNet Xception)
2. B2: frequency_only (16-subband WPD-CNN)
3. B3: naive_concat (Spatial + Frequency direct vector concatenation)
4. B4: fft_baseline (2D Fast Fourier Transform power spectrum)
5. parallel_se (Proposed DualBranchNet with parallel SE gating)
6. cross_gated (Proposed DualBranchNet with true cross-modal joint gating)
+ Evaluates Pretrained Reference (checkpoints/best_model.pth)

Experimental Controls
---------------------
- Identical locked training subset: stratified 10,000 samples (5,000 real, 5,000 fake).
- Identical locked test set: 20,000 samples (10,000 real, 10,000 fake).
- Identical evaluation conditions: JPEG Q=100, Q=50, Q=23.
- Identical optimizer: AdamW (lr=1e-4, weight_decay=1e-4).
- Identical scheduler: CosineAnnealingLR.
- Fixed seed: 42.
- GPU acceleration via CUDA with automatic mixed precision (AMP).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Config, cfg
from dataset import DeepfakeDataset, build_eval_transforms, build_train_transforms
from models import get_model
from utils import compute_metrics, count_parameters, save_checkpoint, seed_everything


class BalancedSubsetDataset(Dataset):
    """Wraps DeepfakeDataset to extract an exact balanced subset of N samples."""

    def __init__(self, base_ds: DeepfakeDataset, max_per_class: int = 5000, seed: int = 42) -> None:
        random.seed(seed)
        reals = [s for s in base_ds.samples if s[1] == 0]
        fakes = [s for s in base_ds.samples if s[1] == 1]

        random.shuffle(reals)
        random.shuffle(fakes)

        selected_reals = reals[:max_per_class]
        selected_fakes = fakes[:max_per_class]

        self.samples = selected_reals + selected_fakes
        random.shuffle(self.samples)
        self.transform = base_ds.transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import cv2
        path, label = self.samples[idx]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, torch.tensor(label, dtype=torch.float32)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-4,
) -> Tuple[float, float]:
    """Train model and measure average step time and peak VRAM."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    step_times: List[float] = []
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Train Ep {epoch+1}/{epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times.append(time.time() - t0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

    avg_step_ms = float(np.mean(step_times) * 1000)
    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
    return avg_step_ms, peak_vram_mb


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model on a DataLoader and return metrics, labels, and probabilities."""
    model.eval()
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
        probs = torch.sigmoid(logits).float().cpu().numpy().ravel()
        all_probs.append(probs)
        all_labels.append(labels.numpy().ravel())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_prob, include_calibration=True)
    return metrics, y_true, y_prob


def run_controlled_suite(
    cfg: Config,
    train_samples_per_class: int = 5000,
    epochs: int = 3,
    output_dir: str = "./eval_results/controlled",
) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}")
    print(f"  CONTROLLED SOTA EXPERIMENTAL SUITE")
    print(f"  Device: {device} | Train subset: {train_samples_per_class*2:,} images | Epochs: {epochs}")
    print(f"{'='*75}\n")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./checkpoints/controlled", exist_ok=True)

    # 1. Prepare Datasets
    print("[1/4] Preparing balanced training subset and locked test loaders...")
    train_transform = build_train_transforms(cfg)
    base_train_ds = DeepfakeDataset(root=cfg.data_root, split="train", transform=train_transform)
    balanced_train_ds = BalancedSubsetDataset(base_train_ds, max_per_class=train_samples_per_class, seed=cfg.seed)
    train_loader = DataLoader(
        balanced_train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )
    print(f"      Training set: {len(balanced_train_ds):,} samples ({train_samples_per_class} real + {train_samples_per_class} fake)")

    # Test loaders for Q=100, Q=50, Q=23 (locked 20,000 samples)
    test_loaders: Dict[str, DataLoader] = {}
    for q_name, q_val in [("q100", 100), ("q50", 50), ("q23", 23)]:
        trans = build_eval_transforms(cfg, jpeg_quality=q_val)
        ds = DeepfakeDataset(root=cfg.data_root, split="test", transform=trans)
        test_loaders[q_name] = DataLoader(
            ds, batch_size=cfg.eval_batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        )
    print(f"      Locked Test set: {len(test_loaders['q100'].dataset):,} images per condition")

    # 2. Define Model Configurations to Benchmark
    models_to_test = [
        ("B1_spatial_only", "spatial_only"),
        ("B2_frequency_only", "frequency_only"),
        ("B3_naive_concat", "naive_concat"),
        ("B4_fft_baseline", "fft_baseline"),
        ("parallel_se", "parallel_se"),
        ("cross_gated", "cross_gated"),
    ]

    all_results: Dict[str, Dict] = {}

    # 3. Train and Benchmark Each Model
    print("\n[2/4] Training and evaluating baseline models under identical conditions...")
    for model_id, model_type in models_to_test:
        print(f"\n--- Model: {model_id} ({model_type}) ---")
        seed_everything(cfg.seed)
        model = get_model(model_type).to(device)
        param_count = count_parameters(model)
        print(f"    Trainable parameters: {param_count:,}")

        # Train
        t_start = time.time()
        step_time_ms, peak_vram = train_model(model, train_loader, device, epochs=epochs, lr=cfg.learning_rate)
        train_duration_s = time.time() - t_start
        print(f"    Training completed in {train_duration_s:.1f}s | Step: {step_time_ms:.1f}ms | Peak VRAM: {peak_vram:.1f}MB")

        # Save Checkpoint
        ckpt_path = f"./checkpoints/controlled/{model_id}.pth"
        save_checkpoint(ckpt_path, model, torch.optim.AdamW(model.parameters()), epoch=epochs-1, best_auc=0.0)

        # Evaluate across tiers
        model_metrics: Dict[str, Dict] = {}
        for q_name, loader in test_loaders.items():
            metrics, y_true, y_prob = evaluate_model(model, loader, device)
            model_metrics[q_name] = metrics
            np.save(os.path.join(output_dir, f"{model_id}_{q_name}_y_prob.npy"), y_prob)
            print(f"    [{q_name}] Acc: {metrics['accuracy']:.4f} | AUC: {metrics['auc_roc']:.4f} | F1: {metrics['f1']:.4f} | ECE: {metrics['ece']:.4f}")

        all_results[model_id] = {
            "parameters": param_count,
            "train_time_seconds": round(train_duration_s, 2),
            "step_time_ms": round(step_time_ms, 2),
            "peak_vram_mb": round(peak_vram, 2),
            "eval_metrics": model_metrics,
        }

    # 4. Include Pretrained Reference Checkpoint if available
    ref_ckpt = "./checkpoints/best_model.pth"
    if os.path.exists(ref_ckpt):
        print("\n--- Pretrained Full-Scale ParallelSE Reference (8 Epochs / 100k samples) ---")
        from utils import load_checkpoint
        ref_model = get_model("parallel_se").to(device)
        load_checkpoint(ref_ckpt, ref_model, device=device)
        ref_metrics: Dict[str, Dict] = {}
        for q_name, loader in test_loaders.items():
            metrics, _, y_prob = evaluate_model(ref_model, loader, device)
            ref_metrics[q_name] = metrics
            print(f"    [{q_name}] Acc: {metrics['accuracy']:.4f} | AUC: {metrics['auc_roc']:.4f} | F1: {metrics['f1']:.4f} | ECE: {metrics['ece']:.4f}")
        all_results["Ref_ParallelSE_100k"] = {
            "parameters": count_parameters(ref_model),
            "train_time_seconds": "Historical (8 epochs on 100k)",
            "step_time_ms": 339.66,
            "peak_vram_mb": 5752.2,
            "eval_metrics": ref_metrics,
        }

    # 5. Export Summary JSON and Table
    summary_path = os.path.join(output_dir, "summary_controlled_experiments.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Machine-readable results saved to: {summary_path}")

    # Print Clean Formatted Comparison Table
    print(f"\n{'='*105}")
    print("  FINAL CONTROLLED SOTA BASELINE COMPARISON TABLE (Q=100 / Clean)")
    print(f"{'='*105}")
    header = f"{'Model ID':<22} | {'Params':>10} | {'Acc':>7} | {'Bal Acc':>7} | {'AUC':>7} | {'PR-AUC':>7} | {'F1':>7} | {'ECE':>7} | {'Step (ms)':>9}"
    print(header)
    print("-" * len(header))
    for m_id, res in all_results.items():
        q100 = res["eval_metrics"]["q100"]
        st = res.get("step_time_ms", 0.0)
        st_str = f"{st:.1f}" if isinstance(st, (int, float)) else str(st)
        print(
            f"{m_id:<22} | "
            f"{res['parameters']:>10,d} | "
            f"{q100['accuracy']:>7.4f} | "
            f"{q100.get('balanced_accuracy', q100['accuracy']):>7.4f} | "
            f"{q100['auc_roc']:>7.4f} | "
            f"{q100.get('pr_auc', 0.0):>7.4f} | "
            f"{q100['f1']:>7.4f} | "
            f"{q100.get('ece', 0.0):>7.4f} | "
            f"{st_str:>9}"
        )
    print(f"{'='*105}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run controlled baseline suite")
    parser.add_argument("--samples-per-class", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="./eval_results/controlled")
    args = parser.parse_args()

    cfg_inst = Config()
    run_controlled_suite(cfg_inst, train_samples_per_class=args.samples_per_class, epochs=args.epochs, output_dir=args.output_dir)
