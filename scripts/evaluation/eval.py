"""
eval.py
=======
Compression-Robustness & Forensic Evaluation Engine.

Evaluates a saved model checkpoint against standard JPEG compression tiers:
* **q100** (c0)  – Uncompressed / Clean baseline (Quality = 100).
* **q50**        – Moderate web compression (Quality = 50).
* **q40** (c40)  – Medium compression (Quality = 40).
* **q23** (c23)  – Heavy compression (Quality = 23).

Key Features
------------
1. **Terminology Accuracy**: Resolves inverted notation (Q=23 is heavy, Q=100 is clean).
2. **Prediction Serialization**: Saves `y_true.npy` and `y_prob.npy` per condition.
3. **Calibration Assessment**: Computes Expected Calibration Error (ECE) and Brier score.
4. **Export Artifacts**: Writes JSON and CSV tables to `eval_results/` for easy graphing.

Usage
-----
::
    python eval.py --checkpoint ./checkpoints/best_model.pth
    python eval.py --checkpoint ./checkpoints/best_model.pth --tiers q100,q50,q23 --save-artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, cfg
from dataset import DeepfakeDataset, build_eval_transforms
from models.dual_branch_net import DualBranchNet
from utils import compute_metrics, load_checkpoint, seed_everything


# --------------------------------------------------------------------------- #
#  Evaluation routine for a single compression tier
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_tier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tier_label: str,
    cfg: Config,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Run inference on a DataLoader and compute classification & calibration metrics.

    Parameters
    ----------
    model : nn.Module
        The trained detector in eval mode.
    loader : DataLoader
        Test DataLoader applying deterministic JPEG degradation.
    device : torch.device
        Inference device.
    tier_label : str
        Human-readable tier name for logging.
    cfg : Config
        Configuration object.

    Returns
    -------
    tuple[dict, ndarray, ndarray]
        Metrics dictionary, ground truth array, and probability array.
    """
    model.eval()
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    pbar = tqdm(loader, desc=f"Eval [{tier_label}]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(cfg.use_amp and device.type == "cuda")):
            logits: torch.Tensor = model(images)  # (B, 1)

        probs = torch.sigmoid(logits).float().cpu().numpy().ravel()
        all_probs.append(probs)
        all_labels.append(labels.numpy().ravel())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_prob, include_calibration=True)
    return metrics, y_true, y_prob


# --------------------------------------------------------------------------- #
#  Main evaluation entry point
# --------------------------------------------------------------------------- #

def main(
    cfg: Config,
    checkpoint_path: str,
    tier_names: List[str],
    output_dir: str = "./eval_results",
    save_artifacts: bool = True,
    fusion_type: str = "parallel_se",
) -> Dict[str, Dict[str, float]]:
    """Benchmark model across specified compression tiers."""
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    # ── Load model ────────────────────────────────────────────────────
    model = DualBranchNet(fusion_type=fusion_type).to(device)
    epoch, best_auc = load_checkpoint(checkpoint_path, model, device=device)
    print(f"[INFO] Checkpoint from epoch {epoch + 1}, validation AUC={best_auc:.4f}")
    model.eval()

    if save_artifacts:
        os.makedirs(output_dir, exist_ok=True)

    # ── Map tier names to JPEG qualities ──────────────────────────────
    tier_quality_map: Dict[str, int] = {}
    for name in tier_names:
        name_clean = name.strip()
        if name_clean in cfg.compression_quality_map:
            tier_quality_map[name_clean] = cfg.compression_quality_map[name_clean][0]
        elif name_clean.isdigit():
            tier_quality_map[f"q{name_clean}"] = int(name_clean)
        else:
            raise ValueError(f"Unrecognized compression tier name: '{name_clean}'")

    results: Dict[str, Dict[str, float]] = {}

    for tier_name, jpeg_quality in tier_quality_map.items():
        tier_label = f"{tier_name} (JPEG Quality={jpeg_quality})"
        print(f"\n{'='*65}")
        print(f"  Compression Tier: {tier_label}")
        print(f"{'='*65}")

        # Build test DataLoader with deterministic JPEG compression
        transform = build_eval_transforms(cfg, jpeg_quality=jpeg_quality)
        test_ds = DeepfakeDataset(
            root=cfg.data_root,
            split="test",
            transform=transform,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

        metrics, y_true, y_prob = evaluate_tier(model, test_loader, device, tier_label, cfg)
        results[tier_name] = metrics

        # Save artifacts
        if save_artifacts:
            np.save(os.path.join(output_dir, f"{tier_name}_y_true.npy"), y_true)
            np.save(os.path.join(output_dir, f"{tier_name}_y_prob.npy"), y_prob)
            with open(os.path.join(output_dir, f"{tier_name}_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

        # Print metrics
        for metric_name, value in metrics.items():
            print(f"  {metric_name:<14s}: {value:.4f}")

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  COMPRESSION ROBUSTNESS & CALIBRATION BENCHMARK SUMMARY")
    print(f"{'='*75}")
    header = f"{'Tier':<8s} | {'Quality':>7s} | {'Acc':>7s} | {'AUC':>7s} | {'F1':>7s} | {'ECE':>7s} | {'Brier':>7s}"
    print(header)
    print("-" * len(header))
    for tier_name, m in results.items():
        q_val = tier_quality_map[tier_name]
        print(
            f"{tier_name:<8s} | "
            f"{q_val:>7d} | "
            f"{m['accuracy']:>7.4f} | "
            f"{m['auc_roc']:>7.4f} | "
            f"{m['f1']:>7.4f} | "
            f"{m.get('ece', 0.0):>7.4f} | "
            f"{m.get('brier_score', 0.0):>7.4f}"
        )
    print(f"{'='*75}")

    if save_artifacts:
        summary_path = os.path.join(output_dir, "summary_benchmark.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Serialized all evaluation artifacts to: {os.path.abspath(output_dir)}")

    return results


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args() -> Tuple[Config, str, List[str], str, bool, str]:
    parser = argparse.ArgumentParser(
        description="Evaluate Deepfake Detector across standardized JPEG compression tiers",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the model checkpoint (.pth) to evaluate.",
    )
    parser.add_argument("--data-root", type=str, default=cfg.data_root)
    parser.add_argument("--batch-size", type=int, default=cfg.eval_batch_size)
    parser.add_argument(
        "--tiers", type=str, default="q100,q50,q23",
        help="Comma-separated compression tiers to benchmark (e.g. 'q100,q50,q23' or 'c0,c40,c23').",
    )
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--no-save-artifacts", action="store_true")
    parser.add_argument("--fusion-type", type=str, default="parallel_se")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()

    config = Config(
        data_root=args.data_root,
        eval_batch_size=args.batch_size,
        use_amp=not args.no_amp,
        seed=args.seed,
    )
    tier_list = [t.strip() for t in args.tiers.split(",") if t.strip()]
    return config, args.checkpoint, tier_list, args.output_dir, not args.no_save_artifacts, args.fusion_type


if __name__ == "__main__":
    _cfg, _ckpt, _tiers, _out, _save, _fusion = parse_args()
    main(_cfg, _ckpt, _tiers, output_dir=_out, save_artifacts=_save, fusion_type=_fusion)
