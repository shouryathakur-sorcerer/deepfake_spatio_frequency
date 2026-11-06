"""
utils.py
========
Shared utilities for training and evaluation:

* **Metrics**: Accuracy, AUC-ROC, Precision, Recall, F1 using ``scikit-learn``.
* **Checkpointing**: Save / load model + optimiser + scheduler state.
* **Reproducibility**: Deterministic seeding for ``random``, ``numpy``,
  ``torch``, and CUDA.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# --------------------------------------------------------------------------- #
#  Reproducibility
# --------------------------------------------------------------------------- #

def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN deterministic mode (may hurt perf slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
#  Metrics & Calibration
# --------------------------------------------------------------------------- #

def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Parameters
    ----------
    y_true : ndarray, shape ``(N,)``
        Ground-truth binary labels ``{0, 1}``.
    y_prob : ndarray, shape ``(N,)``
        Predicted probabilities in ``[0, 1]``.
    n_bins : int
        Number of equal-width probability bins in [0, 1].

    Returns
    -------
    dict[str, float]
        ``{"ece", "mce", "brier_score"}``.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    N = len(y_true)
    if N == 0:
        return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0}

    brier_score = float(np.mean((y_prob - y_true) ** 2))

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            bin_error = np.abs(bin_acc - bin_conf)
            ece += (bin_size / N) * bin_error
            mce = max(mce, float(bin_error))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier_score,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    include_calibration: bool = True,
) -> Dict[str, float]:
    """Compute a full suite of binary-classification and calibration metrics.

    Parameters
    ----------
    y_true : ndarray, shape ``(N,)``
        Ground-truth binary labels ``{0, 1}``.
    y_prob : ndarray, shape ``(N,)``
        Predicted probabilities in ``[0, 1]`` (after sigmoid).
    threshold : float
        Decision threshold for hard predictions.
    include_calibration : bool
        Whether to calculate ECE, MCE, and Brier score.

    Returns
    -------
    dict[str, float]
        ``{"accuracy", "auc_roc", "precision", "recall", "f1", "ece", "brier_score"}``.
    """
    y_pred = (y_prob >= threshold).astype(np.int32)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # AUC-ROC and PR-AUC require at least two distinct classes
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = 0.0

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["pr_auc"] = 0.0

    # Confusion matrix & error rates
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["tp"] = int(tp)
        metrics["tn"] = int(tn)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    except Exception:
        pass

    if include_calibration:
        calib = compute_calibration_metrics(y_true, y_prob)
        metrics.update(calib)

    return metrics


# --------------------------------------------------------------------------- #
#  Checkpointing
# --------------------------------------------------------------------------- #

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    best_auc: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a full training checkpoint to disk.

    Parameters
    ----------
    path : str
        Destination ``.pth`` file path.
    model : nn.Module
        Model whose ``state_dict`` will be saved.
    optimizer : Optimizer
        Optimiser state.
    scheduler : lr_scheduler | None
        Optional learning-rate scheduler state.
    epoch : int
        Current epoch index (0-based).
    best_auc : float
        Best validation AUC-ROC achieved so far (for early-stopping logic).
    extra : dict | None
        Any additional metadata to include.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state: Dict[str, Any] = {
        "epoch": epoch,
        "best_auc": best_auc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, float]:
    """Restore model (and optionally optimiser / scheduler) from a checkpoint.

    Returns
    -------
    tuple[int, float]
        ``(epoch, best_auc)`` from the saved state.
    """
    ckpt: Dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_auc", 0.0)


# --------------------------------------------------------------------------- #
#  Misc
# --------------------------------------------------------------------------- #

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
