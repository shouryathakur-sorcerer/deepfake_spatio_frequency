"""
compute_statistical_significance.py
====================================
Statistical Significance & Reliability Evaluation Engine for Deepfake Detection.

Features:
1. Paired McNemar's Test with Edwards continuity correction.
2. Stratified Bootstrap Resampling (B=200 iterations) for 95% Confidence Intervals:
   - Accuracy, Balanced Accuracy, AUC-ROC, F1-Score, ECE.
3. Paired Bootstrap Test for Area Under ROC Curve (AUC) differences:
   - Tests H0: AUC(A) == AUC(B).
4. Outputs machine-readable JSON and formatted Markdown tables for publication.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.stats as stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def calculate_fast_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = 0.5
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ece = compute_ece(y_true, y_prob)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "auc_roc": float(roc),
        "f1": float(f1),
        "ece": float(ece),
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 200,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute empirical bootstrap percentile confidence intervals."""
    rng = np.random.default_rng(seed)
    indices_pos = np.where(y_true == 1)[0]
    indices_neg = np.where(y_true == 0)[0]
    n_pos = len(indices_pos)
    n_neg = len(indices_neg)

    boot_metrics: Dict[str, List[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "auc_roc": [],
        "f1": [],
        "ece": [],
    }

    for _ in range(n_bootstraps):
        sample_pos = rng.choice(indices_pos, size=n_pos, replace=True)
        sample_neg = rng.choice(indices_neg, size=n_neg, replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])

        yt_sample = y_true[sample_idx]
        yp_sample = y_prob[sample_idx]

        m = calculate_fast_metrics(yt_sample, yp_sample)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])

    alpha = (1.0 - confidence_level) / 2.0
    results = {}
    for k, values in boot_metrics.items():
        arr = np.array(values)
        results[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, 100 * alpha)),
            "ci_upper": float(np.percentile(arr, 100 * (1.0 - alpha))),
        }
    return results


def mcnemar_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Paired McNemar test with Edwards continuity correction."""
    pred_a = (y_prob_a >= threshold).astype(int)
    pred_b = (y_prob_b >= threshold).astype(int)

    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    both_correct = int(np.sum(correct_a & correct_b))
    both_incorrect = int(np.sum(~correct_a & ~correct_b))

    total_discordant = b + c
    if total_discordant == 0:
        chi2_stat = 0.0
        p_val = 1.0
    else:
        chi2_stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        p_val = float(1.0 - stats.chi2.cdf(chi2_stat, df=1))

    return {
        "model_a_only_correct": b,
        "model_b_only_correct": c,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "discordant_pairs": total_discordant,
        "chi2_statistic": chi2_stat,
        "p_value": p_val,
        "significant_p05": p_val < 0.05,
        "significant_p01": p_val < 0.01,
        "significant_p001": p_val < 0.001,
    }


def paired_auc_bootstrap_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    n_bootstraps: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap test for difference in AUC-ROC: delta = AUC(A) - AUC(B)."""
    rng = np.random.default_rng(seed)
    indices_pos = np.where(y_true == 1)[0]
    indices_neg = np.where(y_true == 0)[0]
    n_pos = len(indices_pos)
    n_neg = len(indices_neg)

    delta_aucs = []
    base_auc_a = roc_auc_score(y_true, y_prob_a)
    base_auc_b = roc_auc_score(y_true, y_prob_b)
    observed_delta = float(base_auc_a - base_auc_b)

    for _ in range(n_bootstraps):
        sample_pos = rng.choice(indices_pos, size=n_pos, replace=True)
        sample_neg = rng.choice(indices_neg, size=n_neg, replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])

        yt_s = y_true[sample_idx]
        auc_a = roc_auc_score(yt_s, y_prob_a[sample_idx])
        auc_b = roc_auc_score(yt_s, y_prob_b[sample_idx])
        delta_aucs.append(auc_a - auc_b)

    arr = np.array(delta_aucs)
    ci_lower = float(np.percentile(arr, 2.5))
    ci_upper = float(np.percentile(arr, 97.5))

    shifted_arr = arr - observed_delta
    p_val = float(np.mean(np.abs(shifted_arr) >= np.abs(observed_delta)))

    return {
        "observed_delta_auc": observed_delta,
        "delta_auc_mean": float(np.mean(arr)),
        "delta_auc_std": float(np.std(arr)),
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "p_value": p_val,
        "significant_p05": p_val < 0.05,
    }


def analyze_experiments(
    eval_dir: str = "./eval_results/controlled",
    y_true_path: str = "./eval_results/q100_y_true.npy",
    primary_model: str = "cross_gated",
    baseline_models: List[str] = None,
    tiers: List[str] = None,
    n_bootstraps: int = 200,
) -> Dict:
    if baseline_models is None:
        baseline_models = ["parallel_se", "B3_naive_concat", "B1_spatial_only", "B4_fft_baseline", "B2_frequency_only"]
    if tiers is None:
        tiers = ["q100", "q50", "q23"]

    y_true = np.load(y_true_path).astype(int)
    print(f"[INFO] Loaded ground-truth labels for {len(y_true):,} locked test images.", flush=True)
    print(f"       Class distribution: {np.sum(y_true == 0):,} Real (0), {np.sum(y_true == 1):,} Fake (1)", flush=True)

    output_report = {
        "dataset_size": len(y_true),
        "tiers": tiers,
        "primary_model": primary_model,
        "bootstrap_ci": {},
        "mcnemar_tests": {},
        "auc_difference_tests": {},
    }

    for tier in tiers:
        print(f"\n{'='*75}", flush=True)
        print(f"  TIER: {tier.upper()} STATISTICAL ANALYSIS", flush=True)
        print(f"{'='*75}", flush=True)

        output_report["bootstrap_ci"][tier] = {}
        output_report["mcnemar_tests"][tier] = {}
        output_report["auc_difference_tests"][tier] = {}

        primary_path = os.path.join(eval_dir, f"{primary_model}_{tier}_y_prob.npy")
        if not os.path.exists(primary_path):
            print(f"[WARNING] Primary prediction file not found: {primary_path}. Skipping tier.", flush=True)
            continue
        y_prob_primary = np.load(primary_path)

        print(f"Computing 95% Bootstrap CIs for {primary_model} ({n_bootstraps} iterations)...", flush=True)
        ci_primary = bootstrap_confidence_intervals(y_true, y_prob_primary, n_bootstraps=n_bootstraps)
        output_report["bootstrap_ci"][tier][primary_model] = ci_primary
        acc_ci = ci_primary["accuracy"]
        auc_ci = ci_primary["auc_roc"]
        f1_ci = ci_primary["f1"]
        print(f"  {primary_model}:", flush=True)
        print(f"    Acc:    {acc_ci['mean']*100:.2f}% [{acc_ci['ci_lower']*100:.2f}%, {acc_ci['ci_upper']*100:.2f}%]", flush=True)
        print(f"    AUC:    {auc_ci['mean']:.4f} [{auc_ci['ci_lower']:.4f}, {auc_ci['ci_upper']:.4f}]", flush=True)
        print(f"    F1:     {f1_ci['mean']:.4f} [{f1_ci['ci_lower']:.4f}, {f1_ci['ci_upper']:.4f}]", flush=True)

        for b_name in baseline_models:
            b_path = os.path.join(eval_dir, f"{b_name}_{tier}_y_prob.npy")
            if not os.path.exists(b_path):
                continue
            y_prob_b = np.load(b_path)

            ci_b = bootstrap_confidence_intervals(y_true, y_prob_b, n_bootstraps=n_bootstraps)
            output_report["bootstrap_ci"][tier][b_name] = ci_b

            mcn = mcnemar_test(y_true, y_prob_primary, y_prob_b)
            output_report["mcnemar_tests"][tier][f"{primary_model}_vs_{b_name}"] = mcn

            auc_test = paired_auc_bootstrap_test(y_true, y_prob_primary, y_prob_b, n_bootstraps=n_bootstraps)
            output_report["auc_difference_tests"][tier][f"{primary_model}_vs_{b_name}"] = auc_test

            print(f"\n  Comparison: {primary_model} vs {b_name}:", flush=True)
            print(f"    Acc: {ci_primary['accuracy']['mean']*100:.2f}% vs {ci_b['accuracy']['mean']*100:.2f}%", flush=True)
            print(f"    McNemar Chi2: {mcn['chi2_statistic']:.3f} | p-value: {mcn['p_value']:.4e} (Sig p<0.05: {mcn['significant_p05']})", flush=True)
            print(f"    Discordant pairs: A_correct={mcn['model_a_only_correct']} | B_correct={mcn['model_b_only_correct']}", flush=True)
            print(f"    AUC: {ci_primary['auc_roc']['mean']:.4f} vs {ci_b['auc_roc']['mean']:.4f} (Delta: {auc_test['observed_delta_auc']:+.4f})", flush=True)
            print(f"    AUC 95% CI: [{auc_test['ci_lower_95']:+.4f}, {auc_test['ci_upper_95']:+.4f}] | p-value: {auc_test['p_value']:.4e}", flush=True)

    out_json = os.path.join(eval_dir, "statistical_reliability_report.json")
    with open(out_json, "w") as f:
        json.dump(output_report, f, indent=2)
    print(f"\n[SUCCESS] Statistical report saved to: {out_json}", flush=True)
    return output_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Significance Engine")
    parser.add_argument("--eval-dir", type=str, default="./eval_results/controlled")
    parser.add_argument("--bootstraps", type=int, default=200)
    args = parser.parse_args()

    analyze_experiments(eval_dir=args.eval_dir, n_bootstraps=args.bootstraps)
