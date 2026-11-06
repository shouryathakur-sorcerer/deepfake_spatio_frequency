"""
scratch_inspect_baselines.py
============================
Fast analytical diagnostic script for:
1. FFT baseline (AUC < 0.5) and Frequency-only model diagnostics.
2. Exact statistical tests with Cohen's g, odds ratios, and non-zero p-values.
3. Spatial-only vs Dual-branch comparisons.
"""

import json
import os
import sys
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

y_true = np.load("./eval_results/q100_y_true.npy").astype(int)
eval_dir = "./eval_results/controlled"

print("=" * 80, flush=True)
print("PART 1: FFT BASELINE & FREQUENCY-ONLY MODEL DIAGNOSTICS", flush=True)
print("=" * 80, flush=True)

for m in ["B4_fft_baseline", "B2_frequency_only"]:
    print(f"\n--- Diagnostic: {m} ---", flush=True)
    for tier in ["q100", "q50", "q23"]:
        y_prob = np.load(os.path.join(eval_dir, f"{m}_{tier}_y_prob.npy"))
        prob_real = y_prob[y_true == 0]
        prob_fake = y_prob[y_true == 1]
        auc_val = roc_auc_score(y_true, y_prob)
        print(f"  [{tier}] AUC: {auc_val:.4f} | Mean Prob (Real): {np.mean(prob_real):.4f} (\u00b1{np.std(prob_real):.4f}) | Mean Prob (Fake): {np.mean(prob_fake):.4f} (\u00b1{np.std(prob_fake):.4f})", flush=True)
        print(f"       Prob Min: {np.min(y_prob):.4f}, Max: {np.max(y_prob):.4f}, Median: {np.median(y_prob):.4f}", flush=True)
        
        # Check inverted AUC
        inv_auc = roc_auc_score(y_true, 1.0 - y_prob)
        print(f"       If inverted (1 - p): AUC = {inv_auc:.4f}", flush=True)

print("\n" + "=" * 80, flush=True)
print("PART 2: EXACT STATISTICAL COMPARISONS & EFFECT SIZES", flush=True)
print("=" * 80, flush=True)

def compute_fast_delong_auc_var(y_true, y_prob):
    """Fast analytical variance of AUC based on Mann-Whitney U structure (Hanley & McNeil / DeLong)."""
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    theta = roc_auc_score(y_true, y_prob)
    q1 = theta / (2.0 - theta)
    q2 = 2.0 * theta**2 / (1.0 + theta)
    var = (theta * (1.0 - theta) + (n_pos - 1) * (q1 - theta**2) + (n_neg - 1) * (q2 - theta**2)) / (n_pos * n_neg)
    return theta, max(var, 1e-9)

def compute_paired_stats(y_true, p_a, p_b, name_a, name_b, tier):
    pred_a = (p_a >= 0.5).astype(int)
    pred_b = (p_b >= 0.5).astype(int)
    
    cor_a = (pred_a == y_true)
    cor_b = (pred_b == y_true)
    
    b = int(np.sum(cor_a & ~cor_b)) # A correct, B wrong
    c = int(np.sum(~cor_a & cor_b)) # A wrong, B correct
    both_cor = int(np.sum(cor_a & cor_b))
    both_wrg = int(np.sum(~cor_a & ~cor_b))
    
    total_disc = b + c
    if total_disc == 0:
        chi2 = 0.0
        p_mcnemar = 1.0
        cohens_g = 0.0
        odds_ratio = 1.0
    else:
        chi2 = ((abs(b - c) - 1.0) ** 2) / total_disc
        p_mcnemar = 1.0 - stats.chi2.cdf(chi2, df=1)
        cohens_g = (b / total_disc) - 0.5
        odds_ratio = b / c if c > 0 else float("inf")
        
    auc_a, var_a = compute_fast_delong_auc_var(y_true, p_a)
    auc_b, var_b = compute_fast_delong_auc_var(y_true, p_b)
    delta_auc = auc_a - auc_b
    
    # 50-iteration fast bootstrap for correlation between AUCs
    rng = np.random.default_rng(42)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    boot_deltas = []
    for _ in range(50):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        s = np.concatenate([s_pos, s_neg])
        boot_deltas.append(roc_auc_score(y_true[s], p_a[s]) - roc_auc_score(y_true[s], p_b[s]))
        
    se_delta = float(np.std(boot_deltas))
    ci_low = float(delta_auc - 1.96 * se_delta)
    ci_high = float(delta_auc + 1.96 * se_delta)
    
    z_score = delta_auc / se_delta if se_delta > 0 else 0.0
    p_auc_z = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))
    
    # Format p-values so they are NEVER exactly zero
    p_mcn_str = f"{p_mcnemar:.4e}" if p_mcnemar >= 1e-12 else "< 1.0e-12"
    p_auc_str = f"{p_auc_z:.4e}" if p_auc_z >= 1e-12 else "< 1.0e-12"
    
    print(f"\n[{tier}] {name_a} vs {name_b}:", flush=True)
    print(f"  Acc: {np.mean(cor_a)*100:.2f}% vs {np.mean(cor_b)*100:.2f}% (Delta: {(np.mean(cor_a)-np.mean(cor_b))*100:+.2f}%)", flush=True)
    print(f"  McNemar: b={b}, c={c}, both_correct={both_cor}, both_wrong={both_wrg}", flush=True)
    print(f"  McNemar Chi2 = {chi2:.3f}, p-value = {p_mcn_str}, Odds Ratio = {odds_ratio:.3f}, Cohen's g = {cohens_g:+.4f}", flush=True)
    print(f"  AUC: {auc_a:.4f} vs {auc_b:.4f} (Delta AUC = {delta_auc:+.4f})", flush=True)
    print(f"  AUC 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}], SE = {se_delta:.4f}, Z = {z_score:+.2f}, p-value = {p_auc_str}", flush=True)
    
    return {
        "b": b, "c": c, "both_correct": both_cor, "both_wrong": both_wrg,
        "chi2": float(chi2), "p_mcnemar": p_mcn_str, "odds_ratio": float(odds_ratio), "cohens_g": float(cohens_g),
        "auc_a": float(auc_a), "auc_b": float(auc_b), "delta_auc": float(delta_auc),
        "ci_low": float(ci_low), "ci_high": float(ci_high), "se_delta": float(se_delta),
        "z_score": float(z_score), "p_auc": p_auc_str,
    }

pairs = [
    ("cross_gated", "parallel_se"),
    ("cross_gated", "B3_naive_concat"),
    ("cross_gated", "B1_spatial_only"),
    ("parallel_se", "B1_spatial_only"),
    ("parallel_se", "B3_naive_concat"),
]

detailed_stats = {}
for tier in ["q100", "q50", "q23"]:
    detailed_stats[tier] = {}
    for na, nb in pairs:
        pa = np.load(os.path.join(eval_dir, f"{na}_{tier}_y_prob.npy"))
        pb = np.load(os.path.join(eval_dir, f"{nb}_{tier}_y_prob.npy"))
        detailed_stats[tier][f"{na}_vs_{nb}"] = compute_paired_stats(y_true, pa, pb, na, nb, tier)

with open("./eval_results/controlled/detailed_statistical_tests.json", "w") as f:
    json.dump(detailed_stats, f, indent=2)

print("\n[SUCCESS] Detailed diagnostics and statistical tests completed!", flush=True)
