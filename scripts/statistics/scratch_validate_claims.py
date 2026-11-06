"""
scratch_validate_claims.py
==========================
Comprehensive script to audit and validate all experimental claims
against the actual saved predictions (.npy) in eval_results/ and eval_results/controlled/.
"""

import json
import os
import numpy as np
import scipy.stats as stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)

eval_dir = "./eval_results/controlled"
y_true = np.load("./eval_results/q100_y_true.npy").astype(int)

# Check class balance
n_total = len(y_true)
n_real = int(np.sum(y_true == 0))
n_fake = int(np.sum(y_true == 1))
print(f"Total samples: {n_total}, Real (0): {n_real}, Fake (1): {n_fake}")

tiers = ["q100", "q50", "q23"]
models = [
    "cross_gated",
    "parallel_se",
    "B1_spatial_only",
    "B2_frequency_only",
    "B3_naive_concat",
    "B4_fft_baseline",
]

# Recalculate metrics for all models across all tiers
results = {}

for tier in tiers:
    results[tier] = {}
    for m in models:
        prob_path = os.path.join(eval_dir, f"{m}_{tier}_y_prob.npy")
        if not os.path.exists(prob_path):
            continue
        y_prob = np.load(prob_path)
        
        # Binary threshold = 0.5
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Confusion matrix:
        # tn: real predicted as real (y_true=0, y_pred=0)
        # fp: real predicted as fake (y_true=0, y_pred=1)
        # fn: fake predicted as real (y_true=1, y_pred=0)
        # tp: fake predicted as fake (y_true=1, y_pred=1)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        acc = (tp + tn) / n_total
        bal_acc = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) # False positive rate = FP / Real
        fnr = fn / (fn + tp) # False negative rate = FN / Fake
        
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc_auc = 0.5
            
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(r_curve, p_curve)
        
        # Calibration ECE (15 bins)
        bins = np.linspace(0, 1, 16)
        ece = 0.0
        for i in range(15):
            in_b = (y_prob > bins[i]) & (y_prob <= bins[i+1])
            prop = np.mean(in_b)
            if prop > 0:
                acc_b = np.mean(y_true[in_b])
                conf_b = np.mean(y_prob[in_b])
                ece += np.abs(acc_b - conf_b) * prop
                
        brier = brier_score_loss(y_true, y_prob)
        
        # Optimal threshold (Youden's J statistic)
        fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr_curve - fpr_curve
        opt_idx = np.argmax(j_scores)
        opt_thresh = float(thresholds[opt_idx])
        y_pred_opt = (y_prob >= opt_thresh).astype(int)
        opt_acc = accuracy_score(y_true, y_pred_opt)
        opt_bal_acc = balanced_accuracy_score(y_true, y_pred_opt)
        opt_f1 = f1_score(y_true, y_pred_opt)
        opt_prec = precision_score(y_true, y_pred_opt)
        opt_rec = recall_score(y_true, y_pred_opt)
        
        results[tier][m] = {
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Accuracy": float(acc),
            "Balanced_Acc": float(bal_acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1),
            "FPR": float(fpr),
            "FNR": float(fnr),
            "ROC_AUC": float(roc_auc),
            "PR_AUC": float(pr_auc),
            "ECE": float(ece),
            "Brier": float(brier),
            "Optimal_Threshold": opt_thresh,
            "Opt_Accuracy": float(opt_acc),
            "Opt_Balanced_Acc": float(opt_bal_acc),
            "Opt_Precision": float(opt_prec),
            "Opt_Recall": float(opt_rec),
            "Opt_F1": float(opt_f1),
            "Mean_Predicted_Prob": float(np.mean(y_prob)),
            "Prob_Min": float(np.min(y_prob)),
            "Prob_Max": float(np.max(y_prob)),
        }

print("\n=== SUMMARY RECALCULATION TABLE ===")
for tier in tiers:
    print(f"\n--- TIER: {tier} ---")
    print(f"{'Model':<20} | {'Acc':>7} | {'BalAcc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'FPR':>7} | {'FNR':>7} | {'ROC-AUC':>7} | {'PR-AUC':>7} | {'OptThresh':>9} | {'OptAcc':>7}")
    print("-" * 115)
    for m in models:
        r = results[tier][m]
        print(f"{m:<20} | {r['Accuracy']*100:>6.2f}% | {r['Balanced_Acc']*100:>6.2f}% | {r['Precision']*100:>6.2f}% | {r['Recall']*100:>6.2f}% | {r['F1']:>7.4f} | {r['FPR']*100:>6.2f}% | {r['FNR']*100:>6.2f}% | {r['ROC_AUC']:>7.4f} | {r['PR_AUC']:>7.4f} | {r['Optimal_Threshold']:>9.4f} | {r['Opt_Accuracy']*100:>6.2f}%")

# Save recalculations to json
with open("./eval_results/controlled/audit_recalculations.json", "w") as f:
    json.dump(results, f, indent=2)
