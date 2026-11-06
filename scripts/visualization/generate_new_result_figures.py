"""
generate_new_result_figures.py
==============================
Publication-grade figure generator for the new empirical validation and baseline audit.
Reads the exact saved predictions from ./eval_results/controlled/ and produces high-resolution
figures (300 DPI PNG + vector PDF) in ./figures/.

Figures Generated:
1. figures/new_controlled_roc_curves.png / .pdf
2. figures/new_q23_tradeoff_analysis.png / .pdf
3. figures/new_confusion_matrices_q23.png / .pdf
4. figures/new_compression_degradation.png / .pdf
5. figures/new_calibration_diagrams.png / .pdf
"""

import os
import tempfile

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_deepfake_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    roc_auc_score,
)

# IEEE Publication Style Settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

os.makedirs("./figures", exist_ok=True)
eval_dir = "./eval_results/controlled"
y_true = np.load("./eval_results/q100_y_true.npy").astype(int)

MODEL_CONFIGS = {
    "parallel_se": {"label": "ParallelSEFusion (Proposed SE)", "color": "#1f77b4", "lw": 2.0, "ls": "-"},
    "cross_gated": {"label": "CrossGatedFusion (Cross-Gated)", "color": "#ff7f0e", "lw": 2.0, "ls": "-"},
    "B1_spatial_only": {"label": "B1: Spatial Only (Xception)", "color": "#2ca02c", "lw": 1.5, "ls": "--"},
    "B3_naive_concat": {"label": "B3: Naive Concat (No Gating)", "color": "#9467bd", "lw": 1.5, "ls": "-."},
    "B2_frequency_only": {"label": "B2: Frequency Only (WPD-CNN)", "color": "#8c564b", "lw": 1.2, "ls": ":"},
    "B4_fft_baseline": {"label": "B4: 2D FFT Power Spectrum", "color": "#7f7f7f", "lw": 1.0, "ls": ":"},
}

TIERS = [("q100", "Clean Baseline (Q=100)"), ("q50", "Moderate Web (Q=50)"), ("q23", "Heavy JPEG (Q=23)")]

# ==============================================================================
# FIGURE 1: MULTI-TIER ROC CURVES COMPARISON
# ==============================================================================
print("[1/5] Generating Multi-Tier ROC Curves Comparison...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

for ax_idx, (tier_id, tier_name) in enumerate(TIERS):
    ax = axes[ax_idx]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Chance (AUC=0.500)")
    
    for m_id, cfg in MODEL_CONFIGS.items():
        fpath = os.path.join(eval_dir, f"{m_id}_{tier_id}_y_prob.npy")
        if os.path.exists(fpath):
            y_prob = np.load(fpath)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            score = roc_auc_score(y_true, y_prob)
            ax.plot(
                fpr, tpr,
                label=f"{cfg['label']} (AUC={score:.4f})",
                color=cfg["color"],
                lw=cfg["lw"],
                ls=cfg["ls"],
            )
            
    ax.set_title(tier_name, fontweight="bold")
    ax.set_xlabel("False Positive Rate (FPR)")
    if ax_idx == 0:
        ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="lower right")

fig.suptitle("Controlled Receiver Operating Characteristic (ROC) Comparison across JPEG Quality Tiers", y=1.02, fontweight="bold")
plt.tight_layout()
fig.savefig("./figures/new_controlled_roc_curves.png", bbox_inches="tight")
fig.savefig("./figures/new_controlled_roc_curves.pdf", bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# FIGURE 2: THE Q=23 PRECISION-RECALL & PROBABILITY DISTRIBUTION TRADE-OFF
# ==============================================================================
print("[2/5] Generating Q=23 Trade-Off Analysis Figure...", flush=True)
fig, (ax_pr, ax_dist) = plt.subplots(1, 2, figsize=(12, 4.8))

# Panel A: Precision-Recall Curves at Q=23
for m_id in ["parallel_se", "cross_gated", "B1_spatial_only", "B3_naive_concat"]:
    cfg = MODEL_CONFIGS[m_id]
    y_prob = np.load(os.path.join(eval_dir, f"{m_id}_q23_y_prob.npy"))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_score = auc(rec, prec)
    ax_pr.plot(
        rec, prec,
        label=f"{cfg['label']} (PR-AUC={pr_score:.4f})",
        color=cfg["color"],
        lw=cfg["lw"],
        ls=cfg["ls"],
    )

ax_pr.set_title("A: Precision-Recall Curves at Heavy Compression (Q=23)", fontweight="bold")
ax_pr.set_xlabel("Recall (Sensitivity)")
ax_pr.set_ylabel("Precision (Positive Predictive Value)")
ax_pr.set_xlim([-0.02, 1.02])
ax_pr.set_ylim([0.45, 1.02])
ax_pr.legend(loc="lower left")

# Panel B: Score Distribution Histograms at Q=23
p_cg = np.load(os.path.join(eval_dir, "cross_gated_q23_y_prob.npy"))
p_pse = np.load(os.path.join(eval_dir, "parallel_se_q23_y_prob.npy"))

bins = np.linspace(0, 1, 40)
ax_dist.hist(p_cg[y_true == 1], bins=bins, alpha=0.55, color="#ff7f0e", label="CrossGated: Fake Faces (Mean=0.354)", density=True)
ax_dist.hist(p_cg[y_true == 0], bins=bins, alpha=0.35, color="#1f77b4", label="CrossGated: Real Faces (Mean=0.082)", density=True)
ax_dist.axvline(0.5, color="red", linestyle="--", lw=1.5, label="Default Threshold \u03c4 = 0.5")
ax_dist.axvline(0.224, color="orange", linestyle=":", lw=1.5, label="CrossGated Optimal \u03c4* = 0.224")

ax_dist.set_title("B: CrossGated Probability Shift at Q=23 (Shrinkage to 0)", fontweight="bold")
ax_dist.set_xlabel("Predicted Probability P(Fake)")
ax_dist.set_ylabel("Probability Density")
ax_dist.legend(loc="upper right", fontsize=8)

plt.tight_layout()
fig.savefig("./figures/new_q23_tradeoff_analysis.png", bbox_inches="tight")
fig.savefig("./figures/new_q23_tradeoff_analysis.pdf", bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# FIGURE 3: SIDE-BY-SIDE CONFUSION MATRICES AT Q=23
# ==============================================================================
print("[3/5] Generating Q=23 Confusion Matrices Comparison...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

models_cm = [
    ("cross_gated", "CrossGatedFusion (Q=23)\nHigh Precision / High FNR"),
    ("parallel_se", "ParallelSEFusion (Q=23)\nHigh Recall / High FPR"),
    ("B1_spatial_only", "Spatial Only Xception (Q=23)\nBalanced Spatial Baseline"),
]

for idx, (m_id, title) in enumerate(models_cm):
    ax = axes[idx]
    y_prob = np.load(os.path.join(eval_dir, f"{m_id}_q23_y_prob.npy"))
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm = np.array([[tn, fp], [fn, tp]])
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Real (0)", "Pred Fake (1)"])
    ax.set_yticklabels(["True Real (0)", "True Fake (1)"])
    ax.set_title(title, fontweight="bold", fontsize=9.5)
    
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_color = "white" if pct > 50 else "black"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)

plt.tight_layout()
fig.savefig("./figures/new_confusion_matrices_q23.png", bbox_inches="tight")
fig.savefig("./figures/new_confusion_matrices_q23.pdf", bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# FIGURE 4: COMPRESSION ROBUSTNESS DEGRADATION BAR CHARTS
# ==============================================================================
print("[4/5] Generating Compression Degradation Bar Chart...", flush=True)
fig, (ax_acc, ax_auc) = plt.subplots(1, 2, figsize=(14, 4.8))

bar_models = ["parallel_se", "cross_gated", "B1_spatial_only", "B3_naive_concat", "B2_frequency_only"]
labels_bar = ["ParallelSE", "CrossGated", "Spatial Only", "Naive Concat", "WPD Freq Only"]
colors_bar = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

x = np.arange(len(bar_models))
width = 0.25

acc_q100 = [93.76, 93.13, 92.93, 92.12, 61.90]
acc_q50 = [88.81, 89.66, 89.61, 89.85, 56.01]
acc_q23 = [84.69, 82.59, 87.06, 86.22, 52.45]

auc_q100 = [0.9855, 0.9851, 0.9830, 0.9822, 0.7479]
auc_q50 = [0.9727, 0.9695, 0.9681, 0.9671, 0.6541]
auc_q23 = [0.9495, 0.9382, 0.9473, 0.9378, 0.5972]

# Accuracy panel
ax_acc.bar(x - width, acc_q100, width, label="Q=100 (Clean)", color="#2ca02c", alpha=0.85)
ax_acc.bar(x, acc_q50, width, label="Q=50 (Web)", color="#1f77b4", alpha=0.85)
ax_acc.bar(x + width, acc_q23, width, label="Q=23 (Heavy)", color="#d62728", alpha=0.85)
ax_acc.set_ylabel("Accuracy (%)")
ax_acc.set_title("A: Classification Accuracy across Compression Tiers", fontweight="bold")
ax_acc.set_xticks(x)
ax_acc.set_xticklabels(labels_bar, rotation=15)
ax_acc.set_ylim([45, 100])
ax_acc.legend(loc="lower left")

# ROC-AUC panel
ax_auc.bar(x - width, auc_q100, width, label="Q=100 (Clean)", color="#2ca02c", alpha=0.85)
ax_auc.bar(x, auc_q50, width, label="Q=50 (Web)", color="#1f77b4", alpha=0.85)
ax_auc.bar(x + width, auc_q23, width, label="Q=23 (Heavy)", color="#d62728", alpha=0.85)
ax_auc.set_ylabel("ROC-AUC")
ax_auc.set_title("B: ROC-AUC Discriminability across Compression Tiers", fontweight="bold")
ax_auc.set_xticks(x)
ax_auc.set_xticklabels(labels_bar, rotation=15)
ax_auc.set_ylim([0.50, 1.02])
ax_auc.legend(loc="lower left")

plt.tight_layout()
fig.savefig("./figures/new_compression_degradation.png", bbox_inches="tight")
fig.savefig("./figures/new_compression_degradation.pdf", bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# FIGURE 5: CALIBRATION RELIABILITY DIAGRAMS (ECE)
# ==============================================================================
print("[5/5] Generating Calibration Reliability Diagrams...", flush=True)
fig, (ax_cal_clean, ax_cal_q23) = plt.subplots(1, 2, figsize=(12, 4.8))

def plot_calibration(ax, y_true, y_prob, label, color):
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accs = []
    confs = []
    for i in range(len(bins) - 1):
        in_b = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if np.sum(in_b) > 0:
            accs.append(np.mean(y_true[in_b]))
            confs.append(np.mean(y_prob[in_b]))
        else:
            accs.append(np.nan)
            confs.append(bin_centers[i])
    ax.plot(confs, accs, marker="o", label=label, color=color, lw=1.8)

# Clean Q=100
ax_cal_clean.plot([0, 1], [0, 1], "k--", lw=1.0, label="Perfect Calibration")
plot_calibration(ax_cal_clean, y_true, np.load(os.path.join(eval_dir, "parallel_se_q100_y_prob.npy")), "ParallelSE (ECE=0.0209)", "#1f77b4")
plot_calibration(ax_cal_clean, y_true, np.load(os.path.join(eval_dir, "cross_gated_q100_y_prob.npy")), "CrossGated (ECE=0.0436)", "#ff7f0e")
plot_calibration(ax_cal_clean, y_true, np.load(os.path.join(eval_dir, "B1_spatial_only_q100_y_prob.npy")), "Spatial Only (ECE=0.0353)", "#2ca02c")
ax_cal_clean.set_title("A: Reliability Diagram at Q=100 (Clean)", fontweight="bold")
ax_cal_clean.set_xlabel("Mean Predicted Confidence")
ax_cal_clean.set_ylabel("Empirical Accuracy")
ax_cal_clean.legend(loc="upper left")

# Q=23 Heavy
ax_cal_q23.plot([0, 1], [0, 1], "k--", lw=1.0, label="Perfect Calibration")
plot_calibration(ax_cal_q23, y_true, np.load(os.path.join(eval_dir, "parallel_se_q23_y_prob.npy")), "ParallelSE (ECE=0.1131)", "#1f77b4")
plot_calibration(ax_cal_q23, y_true, np.load(os.path.join(eval_dir, "cross_gated_q23_y_prob.npy")), "CrossGated (ECE=0.1171)", "#ff7f0e")
plot_calibration(ax_cal_q23, y_true, np.load(os.path.join(eval_dir, "B1_spatial_only_q23_y_prob.npy")), "Spatial Only (ECE=0.0359)", "#2ca02c")
ax_cal_q23.set_title("B: Reliability Diagram at Q=23 (Heavy JPEG)", fontweight="bold")
ax_cal_q23.set_xlabel("Mean Predicted Confidence")
ax_cal_q23.set_ylabel("Empirical Accuracy")
ax_cal_q23.legend(loc="upper left")

plt.tight_layout()
fig.savefig("./figures/new_calibration_diagrams.png", bbox_inches="tight")
fig.savefig("./figures/new_calibration_diagrams.pdf", bbox_inches="tight")
plt.close(fig)

print("\n[SUCCESS] All 5 publication-grade figure sets generated successfully in ./figures/!", flush=True)
