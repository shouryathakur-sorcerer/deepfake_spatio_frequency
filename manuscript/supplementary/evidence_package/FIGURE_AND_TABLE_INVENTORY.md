# Figure and Table Inventory

**Document Purpose:** Exhaustive catalogue of all generated visual figures, data tables, and their supporting empirical evidence in the deepfake spatio-frequency forensics workspace.

---

## 1. Visual Figure Inventory

All figures are compiled in both 300 DPI publication-ready PNG and lossless vector PDF formats in the `figures/` directory.

| Figure Filename | Format | Dimensions / DPI | Generating Script | Associated Model & Data Sources | Exact Visual Content & Supported Claim |
| :--- | :---: | :---: | :--- | :--- | :--- |
| `figures/new_controlled_roc_curves.png` (`.pdf`) | PNG / PDF | 3300 × 1100 (300 DPI) | `generate_new_result_figures.py` | All 6 Controlled Baselines (B1–B4, ParallelSE, CrossGated) across `q100`, `q50`, `q23` | **3-Panel Controlled ROC Curves:** Demonstrates true positive vs. false positive rate trade-offs. Highlights that `ParallelSEFusion` achieves higher AUC (0.9495) than `CrossGatedFusion` (0.9382) under severe compression ($Q=23$), while both beat naive concatenation (0.9378). Shows B4 (2D FFT) failing ($AUC < 0.5$). |
| `figures/new_q23_tradeoff_analysis.png` (`.pdf`) | PNG / PDF | 3600 × 1200 (300 DPI) | `generate_new_result_figures.py` | `parallel_se_q23_y_prob.npy` vs. `cross_gated_q23_y_prob.npy` ($N=20,000$) | **Forensic Operating Point Analysis at $Q=23$:** 3-panel figure featuring (1) Predicted probability density histograms, (2) Threshold sweep ($\tau \in [0, 1]$) curves for Accuracy, Precision, Recall, and F1, and (3) Precision-Recall curves. Proves `CrossGatedFusion`'s high precision (94.21%) is an artifact of conservative probability shrinkage ($\mu_{\text{fake}}=0.354$), causing a 30.55% False Negative Rate at default $\tau=0.5$. |
| `figures/new_confusion_matrices_q23.png` (`.pdf`) | PNG / PDF | 3600 × 1100 (300 DPI) | `generate_new_result_figures.py` | `cross_gated`, `parallel_se`, `B1_spatial_only` at $Q=23$ | **Heavy Compression Confusion Matrix Comparison:** 1×3 annotated heatmap showing raw counts. CrossGated: $\text{FN}=3,055$ (missed fakes), $\text{FP}=427$; ParallelSE: $\text{FN}=499$, $\text{FP}=2,563$; SpatialOnly: $\text{FN}=988$, $\text{FP}=1,601$. Proves the exact error profile disparity. |
| `figures/new_compression_degradation.png` (`.pdf`) | PNG / PDF | 3000 × 2400 (300 DPI) | `generate_new_result_figures.py` | All 6 Controlled Baselines across $Q \in \{100, 50, 23\}$ | **4-Panel Robustness Trajectory:** Plots degradation curves across compression tiers for (A) Accuracy, (B) ROC-AUC, (C) Precision, and (D) Recall. Shows that Spatial-Only degrades by 5.87% in accuracy, while CrossGated drops 10.54% and ParallelSE drops 9.07% due to frequency quantization noise in 2-epoch training. |
| `figures/new_calibration_diagrams.png` (`.pdf`) | PNG / PDF | 3600 × 1200 (300 DPI) | `generate_new_result_figures.py` | Predictions and probabilities across $Q=100, 50, 23$ | **Forensic Reliability Diagrams & ECE:** Reliability curves (10 bins) comparing confidence vs. empirical accuracy. Quantifies Expected Calibration Error (ECE) and Brier scores, proving how compression shifts calibration boundaries. |
| `figures/roc_curves.png` (`.pdf`) | PNG / PDF | 2400 × 1800 (300 DPI) | `generate_thesis_plots.py` | Full-Scale Reference Model (`checkpoints/best_model.pth`, 8 epochs on 100k images) | **Multi-Tier Full-Scale ROC Curves:** Demonstrates near-perfect separation on 20,000 images: $Q=100$ ($AUC=0.99995$), $Q=50$ ($AUC=0.99977$), $Q=23$ ($AUC=0.99904$). Shows model convergence after full training. |
| `figures/confusion_matrices.png` (`.pdf`) | PNG / PDF | 3600 × 1200 (300 DPI) | `generate_thesis_plots.py` | Full-Scale Reference Model across $Q=100, 50, 23$ | **Full-Scale Confusion Matrices:** 1×3 grid showing 99.52% accuracy at $Q=100$ (FP=90, FN=6), 98.72% at $Q=50$ (FP=243, FN=14), and 97.94% at $Q=23$ (FP=366, FN=47). |
| `figures/score_distributions.png` (`.pdf`) | PNG / PDF | 3600 × 1200 (300 DPI) | `generate_thesis_plots.py` | Full-Scale Reference Model probability arrays | **Bimodal Output Probability Distributions:** Histograms showing sharp separation between real faces ($\approx 0.00$) and fake faces ($\approx 1.00$) on the fully trained reference model. |
| `figures/metrics_comparison_bar.png` (`.pdf`) | PNG / PDF | 3000 × 1800 (300 DPI) | `generate_thesis_plots.py` | Full-Scale Reference Model summary metrics | **Grouped Bar Chart:** Demonstrates resilience across metric families (Accuracy, Precision, Recall, F1) for the 8-epoch trained reference detector. |

---

## 2. Quantitative Data Table Inventory

| Table Description | Primary File Location | Data Dimensions | Underlying Experiments & Models Covered |
| :--- | :--- | :--- | :--- |
| **Comprehensive Metric Benchmark** | `eval_results/controlled/summary_controlled_experiments.json`<br>`audit/manuscript_evidence_package/RESULTS_TABLES.csv` | 312 rows | All 7 models (6 controlled + 1 full-scale reference), 3 compression tiers ($Q=100, 50, 23$), 12 metrics per condition. |
| **Controlled Contingency Counts** | `eval_results/controlled/audit_recalculations.json`<br>`audit/manuscript_evidence_package/CONFUSION_MATRICES.csv` | 21 rows | Exact TP, TN, FP, FN, FPR, FNR counts at fixed threshold $\tau=0.5$ on $N=20,000$ images. |
| **Paired Statistical Significance Matrix** | `eval_results/controlled/detailed_statistical_tests.json`<br>`audit/manuscript_evidence_package/STATISTICAL_COMPARISONS.csv` | 30 rows | 5 critical model pairs across 3 tiers: McNemar $\chi^2$, analytical $p$-values, Odds Ratios, Cohen's $g$, Paired $\Delta\text{AUC}$, 95% CIs, and $Z$-scores. |
| **Structured Ablation Table** | `audit/manuscript_evidence_package/ABLATION_RESULTS.csv` | 6 rows | Formal status of Ablations A1 (WPD vs DWT), A2 (Basis sensitivity), A3 (Representation space), and A4 (Fusion mechanisms). |
| **Computational Hardware Benchmarks** | `audit/manuscript_evidence_package/COMPUTATIONAL_BENCHMARKS.csv` | 20 rows | Exact trainable parameters, batch-1 latency, batch-64 latency, throughput, training step duration, and peak VRAM measured on RTX 4060 GPU. |
| **Audited Reference & Literature Matrix** | `audit/manuscript_evidence_package/REFERENCE_AUDIT.csv` | 15 rows | Complete metadata, DOI links, intended claims supported, and relationship to proposed method for the 15 required literature citations. |

---

## 3. Data Integrity & Reproducibility Audit

1. **Prediction Arrays:** Every single metric reported in the tables corresponds to raw NumPy prediction files stored on disk (`eval_results/controlled/*_y_prob.npy` and `eval_results/*_y_prob.npy`), each containing exactly 20,000 floating-point probabilities.
2. **Deterministic Evaluation:** Ground truth labels (`eval_results/q100_y_true.npy`) match `DeepfakeDataset.LABEL_MAP` (`real = 0, fake = 1`).
3. **No Synthetic Extrapolations:** No missing cells were imputed or populated with heuristic guesses. Where experiments were restricted to tensor-shape dry-runs (A1–A3), they are explicitly documented as implementation verifications.
