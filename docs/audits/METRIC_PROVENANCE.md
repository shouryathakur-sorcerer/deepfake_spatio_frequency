# Metric Provenance & Result Reconciliation Report
**Project:** Spatial–Frequency Deepfake Image Classification with Wavelet-Packet Features and Bidirectional Cross-Attention  
**Document Purpose:** Exhaustive forensic audit, reconciliation, and artifact-level traceability of all experimental performance figures across project documents, faculty reviews, and codebase result files.

---

## 1. Executive Summary: The AUC Discrepancy Resolved

The research project history contains two sharply contrasting sets of performance figures:

1. **The Teacher's Review (`review 1.docx`):**
   - Evaluated the earlier manuscript reporting:
     - JPEG $Q = 100$: Accuracy = $99.52\%$, ROC-AUC = $0.99995$
     - JPEG $Q = 23$: Accuracy = $97.94\%$, ROC-AUC = $0.99904$
2. **The Anonymous Draft Manuscript (`deepfake.pdf`):**
   - Reported:
     - JPEG $Q = 50$: ROC-AUC = $0.937$
     - JPEG $Q = 23$: ROC-AUC = $0.888$
     - JPEG $Q = 10$: Qualitative decline, no numeric value supplied.

### The Forensic Discovery & Root Cause
Through systematic inspection of all JSON artifacts (`eval_results/summary_benchmark.json`, `eval_results/controlled/summary_controlled_experiments.json`, `eval_results/controlled/audit_recalculations.json`) and audit reports, the exact provenance of both sets of figures has been definitively resolved:

1. **Set 1 (0.99995 / 0.99904):** These are **genuine, verifiable results** from the project's original 8-epoch training run on the full 100,000-image training split of Kaggle *140k Real and Fake Faces* (FFHQ vs. StyleGAN2) using `ParallelSEFusion` with on-the-fly random JPEG augmentation ($Q \in [60, 100]$). The exact machine-readable numbers reside in [`eval_results/summary_benchmark.json`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/eval_results/summary_benchmark.json).
2. **Set 2 (0.937 / 0.888):** These numbers arose from a **metric transposition error** during the drafting of `deepfake.pdf`. In the controlled baseline experiments on 5,000 images without augmentation (`eval_results/controlled/summary_controlled_experiments.json`), the `ParallelSEFusion` model achieved an **Accuracy of 93.76% (0.9376)** at $Q=100$ and an **Accuracy of 88.81% (0.8881)** at $Q=50$. Simultaneously, the baseline `B3_naive_concat` model achieved an **ROC-AUC of 0.9378** at $Q=23$. The previous author conflated these accuracy percentages with ROC-AUC values, truncating $93.76\%$ to $0.937$ and $88.81\%$ to $0.888$ and labeling them as ROC-AUC at $Q=50$ and $Q=23$.
3. **The True Controlled Baseline AUCs:** In the controlled 5k-sample experiment, the true ROC-AUCs of `ParallelSEFusion` are **0.9855** at $Q=100$, **0.9727** at $Q=50$, and **0.9495** at $Q=23$; for `CrossGatedFusion`, they are **0.9851** at $Q=100$, **0.9695** at $Q=50$, and **0.9382** at $Q=23$.

These two experimental configurations represent fundamentally different training budgets and data exposures; they must never be presented as a single experiment.

---

## 2. Complete Metric-Provenance Master Table

| Metric | Dataset | Checkpoint | Training Configuration | Test Split | JPEG Q | Source Artifact | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **ROC-AUC = 0.99995** | FFHQ vs StyleGAN2 (140k) | `best_model.pth` | 100k images, 8 epochs, AdamW, JPEG aug ($Q \in [60, 100]$) | 20k test | 100 | `eval_results/summary_benchmark.json` | **Directly Verified** (`0.999951625`) |
| **Accuracy = 99.52%** | FFHQ vs StyleGAN2 (140k) | `best_model.pth` | 100k images, 8 epochs, AdamW, JPEG aug ($Q \in [60, 100]$) | 20k test | 100 | `eval_results/summary_benchmark.json` | **Directly Verified** (`0.995200`) |
| **ROC-AUC = 0.99977** | FFHQ vs StyleGAN2 (140k) | `best_model.pth` | 100k images, 8 epochs, AdamW, JPEG aug ($Q \in [60, 100]$) | 20k test | 50 | `eval_results/summary_benchmark.json` | **Directly Verified** (`0.999773885`) |
| **Accuracy = 98.72%** | FFHQ vs StyleGAN2 (140k) | `best_model.pth` | 100k images, 8 epochs, AdamW, JPEG aug ($Q \in [60, 100]$) | 20k test | 50 | `eval_results/summary_benchmark.json` | **Directly Verified** (`0.987150`) |
| **ROC-AUC = 0.99904** | FFHQ vs StyleGAN2 (140k) | `best_model.pth` | 100k images, 8 epochs, AdamW, JPEG aug ($Q \in [60, 100]$) | 20k test | 23 | `eval_results/summary_benchmark.json` | **Directly Verified** (`0.999043625`) |
| **Accuracy = 97.94%** | FFHQ vs StyleGAN2 (140k) | `best_model.pth` | 100k images, 8 epochs, AdamW, JPEG aug ($Q \in [60, 100]$) | 20k test | 23 | `eval_results/summary_benchmark.json` | **Directly Verified** (`0.979350`) |
| **ROC-AUC = 0.9855** | FFHQ vs StyleGAN2 (140k) | `parallel_se.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 100 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.985501`) |
| **Accuracy = 93.76%** | FFHQ vs StyleGAN2 (140k) | `parallel_se.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 100 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.937600`) *(Misquoted as AUC 0.937 in `deepfake.pdf`)* |
| **ROC-AUC = 0.9727** | FFHQ vs StyleGAN2 (140k) | `parallel_se.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 50 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.972744`) |
| **Accuracy = 88.81%** | FFHQ vs StyleGAN2 (140k) | `parallel_se.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 50 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.888100`) *(Misquoted as AUC 0.888 in `deepfake.pdf`)* |
| **ROC-AUC = 0.9495** | FFHQ vs StyleGAN2 (140k) | `parallel_se.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 23 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.949451`) |
| **Accuracy = 84.69%** | FFHQ vs StyleGAN2 (140k) | `parallel_se.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 23 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.846900`) |
| **ROC-AUC = 0.9851** | FFHQ vs StyleGAN2 (140k) | `cross_gated.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 100 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.985120`) |
| **Accuracy = 93.13%** | FFHQ vs StyleGAN2 (140k) | `cross_gated.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 100 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.931300`) |
| **ROC-AUC = 0.9695** | FFHQ vs StyleGAN2 (140k) | `cross_gated.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 50 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.969529`) |
| **Accuracy = 89.67%** | FFHQ vs StyleGAN2 (140k) | `cross_gated.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 50 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.896650`) |
| **ROC-AUC = 0.9382** | FFHQ vs StyleGAN2 (140k) | `cross_gated.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 23 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.938212`) |
| **Accuracy = 84.81%** | FFHQ vs StyleGAN2 (140k) | `cross_gated.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 23 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.848100`) |
| **ROC-AUC = 0.9830** | FFHQ vs StyleGAN2 (140k) | `B1_spatial_only.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 100 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.982987`) |
| **Accuracy = 92.93%** | FFHQ vs StyleGAN2 (140k) | `B1_spatial_only.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 100 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.929300`) |
| **ROC-AUC = 0.9681** | FFHQ vs StyleGAN2 (140k) | `B1_spatial_only.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 50 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.968103`) |
| **Accuracy = 89.61%** | FFHQ vs StyleGAN2 (140k) | `B1_spatial_only.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 50 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.896150`) |
| **ROC-AUC = 0.9473** | FFHQ vs StyleGAN2 (140k) | `B1_spatial_only.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 23 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.947328`) |
| **Accuracy = 87.06%** | FFHQ vs StyleGAN2 (140k) | `B1_spatial_only.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 23 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.870550`) |
| **ROC-AUC = 0.7479** | FFHQ vs StyleGAN2 (140k) | `B2_frequency_only.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 100 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.747937`) |
| **Accuracy = 61.90%** | FFHQ vs StyleGAN2 (140k) | `B2_frequency_only.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 100 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.619000`) |
| **ROC-AUC = 0.6541** | FFHQ vs StyleGAN2 (140k) | `B2_frequency_only.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 50 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.654143`) |
| **Accuracy = 56.01%** | FFHQ vs StyleGAN2 (140k) | `B2_frequency_only.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 50 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.560050`) |
| **ROC-AUC = 0.5972** | FFHQ vs StyleGAN2 (140k) | `B2_frequency_only.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 23 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.597154`) |
| **Accuracy = 52.88%** | FFHQ vs StyleGAN2 (140k) | `B2_frequency_only.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 23 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.528750`) |
| **ROC-AUC = 0.9822** | FFHQ vs StyleGAN2 (140k) | `B3_naive_concat.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 100 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.982203`) |
| **Accuracy = 92.13%** | FFHQ vs StyleGAN2 (140k) | `B3_naive_concat.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 100 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.921300`) |
| **ROC-AUC = 0.9671** | FFHQ vs StyleGAN2 (140k) | `B3_naive_concat.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 50 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.967074`) |
| **Accuracy = 89.86%** | FFHQ vs StyleGAN2 (140k) | `B3_naive_concat.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 50 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.898600`) |
| **ROC-AUC = 0.9378** | FFHQ vs StyleGAN2 (140k) | `B3_naive_concat.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 23 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.937820`) *(Misattributed to Q=50 in `deepfake.pdf`)* |
| **Accuracy = 86.23%** | FFHQ vs StyleGAN2 (140k) | `B3_naive_concat.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 23 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.862250`) |
| **ROC-AUC = 0.4554** | FFHQ vs StyleGAN2 (140k) | `B4_fft_baseline.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 100 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.455402`) *(Fails completely)* |
| **Accuracy = 50.00%** | FFHQ vs StyleGAN2 (140k) | `B4_fft_baseline.pth` | 5k images, 2 epochs, AdamW, no JPEG aug | 20k test | 100 | `eval_results/controlled/summary_controlled_experiments.json` | **Directly Verified** (`0.500000`) |
| **JPEG Q = 10 AUC** | FFHQ vs StyleGAN2 (140k) | Any | Dense sweep Q=10 | 20k test | 10 | None in current JSONs | **Unrecovered / Missing** *(Qualitative only)* |
| **FaceForensics++ c23** | FF++ H.264 Video | None | Video extraction scripts only | Planned | c23 | None | **Planned / Not Run** |
| **Celeb-DF v2** | Celeb-DF | None | External transfer | Planned | Raw | None | **Planned / Not Run** |
| **Team 14 Full Model** | FFHQ vs StyleGAN2 | Initialized | 6-variant ablation suite | 20k test | 100 | Code verified (13/13 tests) | **Awaiting Matched Training** |

---

## 3. Scientific Implications & Journal Manuscript Guidance

1. **Reconciliation in the Paper:**
   The paper must present both experimental regimes with complete transparency:
   - **Regime A (Descriptive Upper-Bound Run):** Full-scale 100,000-image training with 8 epochs and data augmentation achieves near-ceiling performance on StyleGAN2 ($0.99995$ AUC at $Q=100$, $0.99904$ AUC at $Q=23$). This demonstrates that the dual-branch framework can memorize and detect in-distribution generative artifacts with near perfection.
   - **Regime B (Controlled 5k Baseline Comparison):** Under strictly matched compute budgets (5k training samples, 2 epochs, no augmentation), the comparative dynamics between branches are exposed:
     - At $Q=100$: `ParallelSE` ($0.9855$ AUC, $93.76\%$ Acc) and `CrossGated` ($0.9851$ AUC, $93.13\%$ Acc) both outperform `B1_spatial_only` ($0.9830$ AUC, $92.93\%$ Acc) and `B3_naive_concat` ($0.9822$ AUC, $92.13\%$ Acc).
     - At $Q=23$: `ParallelSE` ($0.9495$ AUC) maintains superior rank ordering over `CrossGated` ($0.9382$ AUC), `B3_naive_concat` ($0.9378$ AUC), and `B2_frequency_only` ($0.5972$ AUC).
     - Crucial Scientific Negative Result: Without compression-aware training, spatial-only accuracy at $Q=23$ ($87.06\%$) exceeds `ParallelSE` accuracy ($84.69\%$) at the default 0.5 threshold. This proves that frequency features do not automatically confer compression resilience without compression-aware training.
2. **Elimination of the Transposition Error:**
   The manuscript text must be corrected to remove the false statement that "ROC-AUC was 0.937 at Q=50 and 0.888 at Q=23". Instead, the paper will cite the actual verified ROC-AUCs ($0.9727$ at $Q=50$ and $0.9495$ at $Q=23$ for controlled `ParallelSE`; and $0.99977$ / $0.99904$ for the full-dataset run).

---
*End of Metric Provenance & Result Reconciliation Report.*
