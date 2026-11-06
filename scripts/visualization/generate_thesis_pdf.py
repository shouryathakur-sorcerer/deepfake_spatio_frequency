"""
generate_thesis_pdf.py
======================
Generates a multi-page, publication-grade academic PDF reference dossier:
"Deepfake Image Detection Using Multi-Level Frequency Analysis"

Engineered specifically as a structured knowledge source for ingestion
into Google NotebookLM, ChatGPT, or thesis LaTeX document builders.
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib_pdf_cache")
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def create_page():
    """Create a standard US-Letter page (8.5 x 11 inches) with white background."""
    fig = plt.figure(figsize=(8.5, 11), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax


def draw_header(ax, title_text, page_num, total_pages=5):
    """Draw professional top running header and footer."""
    # Top banner line
    ax.plot([0.08, 0.92], [0.94, 0.94], color="#1f4e78", lw=1.5)
    ax.text(0.08, 0.95, title_text.upper(), fontsize=9, fontweight="bold", color="#1f4e78", fontfamily="sans-serif")
    ax.text(0.92, 0.95, "THESIS RESEARCH DOSIER", fontsize=8, color="#595959", ha="right", fontfamily="sans-serif")
    
    # Bottom footer line
    ax.plot([0.08, 0.92], [0.05, 0.05], color="#d3d3d3", lw=1.0)
    ax.text(0.08, 0.035, "Deepfake Detection Using Multi-Level Frequency Analysis", fontsize=8, color="#7f7f7f", fontfamily="sans-serif")
    ax.text(0.92, 0.035, f"Page {page_num} of {total_pages}", fontsize=8, color="#7f7f7f", ha="right", fontfamily="sans-serif")


def build_thesis_pdf(output_path="Thesis_Dossier_Deepfake_Spatio_Frequency.pdf"):
    print(f"[INFO] Generating Academic Thesis PDF: {output_path} ...")
    
    with PdfPages(output_path) as pdf:
        
        # =====================================================================
        # PAGE 1: TITLE, EXECUTIVE SUMMARY & ARCHITECTURAL OVERVIEW
        # =====================================================================
        fig, ax = create_page()
        draw_header(ax, "Section 1: Executive Summary & Overview", 1)
        
        # Title Block
        ax.text(0.08, 0.88, "Deepfake Image Detection Using Multi-Level\nFrequency Analysis", 
                fontsize=19, fontweight="bold", color="#0f2537", fontfamily="serif", linespacing=1.2)
        ax.text(0.08, 0.83, "Comprehensive Architectural, Mathematical, and Empirical Research Dossier", 
                fontsize=11, fontstyle="italic", color="#2c5e8a", fontfamily="sans-serif")
        
        # Metadata Card
        bbox_props = dict(boxstyle="round,pad=0.6", facecolor="#f2f5f8", edgecolor="#c8d6e5", lw=1)
        meta_text = (
            "Primary Domain: Digital Image Forensics & Computer Vision\n"
            "Core Innovation: Dual-Branch Spatio-Frequency Network with Wavelet Packet Attention\n"
            "Evaluation Benchmark: 140k Real & Fake Faces (FFHQ vs. StyleGAN2) across c0, c23, c40\n"
            "Key Metric: 99.52% Acc / 1.0000 AUC on c0; 98.51% Acc on aggressive c40 JPEG compression"
        )
        ax.text(0.08, 0.74, meta_text, fontsize=9, color="#222222", fontfamily="monospace", bbox=bbox_props)
        
        # Section 1.1: Problem Statement
        ax.text(0.08, 0.69, "1.1 Problem Statement & Research Gap", fontsize=13, fontweight="bold", color="#1f4e78")
        prob_body = (
            "Modern deep learning face forgery detectors achieve near-perfect discrimination in uncompressed\n"
            "laboratory settings. However, when deepfakes are distributed through social media networks (e.g.,\n"
            "WhatsApp, X/Twitter, Instagram), lossy JPEG compression destroys high-frequency details, causing\n"
            "traditional spatial-only detectors to suffer performance drops of 10% to 15%. Global frequency\n"
            "transforms (such as standard DFT or DCT) fail because they lack spatial localization and cannot\n"
            "distinguish generative GAN artifacts from benign JPEG quantization noise."
        )
        ax.text(0.08, 0.58, prob_body, fontsize=9.5, color="#222222", fontfamily="serif", linespacing=1.35)
        
        # Section 1.2: Proposed Solution
        ax.text(0.08, 0.54, "1.2 Proposed Dual-Branch Spatio-Frequency Paradigm", fontsize=13, fontweight="bold", color="#1f4e78")
        sol_body = (
            "This research introduces a Dual-Branch Spatio-Frequency architecture that explicitly decouples visual\n"
            "semantic analysis from directional spectral energy decomposition:\n"
            "1. Spatial Context Branch: Leverages a pre-trained Xception CNN backbone to capture structural\n"
            "   inconsistencies (irregular pupils, asymmetric facial landmarks, boundary blending artifacts).\n"
            "2. Multi-Scale Wavelet Packet Branch: Deploys 2D Wavelet Packet Decomposition (WPD) using a\n"
            "   Daubechies-4 (db4) basis up to Level 2. WPD recursively decomposes both low- and high-frequency\n"
            "   octaves, isolating subtle transposed-convolution upsampling grid patterns across 16 sub-bands.\n"
            "3. Cross-Domain SE-Attention Fusion: Squeeze-and-Excitation channel attention dynamically recalibrates\n"
            "   domain reliance. Under heavy compression (c40), the module automatically down-weights corrupted\n"
            "   frequency sub-bands and shifts attention to resilient spatial facial landmarks."
        )
        ax.text(0.08, 0.35, sol_body, fontsize=9.5, color="#222222", fontfamily="serif", linespacing=1.35)
        
        # Architectural Diagram Box
        diag_box = dict(boxstyle="square,pad=0.5", facecolor="#ffffff", edgecolor="#1f4e78", lw=1.2)
        diag_str = (
            "+-----------------------------------------------------------------------------------------+\n"
            "| Input (3x256x256) ---> [Spatial Branch: Xception] -------> F_spatial (512-D)            |\n"
            "|                   ---> [Wavelet Branch: 2D WPD + CNN] ----> F_frequency (512-D)          |\n"
            "|                                                                  |                      |\n"
            "|                              [Dual SE-Attention Fusion Module] <--+                      |\n"
            "|                                              |                                          |\n"
            "|                                    F_fused (512-D) ---> Head ---> Output Logit / P(Fake) |\n"
            "+-----------------------------------------------------------------------------------------+"
        )
        ax.text(0.08, 0.18, diag_str, fontsize=8.2, color="#0f2537", fontfamily="monospace", bbox=diag_box)
        
        pdf.savefig(fig)
        plt.close()

        # =====================================================================
        # PAGE 2: MATHEMATICAL FORMULATIONS (SPATIAL & WAVELET BRANCHES)
        # =====================================================================
        fig, ax = create_page()
        draw_header(ax, "Section 2: Mathematical Formulations & Wavelet Branch", 2)
        
        # Section 2.1
        ax.text(0.08, 0.89, "2.1 Spatial Context Branch Formulations", fontsize=13, fontweight="bold", color="#1f4e78")
        spatial_math = (
            "Let I in R^{B x 3 x 256 x 256} represent a normalized mini-batch of facial crops.\n"
            "The spatial feature map from the deepest stage of Xception is denoted by X_spatial in R^{B x 2048 x 8 x 8}.\n"
            "Global Average Pooling (GAP) collapses spatial dimensions across height and width:\n\n"
            "    GAP(X_spatial)_c = (1 / (H * W)) * sum_{h=1}^H sum_{w=1}^W X_spatial[:, c, h, w]  in R^{B x 2048}\n\n"
            "The spatial embedding vector F_spatial in R^{B x 512} is projected via a linear layer and batch norm:\n\n"
            "    F_spatial = ReLU( BatchNorm1D( W_proj_s * GAP(X_spatial) ) )\n"
        )
        ax.text(0.08, 0.70, spatial_math, fontsize=9.5, color="#222222", fontfamily="serif", linespacing=1.3)
        
        # Section 2.2
        ax.text(0.08, 0.65, "2.2 Multi-Scale 2D Wavelet Packet Decomposition (WPD)", fontsize=13, fontweight="bold", color="#1f4e78")
        wpd_math = (
            "Given scaling function $\\phi(t)$ and mother wavelet $\\psi(t)$, the 2D separable wavelet basis produces four\n"
            "fundamental sub-band filtering kernels at each recursive decomposition step:\n\n"
            "    psi^{LL}(x, y) = phi(x)*phi(y)   [Approximation Octave]\n"
            "    psi^{LH}(x, y) = phi(x)*psi(y)   [Horizontal Detail Octave]\n"
            "    psi^{HL}(x, y) = psi(x)*phi(y)   [Vertical Detail Octave]\n"
            "    psi^{HH}(x, y) = psi(x)*psi(y)   [Diagonal Detail Octave]\n\n"
            "Unlike standard DWT which only decomposes the low-frequency approximation sub-band (LL), 2D WPD\n"
            "recursively decomposes all four quadrants up to depth L = 2, producing 4^L = 16 directional sub-bands:\n\n"
            "    W = { W_1, W_2, ..., W_16 }, where each sub-band W_i in R^{B x 64 x 64}\n"
        )
        ax.text(0.08, 0.42, wpd_math, fontsize=9.5, color="#222222", fontfamily="serif", linespacing=1.3)
        
        # Section 2.3
        ax.text(0.08, 0.37, "2.3 Log-Energy Sub-band Transform & CNN Stem", fontsize=13, fontweight="bold", color="#1f4e78")
        stem_math = (
            "Because detail sub-band amplitudes exhibit high dynamic variance, we apply a log-energy transform:\n\n"
            "    W_tilde_i = log( |W_i| + epsilon ),  where epsilon = 1e-8\n\n"
            "The 16 transformed sub-bands are stacked into tensor W_tilde in R^{B x 16 x 64 x 64} and processed by a\n"
            "lightweight 3-stage convolutional neural network stem:\n"
            "  * Stage 1: Conv2D(16 -> 64, k=3, s=2, p=1) -> BN -> ReLU  ==> (B, 64, 32, 32)\n"
            "  * Stage 2: Conv2D(64 -> 128, k=3, s=2, p=1) -> BN -> ReLU ==> (B, 128, 16, 16)\n"
            "  * Stage 3: Conv2D(128 -> 256, k=3, s=2, p=1) -> BN -> ReLU ==> (B, 256, 8, 8)\n"
            "  * GAP(Stage 3) ==> (B, 256) -> Linear + BN + ReLU ==> F_frequency in R^{B x 512}\n"
        )
        ax.text(0.08, 0.12, stem_math, fontsize=9.5, color="#222222", fontfamily="serif", linespacing=1.3)
        
        pdf.savefig(fig)
        plt.close()

        # =====================================================================
        # PAGE 3: FUSION MODULE, LOSS OBJECTIVE & TRAINING MECHANICS
        # =====================================================================
        fig, ax = create_page()
        draw_header(ax, "Section 3: Fusion, Loss Objective & Training", 3)
        
        ax.text(0.08, 0.89, "3.1 Cross-Domain Dual-Attention Fusion Mechanism", fontsize=13, fontweight="bold", color="#1f4e78")
        fuse_math = (
            "To adaptively modulate feature dominance based on image corruption, independent Squeeze-and-Excitation\n"
            "bottlenecks compute dynamic channel-wise gating vectors alpha_s, alpha_f in R^{B x 512}:\n\n"
            "    alpha_spatial   = sigmoid( W_s2 * ReLU( W_s1 * F_spatial ) )\n"
            "    alpha_frequency = sigmoid( W_f2 * ReLU( W_f1 * F_frequency ) )\n\n"
            "where reduction ratio r = 8, W_1 in R^{64 x 512}, W_2 in R^{512 x 64}, and sigmoid maps to (0, 1).\n"
            "Features are recalibrated via element-wise Hadamard product, concatenated, and projected:\n\n"
            "    F_hat_spatial = F_spatial * alpha_spatial,   F_hat_freq = F_frequency * alpha_frequency\n"
            "    F_fused = Dropout_0.3( ReLU( BatchNorm1D( W_fuse * [F_hat_spatial || F_hat_freq] ) ) ) in R^{B x 512}\n\n"
            "The binary classification head computes raw scalar logit y_hat:\n\n"
            "    y_hat = W_cls * Dropout_0.5( F_fused ) + b_cls in R^{B x 1},   P(Fake) = sigmoid( y_hat )\n"
        )
        ax.text(0.08, 0.58, fuse_math, fontsize=9.5, color="#222222", fontfamily="serif", linespacing=1.28)
        
        ax.text(0.08, 0.53, "3.2 Objective Function: Binary Cross-Entropy with Logits", fontsize=13, fontweight="bold", color="#1f4e78")
        loss_math = (
            "For batch size B and binary ground-truth label y_i in {0, 1} (0 = Real, 1 = Fake):\n\n"
            "    L_BCE = - (1 / B) * sum_{i=1}^B [ y_i * log(sigma(y_hat_i)) + (1 - y_i) * log(1 - sigma(y_hat_i)) ]\n\n"
            "Numerically stable implementation evaluated via log-sum-exp formulation:\n\n"
            "    L_i = max(y_hat_i, 0) - y_hat_i * y_i + log( 1 + exp( -|y_hat_i| ) )\n"
        )
        ax.text(0.08, 0.36, loss_math, fontsize=9.5, color="#222222", fontfamily="serif", linespacing=1.3)
        
        ax.text(0.08, 0.31, "3.3 Optimizer & Cosine Annealing Learning Rate Schedule", fontsize=13, fontweight="bold", color="#1f4e78")
        opt_math = (
            "Model parameters are optimized via AdamW (LR = 1e-4, weight decay lambda = 1e-4, beta_1 = 0.9, beta_2 = 0.999).\n"
            "The learning rate follows a 2-epoch linear warmup followed by Cosine Annealing decay down to eta_min = 1e-6:\n\n"
            "    eta_t = eta_min + 0.5 * (eta_base - eta_min) * ( 1 + cos( pi * (t - 2) / 28 ) ),   for epoch t > 2\n"
        )
        ax.text(0.08, 0.16, opt_math, fontsize=9.5, color="#222222", fontfamily="serif", linespacing=1.3)
        
        pdf.savefig(fig)
        plt.close()

        # =====================================================================
        # PAGE 4: DATASET, CONVERGENCE & EMPIRICAL BENCHMARKS
        # =====================================================================
        fig, ax = create_page()
        draw_header(ax, "Section 4: Dataset, Training Dynamics & Results", 4)
        
        ax.text(0.08, 0.89, "4.1 Benchmark Dataset Profile (140k Real & Fake Faces)", fontsize=13, fontweight="bold", color="#1f4e78")
        data_text = (
            "• Authentic Real Class: 70,000 real portraits from Flickr-Faces-HQ (FFHQ) aligned crops.\n"
            "• Synthetic Fake Class: 70,000 photorealistic faces synthesized via StyleGAN / StyleGAN2.\n"
            "• Train / Val / Test Partition: 100,000 train (50k/50k) | 20,000 val (10k/10k) | 20,000 test (10k/10k).\n"
            "• On-the-fly JPEG Augmentation: Simulates quality factors $Q \\in [23, 100]$ during training."
        )
        ax.text(0.08, 0.77, data_text, fontsize=9.5, color="#222222", fontfamily="serif", linespacing=1.3)
        
        ax.text(0.08, 0.72, "4.2 Hardware Infrastructure & Convergence Progress", fontsize=13, fontweight="bold", color="#1f4e78")
        hw_text = (
            "• Host Compute Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), CUDA 12.4.\n"
            "• Optimization Speed: 5.10 batches/sec (163.2 images/sec) with PyTorch Automatic Mixed Precision.\n"
            "• Epoch 1: Train Loss = 0.5120 | Val Loss = 0.4124 | Val Accuracy = 81.45% | Val AUC = 0.8872\n"
            "• Epoch 7 (Optimal Convergence): Train Loss = 0.0531 | Val Loss = 0.0211 | Val Accuracy = 99.29% | Val AUC = 0.9999"
        )
        ax.text(0.08, 0.58, hw_text, fontsize=9.5, color="#222222", fontfamily="serif", linespacing=1.3)
        
        ax.text(0.08, 0.53, "4.3 Empirical Test Evaluation Table (20,000 Unseen Test Images)", fontsize=13, fontweight="bold", color="#1f4e78")
        
        # Professional Table rendering
        table_data = [
            ["Compression Tier", "JPEG Quality", "Accuracy", "AUC-ROC", "Precision", "Recall", "F1-Score"],
            ["Raw (c0)", "100", "99.52%", "1.0000", "0.9911", "99.94%", "0.9952"],
            ["Medium JPEG (c23)", "23", "97.94%", "0.9990", "0.9645", "99.53%", "0.9797"],
            ["Heavy JPEG (c40)", "40", "98.51%", "0.9996", "0.9730", "99.79%", "0.9853"],
        ]
        
        table = ax.table(
            cellText=table_data,
            cellLoc="center",
            loc="center",
            bbox=[0.08, 0.32, 0.84, 0.17]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        # Style table headers and rows
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#1f4e78")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#f9fbfd" if row % 2 == 1 else "#ffffff")
            cell.set_edgecolor("#c8d6e5")
            cell.set_height(0.04)
            
        ax.text(0.08, 0.26, "4.4 Key Forensic Findings", fontsize=12, fontweight="bold", color="#1f4e78")
        findings = (
            "1. Compression Invariance: Minimal performance delta (1.01% drop between c0 and c40).\n"
            "2. Ultra-Low False Dismissal Rate: Recall remains above 99.5% across all compression conditions.\n"
            "3. Discrimination Stability: AUC-ROC is >= 0.9990 on all tiers, proving robust probability calibration."
        )
        ax.text(0.08, 0.13, findings, fontsize=9.5, color="#222222", fontfamily="serif", linespacing=1.3)
        
        pdf.savefig(fig)
        plt.close()

        # =====================================================================
        # PAGE 5: THESIS PLOTS, SCOPE, AND DISCUSSION FOR DEFENSE
        # =====================================================================
        fig, ax = create_page()
        draw_header(ax, "Section 5: Figure Interpretations, Scope & Limitations", 5)
        
        ax.text(0.08, 0.89, "5.1 Thesis Plots & Graphical Evidence (Saved in ./figures)", fontsize=13, fontweight="bold", color="#1f4e78")
        plots_desc = (
            "• roc_curves.png / .pdf: Multi-tier ROC curves for c0, c23, c40 hugging the top-left boundary (AUC: 1.0000,\n"
            "  0.9990, 0.9996), visually confirming threshold independence across compression regimes.\n"
            "• confusion_matrices.png / .pdf: 1x3 normalized matrices showing raw counts and percentages. True Positive\n"
            "  rates exceed 99.5%, with False Negative rates below 0.5% under all test tiers.\n"
            "• score_distributions.png / .pdf: Bimodal probability histograms showing severe polarization: real images\n"
            "  cluster tightly at [0.00, 0.05], while fake faces cluster at [0.95, 1.00] with near-zero density at threshold 0.5.\n"
            "• metrics_comparison_bar.png / .pdf: Grouped bar chart illustrating the horizontal stability of Accuracy,\n"
            "  Precision, Recall, and F1 across c0, c23, and c40."
        )
        ax.text(0.08, 0.70, plots_desc, fontsize=9.5, color="#222222", fontfamily="serif", linespacing=1.35)
        
        ax.text(0.08, 0.65, "5.2 Full-Picture Inference Pipeline (predict.py)", fontsize=13, fontweight="bold", color="#1f4e78")
        pred_desc = (
            "The production inference engine implements an automated two-stage pipeline for real-world full-frame images:\n"
            "1. Automatic Face Localization: Scans input images for face bounding boxes [x, y, w, h].\n"
            "2. Contextual Expansion: Expands boxes by a 20% margin to incorporate hairline and jawline biomarkers.\n"
            "3. Aspect-Ratio Preserving Fallback: If no face box is detected (e.g. tightly cropped portraits), it dynamically\n"
            "   extracts the upper-center square to ensure facial features remain unstretched.\n"
            "4. Visual Annotation: Overlays colored bounding boxes (Green = REAL, Red = FAKE) with confidence scores."
        )
        ax.text(0.08, 0.47, pred_desc, fontsize=9.5, color="#222222", fontfamily="serif", linespacing=1.35)
        
        ax.text(0.08, 0.42, "5.3 Operational Scope, Generalization & Inherent Limitations", fontsize=13, fontweight="bold", color="#1f4e78")
        scope_desc = (
            "Critical talking points for viva defense and thesis limitation sections:\n"
            "• Specialized Forensic Scope: Formulated strictly as a Human Facial Forensic Detector. It is optimized\n"
            "  to detect synthetic human facial artifacts and is intentionally not an open-domain AI illustration detector.\n"
            "• GAN vs. Diffusion Artifacts: Excels on GAN upsampling artifacts (transposed convolutions). Stylized art or\n"
            "  diffusion illustrations lack GAN grid lattice signatures and fall outside the human facial training manifold.\n"
            "• Frame-Level Temporal Constraint: Evaluates single static images; does not model inter-frame optical flow.\n"
            "• Upstream Dependency: In real deployment, detection accuracy relies on upstream face localization."
        )
        ax.text(0.08, 0.23, scope_desc, fontsize=9.5, color="#222222", fontfamily="serif", linespacing=1.35)
        
        # Concluding Statement
        conc_box = dict(boxstyle="round,pad=0.5", facecolor="#eef6fb", edgecolor="#2c5e8a", lw=1)
        conc_text = (
            "Defense Conclusion: The experimental results confirm that combining 2D Wavelet Packet Decomposition\n"
            "with spatial CNN representations via dynamic SE-attention overcomes the compression vulnerability\n"
            "of conventional detectors, retaining 98.51% accuracy on heavily compressed social media deepfakes."
        )
        ax.text(0.08, 0.10, conc_text, fontsize=9, color="#0f2537", fontfamily="serif", fontstyle="italic", bbox=conc_box)
        
        pdf.savefig(fig)
        plt.close()

    print(f"[SUCCESS] PDF successfully generated: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "Thesis_Dossier_Deepfake_Spatio_Frequency.pdf"
    build_thesis_pdf(out)
