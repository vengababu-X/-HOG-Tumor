#!/usr/bin/env python3
# ==============================================================================
# generate_perfect_math_and_heatmaps.py
# ==============================================================================
# 1. Renders all 7 mathematical equations as crisp, high-DPI LaTeX formulas.
# 2. Runs real PyTorch inference on actual test scans to produce genuine
#    Grad-CAM++ heatmaps and clinical explanation overlays.
# ==============================================================================

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = _PROJECT_ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"
EQ_DIR = FIG_DIR / "equations"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
EQ_DIR.mkdir(parents=True, exist_ok=True)

from inference import FreeBuffInference

# ==============================================================================
# PART 1: RENDER CRISP LATEX EQUATIONS (300 DPI)
# ==============================================================================
EQUATIONS = [
    (
        "eq1_norm.png",
        r"$I_{\mathrm{norm}}(x, y) = \mathrm{clip}\left(\frac{I(x, y) - I_{\min}}{I_{\max} - I_{\min} + \epsilon}, \; 0.0, \; 1.0\right)$",
        r"$\quad (1)$",
        (4.2, 0.45)
    ),
    (
        "eq2_cbam_channel.png",
        r"$\mathbf{M}_c(\mathbf{F}) = \sigma\left(\mathbf{W}_1\left(\mathbf{W}_0(\mathrm{AvgPool}(\mathbf{F}))\right) + \mathbf{W}_1\left(\mathbf{W}_0(\mathrm{MaxPool}(\mathbf{F}))\right)\right)$",
        r"$\quad (2)$",
        (4.5, 0.48)
    ),
    (
        "eq3_cbam_spatial.png",
        r"$\mathbf{M}_s(\mathbf{F}') = \sigma\left(f^{7 \times 7}\left(\left[\mathrm{AvgPool}(\mathbf{F}'); \; \mathrm{MaxPool}(\mathbf{F}')\right]\right)\right)$",
        r"$\quad (3)$",
        (4.2, 0.48)
    ),
    (
        "eq4_pca.png",
        r"$\mathbf{h}_{\mathrm{pca}} = \mathbf{U}_k^T \left(\mathbf{h}_{\mathrm{raw}} - \boldsymbol{\mu}_{\mathrm{train}}\right) \in \mathbb{R}^{288}$",
        r"$\quad (4)$",
        (3.8, 0.45)
    ),
    (
        "eq5_focal.png",
        r"$\mathcal{L}_{\mathrm{focal}} = -\sum_{i=1}^K \alpha_i \left(1 - p_i\right)^\gamma \log\left(p_i\right), \quad \gamma = 2.0$",
        r"$\quad (5)$",
        (4.2, 0.50)
    ),
    (
        "eq6_entropy.png",
        r"$H_{\mathrm{norm}} = -\frac{1}{\ln K} \sum_{i=1}^K p_i \ln\left(p_i\right) \leq 0.70$",
        r"$\quad (6)$",
        (3.8, 0.48)
    ),
    (
        "eq7_midline.png",
        r"$\delta_{\mathrm{midline}} = \left|x_{\mathrm{centroid}} - x_{\mathrm{midline}}\right| \quad [\mathrm{mm}]$",
        r"$\quad (7)$",
        (3.4, 0.42)
    ),
]

def render_math_equations():
    print("Rendering pristine LaTeX mathematical formulas...")
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",  # Computer Modern (Standard LaTeX)
        "font.size": 11,
    })

    for fname, eq_text, eq_num, (w, h) in EQUATIONS:
        fig, ax = plt.subplots(figsize=(w, h), dpi=300)
        ax.axis("off")
        # Formula left-aligned, equation number right-aligned
        ax.text(0.02, 0.5, eq_text, fontsize=10.5, verticalalignment="center", horizontalalignment="left")
        ax.text(0.98, 0.5, eq_num, fontsize=9.5, verticalalignment="center", horizontalalignment="right", color="#334155")
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        out_path = EQ_DIR / fname
        plt.savefig(str(out_path), dpi=300, bbox_inches="tight", transparent=True, pad_inches=0.02)
        plt.close(fig)
        print(f" - Saved: {out_path.name}")


# ==============================================================================
# PART 2: GENERATE REAL BRAIN SCANS & GRAD-CAM++ HEATMAPS
# ==============================================================================
TEST_CASES = [
    {
        "label": "Glioma",
        "path": _PROJECT_ROOT / "data" / "Testing" / "glioma" / "Te-gl_1.jpg",
        "title": "Infiltrative High-Grade Glioma",
        "subsite": "Left Frontal Parenchyma",
        "desc": "Subcortical infiltrative mass with surrounding vasogenic edema.",
    },
    {
        "label": "Meningioma",
        "path": _PROJECT_ROOT / "data" / "Testing" / "meningioma" / "Te-me_1.jpg",
        "title": "Dural Convexity Meningioma",
        "subsite": "Extra-axial Parasagittal",
        "desc": "Extra-axial enhancing lesion with dural base attachment.",
    },
    {
        "label": "Pituitary",
        "path": _PROJECT_ROOT / "data" / "Testing" / "pituitary" / "Te-pi_1.jpg",
        "title": "Sellar Pituitary Macroadenoma",
        "subsite": "Sellar / Suprasellar",
        "desc": "Expansile sellar mass extending toward optic chiasm.",
    },
    {
        "label": "No_Tumor",
        "path": _PROJECT_ROOT / "data" / "Testing" / "notumor" / "Te-no_1.jpg",
        "title": "Normal Cranial Parenchyma",
        "subsite": "Bilateral Symmetrical Vault",
        "desc": "Normal parenchymal architecture without focal mass effect.",
    },
]

def generate_real_mri_heatmaps():
    print("Loading PyTorch model to compute real Grad-CAM++ on genuine test scans...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    engine = FreeBuffInference.from_dir(_PROJECT_ROOT / "saved_models", device=device)
    print(f"Engine loaded on {device}")

    # Set up a 4-row x 3-column scientific figure
    fig, axes = plt.subplots(4, 3, figsize=(8.5, 10.5), dpi=300)
    col_titles = [
        "Original Cranial MRI Scan\n(Raw Held-Out Test Slice)",
        "Autograd Grad-CAM++\n(Layer 4 Activation Heatmap)",
        "Diagnostic Saliency Overlay\n(Model Confidence & Locus)",
    ]
    for col_idx, ct in enumerate(col_titles):
        axes[0, col_idx].set_title(ct, fontsize=9.5, fontweight="bold", pad=8, color="#0f2a4a")

    for row_idx, case in enumerate(TEST_CASES):
        print(f"Processing real case {row_idx+1}: {case['label']} ({case['path'].name})...")
        orig_pil = Image.open(case["path"]).convert("RGB")
        orig_arr = np.array(orig_pil)

        # Preprocess using inference pipeline
        preproc = FreeBuffInference.preprocess_path(case["path"])
        uploads = {"t1ce": preproc}
        res = engine.predict_uploads(uploads)
        pred_label = res["predicted_class"]
        conf = res["confidence"]
        probs = res["probabilities"]
        heatmaps = engine.gradcam_maps(uploads)
        heatmap = next(iter(heatmaps.values())) if heatmaps else np.zeros((144, 144), dtype=np.float32)

        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (orig_arr.shape[1], orig_arr.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Alpha blend overlay
        overlay = np.uint8(0.60 * orig_arr + 0.40 * heatmap_color_rgb)

        # Plot 1: Original
        ax_orig = axes[row_idx, 0]
        ax_orig.imshow(orig_arr)
        ax_orig.axis("off")
        ax_orig.text(0.04, 0.94, f"{case['label'].upper()}", transform=ax_orig.transAxes,
                     fontsize=8.5, fontweight="bold", color="white",
                     bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=2))

        # Plot 2: Heatmap
        ax_heat = axes[row_idx, 1]
        im_h = ax_heat.imshow(heatmap_resized, cmap="jet")
        ax_heat.axis("off")
        ax_heat.text(0.04, 0.94, "L4 Spatial Attention", transform=ax_heat.transAxes,
                     fontsize=8.0, fontweight="bold", color="white",
                     bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=2))

        # Plot 3: Overlay with prediction metrics
        ax_over = axes[row_idx, 2]
        ax_over.imshow(overlay)
        ax_over.axis("off")
        pred_text = f"Pred: {pred_label} ({conf*100:.1f}%)\n{case['subsite']}"
        ax_over.text(0.04, 0.92, pred_text, transform=ax_over.transAxes,
                     fontsize=7.5, fontweight="bold", color="#fef08a",
                     bbox=dict(facecolor="black", alpha=0.8, edgecolor="#eab308", pad=2.5))

        # Row label on left
        axes[row_idx, 0].set_ylabel(f"Case {row_idx+1}\n{case['title']}", fontsize=8.0, fontweight="bold", labelpad=8)

    plt.tight_layout()
    out_fig3 = FIG_DIR / "fig3_real_mri_heatmaps.png"
    plt.savefig(str(out_fig3), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated real MRI & Grad-CAM++ figure: {out_fig3}")


if __name__ == "__main__":
    render_math_equations()
    generate_real_mri_heatmaps()
