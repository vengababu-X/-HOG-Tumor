#!/usr/bin/env python3
# ==============================================================================
# NeuroScan CDS — Academic Manuscript Generator (.docx & .pdf)
# ==============================================================================
# Generates a publication-ready, IEEE/Scopus formatted journal manuscript in
# both Microsoft Word (.docx) and Adobe PDF (.pdf) formats.
# ==============================================================================
from __future__ import annotations

import os
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

_PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = _PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

DOCX_OUT = DOCS_DIR / "NeuroScan_CDS_IEEE_Journal_Manuscript.docx"
PDF_OUT = DOCS_DIR / "NeuroScan_CDS_IEEE_Journal_Manuscript.pdf"

TITLE = "NeuroScan CDS: An Explainable Spatial-Texture Dual-Branch Architecture With 3D Multi-Planar Reconstruction and Out-of-Distribution Safeguards for Brain Tumor Decision Support"
SUBTITLE = "IEEE Transactions on Medical Imaging / IEEE Journal of Biomedical and Health Informatics"

ABSTRACT_TEXT = (
    "Reliable differentiation of intracranial neoplasms from Magnetic Resonance Imaging (MRI) is "
    "clinically crucial for pre-operative neurosurgical planning and radiotherapy triage. However, contemporary "
    "computer-aided diagnosis models frequently suffer from unquantified cross-split dataset leakage, opaque feature representations, "
    "absence of 3D multi-planar context, and catastrophic false-confidence hallucinations on non-cranial or out-of-distribution inputs. "
    "We present NeuroScan CDS, an explainable, dual-branch spatial-texture deep-learning framework designed for clinical decision support. "
    "The primary spatial branch employs an ImageNet-pretrained ResNet-50 backbone augmented with a Convolutional Block Attention Module (CBAM) "
    "to adaptively refine channel-wise and spatial responses at layer 4, projected to a 256-dimensional latent representation. A concurrent texture "
    "branch extracts 2,048-dimensional Histogram of Oriented Gradients (HOG) descriptors, compressed via Principal Component Analysis (PCA) to 288 dimensions "
    "and mapped through a non-linear texture encoder to 512 dimensions. The concatenated 768-dimensional hybrid manifold is optimized using Focal Loss "
    "(γ = 2.0) with class-frequency weighting, decoupled AdamW learning rates, and Test-Time Augmentation (TTA). The architecture integrates an interactive "
    "3D Multi-Planar Reconstruction (MPR) engine across Axial, Coronal, and Sagittal planes, an automated pre-inference Image Quality Assurance (QA) routine, "
    "and a normalized Shannon diagnostic entropy guard to prevent out-of-distribution (OOD) misclassifications. Evaluated on a held-out cohort of N = 1,600 multi-class "
    "MRI scans spanning Glioma, Meningioma, No_Tumor, and Pituitary pathologies, NeuroScan CDS achieved 95.50% test accuracy, 0.9547 macro F1-score, and 0.9919 macro ROC-AUC. "
    "A forensic pixel-hash audit identified 114 cross-split duplicate images in the benchmark; on the verified uncontaminated test subset (N = 1,486), the architecture "
    "demonstrated robust generalization at 95.42% accuracy and 0.9518 macro F1. By combining gradient-stable Grad-CAM++ saliency, neurosurgical subsite triage, and "
    "an immutable JSON-Lines audit trail, NeuroScan CDS establishes an auditable foundation for prospective hospital reader trials."
)

KEYWORDS = [
    "Brain Tumor Classification",
    "Convolutional Block Attention Module (CBAM)",
    "Dual-Branch Neural Network",
    "Grad-CAM++ Explainability",
    "Histogram of Oriented Gradients (HOG)",
    "Out-of-Distribution (OOD) Detection",
    "3D Multi-Planar Reconstruction (MPR)",
    "Clinical Decision Support (CDS)",
    "Medical Image Analysis",
]

# ==============================================================================
# 1. BUILD WORD DOCUMENT (.docx)
# ==============================================================================
def build_docx():
    doc = Document()

    # Set Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Title
    title_p = doc.add_paragraph()
    title_run = title_p.add_run(TITLE)
    title_run.font.name = "Times New Roman"
    title_run.font.size = Pt(18)
    title_run.bold = True
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Subtitle / Target Venue
    sub_p = doc.add_paragraph()
    sub_run = sub_p.add_run(f"Target Submission: {SUBTITLE}")
    sub_run.font.name = "Times New Roman"
    sub_run.font.size = Pt(11)
    sub_run.italic = True
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Author Affiliations
    auth_p = doc.add_paragraph()
    auth_run = auth_p.add_run(
        "Academic Capstone & Clinical Research Consortium\n"
        "Department of Computer Science & Engineering | Department of Radiodiagnosis & Neurosurgery\n"
        "Multispeciality Hospital Collaborative Research Group\n"
    )
    auth_run.font.name = "Times New Roman"
    auth_run.font.size = Pt(10)
    auth_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # Abstract Box
    abs_heading = doc.add_paragraph()
    abs_h_run = abs_heading.add_run("Abstract—")
    abs_h_run.font.name = "Times New Roman"
    abs_h_run.font.size = Pt(10.5)
    abs_h_run.bold = True

    abs_run = abs_heading.add_run(ABSTRACT_TEXT)
    abs_run.font.name = "Times New Roman"
    abs_run.font.size = Pt(10)
    abs_heading.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Keywords
    kw_p = doc.add_paragraph()
    kw_h_run = kw_p.add_run("Index Terms—")
    kw_h_run.font.name = "Times New Roman"
    kw_h_run.font.size = Pt(10)
    kw_h_run.bold = True
    kw_run = kw_p.add_run(", ".join(KEYWORDS))
    kw_run.font.name = "Times New Roman"
    kw_run.font.size = Pt(10)
    kw_p.paragraph_format.space_after = Pt(18)

    # Section I: Introduction
    sec1 = doc.add_paragraph()
    sec1_run = sec1.add_run("I. INTRODUCTION")
    sec1_run.font.name = "Times New Roman"
    sec1_run.font.size = Pt(12)
    sec1_run.bold = True

    p_intro = doc.add_paragraph()
    p_intro.add_run(
        "Primary intracranial neoplasms, including diffuse gliomas, meningiomas, and pituitary neuroendocrine tumors, "
        "constitute one of the most clinically challenging categories of solid tumors. In modern neuro-oncological management, "
        "Magnetic Resonance Imaging (MRI) serves as the definitive non-invasive imaging modality for screening, surgical trajectory planning, "
        "and therapeutic monitoring. Despite advancements in scanner hardware, manual interpretation of cranial scans remains labor-intensive, "
        "inter-observer variable, and heavily dependent on sub-specialty radiological training. Atypical presentations, such as high-grade gliomas "
        "mimicking dural-based meningiomas, frequently pose significant diagnostic dilemmas.\n\n"
        "While deep learning has catalyzed rapid progress in automated medical image analysis, prevailing academic benchmarks suffer from four fundamental limitations:\n"
        "1) Unverified Data Leakage: Loose-slice benchmark datasets frequently contain identical-duplicate images across train and test partitions, artificially inflating reported accuracies.\n"
        "2) Closed-World Hallucination: Standard multi-class classifiers assign high-confidence diagnostic predictions to corrupted, noisy, or completely non-cranial images without out-of-distribution awareness.\n"
        "3) Spatial-Texture Disconnect: Pure convolutional architectures frequently neglect fine-grained subcortical texture orientation, while pure Vision Transformers struggle with inductive bias on small medical sample cohorts.\n"
        "4) 2D Isolation: Radiologists require multi-planar 3D volumetric context (Axial, Coronal, Sagittal) rather than isolated 2D slice predictions.\n\n"
        "To resolve these challenges, this study introduces NeuroScan CDS, a clinically aligned decision-support architecture uniting dual-branch spatial-texture fusion, "
        "rigorous empirical data auditing, 3D Multi-Planar Reconstruction (MPR), and pre-operative neurosurgical anatomical subsite mapping."
    )
    p_intro.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Section II: Proposed Methodology
    sec2 = doc.add_paragraph()
    sec2_run = sec2.add_run("II. SYSTEM ARCHITECTURE & MATHEMATICAL FORMULATION")
    sec2_run.font.name = "Times New Roman"
    sec2_run.font.size = Pt(12)
    sec2_run.bold = True

    p_meth = doc.add_paragraph()
    p_meth.add_run(
        "A. Dual-Branch Hybrid Representation\n"
        "NeuroScan CDS operates on a dual-branch feature extraction pipeline. The spatial branch utilizes a ResNet-50 backbone pretrained on ImageNet. "
        "At layer 4, intermediate feature maps F ∈ R^(C×H×W) (where C = 2048, H = W = 5 at 144×144 resolution) are refined through a Convolutional Block Attention Module (CBAM). "
        "CBAM sequentially computes channel attention M_c(F) through shared multi-layer perceptrons over average and max pooled descriptors, followed by spatial attention M_s(F') "
        "via a 7×7 dilated convolution. The spatial embedding is pooled and projected to a 256-dimensional vector.\n\n"
        "Concurrently, the texture branch extracts handcrafted Histogram of Oriented Gradients (HOG) over the greyscale transformed input using 8 orientation bins, "
        "16×16 pixel cells, and 2×2 cell blocks, yielding a raw 2,048-dimensional texture descriptor. To prevent curse-of-dimensionality degradation, the vector is projected "
        "via Principal Component Analysis (PCA) to 288 orthogonal components, subsequently passed through a multi-layer texture encoder to produce a 512-dimensional embedding.\n\n"
        "B. Fusion and Non-Inplace autograd Traversal\n"
        "The deep spatial vector (256-d) and PCA-HOG texture vector (512-d) are concatenated into a unified 768-dimensional hybrid manifold. "
        "Crucially, all intermediate activations maintain strict non-inplace semantics (inplace=False) to ensure unobstructed autograd graph traversal during Grad-CAM++ saliency computation.\n\n"
        "C. Optimization via Multi-Class Focal Loss\n"
        "To penalize hard misclassifications and mitigate class frequency imbalance, the network is trained by minimizing Focal Loss:\n"
        "   L_focal = - ∑ [α_i * (1 - p_i)^γ * log(p_i)]\n"
        "where focusing parameter γ = 2.0, optimized using decoupled AdamW learning rates (10^-5 for backbone, 10^-4 for classification head) and Cosine Annealing."
    )
    p_meth.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Section III: Experimental Results Table
    sec3 = doc.add_paragraph()
    sec3_run = sec3.add_run("III. EXPERIMENTAL RESULTS & ABLATION ANALYSIS")
    sec3_run.font.name = "Times New Roman"
    sec3_run.font.size = Pt(12)
    sec3_run.bold = True

    doc.add_paragraph("TABLE I. HELD-OUT PER-CLASS DIAGNOSTIC PERFORMANCE (N = 1,600 EVALUATION COHORT)")
    table = doc.add_table(rows=6, cols=6)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Pathology Class", "Test Support", "Precision", "Recall", "F1-Score", "One-vs-Rest AUC"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True

    data_rows = [
        ["Glioma", "400", "0.9831", "0.8750", "0.9259", "0.9786"],
        ["Meningioma", "400", "0.8998", "0.9650", "0.9312", "0.9914"],
        ["No_Tumor", "400", "0.9707", "0.9950", "0.9827", "0.9993"],
        ["Pituitary", "400", "0.9728", "0.9850", "0.9789", "0.9994"],
        ["Macro Average / Overall", "1,600", "0.9566", "0.9550", "0.9547", "0.9919"],
    ]
    for row_idx, r_data in enumerate(data_rows, start=1):
        for col_idx, text in enumerate(r_data):
            table.cell(row_idx, col_idx).text = text

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Section IV: Data Leakage & Saliency Audit
    sec4 = doc.add_paragraph()
    sec4_run = sec4.add_run("IV. FORENSIC AUDITING: DATA LEAKAGE & SALIENCY BOUNDARIES")
    sec4_run.font.name = "Times New Roman"
    sec4_run.font.size = Pt(12)
    sec4_run.bold = True

    p_audit = doc.add_paragraph()
    p_audit.add_run(
        "A forensic pixel-level SHA-1 hash analysis across data/Training and data/Testing partitions revealed that 114 out of 1,600 test images (7.12%) "
        "were byte-identical duplicates of training images, with 100 duplicate instances concentrated in the Meningioma class (25.0% of that cohort). "
        "When re-evaluating strictly on the verified leak-free subset (N = 1,486), NeuroScan CDS retained an accuracy of 95.42% and macro F1 of 0.9518, "
        "confirming that the core architectural discriminative power survives rigorous data cleaning.\n\n"
        "Furthermore, quantitative evaluation of Grad-CAM++ activation maps indicated that at 144×144 resolution, layer 4 spatial grids (5×5) yield a mean "
        "intensity separation of |Δ| = 0.025 compared to randomized intracranial patches. This demonstrates that while Grad-CAM++ provides effective "
        "coarse visual verification of the lesion locus, it cannot substitute for dedicated voxel-level segmentation masks."
    )
    p_audit.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Section V: Conclusion
    sec5 = doc.add_paragraph()
    sec5_run = sec5.add_run("V. CONCLUSION & CLINICAL TRIAL PROTOCOL")
    sec5_run.font.name = "Times New Roman"
    sec5_run.font.size = Pt(12)
    sec5_run.bold = True

    p_concl = doc.add_paragraph()
    p_concl.add_run(
        "NeuroScan CDS demonstrates that uniting deep spatial attention (ResNet-50+CBAM) with handcrafted texture features (HOG-PCA) achieves 95.50% multi-class "
        "diagnostic consistency on intracranial MRI. By coupling transparent audit reporting, out-of-distribution Shannon entropy safeguards, "
        "3D Multi-Planar Reconstruction (MPR), and pre-operative neurosurgical subsite planning, this platform transitions computer-aided neuro-oncology "
        "from opaque black-box heuristics toward auditable clinical reality. Future work focuses on an active N = 100 prospective multi-reader hospital pilot trial "
        "under an Institutional Review Board (IRB) framework."
    )
    p_concl.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    doc.save(str(DOCX_OUT))
    print(f"Generated DOCX: {DOCX_OUT}")


# ==============================================================================
# 2. BUILD PDF DOCUMENT (.pdf) VIA REPORTLAB
# ==============================================================================
def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_OUT),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()

    # Custom Academic Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f2a4a"),
        spaceAfter=6,
    )
    sub_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=12,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "DocHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0f2a4a"),
        spaceBefore=10,
        spaceAfter=4,
    )
    abstract_style = ParagraphStyle(
        "DocAbstract",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        alignment=TA_JUSTIFY,
        leftIndent=18,
        rightIndent=18,
        spaceAfter=8,
    )

    story = []
    story.append(Paragraph(TITLE, title_style))
    story.append(Paragraph(f"Target Venue: {SUBTITLE}", sub_style))
    story.append(Paragraph("<b>Academic Capstone & Clinical Research Consortium</b><br/>Department of Computer Science & Engineering | Department of Radiodiagnosis & Neurosurgery<br/>Multispeciality Hospital Collaborative Research Group", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=10))

    # Abstract
    story.append(Paragraph(f"<b><i>Abstract</i>—{ABSTRACT_TEXT}</b>", abstract_style))
    story.append(Paragraph(f"<b><i>Index Terms</i>—</b>{', '.join(KEYWORDS)}", abstract_style))
    story.append(Spacer(1, 8))

    # Section I
    story.append(Paragraph("I. INTRODUCTION", heading_style))
    story.append(Paragraph(
        "Primary intracranial neoplasms, including diffuse gliomas, meningiomas, and pituitary neuroendocrine tumors, "
        "represent high-stakes pathologies requiring rapid, accurate radiological characterization. In modern neuro-oncological management, "
        "Magnetic Resonance Imaging (MRI) serves as the primary non-invasive modality. However, manual interpretation is constrained by reader fatigue, "
        "inter-observer variability, and overlapping radiological features. While machine learning offers high classification consistency, existing "
        "paradigms frequently exhibit four major vulnerabilities: (1) unverified cross-split dataset leakage, (2) closed-world hallucinations on non-cranial "
        "inputs, (3) neglect of handcrafted subcortical texture descriptors, and (4) absence of tri-planar 3D surgical context.",
        body_style
    ))

    # Section II
    story.append(Paragraph("II. SYSTEM ARCHITECTURE & PROPOSED METHODOLOGY", heading_style))
    story.append(Paragraph(
        "NeuroScan CDS unites a dual-branch hybrid network: Branch A combines an ImageNet-pretrained ResNet-50 backbone with a Convolutional Block "
        "Attention Module (CBAM) at layer 4 (256-d embedding). Branch B computes 2,048-d Histogram of Oriented Gradients (HOG) compressed via PCA to 288 dimensions "
        "and embedded through a non-linear texture MLP (512-d). The concatenated 768-d hybrid manifold is optimized using Focal Loss (γ = 2.0), decoupled AdamW rates, "
        "and Test-Time Augmentation (TTA). The architecture integrates a 3D Multi-Planar Reconstruction (MPR) engine and a normalized Shannon diagnostic entropy guard.",
        body_style
    ))

    # Table
    story.append(Paragraph("TABLE I. HELD-OUT PER-CLASS DIAGNOSTIC PERFORMANCE (N = 1,600 COHORT)", heading_style))
    table_data = [
        ["Pathology Class", "Support", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        ["Glioma", "400", "0.9831", "0.8750", "0.9259", "0.9786"],
        ["Meningioma", "400", "0.8998", "0.9650", "0.9312", "0.9914"],
        ["No_Tumor", "400", "0.9707", "0.9950", "0.9827", "0.9993"],
        ["Pituitary", "400", "0.9728", "0.9850", "0.9789", "0.9994"],
        ["Macro Average", "1,600", "0.9566", "0.9550", "0.9547", "0.9919"],
    ]
    t = Table(table_data, colWidths=[120, 60, 75, 75, 75, 75])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2a4a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # Section III & IV
    story.append(Paragraph("III. FORENSIC AUDITING & CLINICAL INTEGRATION", heading_style))
    story.append(Paragraph(
        "A forensic pixel-level SHA-1 hash analysis revealed 114 cross-split duplicate images in the public benchmark (7.12% duplicate exposure). "
        "Evaluating strictly on the verified uncontaminated test subset (N = 1,486), NeuroScan CDS retained 95.42% accuracy and 0.9518 macro F1. "
        "The system incorporates an on-premise PACS DICOM bridge, a pre-operative neurosurgical subsite planner, and an automated Institutional Review "
        "Board (IRB) clinical trial dossier for prospective N = 100 multispeciality hospital validation.",
        body_style
    ))

    story.append(Paragraph("IV. CONCLUSION", heading_style))
    story.append(Paragraph(
        "NeuroScan CDS bridges the chasm between academic machine learning benchmarks and clinical medical reality. Through rigorous data auditing, "
        "3D Multi-Planar Reconstruction, out-of-distribution safeguards, and transparent surgical planning, the platform delivers an auditable "
        "foundation for clinical adoption.",
        body_style
    ))

    doc.build(story)
    print(f"Generated PDF: {PDF_OUT}")

if __name__ == "__main__":
    build_docx()
    build_pdf()

