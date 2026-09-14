#!/usr/bin/env python3
# ==============================================================================
# FreeBuff v4 — Utility Modules: ReportLab PDF Generation, Disclaimers, Helpers
# ==============================================================================
# Provides hospital-grade PDF report generation utilities, regulatory/medical
# disclaimers, and clinical report assembly from inference results.
# ==============================================================================
from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / headless env
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak,
    KeepTogether,
    HRFlowable,
)

from config import (
    CLASSES,
    NUM_CLASSES,
    INPUT_SIZE,
    REGULATORY_DISCLAIMER,
    REGULATORY_SHORTHAND,
    GRADCAM_ALPHA,
    GRADCAM_COLORMAP,
    GRADCAM_INTERPOLATION_MODE,
)

logger = logging.getLogger("freebuff_v4.utils")


# ==============================================================================
# PDF STYLES (SINGLE SOURCE OF TRUTH)
# ==============================================================================
def build_pdf_styles() -> Dict[str, ParagraphStyle]:
    """
    Build and return a dictionary of ReportLab paragraph styles used across
    PDF reports.

    Returns:
        Dict from style name -> ParagraphStyle instance.
    """
    styles = getSampleStyleSheet()
    base_dir = os.path.dirname(__file__)

    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0B3D91"),
            alignment=TA_CENTER,
            spaceAfter=12,
            fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#2C3E50"),
            alignment=TA_CENTER,
            spaceAfter=8,
            fontName="Helvetica-Oblique",
        ),
        "heading1": ParagraphStyle(
            "Heading1Custom",
            parent=styles["Heading1"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0B3D91"),
            spaceBefore=12,
            spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "heading2": ParagraphStyle(
            "Heading2Custom",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#2C3E50"),
            spaceBefore=8,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.black,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "body_small": ParagraphStyle(
            "BodySmall",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#34495E"),
            alignment=TA_JUSTIFY,
            spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=3,
        ),
        "disclaimer": ParagraphStyle(
            "DisclaimerStyle",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#7B241C"),
            backColor=colors.HexColor("#FDEDEC"),
            borderColor=colors.HexColor("#E74C3C"),
            borderWidth=1,
            borderPadding=8,
            alignment=TA_JUSTIFY,
            spaceBefore=8,
            spaceAfter=8,
        ),
        "footer": ParagraphStyle(
            "FooterStyle",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.white,
            fontName="Helvetica-Bold",
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
        ),
        "patient_label": ParagraphStyle(
            "PatientLabel",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#2C3E50"),
        ),
    }


STYLES = build_pdf_styles()


# ==============================================================================
# IMAGE PROCESSING UTILITIES FOR PDF EMBEDDING
# ==============================================================================
def pil_to_reportlab_image(pil_img: PILImage.Image, max_width: float = 6.0 * inch,
                           max_height: float = 4.5 * inch) -> RLImage:
    """
    Convert a PIL Image to a reportlab Image flowable, scaling to fit within
    max_width and max_height while preserving aspect ratio.

    Args:
        pil_img: PIL Image (RGB/RGBA).
        max_width: Maximum display width in points.
        max_height: Maximum display height in points.

    Returns:
        RLImage instance ready for PDF canvas drawing.
    """
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        background = PILImage.new("RGB", pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.convert("RGBA") if pil_img.mode in ("RGBA", "LA") else pil_img)
        pil_img = background
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    width, height = pil_img.size
    scale = min(max_width / width, max_height / height, 1.0)
    new_width = width * scale
    new_height = height * scale

    # Convert to PNG bytes and feed to RLImage
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    rl_img = RLImage(buf, width=new_width, height=new_height)
    return rl_img


def numpy_to_pil_uint8(arr: np.ndarray) -> PILImage.Image:
    """
    Convert a numpy float array (H, W, 3) in [0,1] or [0,255] to a PIL RGB
    image (uint8).
    """
    arr = np.asarray(arr)
    if arr.max() <= 1.0:
        arr = (arr * 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return PILImage.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        return PILImage.fromarray(arr, mode="RGB" if arr.shape[2] == 3 else "RGBA")
    raise ValueError(f"Cannot convert array of shape {arr.shape} to PIL image.")


# ==============================================================================
# STATISTICALLY NORMALIZED PLOT GENERATOR (USED IN PDF REPORTS)
# ==============================================================================
def build_confidence_bar_chart(
    class_names: List[str],
    confidences: np.ndarray,
    cmap_name: str = "Blues",
    figsize: Tuple[int, int] = (8, 4),
) -> PILImage.Image:
    """
    Build a matplotlib bar chart of per-class prediction confidence and return
    a PIL image suitable for embedding in PDF reports.

    Args:
        class_names: Class name strings.
        confidences: Confidence values (len = len(class_names)).
        cmap_name: Matplotlib colormap name for bar coloring.
        figsize: Figure size in inches.

    Returns:
        PIL RGB image.
    """
    confidences = np.asarray(confidences, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.get_cmap(cmap_name)(np.linspace(0.3, 0.9, len(class_names)))
    x = np.arange(len(class_names))
    bars = ax.bar(x, confidences, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1.0)
    ax.set_title("Prediction Confidence by Class")
    for bar, val in zip(bars, confidences):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf)


def build_confusion_matrix_table(
    cmatrix: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
) -> Table:
    """
    Build a ReportLab Table from a confusion matrix (counts or normalized).

    Args:
        cmatrix: (C, C) confusion matrix array.
        class_names: Class names for row/col headers.
        normalize: If True, format cells as percentages with 1 decimal.

    Returns:
        ReportLab Table flowable.
    """
    n = len(class_names)
    header = [Paragraph("True \\ Pred", STYLES["table_header"])]
    for name in class_names:
        header.append(Paragraph(name, STYLES["table_header"]))
    header.append(Paragraph("Row Sum", STYLES["table_header"]))

    data = [header]
    for i, true_name in enumerate(class_names):
        row = [Paragraph(true_name, STYLES["table_cell"])]
        row_sum = 0
        for j in range(n):
            val = cmatrix[i, j]
            if normalize:
                total = cmatrix[i].sum()
                display = val / total if total > 0 else 0.0
                txt = f"{display:.1%}"
            else:
                txt = f"{int(val)}"
            row.append(Paragraph(txt, STYLES["table_cell"]))
            row_sum += val
        if normalize:
            row_sum = cmatrix[i].sum()
            row.append(Paragraph(f"{row_sum:.0f}" if row_sum > 0 else "0", STYLES["table_cell"]))
        else:
            row.append(Paragraph(f"{int(row_sum)}", STYLES["table_cell"]))
        data.append(row)

    # Column sums footer
    col_sums = cmatrix.sum(axis=0)
    footer = [Paragraph("Col Sum", STYLES["table_cell"])]
    for j in range(n):
        if normalize:
            footer.append(Paragraph(f"{col_sums[j]/col_sums.sum():.1%}" if col_sums.sum()>0 else "0%",
                                    STYLES["table_cell"]))
        else:
            footer.append(Paragraph(f"{int(col_sums[j])}", STYLES["table_cell"]))
    footer.append(Paragraph(f"{int(col_sums.sum())}", STYLES["table_cell"]))
    data.append(footer)

    col_widths = [1.8 * inch] + [1.0 * inch] * n + [1.0 * inch]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EBF5FB")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#0B3D91")),
    ]
    t.setStyle(TableStyle(style_cmds))
    return t


# ==============================================================================
# PDF REPORT BUILDER (CORE GENERATOR)
# ==============================================================================
class PDFReportGenerator:
    """
    Hospital-grade PDF report generator using ReportLab.

    Assembles patient metrics, confidence spectra, Grad-CAM++ overlays,
    WHO staging information, clinical reasoning text, and disclaimers into a
    clean, downloadable document.
    """

    def __init__(
        self,
        report_dir: Optional[Union[str, Path]] = None,
        styles: Optional[Dict[str, ParagraphStyle]] = None,
    ) -> None:
        self.report_dir = Path(report_dir) if report_dir else Path("clinical_reports/pdf_reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.styles = styles or STYLES
        self.elements: List[Any] = []

    def _add_spacer(self, height: float = 6) -> None:
        self.elements.append(Spacer(1, height))

    def _add_horizontal_rule(self, color: colors.Color = colors.HexColor("#0B3D91"),
                             thickness: int = 1) -> None:
        self.elements.append(HRFlowable(width="100%", thickness=thickness, color=color,
                                        spaceBefore=4, spaceAfter=4))

    # ------------------------------------------------------------------
    def add_cover_page(
        self,
        patient_id: Optional[str] = None,
        patient_name: Optional[str] = None,
        exam_date: Optional[str] = None,
        modality: str = "MRI Brain",
        scan_description: str = "Brain MRI — T1 / T2 / FLAIR (classification slice)",
        created_by: str = "FreeBuff v4 Research System",
    ) -> None:
        """
        Add a cover page with hospital-style header, patient info block, and
        report metadata.

        Args:
            patient_id: Optional patient identifier.
            patient_name: Optional patient name (de-identified recommended).
            exam_date: Date string for the exam.
            modality: Imaging modality description.
            scan_description: Free-text scan description.
            created_by: Generating system name.
        """
        self.elements = []
        self._add_horizontal_rule()
        self.elements.append(Paragraph("FreeBuff v4", self.styles["title"]))
        self.elements.append(Paragraph(
            "Multi-Modal Brain Tumor Classification & Clinical Decision Support",
            self.styles["subtitle"]))
        self._add_horizontal_rule()
        self._add_spacer(10)

        self.elements.append(Paragraph("Diagnostic Imaging Classification Report", self.styles["heading1"]))
        self._add_spacer(6)

        meta_table_data = [
            [Paragraph("<b>Generated By</b>", self.styles["patient_label"]),
             Paragraph(created_by, self.styles["body"])],
            [Paragraph("<b>Report Date/Time</b>", self.styles["patient_label"]),
             Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.styles["body"])],
            [Paragraph("<b>Modality</b>", self.styles["patient_label"]),
             Paragraph(modality, self.styles["body"])],
            [Paragraph("<b>Scan Description</b>", self.styles["patient_label"]),
             Paragraph(scan_description, self.styles["body"])],
        ]
        if patient_id:
            meta_table_data.append(
                [Paragraph("<b>Patient ID</b>", self.styles["patient_label"]),
                 Paragraph(patient_id, self.styles["body"])])
        if patient_name:
            meta_table_data.append(
                [Paragraph("<b>Patient Name</b>", self.styles["patient_label"]),
                 Paragraph(patient_name, self.styles["body"])])
        if exam_date:
            meta_table_data.append(
                [Paragraph("<b>Exam Date</b>", self.styles["patient_label"]),
                 Paragraph(exam_date, self.styles["body"])])

        t = Table(meta_table_data, colWidths=[2.2 * inch, 4.3 * inch])
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EBF5FB")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5D8DC")),
        ]))
        self.elements.append(t)
        self._add_spacer(10)
        self.elements.append(PageBreak())

    # ------------------------------------------------------------------
    def add_prediction_summary(
        self,
        prediction: str,
        confidence: float,
        probabilities: Optional[Sequence[float]] = None,
        logits: Optional[Sequence[float]] = None,
        pred_class_idx: Optional[int] = None,
        image_source: Optional[str] = None,
        tta_used: bool = False,
    ) -> None:
        """
        Add a detailed classification summary block.

        Args:
            prediction: Predicted class name.
            confidence: Maximum softmax probability in [0, 1].
            probabilities: Per-class probability list (length = NUM_CLASSES).
            logits: Per-class logit values (optional).
            pred_class_idx: Predicted class index.
            image_source: Source image path or identifier.
            tta_used: Whether TTA was applied.
        """
        self.elements.append(Paragraph("CLASSIFICATION RESULTS", self.styles["heading1"]))
        self._add_horizontal_rule()

        confidence_band = ("High" if confidence >= 0.90 else
                           "Moderate" if confidence >= 0.75 else
                           "Low" if confidence >= 0.60 else "Very Low")

        result_lines = [
            f"<b>Predicted Class:</b> {prediction}",
            f"<b>Prediction Confidence:</b> {confidence:.4f} ({confidence:.2%}) — Confidence Band: {confidence_band}",
            f"<b>Predicted Class Index:</b> {pred_class_idx}",
            f"<b>Test-Time Augmentation (TTA) Used:</b> {tta_used}",
        ]
        if image_source:
            result_lines.append(f"<b>Source Image:</b> {image_source}")

        for line in result_lines:
            self.elements.append(Paragraph(line, self.styles["body"]))

        # Probabilities table
        self.elements.append(Spacer(1, 6))
        self.elements.append(Paragraph("Per-Class Probability Distribution", self.styles["heading2"]))
        prob_header = [Paragraph("Class", self.styles["table_header"]),
                       Paragraph("Probability", self.styles["table_header"]),
                       Paragraph("Percentage", self.styles["table_header"])]
        prob_data = [prob_header]
        if probabilities is None:
            probabilities = [0.0] * NUM_CLASSES
        for idx, (cls, prob) in enumerate(zip(CLASSES, probabilities)):
            prob_data.append([
                Paragraph(cls, self.styles["table_cell"]),
                Paragraph(f"{prob:.4f}", self.styles["table_cell"]),
                Paragraph(f"{prob:.2%}", self.styles["table_cell"]),
            ])
        prob_table = Table(prob_data, colWidths=[2.2*inch, 1.8*inch, 1.8*inch])
        prob_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EBF5FB")]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        self.elements.append(prob_table)
        self._add_spacer(8)

        # Logits table (if provided)
        if logits is not None and len(logits) == NUM_CLASSES:
            self.elements.append(Paragraph("Raw Logits (Pre-Softmax)", self.styles["heading2"]))
            logit_data = [[Paragraph("Class", self.styles["table_header"]),
                           Paragraph("Logit", self.styles["table_header"])]]
            for cls, logit in zip(CLASSES, logits):
                logit_data.append([Paragraph(cls, self.styles["table_cell"]),
                                   Paragraph(f"{logit:.4f}", self.styles["table_cell"])])
            logit_table = Table(logit_data, colWidths=[2.5*inch, 2.5*inch])
            logit_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            self.elements.append(logit_table)
            self._add_spacer(8)

        self.elements.append(PageBreak())

    # ------------------------------------------------------------------
    def add_confidence_chart(self, probabilities: Sequence[float]) -> None:
        """
        Add a bar chart image of per-class confidence into the report.

        Args:
            probabilities: Per-class probability sequence.
        """
        self.elements.append(Paragraph("CONFIDENCE SPECTRUM VISUALIZATION", self.styles["heading1"]))
        self._add_horizontal_rule()
        chart_img = build_confidence_bar_chart(CLASSES, np.asarray(probabilities, dtype=float))
        self.elements.append(pil_to_reportlab_image(chart_img, max_width=6.2*inch, max_height=3.8*inch))
        self._add_spacer(8)
        self.elements.append(Paragraph(
            "The bar chart above displays the predicted probability (confidence) assigned to each "
            "tumour class by the ensemble decision pathway. Higher bars indicate stronger model "
            "assertion for the corresponding diagnostic category.", self.styles["body_small"]))
        self._add_spacer(8)

    # ------------------------------------------------------------------
    def add_gradcam_overlay(self, gradcam_image: Optional[PILImage.Image] = None,
                            gradcam_path: Optional[Union[str, Path]] = None,
                            caption: Optional[str] = None) -> None:
        """
        Embed a Grad-CAM++ heatmap overlay into the PDF report.

        Args:
            gradcam_image: PIL Image of the overlay (if available in memory).
            gradcam_path: Path to a saved overlay image file.
            caption: Optional caption text below the image.
        """
        self.elements.append(Paragraph("GRAD-CAM++ EXPLAINABILITY HEATMAP", self.styles["heading1"]))
        self._add_horizontal_rule()

        if gradcam_path is not None:
            path = Path(gradcam_path)
            if path.exists():
                pil_img = PILImage.open(path)
                self.elements.append(pil_to_reportlab_image(pil_img, max_width=6.2*inch, max_height=4.5*inch))
            else:
                self.elements.append(Paragraph(f"[Grad-CAM overlay not found at: {path}]", self.styles["body_small"]))
        elif gradcam_image is not None:
            self.elements.append(pil_to_reportlab_image(gradcam_image, max_width=6.2*inch, max_height=4.5*inch))
        else:
            self.elements.append(Paragraph("[No Grad-CAM++ overlay available for this case.]",
                                            self.styles["body_small"]))

        if caption:
            self.elements.append(Paragraph(caption, self.styles["body_small"]))
        self._add_spacer(8)
        self.elements.append(Paragraph(
            "The heatmap above visualizes the image regions most influential in the model's "
            "classification decision (Grad-CAM++). Warmer colors indicate higher importance. "
            "This visualization is intended to assist clinician review and is NOT a standalone "
            "diagnostic indicator.", self.styles["body_small"]))
        self._add_spacer(8)

    # ------------------------------------------------------------------
    def add_who_staging(self, stage_info: Dict[str, str], class_name: str) -> None:
        """
        Add WHO grading and staging section.

        Args:
            stage_info: Dict with grade_label, typical_imaging_features,
                malignancy, growth_pattern, recommended_intervention_protocol.
            class_name: Tumour class name for context.
        """
        self.elements.append(Paragraph("WHO GRADE & STAGING ASSESSMENT", self.styles["heading1"]))
        self._add_horizontal_rule()
        self.elements.append(Paragraph(
            f"Tumour Class: <b>{class_name}</b>", self.styles["body"]))
        self._add_spacer(4)

        staging_fields = [
            ("Grade Label", stage_info.get("grade_label", "N/A")),
            ("Typical Imaging Features", stage_info.get("typical_imaging_features", "N/A")),
            ("Malignancy Profile", stage_info.get("malignancy", "N/A")),
            ("Growth Pattern", stage_info.get("growth_pattern", "N/A")),
            ("Recommended Intervention Protocol",
             stage_info.get("recommended_intervention_protocol", "N/A")),
        ]
        for field, value in staging_fields:
            self.elements.append(Paragraph(f"<b>{field}:</b> {value}", self.styles["body"]))
        self._add_spacer(8)

    # ------------------------------------------------------------------
    def add_clinical_reasoning(self, reasoning_text: str) -> None:
        """
        Add clinical reasoning narrative text to the report.

        Args:
            reasoning_text: Multi-line reasoning string.
        """
        self.elements.append(Paragraph("CLINICAL REASONING NARRATIVE", self.styles["heading1"]))
        self._add_horizontal_rule()
        for paragraph in reasoning_text.split("\n\n"):
            para = paragraph.strip()
            if para:
                self.elements.append(Paragraph(para, self.styles["body"]))
        self._add_spacer(8)

    # ------------------------------------------------------------------
    def add_disclaimer(self, include_full: bool = True) -> None:
        """
        Add mandatory regulatory and medical disclaimer block.

        Args:
            include_full: If True, include the full disclaimer; otherwise only
                the shorthand version.
        """
        self.elements.append(Paragraph("REGULATORY & MEDICAL DISCLAIMER", self.styles["heading1"]))
        self._add_horizontal_rule()
        if include_full:
            self.elements.append(Paragraph(REGULATORY_DISCLAIMER, self.styles["disclaimer"]))
        else:
            self.elements.append(Paragraph(REGULATORY_SHORTHAND, self.styles["disclaimer"]))
        self._add_spacer(6)
        self.elements.append(Paragraph(
            "This report was generated by a research-grade computer-aided classification system "
            "and must be reviewed, verified, and interpreted by a licensed medical professional "
            "in the context of the complete clinical presentation.", self.styles["disclaimer"]))

    # ------------------------------------------------------------------
    def build_and_save(
        self,
        filename: Optional[str] = None,
        patient_id: Optional[str] = None,
        page_size: Tuple[float, float] = A4,
        author: str = "FreeBuff v4",
        title: str = "Brain Tumor Classification Report",
        subject: str = "Diagnostic Imaging Classification",
    ) -> Path:
        """
        Build the complete PDF document from accumulated elements and save to
        the report directory.

        Args:
            filename: Optional output filename (without .pdf suffix). If None,
                a timestamped UUID-based name is generated.
            patient_id: Optional patient identifier for the filename stem.
            page_size: ReportLab page size tuple.
            author: PDF metadata author.
            title: PDF metadata title.
            subject: PDF metadata subject.

        Returns:
            Path to the saved PDF file.
        """
        from reportlab.platypus import SimpleDocTemplate

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pid = (patient_id or "unknown")[:20].replace(" ", "_")
            safe_pid = "".join(c if c.isalnum() else "_" for c in pid)
            filename = f"FreeBuff_v4_Report_{safe_pid}_{ts}.pdf"

        out_path = self.report_dir / filename
        if out_path.suffix != ".pdf":
            out_path = out_path.with_suffix(".pdf")

        out_path_str = str(out_path)
        doc = SimpleDocTemplate(
            out_path_str,
            pagesize=page_size,
            leftMargin=0.7*inch,
            rightMargin=0.7*inch,
            topMargin=0.6*inch,
            bottomMargin=0.6*inch,
            title=title,
            author=author,
            subject=subject,
        )
        doc.build(self.elements, onFirstPage=self._page_decorator, onLaterPages=self._page_decorator)
        logger.info("PDF report saved: %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    @staticmethod
    def _page_decorator(canvas: Any, doc: Any) -> None:
        """
        Draw header/footer decorations on every page.
        """
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(0.7*inch, 0.4*inch,
                          "FreeBuff v4 Research System — NOT FOR CLINICAL DIAGNOSTIC USE")
        canvas.drawRightString(doc.pagesize[0] - 0.7*inch, 0.4*inch,
                               f"Page {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#0B3D91"))
        canvas.setLineWidth(0.5)
        canvas.line(0.7*inch, 0.55*inch, doc.pagesize[0] - 0.7*inch, 0.55*inch)
        canvas.restoreState()


# ==============================================================================
# CONVENIENCE GENERATOR (ASSEMBLY HELPER)
# ==============================================================================
def generate_pdf_report(
    inference_results: Dict[str, Any],
    stage_info: Optional[Dict[str, str]] = None,
    gradcam_image: Optional[PILImage.Image] = None,
    gradcam_path: Optional[Union[str, Path]] = None,
    patient_id: Optional[str] = None,
    patient_name: Optional[str] = None,
    exam_date: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Convenience function that assembles and saves a complete PDF report from
    an inference results dictionary.

    Args:
        inference_results: Dict returned by run_inference_on_image or similar,
            expected keys: prediction, confidence, probabilities, logits,
            pred_class_idx, image_source, tta_used, reasoning_text.
        stage_info: Optional WHO staging dict (auto-looked-up if None).
        gradcam_image: Optional PIL overlay image.
        gradcam_path: Optional path to saved Grad-CAM overlay.
        patient_id: Optional patient ID.
        patient_name: Optional patient name.
        exam_date: Optional exam date string.
        output_dir: Optional report output directory.
        filename: Optional output PDF filename stem.

    Returns:
        Path to the saved PDF.
    """
    from inference import lookup_who_stage, generate_diagnostic_summary
    from config import CLASSES

    generator = PDFReportGenerator(report_dir=output_dir)
    pred = inference_results.get("prediction", "Unknown")
    conf = inference_results.get("confidence", 0.0)
    probs = inference_results.get("probabilities", [0.0] * NUM_CLASSES)
    logits = inference_results.get("logits", [0.0] * NUM_CLASSES)
    pred_idx = inference_results.get("pred_class_idx", 0)
    img_src = inference_results.get("image_source", None)
    tta_used = inference_results.get("tta_used", False)
    reasoning = inference_results.get("reasoning_text", "")
    if not reasoning:
        reasoning = generate_diagnostic_summary(inference_results, img_src)

    if stage_info is None:
        stage_info = lookup_who_stage(pred)

    generator.add_cover_page(
        patient_id=patient_id,
        patient_name=patient_name,
        exam_date=exam_date,
        modality="MRI Brain",
        scan_description=f"Brain MRI slice — classification target: {pred}",
        created_by="FreeBuff v4 Research System",
    )
    generator.add_prediction_summary(
        prediction=pred,
        confidence=conf,
        probabilities=probs,
        logits=logits,
        pred_class_idx=pred_idx,
        image_source=img_src,
        tta_used=tta_used,
    )
    generator.add_confidence_chart(probs)
    generator.add_gradcam_overlay(gradcam_image=gradcam_image, gradcam_path=gradcam_path)
    generator.add_who_staging(stage_info, pred)
    generator.add_clinical_reasoning(reasoning)
    generator.add_disclaimer(include_full=True)

    return generator.build_and_save(filename=filename, patient_id=patient_id)


# ==============================================================================
# METADATA HASHING FOR AUDIT TRAIL
# ==============================================================================
def compute_report_checksum(filepath: Union[str, Path]) -> str:
    """
    Compute a SHA-256 checksum for a generated PDF file to support audit trail
    and de-duplication.

    Args:
        filepath: Path to the PDF file.

    Returns:
        Hex digest string.
    """
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ==============================================================================
# BULK REPORT GENERATION
# ==============================================================================
def generate_bulk_pdf_reports(
    results_list: List[Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    save_gradcam: bool = True,
) -> List[Path]:
    """
    Generate PDF reports for a list of inference results.

    Args:
        results_list: List of inference result dicts.
        output_dir: Output directory for PDF reports.
        save_gradcam: Whether to include Grad-CAM overlays where present.

    Returns:
        List of saved PDF file paths.
    """
    out_paths: List[Path] = []
    for i, res in enumerate(results_list):
        logger.info("Generating PDF report for result %d/%d...", i+1, len(results_list))
        try:
            gradcam_img = res.get("gradcam_overlay") if save_gradcam else None
            gradcam_path = None
            path = generate_pdf_report(
                inference_results=res,
                output_dir=output_dir,
                gradcam_image=gradcam_img,
                patient_id=f"patient_{i+1:04d}",
                exam_date=datetime.now().strftime("%Y-%m-%d"),
            )
            out_paths.append(path)
        except Exception as e:
            logger.error("Failed to generate report for result %d: %s", i, e)
    return out_paths


# ==============================================================================
# MODULE-LEVEL HELPER DOCTESTS (CONCEPTUAL PLACEHOLDERS FOR DOCS)
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Utils module loaded. ReportLab PDF generation utilities ready.")
    # Quick sanity check on styles
    for name, style in STYLES.items():
        logger.debug("Style '%s': fontSize=%s, leading=%s", name, style.fontSize, style.leading)
