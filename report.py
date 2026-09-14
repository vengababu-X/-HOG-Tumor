"""
report.py – Clinical PDF Report Generator (ReportLab)
======================================================

Turns one prediction payload into a polished, downloadable clinical report
containing:

  * Patient / study header + model metadata
  * Primary finding (lesion family + confidence)
  * Full confidence spectrum table (all classes)
  * WHO staging & malignancy profile (see ``staging.py``)
  * Grad-CAM explainability images (one per uploaded sequence)
  * Regulatory disclaimer

Usage
-----
    result = {...}                     # built by the inference engine
    pdf_bytes = build_pdf(result)      # → bytes for Streamlit download

Author: Codebuff AI
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime
from typing import Dict, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

# Brand / palette
BRAND: str = "FreeBuff"
ACCENT: colors.HexColor = colors.HexColor("#0e5a8a")
LIGHT: colors.HexColor = colors.HexColor("#eef4f8")


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "fbTitle", parent=base["Title"], fontSize=20, textColor=ACCENT,
            alignment=TA_CENTER, spaceAfter=2 * mm,
        ),
        "subtitle": ParagraphStyle(
            "fbSub", parent=base["Normal"], fontSize=9.5, textColor=colors.grey,
            alignment=TA_CENTER, spaceAfter=6 * mm,
        ),
        "h2": ParagraphStyle(
            "fbH2", parent=base["Heading2"], fontSize=13, textColor=ACCENT,
            spaceBefore=5 * mm, spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "fbBody", parent=base["BodyText"], fontSize=9.5, leading=13,
        ),
        "small": ParagraphStyle(
            "fbSmall", parent=base["BodyText"], fontSize=8, textColor=colors.grey,
            leading=10,
        ),
        "finding": ParagraphStyle(
            "fbFinding", parent=base["BodyText"], fontSize=12, leading=16,
            textColor=colors.HexColor("#06324d"), spaceBefore=2 * mm,
        ),
    }


def _header_table(st: Dict[str, ParagraphStyle], result: Dict[str, object]) -> Table:
    """Branded top band with study metadata."""
    rows = [
        [Paragraph("Primary finding", st["small"]),
         Paragraph(str(result.get("predicted_class", "—")).upper(), st["finding"])],
        [Paragraph("Model confidence", st["small"]),
         Paragraph(f"{float(result.get('confidence', 0.0)) * 100:.1f} %", st["finding"])],
        [Paragraph("Sequences analysed", st["small"]),
         Paragraph(", ".join(str(m) for m in result.get("modalities", [])) or "—", st["body"])],
        [Paragraph("Patient / study id", st["small"]),
         Paragraph(str(result.get("patient_id") or "—"), st["body"])],
        [Paragraph("Report generated", st["small"]),
         Paragraph(str(result.get("date") or datetime.now().strftime("%Y-%m-%d %H:%M")),
                   st["body"])],
        [Paragraph("Model version", st["small"]),
         Paragraph(f"v{result.get('model_version', '?')} — 144×144 hybrid CNN+Transformer/HOG/SVM",
                   st["body"])],
    ]
    t = Table(rows, colWidths=[45 * mm, 120 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bcd2e0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _confidence_table(st: Dict[str, ParagraphStyle], probs: Dict[str, float]) -> Table:
    """Confidence-spectrum table across all classes."""
    rows = [["Class", "Probability"]] + [
        [str(label), f"{float(p) * 100:6.2f} %"] for label, p in
        sorted(probs.items(), key=lambda kv: -float(kv[1]))
    ]
    t = Table(rows, colWidths=[80 * mm, 80 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bcd2e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _staging_block(st: Dict[str, ParagraphStyle], staging: Dict[str, object]):
    """Paragraph flowables for the WHO staging payload."""
    parts = [
        Paragraph(f"Typical presentation: <b>{staging.get('typical_grade', '—')}</b>",
                  st["body"]),
        Spacer(1, 2 * mm),
        Paragraph("Grade rationale:", st["body"]),
    ]
    for desc in staging.get("grade_descriptions", []):
        parts.append(Paragraph(f"&bull;&nbsp; {desc}", st["body"]))
    parts += [
        Spacer(1, 2 * mm),
        Paragraph(f"<b>Malignancy profile:</b> {staging.get('malignancy_profile', '—')}",
                  st["body"]),
        Spacer(1, 2 * mm),
        Paragraph(f"<b>Clinical notes:</b> {staging.get('clinical_notes', '—')}",
                  st["body"]),
    ]
    return parts


def _heatmap_images(result: Dict[str, object], max_w: int = 60 * mm) -> list:
    """Platypus Images for the Grad-CAM overlays that actually exist."""
    imgs = []
    paths: Dict[str, Optional[str]] = result.get("heatmap_paths", {}) or {}
    for modality in result.get("modalities", []):
        path = paths.get(str(modality))
        if path and os.path.isfile(path):
            try:
                imgs.append(Image(path, width=max_w, height=max_w))
            except Exception as exc:  # pragma: no cover
                logger.warning("Skipping heatmap %s: %s", path, exc)
    return imgs


def build_pdf(result: Dict[str, object]) -> bytes:
    """
    Render the full clinical report into a PDF in memory.

    Parameters
    ----------
    result : dict
        Keys: ``predicted_class``, ``confidence``, ``probabilities``,
        ``staging``, ``modalities``, ``heatmap_paths``, ``patient_id``,
        ``date``, ``model_version``, ``image_names`` (optional).
    """
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title="FreeBuff Clinical Decision-Support Report",
        author="FreeBuff",
    )

    story: list = [
        Paragraph(f"{BRAND} — Brain Tumour Diagnosis &amp; Clinical Staging", st["title"]),
        Paragraph(
            "Hybrid deep-learning decision support · 144×144 multimodal MRI · "
            "WHO staging reference (educational)",
            st["subtitle"],
        ),
        _header_table(st, result),
        Spacer(1, 3 * mm),
        Paragraph("Confidence spectrum", st["h2"]),
        _confidence_table(st, dict(result.get("probabilities", {}))),
    ]

    staging = result.get("staging") or {}
    if staging:
        story += [Paragraph("WHO grading &amp; malignancy profile", st["h2"])]
        story += _staging_block(st, staging)

    heatmaps = _heatmap_images(result)
    if heatmaps:
        story += [Paragraph("Explainability — Grad-CAM overlays (per sequence)", st["h2"])]
        row: list = []
        for img in heatmaps:
            row.append(img)
            if len(row) == 2:
                story.append(Table([row], hAlign="LEFT"))
                row = []
        if row:
            story.append(Table([row], hAlign="LEFT"))

    disclaimer = (staging or {}).get(
        "disclaimer",
        "Decision-support only; final WHO grading requires histopathology "
        "and molecular testing by a neuropathologist.",
    )
    story += [
        PageBreak(),
        Paragraph("Disclaimer &amp; limitations", st["h2"]),
        Paragraph(disclaimer, st["small"]),
        Spacer(1, 3 * mm),
        Paragraph(
            "This report was generated automatically by an AI research "
            "platform and is provided for research and educational purposes. "
            "It must not replace a formal radiological or pathological "
            "report and is not intended for direct clinical decision-making.",
            st["small"],
        ),
    ]

    doc.build(story)
    data = buf.getvalue()
    logger.info("PDF report rendered (%d bytes).", len(data))
    return data


def save_pdf(result: Dict[str, object], save_path: str) -> str:
    """Convenience wrapper writing the PDF to disk (evaluate/CLI flows)."""
    data = build_pdf(result)
    with open(save_path, "wb") as fh:
        fh.write(data)
    logger.info("Saved PDF → %s", save_path)
    return save_path
