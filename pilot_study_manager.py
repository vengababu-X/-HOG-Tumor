#!/usr/bin/env python3
# ==============================================================================
# NeuroScan CDS — Clinical Pilot Reader Study Manager
# ==============================================================================
# Manages structured evaluation responses from clinical and non-clinical readers
# participating in pilot evaluation trials. Records usability, diagnostic
# concurrence, saliency alignment, and 3D utility into an exportable CSV.
# ==============================================================================
from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("neuroscan_cds.pilot_study")

_PROJECT_ROOT = Path(__file__).resolve().parent
PILOT_LOG_DIR = _PROJECT_ROOT / "logs"
PILOT_CSV_FILE = PILOT_LOG_DIR / "pilot_study_responses.csv"


def _ensure_csv() -> Path:
    PILOT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not PILOT_CSV_FILE.exists():
        headers = [
            "timestamp_utc",
            "audit_id",
            "reader_id",
            "reader_background",
            "predicted_class",
            "model_confidence",
            "reader_agreement",
            "saliency_accuracy_rating",
            "mpr_3d_utility_rating",
            "safeguards_clarity_rating",
            "overall_usability_score",
            "qualitative_feedback",
        ]
        with open(PILOT_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    return PILOT_CSV_FILE


def record_pilot_response(
    audit_id: str,
    reader_id: str,
    reader_background: str,
    predicted_class: str,
    model_confidence: float,
    reader_agreement: str,
    saliency_rating: int,
    mpr_rating: int,
    safeguards_rating: int,
    usability_score: int,
    feedback: str = "",
) -> bool:
    """Record a reader pilot study survey response."""
    log_path = _ensure_csv()
    row = [
        datetime.now(timezone.utc).isoformat(),
        audit_id or "UNASSIGNED",
        reader_id.strip() or "ANON_READER",
        reader_background,
        predicted_class,
        f"{model_confidence:.4f}",
        reader_agreement,
        saliency_rating,
        mpr_rating,
        safeguards_rating,
        usability_score,
        feedback.strip().replace("\n", " "),
    ]
    try:
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        logger.info("Pilot study response recorded for reader %s.", reader_id)
        return True
    except Exception as exc:
        logger.error("Failed to write pilot study response: %s", exc)
        return False


def get_pilot_study_summary() -> Dict[str, Any]:
    """Calculate aggregate summary statistics of recorded pilot trials."""
    if not PILOT_CSV_FILE.exists():
        return {"total_responses": 0, "avg_usability": 0.0, "avg_saliency": 0.0, "avg_mpr": 0.0}

    rows: List[Dict[str, str]] = []
    try:
        with open(PILOT_CSV_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception:
        return {"total_responses": 0, "avg_usability": 0.0, "avg_saliency": 0.0, "avg_mpr": 0.0}

    if not rows:
        return {"total_responses": 0, "avg_usability": 0.0, "avg_saliency": 0.0, "avg_mpr": 0.0}

    n = len(rows)
    usability = sum(float(r.get("overall_usability_score", 0)) for r in rows) / n
    saliency = sum(float(r.get("saliency_accuracy_rating", 0)) for r in rows) / n
    mpr = sum(float(r.get("mpr_3d_utility_rating", 0)) for r in rows) / n
    agreed = sum(1 for r in rows if "Agree" in r.get("reader_agreement", ""))

    return {
        "total_responses": n,
        "avg_usability": round(usability, 2),
        "avg_saliency": round(saliency, 2),
        "avg_mpr": round(mpr, 2),
        "concurrence_rate_percent": round((agreed / n) * 100.0, 1),
    }

