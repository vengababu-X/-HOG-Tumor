"""
run_cli.py — FreeBuff v4 live-demo CLI (clinical decision support)

Prompts the operator for T1c / T2 / FLAIR scan paths, feeds them to the
FreeBuffInference engine, and prints a professional diagnosis breakdown
suitable for a capstone presentation.

Usage:
    python run_cli.py

The script is intentionally standalone: it imports FreeBuffInference from
inference.py but never touches Streamlit, joblib-pickled explainers, or any
web-UI code path.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import numpy as np
import torch

# The WHO staging text and the disclaimers contain typographic characters
# (e.g. "≈", en dashes) that the Windows console's default cp1252 codepage
# cannot encode, which aborted the run mid-report with UnicodeEncodeError.
# Re-encode stdout/stderr as UTF-8 and degrade unrepresentable glyphs instead
# of raising.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # non-reconfigurable stream
        pass


def report_compute_device() -> None:
    """Print the device this demo will run on.

    This used to configure TensorFlow GPU memory growth, which was left over
    from the v1 Keras pipeline: the v4 model is PyTorch, TensorFlow is not a
    dependency, and importing it made this entry point fail outright.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[*] GPU detected: {name} ({total:.1f} GB) — running on cuda:0.")
    else:
        print("[*] No CUDA GPU detected — running on CPU.")


def prompt_path(label: str) -> Optional[str]:
    """Ask the operator for a single file path; return None on blank input."""
    raw = input(f"{label} (or press Enter to skip): ").strip().strip('"\'')
    if not raw:
        return None
    return os.path.abspath(raw)


def describe_file(path: str) -> str:
    """Human-friendly descriptor for a scan path, for the audit trail."""
    base = os.path.basename(path)
    size = os.path.getsize(path) if os.path.exists(path) else None
    if size is None:
        return base
    if size < 1_000:
        return f"{base} ({size} B)"
    if size < 1_000_000:
        return f"{base} ({size / 1_000:.1f} KB)"
    return f"{base} ({size / 1_000_000:.2f} MB)"


def main() -> None:
    print("=" * 64)
    print("         FreeBuff v4 — Clinical Decision Support Demo          ")
    print("         Multi-Modal Brain Tumor Classification                ")
    print("         (Glioma / Meningioma / Pituitary / No Tumor)         ")
    print("=" * 64)
    print()

    report_compute_device()
    print()

    models_dir = os.path.abspath("saved_models")
    if not os.path.isdir(models_dir):
        print(f"[!] Error: 'saved_models/' not found at: {models_dir}")
        print("    Place your trained artefact bundle there and re-run.")
        sys.exit(1)

    print(f"[*] Loading inference engine from: {models_dir}")
    try:
        from inference import FreeBuffInference

        engine = FreeBuffInference.from_dir(models_dir)
        print("[+] Inference engine loaded successfully.")
        print(f"    Classes : {', '.join(engine.class_names)}")
        print(f"    Version : v{engine.manifest.get('artifact_version', '?')}")
        print()
    except Exception as exc:
        print(f"[!] Failed to load inference engine: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("[?] Provide the MRI sequences for this case.")
    print("    FreeBuff expects three sequences — T1c, T2, and FLAIR.")
    print("    Any missing sequence is auto-replicated from a present one")
    print("    (this is logged, never silent).")
    print()
    print("    Accepted formats: .png, .jpg, .jpeg, .bmp, .tif/.tiff, .dcm")
    print()

    t1c_path = prompt_path("Path to T1c (T1-weighted contrast-enhanced)")
    t2_path = prompt_path("Path to T2 (T2-weighted)")
    flair_path = prompt_path("Path to FLAIR (fluid-attenuated inversion recovery)")

    provided = {p for p in (t1c_path, t2_path, flair_path) if p}
    if not provided:
        print()
        print("[!] No scan paths provided — nothing to infer. Exiting.")
        sys.exit(0)

    print()
    print("-" * 64)
    print("  Submitted scans")
    print("-" * 64)
    for label, path in [
        ("T1c", t1c_path),
        ("T2", t2_path),
        ("FLAIR", flair_path),
    ]:
        if path is None:
            print(f"  {label:7s}: (not provided)")
        elif not os.path.isfile(path):
            print(f"  {label:7s}: {describe_file(path)}  [NOT FOUND — will be replicated]")
        else:
            print(f"  {label:7s}: {describe_file(path)}")
    print()

    # Build the uploads dict using the engine's own preprocessing contract.
    uploads: Dict[str, Optional[np.ndarray]] = {"t1ce": None, "t2": None, "flair": None}
    for modality, path in [("t1ce", t1c_path), ("t2", t2_path), ("flair", flair_path)]:
        if path and os.path.isfile(path):
            try:
                uploads[modality] = engine.preprocess_path(path)
            except Exception as exc:
                print(f"[!] Failed to preprocess {modality} ({os.path.basename(path)}): {exc}")
                uploads[modality] = None

    missing_after_preprocess = [m for m, a in uploads.items() if a is None]
    # bool() of a multi-element numpy array raises, so any() over the arrays
    # cannot be used here — test for presence instead.
    if all(a is None for a in uploads.values()):
        print("[!] No usable scan could be preprocessed. Exiting.")
        sys.exit(1)

    if missing_after_preprocess:
        print(f"[*] The following sequences will be replicated at inference time: {missing_after_preprocess}")
        print()

    print("-" * 64)
    print("  Running inference (with Test-Time Augmentation)             ")
    print("-" * 64)
    try:
        result = engine.predict_uploads(uploads, use_tta=True)
    except Exception as exc:
        print(f"[!] Inference failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]
    staging = result["staging"]

    print()
    print("=" * 64)
    print("                    PREDICTION RESULTS                         ")
    print("=" * 64)
    print()
    print(f"  Predicted diagnosis : {pred_class.upper()}")
    print(f"  Confidence          : {confidence * 100:.1f}%")
    print()
    print("  Class probabilities :")
    for cls in engine.class_names:
        bar = "#" * int(probabilities[cls] * 40)
        print(f"    {cls:12s} {probabilities[cls] * 100:6.1f}%  {bar}")
    print()
    print("-" * 64)
    print("  WHO staging profile")
    print("-" * 64)
    print(f"  Typical grade  : {staging.get('typical_grade', 'n/a')}")
    print(f"  Grade range    : {staging.get('grade_range', 'n/a')}")
    print(f"  Malignancy     : {staging.get('malignancy_profile', 'n/a')}")
    print()
    # clinical_notes is a single string from the staging engine; iterating it
    # directly printed one character per bullet.
    notes = staging.get("clinical_notes") or []
    if isinstance(notes, str):
        notes = [notes]
    if notes:
        print("  Clinical notes  :")
        for note in notes:
            print(f"    - {note}")
        print()
    print("-" * 64)
    print("  Disclaimer")
    print("-" * 64)
    disclaimer = staging.get("disclaimer") or (
        "Decision-support only. Imaging-based staging is not a substitute "
        "for histopathology and molecular testing."
    )
    for line in disclaimer.split(". "):
        line = line.strip()
        if line:
            print(f"    {line}.")
    print()
    print("=" * 64)
    print("  Demo complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
