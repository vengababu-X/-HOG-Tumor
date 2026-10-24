#!/usr/bin/env python
"""
FreeBuff v4 — regression tests for class-label handling.

WHY THIS FILE EXISTS
--------------------
The canonical labels in `config.CLASSES` are "Glioma", "Meningioma",
"No_Tumor", "Pituitary". Several internal lookup tables are keyed on the
lowercase, unpunctuated forms ("glioma", "notumor"). Four separate defects
came from comparing one against the other without normalising, and every one
of them failed *silently* — no exception, just a wrong answer:

  1. `TUMOR_CLASS_INDICES` compared `name.lower() != "notumor"` against
     "no_tumor", so it excluded nothing. Aggregate tumour probability summed
     to 1.0 on every scan, the >= 30% sensitivity alert fired on all 1,600
     test images, and the "surface the most likely tumour class" override
     picked No_Tumor itself — a no-op safeguard.
  2. `staging._profile_for` returned the empty "unknown" profile for every
     canonical label, so WHO staging silently degraded.
  3. `staging.malignancy_band` returned "LOW" for everything — badging a
     glioma as low-malignancy and a healthy scan as low rather than none.
  4. `narrative` looked up `probs.get("glioma")` in a dict keyed "Glioma", so
     the clinical narrative reported 0.0% for all four classes.

None of these were caught by the existing suites, because those assert on
shapes and exit codes rather than on label semantics. These tests assert the
semantics directly.

Run:
    python test_class_labels.py
"""
from __future__ import annotations

import sys

import torch

from config import CLASSES

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  {status}  {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


print("=" * 74)
print("FreeBuff v4 — class-label regression tests")
print("=" * 74)
print(f"canonical CLASSES: {CLASSES}\n")

# ---------------------------------------------------------------- defect 1
print("1. Tumour class indices must exclude No_Tumor")
from inference import TUMOR_CLASS_INDICES, clinical_decision, tumor_safety_alert

no_tumor_idx = CLASSES.index("No_Tumor")
check("No_Tumor is excluded from TUMOR_CLASS_INDICES",
      no_tumor_idx not in TUMOR_CLASS_INDICES,
      f"indices={TUMOR_CLASS_INDICES}")
check("exactly the three tumour classes are included",
      len(TUMOR_CLASS_INDICES) == len(CLASSES) - 1)

# A healthy scan must NOT raise the tumour alert.
healthy = torch.tensor([0.01, 0.01, 0.97, 0.01])
d = clinical_decision(healthy)
check("healthy scan raises no tumour alert", not d["potential_tumor"],
      f"tumour_p={d['tumor_probability']:.3f}")
check("healthy scan keeps the No_Tumor prediction",
      d["pred_class"] == no_tumor_idx and not d["threshold_override"])

# A missed tumour (argmax No_Tumor, but 40% tumour mass) MUST override.
missed = torch.tensor([0.20, 0.15, 0.60, 0.05])
d = clinical_decision(missed)
check("missed tumour raises the alert", d["potential_tumor"],
      f"tumour_p={d['tumor_probability']:.3f}")
check("missed tumour overrides to a real tumour class",
      d["threshold_override"] and d["pred_class"] != no_tumor_idx,
      f"pred={CLASSES[d['pred_class']]}")
check("aggregate tumour probability is not always 1.0",
      abs(tumor_safety_alert(healthy)["tumor_probability"] - 1.0) > 1e-6)

# ---------------------------------------------------------------- defect 2
print("\n2. WHO staging must resolve canonical labels")
from staging import interpret_prediction, malignancy_band

for cls in CLASSES:
    probs = {c: (0.9 if c == cls else 0.1 / 3) for c in CLASSES}
    res = interpret_prediction(cls, 0.9, probs)
    grade = str(res.get("typical_grade", ""))
    check(f"{cls} resolves a real staging profile",
          bool(grade) and "not mapped" not in grade.lower(),
          grade[:46])

# ---------------------------------------------------------------- defect 3
print("\n3. Malignancy band must reflect the class")
expected = {"Glioma": "HIGH", "Meningioma": "LOW", "No_Tumor": "NONE", "Pituitary": "LOW"}
for cls, want in expected.items():
    got = malignancy_band(cls)
    check(f"malignancy_band({cls}) == {want}", got == want, f"got {got!r}")
check("spelling variants agree",
      malignancy_band("no_tumor") == malignancy_band("No_Tumor") == malignancy_band("notumor"))

# ---------------------------------------------------------------- defect 4
print("\n4. Narrative must read the probability dictionary")
from narrative import generate_narrative

payload = {
    "predicted_class": "Glioma",
    "confidence": 0.95,
    "probabilities": {"Glioma": 0.95, "Meningioma": 0.03, "No_Tumor": 0.01, "Pituitary": 0.01},
    "modalities": ["t1ce"],
    "tumor_probability": 0.99,
}
text = generate_narrative(payload)["narrative"]
check("narrative does not report every class as 0.0%",
      text.count("0.0%") < 4, f"0.0% appears {text.count('0.0%')}x")
check("narrative reports the dominant class probability", "95.0%" in text)

# ------------------------------------------------------------------ summary
print("\n" + "=" * 74)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"   - {f}")
    print("=" * 74)
    sys.exit(1)
print("ALL CLASS-LABEL REGRESSION TESTS PASSED")
print("=" * 74)
