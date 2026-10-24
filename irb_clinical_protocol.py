#!/usr/bin/env python3
# ==============================================================================
# NeuroScan CDS — Institutional Review Board (IRB) Protocol & Funding Dossier
# ==============================================================================
# Generates a formalized, hospital-grade Investigator-Initiated Trial Proposal
# for submission to Hospital Ethics Committees (IEC / IRB) and Hospital
# Research Funding Boards.
# ==============================================================================
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def generate_irb_protocol_text(
    hospital_name: str = "Tertiary Multispeciality Medical Center",
    pi_name: str = "Chief of Neuroradiology / Neurosurgery",
    department: str = "Department of Neurosurgery & Radiodiagnosis",
) -> str:
    """Generate the full text of the Investigator-Initiated IRB Trial Dossier."""
    return f"""================================================================================
INSTITUTIONAL ETHICS COMMITTEE (IEC) / IRB CLINICAL TRIAL PROPOSAL
PROJECT FUNDING & MULTI-READER CLINICAL PILOT PROTOCOL
================================================================================
Protocol ID       : NEUROSCAN-IRB-2026-v4
Date of Submission: {datetime.now().strftime("%B %d, %Y")}
Study Title       : Prospective Evaluation of NeuroScan CDS as an On-Premise
                    Artificial Intelligence Decision-Support System for Brain
                    Tumor Classification, 3D Staging, and Surgical Triage
Clinical Site     : {hospital_name}
Department        : {department}
Principal Investigator (PI): {pi_name}
Technical Developers: Antigravity Neuro-AI Consortium / Engineering Capstone Team
Regulatory Status : Investigational Medical Device Software (SaMD - CDS Class B/IIa)
================================================================================

1. EXECUTIVE SUMMARY & CLINICAL PROBLEM
Intracranial neoplasms (gliomas, meningiomas, pituitary tumors) represent high-stakes
pathologies requiring rapid, accurate radiological characterization. In high-volume
tertiary multispeciality centers, non-specialist off-hours triage and inter-reader
variability create clinical bottlenecks. NeuroScan CDS is an on-premise, GPU-accelerated
dual-branch neural network (ResNet-50+CBAM x HOG-PCA) designed as a non-autonomous
second-read aid that operates directly on local hospital PACS DICOM data without
cloud exposure.

2. SPECIFIC AIMS & STUDY HYPOTHESES
- Primary Aim: Determine whether radiologist diagnostic accuracy (macro F1-score)
  is significantly improved when assisted by NeuroScan CDS compared to unassisted
  evaluation across consecutive intracranial tumor callsets.
- Secondary Aim: Measure the reduction in mean diagnostic review time per scan
  and quantify inter-reader agreement (Cohen's Kappa κ) among junior residents
  vs. senior attending neuroradiologists.
- Surgical Correlation Aim: Correlate pre-operative AI anatomical subsite and
  3D volumetric estimations against intra-operative neurosurgical findings.

3. METHODOLOGY & TRIAL DESIGN
- Study Design: Prospective, multi-reader, sequential observer trial.
- Sample Size: N = 100 consecutive adult cranial MRI examinations presenting
  with suspected intracranial masses.
- Inclusion Criteria:
  1. Adult patients (≥ 18 years) undergoing diagnostic brain MRI (T1, T2, FLAIR, T1+C).
  2. Suspected intra-axial, extra-axial, or sellar neoplasm.
  3. Subsequent histopathological / biopsy confirmation available for surgical cases.
- Exclusion Criteria:
  1. Severe motion artifact rendering images non-diagnostic (QA blur score < 5.0).
  2. Prior gross total resection with extensive craniectomy distortion.

4. ETHICAL CONSIDERATIONS, DATA INTEGRITY & ZERO PHI POLICY
- 100% On-Premise Execution: No image bytes, DICOM tags, or patient records leave
  the hospital firewall.
- Automatic De-Identification: All DICOM Protected Health Information (PHI) is
  scrubbed locally conforming to HIPAA Safe Harbor and Indian DPDPA standards.
- Immutable Audit Trail: Every transaction generates a cryptographic SHA-256 hash
  and unique transaction UUID logged in an append-only audit registry.

5. PROPOSED BUDGET & MILESTONE SCHEDULE FOR PROJECT FUNDING
- Milestone 1 (Months 1–2): On-premise PACS DICOM listener (C-STORE SCP) deployment
  on local hospital workstation with real-time reading room integration.
- Milestone 2 (Months 3–6): Prospective 100-case reader study execution; statistical
  analysis of sensitivity, specificity, and time-to-decision.
- Milestone 3 (Months 7–9): Post-surgical histopathological & molecular correlation
  (IDH1/2 mutation status, 1p/19q codeletion) and co-authored publication in a
  peer-reviewed radiology / neurosurgery journal.

6. SIGNATURE & INSTITUTIONAL ENDORSEMENT

__________________________________          __________________________________
Principal Investigator Signature            Hospital Ethics Committee Chair
Date:                                       Date:
================================================================================
"""

