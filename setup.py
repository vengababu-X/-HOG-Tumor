#!/usr/bin/env python3
"""
setup.py – FreeBuff Automated Environment Setup & CUDA Doctor
=============================================================

One command provisions the whole local platform on the user's machine:

    python setup.py           # → checks GPU, then installs every dependency
    python setup.py --check   # → environment doctor only (no installation)
    python setup.py --no-gpu  # → force CPU-only install plan

What it does
------------
1. Reports Python / pip versions.  TensorFlow only ships wheels for
   Python 3.9–3.12; when a newer interpreter (3.13/3.14) is detected and
   conda/mamba is available, setup.py **automatically creates a dedicated
   environment** (default name ``freebuff``, Python 3.11) and re-runs
   itself inside it — so the user never has to think about interpreter
   compatibility.  Disable with ``--no-bootstrap``.
2. Detects a CUDA-capable NVIDIA GPU through (in order):
     nvidia-smi / nvcc / PyTorch (torch.cuda) / TensorFlow devices
3. Writes ``requirements.txt`` with the exact platform dependency set.
4. Installs everything via ``pip install -r requirements.txt`` (non-GPU
   fallback is automatic — the stock TensorFlow wheel bundles CUDA).

Author: Codebuff AI
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Dependency manifest (Python ≥3.9, ≤3.12 recommended for TensorFlow)
# ---------------------------------------------------------------------------
# Since TensorFlow 2.16 the CUDA libraries ship as separate `nvidia-*` pip
# packages pulled by the `[and-cuda]` extra.  Plain `tensorflow` silently
# installs a CPU-only build, so the GPU variant is only emitted when an
# NVIDIA GPU is detected (write_requirements below).
TF_GPU_REQ: str = "tensorflow[and-cuda]>=2.16,<2.19"
TF_CPU_REQ: str = "tensorflow>=2.15,<2.19"

CORE_REQS: List[str] = [
    "tensorflow",                   # placeholder — replaced with TF_*_REQ
    "streamlit>=1.30",
    "scikit-learn>=1.3",
    "opencv-python>=4.8",
    "scikit-image>=0.21",
    "pillow>=10.0",
    "pandas>=2.0",
    "numpy>=1.24,<2.0",
    "matplotlib>=3.7",
    "seaborn>=0.13",
    "joblib>=1.3",
    "reportlab>=4.0",               # clinical PDF generator
    "altair>=5.0",                  # confidence-spectrum charts (Streamlit)
]
OPTIONAL_REQS: Dict[str, str] = {
    "albumentations": ">=1.3",      # medical-MRI augmentation
    "nibabel": ">=5.0",             # BraTS .nii.gz → matched-slice exporter
}

# Conda environment created automatically when the interpreter is unsupported.
ENV_NAME: str = "freebuff"
ENV_PYTHON: str = "3.11"            # best wheel coverage for tensorflow 2.15–2.18
_BOOTSTRAP_GUARD: str = "FREEBUFF_BOOTSTRAPPED"


def python_ok() -> Tuple[bool, str]:
    """Python 3.9–3.12 → OK; newer versions warn (no TensorFlow wheels yet)."""
    v = sys.version_info
    if (3, 9) <= (v.major, v.minor) <= (3, 12):
        return True, f"Python {v.major}.{v.minor}.{v.micro} — supported ✓"
    return False, (f"Python {v.major}.{v.minor}.{v.micro} detected — TensorFlow has no "
                   f"wheels for it yet. Please install Python 3.9–3.12 "
                   f"(https://www.python.org/downloads/) and re-run setup.py.")


def has_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def gpu_report() -> Dict[str, object]:
    """Best-effort GPU / CUDA detection across several probing tools."""
    report: Dict[str, object] = {"cuda_gpu": False, "probes": {}}

    # 1. nvidia-smi — definitive when present
    if has_cmd("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20,
            )
            lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
            report["probes"]["nvidia-smi"] = lines
            report["cuda_gpu"] = len(lines) > 0
        except Exception as exc:
            report["probes"]["nvidia-smi"] = f"error: {exc}"

    # 2. nvcc (CUDA toolkit)
    if has_cmd("nvcc"):
        try:
            out = subprocess.run(["nvcc", "--version"], capture_output=True,
                                 text=True, timeout=20)
            report["probes"]["nvcc"] = out.stdout.splitlines()[-2:].__str__()
        except Exception:
            pass

    # 3. PyTorch probe (installed or not)
    try:
        import torch
        report["probes"]["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else "cpu"),
        }
        report["cuda_gpu"] = report["cuda_gpu"] or bool(torch.cuda.is_available())
    except Exception:
        report["probes"]["torch"] = "not installed yet"

    # 4. TensorFlow probe (after install, setup.py --check)
    try:
        import tensorflow as tf
        report["probes"]["tensorflow"] = {
            "version": tf.__version__,
            "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
        }
        report["cuda_gpu"] = report["cuda_gpu"] or bool(
            tf.config.list_physical_devices("GPU"))
    except Exception:
        report["probes"]["tensorflow"] = "not installed yet"
    return report


def write_requirements(include_optional: bool = True,
                       use_gpu: Optional[bool] = None) -> str:
    """
    Materialise requirements.txt for the current platform.

    ``use_gpu=None`` probes the machine; ``True``/``False`` force the
    choice.  With CUDA hardware the TensorFlow ``[and-cuda]`` extra is
    emitted so the GPU build (with bundled nvidia-* libraries) installs.
    """
    if use_gpu is None:
        use_gpu = bool(gpu_report()["cuda_gpu"])
    reqs = [TF_GPU_REQ if use_gpu else TF_CPU_REQ]
    reqs += [r for r in CORE_REQS if not r.startswith("tensorflow")]
    if include_optional:
        reqs += [f"{name}{pin}" for name, pin in OPTIONAL_REQS.items()]
    path = "requirements.txt"
    with open(path, "w") as fh:
        fh.write("# FreeBuff platform requirements (generated by setup.py)\n")
        if use_gpu:
            fh.write("# GPU build: tensorflow[and-cuda] (bundles CUDA 12 libs)\n")
        else:
            fh.write("# CPU-only build (no CUDA GPU detected)\n")
        fh.write("\n".join(reqs) + "\n")
    return path


def doctor() -> Dict[str, object]:
    """Full environment report (never installs anything)."""
    return {
        "platform": platform.platform(),
        "python": python_ok(),
        "pip": has_cmd("pip"),
        "gpu": gpu_report(),
    }


# ---------------------------------------------------------------------------
# Automatic conda-env bootstrap for unsupported interpreters
# ---------------------------------------------------------------------------

def _conda_binary() -> str:
    """Return the conda/mamba binary name, or '' if none is available."""
    for name in ("conda", "mamba"):
        if has_cmd(name):
            return name
    return ""


def bootstrap_env(env_name: str = ENV_NAME, extra_args: List[str] | None = None,
                  ) -> bool:
    """
    Create ``env_name`` with a supported Python and re-execute setup.py inside.

    Returns ``True`` when the environment was (re)created and this process
    re-launched itself within it; the child process then continues the normal
    install flow under a supported interpreter.
    """
    conda = _conda_binary()
    if not conda:
        return False

    print(f"[env] {conda} found — checking for '{env_name}' …")
    try:
        out = subprocess.run([conda, "env", "list"], capture_output=True,
                             text=True, timeout=60)
        exists = any(line.strip().split() and line.split()[-1].endswith(env_name)
                     for line in out.stdout.splitlines()[1:])
    except Exception as exc:
        print(f"[env] could not query environments: {exc}")
        return False

    if not exists:
        print(f"[env] creating conda environment '{env_name}' with Python "
              f"{ENV_PYTHON} (one-time, ~200 MB) …")
        cmd = [conda, "create", "-n", env_name, f"python={ENV_PYTHON}",
               "-y", "--quiet"]
        try:
            subprocess.check_call(cmd, timeout=1800)
        except subprocess.CalledProcessError as exc:
            print(f"[env] environment creation failed: {exc}")
            return False
    else:
        print(f"[env] environment '{env_name}' already present.")

    # Re-run this script inside the environment (guarded against recursion).
    env = dict(os.environ)
    env[_BOOTSTRAP_GUARD] = "1"
    args = [conda, "run", "-n", env_name, "--no-capture-output",
            sys.executable, os.path.abspath(__file__)]
    if extra_args:
        args += extra_args
    print(f"[env] re-running setup inside '{env_name}':\n    "
          + " ".join(args))
    raise SystemExit(subprocess.call(args, env=env))


def install(no_gpu: bool) -> None:
    """Write requirements and pip-install everything into this environment."""
    print("[gpu] probing hardware …")
    gpu = gpu_report()
    for probe, val in gpu["probes"].items():
        print(f"  · {probe}: {val}")

    use_gpu = bool(gpu["cuda_gpu"]) and not no_gpu
    if no_gpu:
        print("[gpu] CPU-only plan requested — GPU acceleration disabled.")
    elif use_gpu:
        print("[gpu] ✔ CUDA-capable GPU detected — TensorFlow will use GPU "
              "acceleration.")
    else:
        print("[gpu] ✖ No CUDA GPU detected — falling back to CPU training "
              "(use --no-gpu for a CPU-only plan, or check your NVIDIA driver "
              "with 'nvidia-smi').")

    path = write_requirements(include_optional=True, use_gpu=use_gpu)
    print(f"[pip] requirements written → {path}")
    print("[pip] installing packages (this downloads several hundred MB) …")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "-r", path]
    subprocess.check_call(cmd)
    print("[pip] done.")

    print("\n[verify] quick import check …")
    for pkg in ("tensorflow", "sklearn", "cv2", "skimage", "streamlit",
                "reportlab"):
        try:
            mod = __import__({"cv2": "cv2"}.get(pkg, pkg), fromlist=["__version__"])
            ver = getattr(mod, "__version__", "?")
            print(f"  · {pkg:<12} {ver}")
        except Exception as exc:
            print(f"  · {pkg:<12} FAILED → {exc}")

    print("\nNext steps:")
    print("  1) python dataset.py           # data sanity (auto-detects layout)")
    print("  2) python train.py --quick     # 1-fold sanity run (≈ minutes)")
    print("  3) python train.py             # full 5-fold training (GPU, hours)")
    print("  4) python evaluate.py          # metrics + Grad-CAM on testing split")
    print("  5) streamlit run app.py        # interactive clinical UI")


def main() -> None:
    ap = argparse.ArgumentParser(description="FreeBuff environment setup")
    ap.add_argument("--check", action="store_true", help="Doctor mode — no install")
    ap.add_argument("--no-gpu", action="store_true", help="Force CPU install plan")
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="Never auto-create a conda environment")
    ap.add_argument("--env-name", default=ENV_NAME,
                    help=f"Conda env name to auto-create (default '{ENV_NAME}')")
    args = ap.parse_args()

    if args.check:
        import json
        print(json.dumps(doctor(), indent=2, default=str))
        return

    ok, msg = python_ok()
    print(f"[python] {msg}")

    if not ok:
        bootstrapped = False
        if not args.no_bootstrap and os.environ.get(_BOOTSTRAP_GUARD) != "1":
            bootstrapped = bootstrap_env(args.env_name, extra_args=sys.argv[1:])
        if not bootstrapped:
            print("[hint] Install a supported interpreter first, e.g. via "
                  "https://www.python.org/downloads/ or conda:\n"
                  "  conda create -n freebuff python=3.11 && conda activate freebuff\n"
                  "then re-run:  python setup.py")
            sys.exit(1)
        # bootstrapped → child process continues; this process exits above.

    install(no_gpu=args.no_gpu)


if __name__ == "__main__":
    main()
