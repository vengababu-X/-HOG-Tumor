#!/usr/bin/env python3
# ==============================================================================
# FreeBuff v4 — Streamlit Frontend
# ==============================================================================
# Connects to the local FastAPI backend at http://127.0.0.1:8000.
# All inference runs server-side — this file does NO model loading.
#
# Run:
#     1. Start the backend:  python app.py
#     2. Start this frontend:  streamlit run frontend.py
# ==============================================================================
from __future__ import annotations

import base64
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import CLASSES, REGULATORY_SHORTHAND, REGULATORY_DISCLAIMER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE_URL = "http://127.0.0.1:8000"

# ==============================================================================
# PAGE CONFIG (must be first Streamlit command)
# ==============================================================================
st.set_page_config(
    page_title="FreeBuff v4 — Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM CSS
# ==============================================================================
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0 0 0;
    }
    .alert-banner {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        margin-bottom: 1rem;
    }
    .success-banner {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
        margin-bottom: 1rem;
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin: 0.4rem 0;
    }
    .prob-label {
        width: 120px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .prob-bar-bg {
        flex: 1;
        height: 22px;
        background: #e9ecef;
        border-radius: 11px;
        overflow: hidden;
        margin: 0 10px;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 11px;
        transition: width 0.6s ease;
    }
    .prob-value {
        width: 60px;
        text-align: right;
        font-weight: 600;
        font-size: 0.9rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# SESSION STATE
# ==============================================================================
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def check_api_health() -> Dict[str, Any]:
    """Ping GET /health and return the parsed response or an error dict."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the API server. Is it running?"}
    except requests.exceptions.Timeout:
        return {"error": "API server timed out."}
    except requests.exceptions.HTTPError as exc:
        return {"error": f"API returned error {exc.response.status_code}"}
    except Exception as exc:
        return {"error": f"Health check failed: {exc}"}


def call_predict_api(
    image_bytes: bytes,
    filename: str,
    use_tta: bool = True,
    generate_gradcam: bool = True,
) -> Dict[str, Any]:
    """POST an image to POST /predict. Returns parsed JSON on success."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
    mime_map = {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "bmp": "image/bmp", "tiff": "image/tiff", "tif": "image/tiff",
    }
    content_type = mime_map.get(ext, "image/png")

    resp = requests.post(
        f"{API_BASE_URL}/predict",
        files={"file": (filename, image_bytes, content_type)},
        data={
            "use_tta": str(use_tta).lower(),
            "generate_gradcam": str(generate_gradcam).lower(),
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def decode_gradcam_image(b64_string: str) -> Optional[Image.Image]:
    """Decode a base64-encoded PNG string to a PIL Image."""
    if not b64_string:
        return None
    try:
        raw = base64.b64decode(b64_string)
        return Image.open(BytesIO(raw))
    except Exception:
        return None


def get_confidence_color(confidence: float) -> str:
    """Hex colour based on confidence level."""
    if confidence >= 0.90:
        return "#2ecc71"
    elif confidence >= 0.75:
        return "#3498db"
    elif confidence >= 0.60:
        return "#f39c12"
    return "#e74c3c"


def get_confidence_label(confidence: float) -> str:
    """Human-readable confidence tier."""
    if confidence >= 0.90:
        return "High"
    elif confidence >= 0.75:
        return "Moderate"
    elif confidence >= 0.60:
        return "Low"
    return "Very Low"


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("# 🧠 FreeBuff v4")
    st.markdown("*Brain Tumor Classification Dashboard*")
    st.markdown("---")

    st.markdown("### ⚙️ API Configuration")
    st.markdown(f"**Backend:** `{API_BASE_URL}`")

    if st.button("🔍 Check API Health", use_container_width=True):
        with st.spinner("Checking API health…"):
            health = check_api_health()
        if "error" in health:
            st.error(health["error"])
        else:
            st.success("API is healthy!")
            st.json(health)

    st.markdown("---")

    st.markdown("### 🎛️ Inference Settings")
    use_tta = st.checkbox(
        "Enable TTA (Test-Time Augmentation)",
        value=True,
        help="Uses 48 augmented views for robust prediction.",
    )
    generate_gradcam = st.checkbox(
        "Generate Grad-CAM++ Heatmap",
        value=True,
        help="Explainability heatmap showing which regions influenced the prediction.",
    )
    gradcam_opacity = st.slider(
        "Heatmap Overlay Opacity",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    )

    st.markdown("---")

    st.markdown("### ℹ️ System Info")
    st.markdown(
        "**Model:** ResNet50 + CBAM + PCA-HOG Fusion\n"
        "**Input:** 144×144 px MRI scans\n"
        "**Classes:** Glioma, Meningioma, No Tumor, Pituitary\n"
        "**TTA Views:** 48 (flips × rotations × scales)"
    )

    st.markdown("---")
    st.markdown(
        f"<small>⚠️ {REGULATORY_SHORTHAND}</small>",
        unsafe_allow_html=True,
    )
    with st.expander("Full Disclaimer"):
        st.text(REGULATORY_DISCLAIMER)


# ==============================================================================
# MAIN CONTENT
# ==============================================================================
st.markdown(
    '<p class="main-header">🧠 Brain Tumor Classification</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">'
    "Upload a brain MRI scan for AI-powered classification with explainability"
    "</p>",
    unsafe_allow_html=True,
)

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "📤 Upload Brain MRI Scan",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    help="Supported formats: PNG, JPG, JPEG, BMP, TIFF",
)

if uploaded_file is None:
    # --- Empty state ---
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 1rem; color: #999;">
            <p style="font-size: 4rem; margin-bottom: 0;">🧠</p>
            <p style="font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem;">
                Upload a Brain MRI Scan
            </p>
            <p style="font-size: 1rem;">
                Drag and drop a PNG, JPG, or JPEG image to begin classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick connection status
    health = check_api_health()
    if "error" in health:
        st.warning(
            f"⚠️ API server not reachable at `{API_BASE_URL}`.\n\n"
            "Start the backend first:\n"
            "```bash\npython app.py\n```"
        )
    else:
        device = health.get("device", "unknown")
        gpu = health.get("gpu_name", "N/A")
        st.info(f"✅ API connected — Device: {device} ({gpu})")

else:
    # --- Image uploaded ---
    image_bytes = uploaded_file.read()
    original_image = Image.open(BytesIO(image_bytes))

    # Preview
    col_preview, col_info = st.columns([2, 1])
    with col_preview:
        st.markdown("#### 📷 Uploaded Image")
        st.image(
            original_image,
            caption=f"{uploaded_file.name} — {original_image.size[0]}×{original_image.size[1]} px",
            use_container_width=True,
        )
    with col_info:
        st.markdown("#### 📋 Image Info")
        st.markdown(f"**Filename:** `{uploaded_file.name}`")
        st.markdown(f"**Format:** {original_image.format or 'Unknown'}")
        st.markdown(f"**Mode:** {original_image.mode}")
        st.markdown(f"**Size:** {original_image.size[0]} × {original_image.size[1]} px")
        st.markdown(f"**File size:** {len(image_bytes) / 1024:.1f} KB")
        if original_image.mode not in ("RGB",):
            st.info(f"ℹ️ Image will be converted from `{original_image.mode}` to `RGB`.")

    st.markdown("---")

    # --- Run Analysis Button ---
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    if not run_button:
        st.info("Upload an image and click **Run Analysis** to begin classification.")
    else:
        # 1. Pre-flight: verify API health
        with st.spinner("Checking API connection…"):
            health = check_api_health()

        if "error" in health:
            st.error(f"❌ API Error: {health['error']}")
            st.info(
                "Ensure the FastAPI server is running:\n"
                "```bash\npython app.py\n```"
            )
            st.stop()

        if not health.get("model_loaded", False):
            st.error("❌ The API server is running but the model is not loaded.")
            st.info("Ensure a valid checkpoint exists at `saved_models/freebuff_v4.pt`.")
            st.stop()

        # 2. Call the API
        tta_label = "with TTA (48 views)" if use_tta else "without TTA (single pass)"
        gradcam_label = " + Grad-CAM++" if generate_gradcam else ""
        with st.spinner(f"Running inference {tta_label}{gradcam_label}… This may take a moment."):
            try:
                t0 = time.time()
                result = call_predict_api(
                    image_bytes=image_bytes,
                    filename=uploaded_file.name,
                    use_tta=use_tta,
                    generate_gradcam=generate_gradcam,
                )
                elapsed_ms = (time.time() - t0) * 1000
                st.session_state.last_result = result
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code
                if status == 422:
                    try:
                        detail = exc.response.json().get("detail", str(exc))
                    except Exception:
                        detail = str(exc)
                    st.error(f"❌ Invalid image: {detail}")
                elif status == 503:
                    st.error("❌ Model not loaded on the server. Check the checkpoint path.")
                else:
                    st.error(f"❌ API returned error {status}: {exc}")
                st.stop()
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the API server. Is it running?")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")
                st.stop()

        # 3. Display results
        result = st.session_state.last_result

        # Safety alert
        if result.get("potential_tumor", False):
            alert = result.get("safety_alert", "Potential Tumor Detected: requires radiologist review.")
            tumor_prob = result.get("tumor_probability", 0.0)
            st.markdown(
                f'<div class="alert-banner">'
                f"⚠️ {alert}<br>"
                f"<small>Aggregate tumor probability: {tumor_prob:.1%}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="success-banner">'
                "✅ No tumor detected with high confidence"
                "</div>",
                unsafe_allow_html=True,
            )

        # Metric cards
        pred_class = result["predicted_class"]
        confidence = result["confidence"]
        conf_label = get_confidence_label(confidence)
        proc_ms = result.get("processing_time_ms", elapsed_ms)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-card"><h3>Prediction</h3>'
                f'<div class="value">{pred_class}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><h3>Confidence</h3>'
                f'<div class="value">{confidence:.1%}</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card"><h3>Confidence Level</h3>'
                f'<div class="value">{conf_label}</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-card"><h3>Inference Time</h3>'
                f'<div class="value">{proc_ms:.0f}ms</div></div>',
                unsafe_allow_html=True,
            )

        # Confidence bar
        st.markdown(f"**Confidence: {confidence:.2%}**")
        st.progress(confidence)
        st.markdown("---")

        # Grad-CAM++ section
        gradcam_b64 = result.get("gradcam_heatmap_base64", "")
        gradcam_image = decode_gradcam_image(gradcam_b64) if gradcam_b64 else None

        if gradcam_image is not None:
            st.markdown("### 🔬 Grad-CAM++ Explainability Heatmap")
            st.markdown(
                "The heatmap highlights which regions of the MRI scan most "
                "influenced the model's prediction."
            )

            col_orig, col_heatmap = st.columns(2)
            with col_orig:
                st.markdown("**Original Image**")
                st.image(original_image, use_container_width=True, caption="Input MRI")
            with col_heatmap:
                st.markdown("**Grad-CAM++ Overlay**")
                if gradcam_opacity < 1.0:
                    orig_rgba = original_image.resize(gradcam_image.size).convert("RGBA")
                    heat_rgba = gradcam_image.resize(original_image.size).convert("RGBA")
                    blended = Image.blend(orig_rgba, heat_rgba, alpha=gradcam_opacity)
                    st.image(blended, use_container_width=True, caption=f"Overlay ({gradcam_opacity:.0%})")
                else:
                    st.image(gradcam_image, use_container_width=True, caption="Full overlay")
        else:
            if generate_gradcam:
                st.info("ℹ️ Grad-CAM++ heatmap was not generated. Check server logs.")

        st.markdown("---")

        # Per-class probabilities
        st.markdown("### 📊 Per-Class Probability Distribution")
        probabilities = result.get("probabilities", [])
        if probabilities:
            sorted_probs = sorted(probabilities, key=lambda x: x["probability"], reverse=True)
            bar_colors = ["#667eea", "#764ba2", "#2ecc71", "#e74c3c"]
            prob_html = ""
            for idx, item in enumerate(sorted_probs):
                cls = item["class_name"]
                prob = item["probability"]
                color = bar_colors[idx % len(bar_colors)]
                is_predicted = cls == pred_class
                border = "border: 2px solid #333;" if is_predicted else ""
                label_extra = " ⭐" if is_predicted else ""
                prob_html += (
                    f'<div class="prob-row">'
                    f'<span class="prob-label">{cls}{label_extra}</span>'
                    f'<div class="prob-bar-bg">'
                    f'<div class="prob-bar-fill" style="width: {prob*100:.1f}%; background: {color}; {border}">'
                    f"</div></div>"
                    f'<span class="prob-value">{prob:.1%}</span>'
                    f"</div>"
                )
            st.markdown(prob_html, unsafe_allow_html=True)

        st.markdown("---")

        # Processing details
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("#### 📋 Processing Details")
            st.markdown(f"- **Model version:** {result.get('model_version', 'N/A')}")
            st.markdown(f"- **TTA enabled:** {'Yes' if result.get('tta_used', False) else 'No'}")
            st.markdown(f"- **TTA transforms:** {result.get('num_tta_transforms', 0)}")
            st.markdown(f"- **Processing time:** {result.get('processing_time_ms', 0):.0f} ms")
            st.markdown(f"- **Tumor probability:** {result.get('tumor_probability', 0):.1%}")

        with col_info2:
            st.markdown("#### 🏥 Safety Assessment")
            has_tumor = result.get("potential_tumor", False)
            st.markdown(f"- **Potential tumor:** {'Yes ⚠️' if has_tumor else 'No ✅'}")
            safety = result.get("safety_alert", "")
            if safety:
                st.warning(safety)
            else:
                st.success("No safety alerts triggered.")

        # Raw JSON (debug)
        with st.expander("🔧 Raw API Response (Debug)", expanded=False):
            debug = {k: v for k, v in result.items() if k != "gradcam_heatmap_base64"}
            if gradcam_b64:
                debug["gradcam_heatmap_base64"] = f"[{len(gradcam_b64)} chars, base64 PNG]"
            st.json(debug)

        # Disclaimer footer
        st.markdown("---")
        st.warning(REGULATORY_SHORTHAND)
        with st.expander("Full Regulatory & Medical Disclaimer"):
            st.text(REGULATORY_DISCLAIMER)

# --- Footer ---
st.markdown("---")
st.caption(
    "FreeBuff v4 (v4.0.0) — Research prototype. Not FDA-cleared or approved. "
    "For research, educational, and informational purposes only."
)
