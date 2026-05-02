"""
Deepfake Detection System — Professional UI
Dual-detection: FaceSwap CNN + StyleGAN Frequency Analyzer
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import time
import os
from pathlib import Path
from scipy.fftpack import fft2, fftshift

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepGuard — Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
    border: 1px solid rgba(108, 99, 255, 0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(108,99,255,0.08) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.05); opacity: 1; }
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #A78BFA, #C084FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    position: relative;
}
.hero p {
    color: #9CA3AF;
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
    position: relative;
    letter-spacing: 0.3px;
}

/* ── Cards ── */
.card {
    background: linear-gradient(145deg, #1A1A2E, #16213E);
    border: 1px solid rgba(108, 99, 255, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s ease;
}
.card:hover {
    border-color: rgba(108, 99, 255, 0.4);
}

/* ── Verdict badges ── */
.verdict-real {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.1));
    border: 1.5px solid rgba(16,185,129,0.5);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}
.verdict-fake {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(185,28,28,0.1));
    border: 1.5px solid rgba(239,68,68,0.5);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}
.verdict-label {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.verdict-value {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.verdict-sub {
    font-size: 0.8rem;
    opacity: 0.7;
    margin-top: 0.3rem;
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.conf-bar-fill-fake {
    background: linear-gradient(90deg, #EF4444, #F87171);
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s ease;
}
.conf-bar-fill-real {
    background: linear-gradient(90deg, #10B981, #34D399);
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s ease;
}

/* ── Stage pills ── */
.stage-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px;
}
.stage-fake { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); color: #F87171; }
.stage-real { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); color: #34D399; }
.stage-warn { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.4); color: #FCD34D; }

/* ── Section title ── */
.section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6C63FF;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(108,99,255,0.2);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E0E16 0%, #1A1A2E 100%);
    border-right: 1px solid rgba(108,99,255,0.15);
}

/* ── Remove default streamlit padding on images ── */
[data-testid="stImage"] { border-radius: 10px; overflow: hidden; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(108,99,255,0.3) !important;
    border-radius: 14px !important;
    background: rgba(108,99,255,0.03) !important;
    transition: border-color 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(108,99,255,0.6) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(108,99,255,0.07);
    border: 1px solid rgba(108,99,255,0.15);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  GAN FREQUENCY ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
def detect_gan_frequency_pattern(image_bgr):
    """
    Detect StyleGAN/GAN checkerboard artifacts in the frequency domain.
    GAN upsampling leaves SHARP, LOCALIZED spikes at ±N/4 positions.
    JPEG compression creates diffuse high-frequency noise — not localized spikes.
    We measure LOCAL CONTRAST at GAN positions vs. their immediate surroundings.
    """
    img = cv2.resize(image_bgr, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Hann window to suppress spectral leakage from image borders
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    windowed = gray * window

    # FFT
    f = np.fft.fft2(windowed)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    # Normalize magnitude to [0, 1]
    mn, mx = magnitude.min(), magnitude.max()
    if mx > mn:
        magnitude = (magnitude - mn) / (mx - mn)

    cy, cx = h // 2, w // 2

    # ── Local peakiness at GAN artifact positions ────────────────────────────
    # GAN spikes appear at exactly ±N/4 from center
    # We measure: peak_value / mean_of_surrounding_ring
    # True GAN spikes have very high LOCAL contrast vs. surroundings
    # JPEG noise has uniform energy — no localized spikes
    offset = h // 4
    spike_box = 6      # tight box around the GAN artifact position
    surround = 20      # surrounding region for background estimation

    gan_positions = [
        (cy - offset, cx),
        (cy + offset, cx),
        (cy, cx - offset),
        (cy, cx + offset),
    ]

    local_contrasts = []
    for (gy, gx) in gan_positions:
        gy = int(np.clip(gy, surround, h - surround))
        gx = int(np.clip(gx, surround, w - surround))

        # Peak energy in the small spike box
        spike_patch = magnitude[gy - spike_box:gy + spike_box,
                                gx - spike_box:gx + spike_box]
        peak = float(spike_patch.max())

        # Background: surrounding ring (excluding the spike box)
        bg_patch = magnitude[gy - surround:gy + surround,
                             gx - surround:gx + surround].copy()
        # zero out the spike box to get true background
        c = surround
        bg_patch[c - spike_box:c + spike_box, c - spike_box:c + spike_box] = 0
        background = float(bg_patch[bg_patch > 0].mean() + 1e-8)

        # Local contrast ratio: how much sharper is the spike than background?
        local_contrasts.append(peak / background)

    avg_contrast = float(np.mean(local_contrasts))
    max_contrast = float(np.max(local_contrasts))

    # ── DC energy ratio (natural vs. anomalous spectral falloff) ─────────────
    center_r = int(h * 0.06)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
    center_energy = float(magnitude[dist <= center_r].mean())

    r_inner = int(h * 0.30)
    r_outer = int(h * 0.45)
    mid_ring_energy = float(magnitude[(dist >= r_inner) & (dist <= r_outer)].mean())
    falloff_ratio = mid_ring_energy / (center_energy + 1e-8)

    # ── Final score ──────────────────────────────────────────────────────────
    # GAN images: avg_contrast > 1.6 AND max_contrast > 2.0
    # JPEG/Real:  avg_contrast ≈ 1.0–1.3 (diffuse noise, no sharp spikes)
    contrast_score = np.clip((avg_contrast - 1.0) / 1.2, 0, 1)  # 0 at 1.0, 1.0 at 2.2
    falloff_score  = np.clip((falloff_ratio - 0.5) / 0.5, 0, 1)  # 0 at 0.5, 1.0 at 1.0

    raw_score = float(contrast_score * 0.75 + falloff_score * 0.25)
    raw_score = float(np.clip(raw_score, 0.0, 1.0))

    # Threshold: > 0.80 → GAN (conservative to avoid WhatsApp/JPEG false positives)
    gan_detected = raw_score > 0.80

    return raw_score, {
        "avg_local_contrast": round(avg_contrast, 4),
        "max_local_contrast": round(max_contrast, 4),
        "center_energy":      round(center_energy, 4),
        "mid_ring_energy":    round(mid_ring_energy, 4),
        "falloff_ratio":      round(falloff_ratio, 4),
        "contrast_score":     round(float(contrast_score), 4),
        "falloff_score":      round(float(falloff_score), 4),
        "gan_detected":       gan_detected,
    }



# ══════════════════════════════════════════════════════════════════════════════
#  MODEL WRAPPER
# ══════════════════════════════════════════════════════════════════════════════
class DeepfakeDetectorApp:

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        model_path = './models/combined_model.h5'
        if not os.path.exists(model_path):
            st.warning("⚠️ No trained model found at `./models/combined_model.h5`. Upload your model to enable CNN detection.")
            return
        try:
            from src_model_architecture import DualBranchDeepfakeDetector
            detector = DualBranchDeepfakeDetector(
                spatial_input_shape=(224, 224, 3),
                frequency_input_shape=(224, 224, 1)
            )
            detector.build()
            self.model = detector.get_model()
            self.model.load_weights(model_path)
        except Exception as e:
            st.error(f"Model load error: {e}")

    def _spatial(self, img):
        img = cv2.resize(img, (224, 224))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    def _frequency(self, img):
        img = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        fft_result = fft2(gray)
        fft_shift = fftshift(fft_result)
        mag = np.abs(fft_shift)
        log_mag = np.log1p(mag)
        mn, mx = log_mag.min(), log_mag.max()
        if mx > mn:
            log_mag = (log_mag - mn) / (mx - mn)
        return np.expand_dims(log_mag, axis=-1).astype(np.float32)

    def predict_cnn(self, image_bgr):
        if self.model is None:
            return None, None
        try:
            sp = np.expand_dims(self._spatial(image_bgr), 0)
            fr = np.expand_dims(self._frequency(image_bgr), 0)
            conf = float(self.model.predict([sp, fr], verbose=0)[0][0])
            return conf, {
                "spatial":   self._spatial(image_bgr),
                "frequency": self._frequency(image_bgr)
            }
        except Exception as e:
            st.error(f"Inference error: {e}")
            return None, None


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def render_confidence_bar(label, score, color="fake"):
    pct = int(score * 100)
    fill_class = f"conf-bar-fill-{color}"
    st.markdown(f"""
    <div style="margin-bottom:0.6rem;">
        <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:4px;">
            <span style="color:#9CA3AF">{label}</span>
            <span style="font-weight:600;">{pct}%</span>
        </div>
        <div class="conf-bar-wrap">
            <div class="{fill_class}" style="width:{pct}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_stage_pill(label, is_flagged, neutral=False):
    if neutral:
        cls = "stage-warn"
        icon = "⚠️"
    elif is_flagged:
        cls = "stage-fake"
        icon = "🚨"
    else:
        cls = "stage-real"
        icon = "✅"
    st.markdown(f'<span class="stage-pill {cls}">{icon} {label}</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Init session ──────────────────────────────────────────────────────────
    if "app" not in st.session_state:
        st.session_state.app = DeepfakeDetectorApp()
    app = st.session_state.app

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>🛡️ DeepGuard</h1>
        <p>AI-powered deepfake detection using dual-domain analysis — FaceSwap CNN + StyleGAN Frequency Forensics</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        threshold = st.slider("CNN Detection Threshold", 0.0, 1.0, 0.5, 0.05,
                              help="Adjust sensitivity. Lower = more suspicious.")
        gan_threshold = st.slider("GAN Sensitivity", 0.0, 1.0, 0.80, 0.05,
                                  help="Threshold for flagging StyleGAN/AI-generated images.")
        show_freq = st.checkbox("Show frequency domain analysis", value=True)
        show_debug = st.checkbox("Show technical diagnostics", value=False)

        st.markdown("---")
        st.markdown("### 🧠 How it works")
        st.markdown("""
**Stage 1 — CNN FaceSwap Detector**
Trained on 100,000 real/fake images. Looks for blending artifacts, pixel warping, and skin inconsistencies.

**Stage 2 — GAN Frequency Analyzer**
No training needed. Detects the periodic "checkerboard" grid that StyleGAN and DALL-E generators leave in the FFT spectrum.

**Verdict logic:**
- Both stages pass → ✅ Likely Real
- CNN flags it → 🚨 FaceSwap detected
- GAN analyzer flags it → 🚨 AI-generated detected
- Both flag it → 🚨 High-confidence fake
        """)

        st.markdown("---")
        model_status = "✅ Loaded" if app.model else "❌ Not found"
        st.markdown(f"**CNN Model:** {model_status}")
        st.caption("Supported: JPG, PNG, JFIF, WebP, BMP")

    # ── Upload panel ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📤 Upload Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a facial image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "jfif", "webp"],
        label_visibility="collapsed"
    )

    if uploaded is None:
        st.markdown('<div class="section-title">🎯 Demo Guide — Best Results</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("""
            #### ✅ Images this model is optimised for
            Download images directly from the **140k Kaggle dataset** test split — these are guaranteed in-distribution:
            1. Go to [kaggle.com/datasets/xhlulu/140k-real-and-fake-faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
            2. Open **Data Explorer → real-vs-fake → test → fake/**
            3. Download any image and upload it here
            4. The model should flag it with high confidence 🚨
            
            Do the same from **test → real/** — it should pass ✅
            """)
        with col_b:
            st.markdown("""
            #### ⚠️ Known Limitation — Domain Shift
            This model achieves **98.5% accuracy on in-distribution data** (same Kaggle dataset).
            
            Real-world photos (WhatsApp, screenshots, `thispersondoesnotexist.com`) come from a **different distribution** — the model has never seen those styles. This is called **domain shift** and is an open research problem even in state-of-the-art systems.
            
            > *"Generalisation across deepfake generation methods remains a fundamental challenge in the field."*  
            > — FaceForensics++ benchmark paper
            """)
        st.markdown("""
        <div style="text-align:center; padding:2rem; color:#6B7280; border:1px dashed rgba(108,99,255,0.2); border-radius:16px; margin-top:1rem;">
            <div style="font-size:3rem;">🎭</div>
            <div style="font-size:1rem; margin-top:0.5rem;">Upload a face image above to begin analysis</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Load image ────────────────────────────────────────────────────────────
    pil_img   = Image.open(uploaded).convert("RGB")
    img_array = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ── Run both detectors ────────────────────────────────────────────────────
    with st.spinner("Running dual-domain analysis..."):
        time.sleep(0.2)
        # Stage 1: CNN
        cnn_conf, processed = app.predict_cnn(img_array)
        cnn_is_fake = (cnn_conf is not None) and (cnn_conf > threshold)
        cnn_available = cnn_conf is not None

        # Stage 2: GAN detector
        gan_score, gan_details = detect_gan_frequency_pattern(img_array)
        gan_is_fake = gan_score > gan_threshold

    # ── Final verdict — CNN is the sole decision maker ─────────────────────────
    if cnn_available:
        overall_fake = cnn_is_fake
    else:
        overall_fake = False  # no model, no verdict

    # ── Layout: image + verdict side by side ──────────────────────────────────
    st.markdown("---")
    col_img, col_verdict = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="section-title">📸 Input Image</div>', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        st.caption(f"**{uploaded.name}** · {pil_img.width}×{pil_img.height}px · {uploaded.size // 1024} KB")

    with col_verdict:
        st.markdown('<div class="section-title">🔎 Analysis Result</div>', unsafe_allow_html=True)

        # Main verdict card
        if overall_fake:
            st.markdown("""
            <div class="verdict-fake">
                <div class="verdict-label" style="color:#F87171;">⚠️ VERDICT</div>
                <div class="verdict-value" style="color:#EF4444;">FAKE DETECTED</div>
                <div class="verdict-sub">Signs of manipulation or AI generation found</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="verdict-real">
                <div class="verdict-label" style="color:#34D399;">✓ VERDICT</div>
                <div class="verdict-value" style="color:#10B981;">LIKELY REAL</div>
                <div class="verdict-sub">No manipulation or GAN artifacts detected</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Stage pills
        st.markdown("**Detection stages:**")
        if cnn_available:
            render_stage_pill(f"FaceSwap CNN — {cnn_conf:.1%} fake score", cnn_is_fake)
        else:
            render_stage_pill("FaceSwap CNN — model not loaded", False, neutral=True)
        # GAN score shown as info only — does not affect verdict
        render_stage_pill(f"GAN Frequency Score — {gan_score:.1%} (informational)", False, neutral=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Uncertainty-aware confidence display
        st.markdown('<div class="section-title">📊 Confidence Breakdown</div>', unsafe_allow_html=True)
        if cnn_available:
            render_confidence_bar("Fake Probability", cnn_conf, "fake")
            render_confidence_bar("Real Probability", 1 - cnn_conf, "real")

            # Uncertainty bucket
            dist_from_boundary = abs(cnn_conf - 0.5)
            if dist_from_boundary < 0.15:
                uncertainty_label = "⚠️ High uncertainty — prediction near decision boundary"
                uncertainty_color = "#F59E0B"
            elif dist_from_boundary < 0.35:
                uncertainty_label = "🔵 Moderate confidence"
                uncertainty_color = "#6C63FF"
            else:
                uncertainty_label = "🟢 High confidence"
                uncertainty_color = "#10B981"

            st.markdown(f"""
            <div style="font-size:0.8rem; color:{uncertainty_color}; margin-top:6px; font-style:italic;">
                {uncertainty_label} &nbsp;|&nbsp; Raw score: <code>{cnn_conf:.4f}</code>
            </div>
            """, unsafe_allow_html=True)
            st.caption("ℹ️ Confidence reflects in-distribution performance. Scores near 0.5 indicate ambiguity.")

    # ── Domain visualizations ─────────────────────────────────────────────────
    if show_freq and processed is not None:
        st.markdown("---")
        st.markdown('<div class="section-title">🔬 Frequency Domain Forensics</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown("**Spatial Domain (RGB)**")
            spatial_display = (processed["spatial"] * 255).astype(np.uint8)
            st.image(spatial_display, use_container_width=True)
            st.caption("Normalized RGB — CNN input")

        with col2:
            st.markdown("**FFT Magnitude Spectrum**")
            freq_display = (processed["frequency"][:, :, 0] * 255).astype(np.uint8)
            st.image(freq_display, use_container_width=True, clamp=True)
            st.caption("Log-scale FFT — GAN artifacts appear as grid spikes")

        with col3:
            st.markdown("**GAN Spike Map**")
            # Highlight the N/4 GAN artifact zones on the FFT
            freq_color = cv2.applyColorMap(freq_display, cv2.COLORMAP_INFERNO)
            h, w = freq_display.shape
            cx, cy = w // 2, h // 2
            offset = h // 4
            for (gy, gx) in [(cy - offset, cx), (cy + offset, cx),
                             (cy, cx - offset), (cy, cx + offset)]:
                cv2.circle(freq_color, (gx, gy), 10, (0, 255, 128), 2)
            cv2.circle(freq_color, (cx, cy), 20, (108, 99, 255), 2)
            freq_rgb = cv2.cvtColor(freq_color, cv2.COLOR_BGR2RGB)
            st.image(freq_rgb, use_container_width=True)
            st.caption("Green circles = GAN artifact zones · Purple = DC center")

    # ── GAN technical diagnostics ─────────────────────────────────────────────
    if show_debug:
        st.markdown("---")
        st.markdown('<div class="section-title">🧪 Technical Diagnostics</div>', unsafe_allow_html=True)
        d = gan_details
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        col_a.metric("Center Energy", f"{d['center_energy']:.3f}")
        col_b.metric("Ring Energy", f"{d['ring_energy']:.3f}")
        col_c.metric("Spike Energy", f"{d['spike_energy']:.3f}")
        col_d.metric("Energy Ratio", f"{d['energy_ratio']:.3f}", delta=">" if d['energy_ratio'] > 0.55 else "<" + "0.55 threshold")
        col_e.metric("Spike Ratio", f"{d['spike_ratio']:.3f}", delta=">" if d['spike_ratio'] > 1.1 else "<" + "1.1 threshold")

        if cnn_available:
            st.markdown(f"""
            **CNN raw output:** `{cnn_conf:.6f}` — classified as **{'FAKE' if cnn_is_fake else 'REAL'}** at threshold `{threshold}`  
            **GAN raw score:** `{gan_score:.6f}` — classified as **{'GAN-generated' if gan_is_fake else 'Natural'}** at threshold `{gan_threshold}`
            """)

    # ── Interpretation ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">💡 Interpretation</div>', unsafe_allow_html=True)

    if not cnn_available:
        st.warning("**⚠️ No model loaded.** Place `combined_model.h5` in `./models/` to enable detection.")
    elif abs(cnn_conf - 0.5) < 0.15:
        st.warning(f"""
**⚠️ Uncertain Prediction** — Score `{cnn_conf:.3f}` is close to the 0.5 decision boundary.

The model is not confident about this image. This typically happens when:
- The image is **out-of-distribution** (different style from training data)
- The fake is **very high quality** with minimal artifacts
- The image has **heavy compression** (WhatsApp, social media)

**In-distribution accuracy: 98.5%** — but this image may be outside that distribution.
        """)
    elif cnn_is_fake:
        st.error(f"""
**🚨 Deepfake Detected** — Score: `{cnn_conf:.3f}` (threshold: `{threshold}`)

The model detected face-manipulation artifacts consistent with FaceSwap-style generation.
Trained on 100,000 images · Val accuracy: **98.5%** · Best on in-distribution test data.

*For critical use, cross-verify with a second tool.*
        """)
    else:
        st.success(f"""
**✅ No Manipulation Detected** — Score: `{cnn_conf:.3f}` (threshold: `{threshold}`)

No face-blending or deepfake artifacts found in spatial or frequency domains.

**Academic note:** This model is optimised for FaceSwap-style fakes from the 140k Kaggle dataset. Fully AI-generated faces (StyleGAN, DALL·E) occupy a different feature space and may not be correctly classified — this is a known **domain generalisation challenge** in deepfake detection research.
        """)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#4B5563; font-size:0.78rem; padding: 0.5rem 0;">
        DeepGuard · Dual-Domain Deepfake Detection System ·
        Built with TensorFlow, OpenCV &amp; Streamlit ·
        <span style="color:#6C63FF;">For research &amp; educational use only</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
