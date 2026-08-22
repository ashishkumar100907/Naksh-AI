import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from gradcam import make_gradcam_heatmap


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NAKSH AI | Astronomical Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS & ASSETS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"
CONFUSION_MATRIX_PATH = BASE_DIR / "model" / "confusion_matrix.png"
ASSETS_DIR = BASE_DIR / "assets"


def get_base64_image(image_path):
    """Convert an image file to a base64 Data URI for embedding in HTML/CSS."""
    path = Path(image_path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = f.read()
    ext = path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"


hero_bg_b64 = get_base64_image(ASSETS_DIR / "hero-space.jpg")
cosmic_banner_b64 = get_base64_image(ASSETS_DIR / "cosmic-banner.jpg")
moon_b64 = get_base64_image(ASSETS_DIR / "moon.jpg")
nebula_b64 = get_base64_image(ASSETS_DIR / "nebula.jpg")
planet_b64 = get_base64_image(ASSETS_DIR / "planet.jpg")
star_b64 = get_base64_image(ASSETS_DIR / "star.jpg")


# ============================================================
# STYLES (CUSTOM COSMIC DESIGN SYSTEM)
# ============================================================

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* GLOBAL RESETS & VARIABLES */
:root {{
    --font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --bg-dark: #070913;
    --card-bg: rgba(14, 19, 36, 0.72);
    --card-border: rgba(255, 255, 255, 0.08);
    --card-border-hover: rgba(124, 92, 255, 0.35);
    --accent-purple: #7c5cff;
    --accent-blue: #3b82f6;
    --accent-cyan: #38bdf8;
    --accent-green: #10b981;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --text-subtle: #64748b;
}}

html, body, [class*="css"] {{
    font-family: var(--font-family) !important;
}}

/* Main App Background */
.stApp {{
    background: 
        radial-gradient(circle at 10% 15%, rgba(124, 92, 255, 0.08), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(59, 130, 246, 0.08), transparent 35%),
        radial-gradient(circle at 50% 85%, rgba(16, 185, 129, 0.05), transparent 40%),
        var(--bg-dark);
    color: var(--text-main);
}}

/* Remove default Streamlit top header whitespace */
[data-testid="stHeader"] {{
    background: transparent !important;
}}

.main .block-container {{
    max-width: 1440px;
    padding-top: 1rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}}

/* Links */
a {{
    color: #60a5fa !important;
    text-decoration: none !important;
    transition: color 0.2s ease;
}}
a:hover {{
    color: #a78bfa !important;
    text-decoration: underline !important;
}}

/* ============================================================
   SIDEBAR STYLING
   ============================================================ */
section[data-testid="stSidebar"] {{
    background: #090c1a !important;
    border-right: 1px solid var(--card-border) !important;
}}

section[data-testid="stSidebar"] * {{
    color: #cbd5e1;
}}

.sb-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 4px 18px;
}}

.sb-logo-icon {{
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: radial-gradient(circle, #7c5cff 0%, #3b82f6 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 16px rgba(124, 92, 255, 0.4);
}}

.sb-brand-text {{
    display: flex;
    flex-direction: column;
}}

.sb-title {{
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
    line-height: 1.1;
}}

.sb-title span {{
    color: #8b5cf6;
}}

.sb-subtitle {{
    font-size: 0.72rem;
    color: var(--text-subtle);
    letter-spacing: 0.04em;
    margin-top: 2px;
}}

.sb-section-title {{
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #64748b;
    text-transform: uppercase;
    margin: 20px 0 10px 4px;
}}

.stButton > button {{
    width: 100%;
    border-radius: 12px;
    height: 44px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #cbd5e1 !important;
    transition: all 0.2s ease !important;
}}

.stButton > button:hover {{
    background: rgba(124, 92, 255, 0.15) !important;
    border-color: rgba(124, 92, 255, 0.4) !important;
    color: #ffffff !important;
}}

.sb-status-card {{
    margin-top: 20px;
    padding: 14px;
    border-radius: 14px;
    background: rgba(16, 185, 129, 0.06);
    border: 1px solid rgba(16, 185, 129, 0.2);
}}

.sb-status-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #10b981;
    text-transform: uppercase;
}}

.sb-status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
}}

.sb-status-sub {{
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 4px;
}}

.sb-footer {{
    margin-top: 24px;
    font-size: 0.72rem;
    color: #475569;
    text-align: left;
    padding-left: 4px;
}}


/* ============================================================
   HERO BANNER
   ============================================================ */
.hero-container {{
    position: relative;
    width: 100%;
    min-height: 280px;
    border-radius: 24px;
    overflow: hidden;
    margin-bottom: 28px;
    border: 1px solid var(--card-border);
    background: 
        linear-gradient(90deg, #090d1f 0%, rgba(9, 13, 31, 0.85) 45%, rgba(9, 13, 31, 0.25) 100%),
        url('{hero_bg_b64}') center right / cover no-repeat;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
}}

.hero-content {{
    padding: 42px 48px;
    max-width: 680px;
    position: relative;
    z-index: 2;
}}

.hero-kicker {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    color: #a78bfa;
    text-transform: uppercase;
    margin-bottom: 12px;
}}

.hero-title {{
    font-size: clamp(2.2rem, 4vw, 3.2rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.03em;
    margin-bottom: 16px;
    color: #ffffff;
}}

.hero-title span {{
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.hero-desc {{
    font-size: 0.95rem;
    line-height: 1.65;
    color: #94a3b8;
    margin-bottom: 24px;
}}

.hero-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 99px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}}

.hero-pulse {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 8px #34d399;
}}


/* ============================================================
   EXPLORE CELESTIAL OBJECTS
   ============================================================ */
.section-header-title {{
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 14px;
}}

/* ============================================================
   MAIN PANELS & CARDS
   ============================================================ */
.panel-card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 24px;
    height: 100%;
}}

.panel-header-title {{
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #94a3b8;
    text-transform: uppercase;
    margin-bottom: 20px;
}}

/* File Upload Custom Styling */
[data-testid="stFileUploader"] {{
    border: 1.5px dashed rgba(96, 165, 250, 0.3) !important;
    border-radius: 16px !important;
    background: rgba(15, 23, 42, 0.4) !important;
    padding: 16px !important;
    text-align: center;
}}

[data-testid="stFileUploader"]:hover {{
    border-color: rgba(167, 139, 250, 0.6) !important;
}}

.primary-btn > button {{
    width: 100%;
    border-radius: 12px;
    height: 48px;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}}

.primary-btn > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.45) !important;
}}


/* Timeline Items (How it works) */
.timeline {{
    display: flex;
    flex-direction: column;
    gap: 0;
    position: relative;
    padding: 8px 0;
}}

.timeline-step {{
    display: flex;
    gap: 16px;
    position: relative;
    padding-bottom: 28px;
}}

.timeline-step:last-child {{
    padding-bottom: 0;
}}

.timeline-number {{
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: rgba(30, 41, 59, 0.8);
    border: 1.5px solid rgba(124, 92, 255, 0.4);
    color: #a78bfa;
    font-size: 0.85rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    z-index: 2;
}}

.timeline-step:not(:last-child)::after {{
    content: "";
    position: absolute;
    top: 38px;
    left: 18px;
    width: 1px;
    height: calc(100% - 38px);
    border-left: 2px dashed rgba(124, 92, 255, 0.25);
}}

.timeline-content {{
    display: flex;
    flex-direction: column;
    justify-content: center;
}}

.timeline-title {{
    font-size: 0.95rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
}}

.timeline-desc {{
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.45;
}}


/* Donut Chart Accuracy Gauge */
.gauge-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 10px 0 20px;
}}

.gauge-circle {{
    position: relative;
    width: 150px;
    height: 150px;
    border-radius: 50%;
    background: conic-gradient(#8b5cf6 0% 98.26%, rgba(255, 255, 255, 0.06) 98.26% 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 30px rgba(139, 92, 246, 0.2);
}}

.gauge-inner {{
    width: 118px;
    height: 118px;
    border-radius: 50%;
    background: #0d1222;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}}

.gauge-val {{
    font-size: 1.4rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
}}

.gauge-lbl {{
    font-size: 0.68rem;
    color: #94a3b8;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

.metric-strip {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 10px;
    width: 100%;
}}

.metric-box {{
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}}

.metric-num {{
    font-size: 1.35rem;
    font-weight: 800;
    color: #ffffff;
}}

.metric-sub {{
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 2px;
}}


/* ============================================================
   BOTTOM BANNER
   ============================================================ */
.bottom-banner {{
    width: 100%;
    margin-top: 28px;
    border-radius: 20px;
    padding: 32px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: 
        linear-gradient(90deg, rgba(8, 12, 28, 0.95) 0%, rgba(8, 12, 28, 0.7) 100%),
        url('{cosmic_banner_b64}') center / cover no-repeat;
    border: 1px solid var(--card-border);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
}}

.banner-left {{
    max-width: 650px;
}}

.banner-title {{
    font-size: 1.15rem;
    font-weight: 750;
    color: #ffffff;
    margin-bottom: 6px;
}}

.banner-sub {{
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.5;
}}

.banner-btn {{
    padding: 10px 24px;
    border-radius: 99px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    background: rgba(255, 255, 255, 0.05);
    color: #ffffff !important;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    cursor: pointer;
    text-decoration: none !important;
    transition: all 0.2s ease;
}}

.banner-btn:hover {{
    background: rgba(255, 255, 255, 0.15);
    border-color: rgba(255, 255, 255, 0.4);
}}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DIALOG POPUPS FOR COMPONENTS & CELESTIAL OBJECTS
# ============================================================

@st.dialog("Celestial Component Details: Moon 🌖")
def show_moon_dialog():
    st.image(str(ASSETS_DIR / "moon.jpg"), caption="Lunar Surface Observation", use_container_width=True)
    st.markdown(
        """
        ### Moon (Natural Satellite)
        The Moon is Earth's only natural satellite, featuring cratered highlands, dark volcanic maria, and high-contrast illuminated rims.
        
        **Model Statistics**:
        - **Precision**: 98.02%
        - **Recall**: 94.29%
        - **F1 Score**: 96.12%
        - **Identification Signals**: High edge contrast, impact craters, albedo variance, spherical silhouette.
        
        Learn more on [NASA Moon Science](https://science.nasa.gov/moon/) or explore [Wikipedia: Moon](https://en.wikipedia.org/wiki/Moon).
        """
    )


@st.dialog("Celestial Component Details: Nebula 🌌")
def show_nebula_dialog():
    st.image(str(ASSETS_DIR / "nebula.jpg"), caption="Interstellar Cloud Observation", use_container_width=True)
    st.markdown(
        """
        ### Nebula (Interstellar Gas & Dust)
        Nebulae are diffuse interstellar clouds of hydrogen, helium, and cosmic dust where stars are born.
        
        **Model Statistics**:
        - **Precision**: 97.92%
        - **Recall**: 96.58%
        - **F1 Score**: 97.24%
        - **Identification Signals**: Multi-spectral gas emissions (H-alpha, OIII), diffuse cloud structures, filamentary gas dust.
        
        Learn more on [NASA Nebulae Science](https://science.nasa.gov/universe/nebulae/) or explore [ESA Hubble Nebulae Gallery](https://esahubble.org/images/archive/category/nebulae/).
        """
    )


@st.dialog("Celestial Component Details: Planet 🪐")
def show_planet_dialog():
    st.image(str(ASSETS_DIR / "planet.jpg"), caption="Planetary Body Observation", use_container_width=True)
    st.markdown(
        """
        ### Planet (Planetary Body)
        Planets are spherical celestial bodies orbiting stars. This class includes gas giants with planetary ring systems (e.g. Saturn, Jupiter) and rocky terrestrial planets.
        
        **Model Statistics**:
        - **Precision**: 97.32%
        - **Recall**: 99.61%
        - **F1 Score**: 98.45%
        - **Identification Signals**: Concentric ring geometries, cloud limb darkening, atmospheric bands.
        
        Learn more on [NASA Planets Overview](https://science.nasa.gov/planets/) or explore [NASA Solar System Exploration](https://solarsystem.nasa.gov/).
        """
    )


@st.dialog("Celestial Component Details: Star 🌟")
def show_star_dialog():
    st.image(str(ASSETS_DIR / "star.jpg"), caption="Stellar Object Observation", use_container_width=True)
    st.markdown(
        """
        ### Star (Luminous Plasma Sphere)
        Stars are luminous spheres of plasma powered by nuclear fusion. In astronomical imagery, stars present as bright point sources with diffraction spikes.
        
        **Model Statistics**:
        - **Precision**: 100.00%
        - **Recall**: 100.00%
        - **F1 Score**: 100.00%
        - **Identification Signals**: Point-source intensity peak, cross diffraction rays, radial glare halo.
        
        Learn more on [NASA Stars Science](https://science.nasa.gov/universe/stars/) or explore [ESA Gaia Mission](https://www.cosmos.esa.int/web/gaia).
        """
    )


@st.dialog("Technology Component: MobileNetV3Small ⚡")
def show_mobilenet_dialog():
    st.markdown(
        """
        ### MobileNetV3Small Architecture
        MobileNetV3Small is a state-of-the-art lightweight deep neural network architecture designed by Google for efficient computer vision.
        
        **Key Architectural Highlights**:
        - **Depthwise Separable Convolutions**: Drastically reduces parameter count while preserving spatial feature maps.
        - **Hard-Swish Activation**: High-speed, non-linear activation functions tailored for mobile/edge deployment.
        - **Squeeze-and-Excitation (SE)**: Channel-wise attention mechanisms emphasizing key astronomical visual features.
        
        **Read Research**: [MobileNetV3 Paper on arXiv](https://arxiv.org/abs/1905.02244) | [TensorFlow MobileNet Docs](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small)
        """
    )


@st.dialog("Technology Component: Transfer Learning 🔄")
def show_transfer_dialog():
    st.markdown(
        """
        ### Transfer Learning Paradigm
        Transfer learning reutilizes pre-trained feature weights from a model trained on a large dataset (ImageNet) to solve specialized tasks.
        
        **Project Benefits**:
        - Reuses early-layer edge, boundary, and curve detectors.
        - Reduces dataset requirements while achieving **98.26% test accuracy**.
        - Prevents overfitting on specialized telescope observations.
        
        **Read Documentation**: [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
        """
    )


@st.dialog("Technology Component: TensorFlow & Keras 🤖")
def show_tf_dialog():
    st.markdown(
        """
        ### TensorFlow & Keras Framework
        TensorFlow is an open-source machine learning framework created by Google for deep neural network training and inference.
        
        **Role in Naksh AI**:
        - Manages tensor operations and GPU/CPU computational graphs.
        - Executes model prediction pipeline (`model.predict()`).
        - Enables automatic differentiation via `tf.GradientTape()` for Grad-CAM.
        
        **Official Links**: [TensorFlow.org](https://www.tensorflow.org/) | [Keras API Documentation](https://keras.io/)
        """
    )


@st.dialog("Technology Component: Grad-CAM Explainability 🔥")
def show_gradcam_dialog():
    st.markdown(
        """
        ### Grad-CAM (Explainable AI)
        Gradient-weighted Class Activation Mapping (Grad-CAM) generates visual explanations for decisions made by convolutional neural networks.
        
        **How it Works**:
        1. Computes gradients of the target class score w.r.t the final convolutional feature maps (`conv_1`).
        2. Applies global average pooling to obtain channel importance weights.
        3. Generates a 2D heatmap overlaying the regions (red/yellow = high attention, blue = low attention).
        
        **Read Research**: [Grad-CAM Research Paper on arXiv](https://arxiv.org/abs/1610.02391)
        """
    )


@st.dialog("Technology Component: Python 🐍")
def show_python_dialog():
    st.markdown(
        """
        ### Python Programming Language
        Python is the primary language powering Naksh AI's core backend, computer vision pipeline, and data analysis workflows.
        
        **Core Libraries Used**:
        - `tensorflow` / `keras`: Deep learning model execution.
        - `numpy`: Array manipulation and tensor processing.
        - `Pillow (PIL)`: Image loading, resizing, and format conversions.
        - `matplotlib`: Colormap generation for Grad-CAM heatmaps.
        - `streamlit`: Real-time web application dashboard interface.
        
        **Official Site**: [Python.org](https://www.python.org/)
        """
    )


@st.dialog("Technology Component: Streamlit 🚀")
def show_streamlit_dialog():
    st.markdown(
        """
        ### Streamlit Web Application Framework
        Streamlit turns Python data scripts into interactive web applications in minutes.
        
        **Role in Naksh AI**:
        - Renders the custom cosmic glassmorphism dashboard layout.
        - Handles live drag-and-drop file uploads.
        - Executes real-time inference and displays Grad-CAM focus maps.
        
        **Official Site**: [Streamlit.io](https://streamlit.io/)
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        f"""
        <div class="sb-brand">
            <div class="sb-logo-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="9"></circle>
                    <path d="M3.6 9h16.8"></path>
                    <path d="M3.6 15h16.8"></path>
                    <circle cx="12" cy="12" r="3" fill="white"></circle>
                </svg>
            </div>
            <div class="sb-brand-text">
                <div class="sb-title">NAKSH <span>AI</span></div>
                <div class="sb-subtitle"><a href="https://science.nasa.gov/" target="_blank">Astronomical Intelligence</a></div>
            </div>
        </div>

        <div class="sb-section-title">NAVIGATION</div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("📷 Image classification", key="nav_img_cls"):
        st.toast("Active section: Image classification")
    if st.button("👁️ Explainable AI (Grad-CAM)", key="nav_exp_ai"):
        show_gradcam_dialog()
    if st.button("📈 Prediction analysis", key="nav_pred_an"):
        show_transfer_dialog()
    if st.button("📊 Model performance", key="nav_mod_perf"):
        st.toast("Accuracy: 98.26% on 690 test images")

    st.markdown('<div class="sb-section-title">RECOGNIZED OBJECTS</div>', unsafe_allow_html=True)

    if st.button("🌖  Moon (Natural Satellite)", key="btn_sb_moon"):
        show_moon_dialog()
    if st.button("🌌  Nebula (Gas & Dust)", key="btn_sb_nebula"):
        show_nebula_dialog()
    if st.button("🪐  Planet (Planetary Body)", key="btn_sb_planet"):
        show_planet_dialog()
    if st.button("🌟  Star (Luminous Plasma)", key="btn_sb_star"):
        show_star_dialog()

    st.markdown('<div class="sb-section-title">MODEL ARCHITECTURE</div>', unsafe_allow_html=True)

    if st.button("⚡ MobileNetV3Small", key="btn_arch_mobilenet"):
        show_mobilenet_dialog()
    if st.button("🔄 Transfer learning", key="btn_arch_transfer"):
        show_transfer_dialog()
    if st.button("🤖 TensorFlow / Keras", key="btn_arch_tf"):
        show_tf_dialog()
    if st.button("🔥 Grad-CAM Explainability", key="btn_arch_gradcam"):
        show_gradcam_dialog()

    st.markdown(
        """
        <div class="sb-status-card">
            <div class="sb-status-header">
                <span class="sb-status-dot"></span>
                ONLINE
            </div>
            <div class="sb-status-sub">Ready for analysis</div>
        </div>

        <div class="sb-footer">
            <a href="https://www.python.org/" target="_blank">Python</a> · 
            <a href="https://www.tensorflow.org/" target="_blank">TensorFlow</a> · 
            <a href="https://streamlit.io/" target="_blank">Streamlit</a><br/>
            Naksh AI v1.0 | All systems operational
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO BANNER
# ============================================================

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-content">
            <div class="hero-kicker">ARTIFICIAL INTELLIGENCE · <a href="https://science.nasa.gov/" target="_blank" style="color: #a78bfa;">ASTRONOMY</a></div>
            <div class="hero-title">Astronomical <span>Vision</span></div>
            <div class="hero-desc">
                Naksh AI is an intelligent telescope assistant built with <a href="https://www.python.org/" target="_blank">Python</a>, <a href="https://www.tensorflow.org/" target="_blank">TensorFlow</a>, and <a href="https://arxiv.org/abs/1905.02244" target="_blank">MobileNetV3Small</a> that analyzes astronomical images and identifies celestial objects with high accuracy, revealing regions that influenced its decision via <a href="https://arxiv.org/abs/1610.02391" target="_blank">Grad-CAM</a>.
            </div>
            <div class="hero-badge">
                <span class="hero-pulse"></span>
                MODEL ONLINE
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EXPLORE CELESTIAL OBJECTS
# ============================================================

st.markdown('<div class="section-header-title">EXPLORE CELESTIAL OBJECTS (CLICK FOR DETAILS & NASA LINKS)</div>', unsafe_allow_html=True)

col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)

with col_exp1:
    st.markdown(
        f"""
        <div class="explore-card" style="margin-bottom: 8px;">
            <img class="explore-img" src="{moon_b64}" />
            <div class="explore-body">
                <div class="explore-name">Moon</div>
                <div class="explore-text">Natural satellite orbiting planets.</div>
                <div style="font-size: 0.78rem; margin-top: 4px;"><a href="https://science.nasa.gov/moon/" target="_blank">NASA Link ↗</a></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Explore Moon 🌖", key="btn_exp_moon"):
        show_moon_dialog()

with col_exp2:
    st.markdown(
        f"""
        <div class="explore-card" style="margin-bottom: 8px;">
            <img class="explore-img" src="{nebula_b64}" />
            <div class="explore-body">
                <div class="explore-name">Nebula</div>
                <div class="explore-text">Interstellar clouds of gas and dust.</div>
                <div style="font-size: 0.78rem; margin-top: 4px;"><a href="https://science.nasa.gov/universe/nebulae/" target="_blank">NASA Link ↗</a></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Explore Nebula 🌌", key="btn_exp_nebula"):
        show_nebula_dialog()

with col_exp3:
    st.markdown(
        f"""
        <div class="explore-card" style="margin-bottom: 8px;">
            <img class="explore-img" src="{planet_b64}" />
            <div class="explore-body">
                <div class="explore-name">Planet</div>
                <div class="explore-text">Large celestial bodies orbiting stars.</div>
                <div style="font-size: 0.78rem; margin-top: 4px;"><a href="https://science.nasa.gov/planets/" target="_blank">NASA Link ↗</a></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Explore Planet 🪐", key="btn_exp_planet"):
        show_planet_dialog()

with col_exp4:
    st.markdown(
        f"""
        <div class="explore-card" style="margin-bottom: 8px;">
            <img class="explore-img" src="{star_b64}" />
            <div class="explore-body">
                <div class="explore-name">Star</div>
                <div class="explore-text">Luminous spheres of plasma and energy.</div>
                <div style="font-size: 0.78rem; margin-top: 4px;"><a href="https://science.nasa.gov/universe/stars/" target="_blank">NASA Link ↗</a></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Explore Star 🌟", key="btn_exp_star"):
        show_star_dialog()


# ============================================================
# MODEL & PREPARATION
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    class_names = load_class_names()
except Exception as exc:
    st.error("Could not load the Naksh AI model.")
    st.exception(exc)
    st.stop()


# ============================================================
# 3-COLUMN DASHBOARD GRID
# ============================================================

st.write("")
col_left, col_mid, col_right = st.columns([1.3, 0.85, 0.85], gap="medium")

# ------------------------------------------------------------
# COLUMN 1: CLASSIFY ASTRONOMICAL IMAGE
# ------------------------------------------------------------

with col_left:
    st.markdown(
        """
        <div class="panel-header-title">CLASSIFY ASTRONOMICAL IMAGE</div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload an astronomical image",
        type=["jpg", "jpeg", "png"],
        help="Drag and drop an image here or click to browse. Supports JPG, JPEG, PNG.",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption=f"Uploaded Observation: {uploaded_file.name}", use_container_width=True)

        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        analyze_btn = st.button("ANALYZE WITH NAKSH AI", type="primary", key="btn_main_analyze")
        st.markdown('</div>', unsafe_allow_html=True)

        if analyze_btn or True:  # Auto-analyze once uploaded
            with st.spinner("Analyzing astronomical observation..."):
                resized = image.resize((224, 224))
                image_array = np.asarray(resized).astype(np.float32)
                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(image_array, verbose=0)
                predicted_index = int(np.argmax(predictions[0]))
                predicted_class = class_names[predicted_index]
                confidence = float(predictions[0][predicted_index])

                UNKNOWN_THRESHOLD = 0.75
                is_unknown = confidence < UNKNOWN_THRESHOLD

                heatmap = None
                if not is_unknown:
                    try:
                        heatmap, _ = make_gradcam_heatmap(image, model, "conv_1")
                    except Exception as err:
                        st.warning(f"Grad-CAM generation issue: {err}")

            st.write("")
            st.markdown(
                """
                <div class="panel-header-title" style="margin-top: 15px;">CLASSIFICATION RESULT</div>
                """,
                unsafe_allow_html=True,
            )

            if is_unknown:
                st.error("UNKNOWN / UNRECOGNIZED IMAGE")
                st.warning(
                    f"Naksh AI could not confidently identify this image. "
                    f"Highest prediction: {predicted_class.upper()} ({confidence * 100:.2f}%)."
                )
            else:
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Detected Object", predicted_class.upper())
                with m2:
                    st.metric("Confidence", f"{confidence * 100:.2f}%")

                st.markdown("**Probability Distribution**")
                for i, name in enumerate(class_names):
                    prob = float(predictions[0][i])
                    st.caption(f"{name.upper()} · {prob * 100:.2f}%")
                    st.progress(prob)

                if heatmap is not None:
                    st.write("")
                    st.markdown("**Explainable AI (<a href='https://arxiv.org/abs/1610.02391' target='_blank'>Grad-CAM</a> Visual Attention)**", unsafe_allow_html=True)

                    heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(
                        image.size, Image.Resampling.BILINEAR
                    )
                    heatmap_array = np.asarray(heatmap_image) / 255.0

                    cmap = plt.get_cmap("jet")
                    colored_heatmap = cmap(heatmap_array)
                    colored_heatmap = np.uint8(colored_heatmap[:, :, :3] * 255)
                    original_array = np.asarray(image)

                    overlay = np.uint8(np.clip(0.55 * original_array + 0.45 * colored_heatmap, 0, 255))

                    gc1, gc2, gc3 = st.columns(3)
                    with gc1:
                        st.image(image, caption="Original", use_container_width=True)
                    with gc2:
                        st.image(colored_heatmap, caption="Heatmap", use_container_width=True)
                    with gc3:
                        st.image(overlay, caption="Overlay", use_container_width=True)

                    buf = io.BytesIO()
                    Image.fromarray(overlay).save(buf, format="PNG")
                    st.download_button(
                        "Download AI Focus Image",
                        buf.getvalue(),
                        "naksh_ai_gradcam.png",
                        "image/png",
                        use_container_width=True,
                    )

# ------------------------------------------------------------
# COLUMN 2: HOW IT WORKS
# ------------------------------------------------------------

with col_mid:
    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-header-title">HOW IT WORKS</div>
            <div class="timeline">
                <div class="timeline-step">
                    <div class="timeline-number">01</div>
                    <div class="timeline-content">
                        <div class="timeline-title">Upload Image</div>
                        <div class="timeline-desc">Provide an astronomical image from your device or telescope.</div>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="timeline-number">02</div>
                    <div class="timeline-content">
                        <div class="timeline-title">AI Classification</div>
                        <div class="timeline-desc"><a href="https://arxiv.org/abs/1905.02244" target="_blank">MobileNetV3Small</a> analyzes the image and predicts class probabilities.</div>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="timeline-number">03</div>
                    <div class="timeline-content">
                        <div class="timeline-title">Explainable AI</div>
                        <div class="timeline-desc"><a href="https://arxiv.org/abs/1610.02391" target="_blank">Grad-CAM</a> highlights regions that influenced the model prediction.</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# COLUMN 3: MODEL PERFORMANCE
# ------------------------------------------------------------

with col_right:
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-header-title">MODEL PERFORMANCE</div>
            <div class="gauge-container">
                <div class="gauge-circle">
                    <div class="gauge-inner">
                        <div class="gauge-val">98.26%</div>
                        <div class="gauge-lbl">Test Accuracy</div>
                    </div>
                </div>
            </div>
            <div class="metric-strip">
                <div class="metric-box">
                    <div class="metric-num">690</div>
                    <div class="metric-sub">Test Images</div>
                </div>
                <div class="metric-box">
                    <div class="metric-num">{len(class_names)}</div>
                    <div class="metric-sub">Classes</div>
                </div>
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 14px; text-align: center;">
                Built with <a href="https://www.tensorflow.org/" target="_blank">TensorFlow</a> & <a href="https://www.python.org/" target="_blank">Python</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BOTTOM BANNER
# ============================================================

st.markdown(
    """
    <div class="bottom-banner">
        <div class="banner-left">
            <div class="banner-title">Advancing Astronomy with Artificial Intelligence</div>
            <div class="banner-sub">
                Combining deep learning, <a href="https://arxiv.org/abs/1905.02244" target="_blank" style="color: #60a5fa;">MobileNetV3 Small Transfer Learning</a>, and <a href="https://arxiv.org/abs/1610.02391" target="_blank" style="color: #60a5fa;">Grad-CAM Explainability</a> to explore the universe through the lens of intelligent systems.
            </div>
        </div>
        <a class="banner-btn" href="https://science.nasa.gov/" target="_blank">LEARN MORE ↗</a>
    </div>
    """,
    unsafe_allow_html=True,
)
