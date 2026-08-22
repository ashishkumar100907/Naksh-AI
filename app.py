import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image

from gradcam import make_gradcam_heatmap


# ============================================================
# PAGE CONFIG (NO EMOJIS)
# ============================================================

st.set_page_config(
    page_title="NAKSH AI | Astronomical Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize page navigation state
if "page" not in st.session_state:
    st.session_state["page"] = "intro"


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
# STYLES (PLANETARY & SPACE THEMED POPUP MODAL DIALOGS)
# ============================================================

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Cinzel:wght@600;800&display=swap');

/* GLOBAL RESETS & VARIABLES */
:root {{
    --font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --bg-dark: #03050d;
    --card-bg: rgba(13, 18, 38, 0.85);
    --card-border: rgba(255, 255, 255, 0.12);
    --card-border-hover: rgba(167, 139, 250, 0.5);
    --accent-purple: #8b5cf6;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-pink: #ec4899;
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
        radial-gradient(circle at 50% 10%, rgba(139, 92, 246, 0.14), transparent 45%),
        radial-gradient(circle at 10% 80%, rgba(59, 130, 246, 0.1), transparent 40%),
        radial-gradient(circle at 90% 85%, rgba(6, 182, 212, 0.1), transparent 40%),
        var(--bg-dark);
    color: var(--text-main);
}}

/* Streamlit Header & Toolbar */
[data-testid="stHeader"] {{
    background: transparent !important;
}}

.main .block-container {{
    max-width: 1480px;
    padding-top: 0.5rem;
    padding-bottom: 2.5rem;
    padding-left: 1.8rem;
    padding-right: 1.8rem;
}}

/* Links */
a {{
    color: #60a5fa !important;
    text-decoration: none !important;
    transition: color 0.2s ease;
}}
a:hover {{
    color: #c084fc !important;
    text-decoration: underline !important;
}}


/* ============================================================
   ASTRONOMICAL PLANETARY THEMED MODAL DIALOG CONTAINER
   ============================================================ */
div[data-testid="stModal"] > div:first-child,
div[role="dialog"] > div:first-child {{
    background: linear-gradient(135deg, rgba(13, 18, 42, 0.96) 0%, rgba(22, 28, 60, 0.95) 50%, rgba(9, 12, 30, 0.98) 100%) !important;
    border: 2px solid rgba(167, 139, 250, 0.6) !important;
    border-radius: 36px 16px 36px 16px !important;
    box-shadow: 
        0 0 0 3px rgba(167, 139, 250, 0.25),
        0 0 50px rgba(139, 92, 246, 0.5),
        0 25px 70px rgba(0, 0, 0, 0.85),
        inset 0 0 30px rgba(139, 92, 246, 0.15) !important;
    backdrop-filter: blur(24px) !important;
    color: #f8fafc !important;
    padding: 26px 30px !important;
    position: relative !important;
}}

/* Modal Header Bar */
div[data-testid="stModal"] h2, div[role="dialog"] h2 {{
    color: #ffffff !important;
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.04em !important;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    border-bottom: 1px solid rgba(167, 139, 250, 0.3) !important;
    padding-bottom: 14px !important;
    margin-bottom: 18px !important;
}}

/* Modal Close Button (×) */
div[data-testid="stModal"] button[aria-label="Close"],
div[role="dialog"] button[aria-label="Close"] {{
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 50% !important;
    color: #ffffff !important;
    transition: all 0.25s ease !important;
}}

div[data-testid="stModal"] button[aria-label="Close"]:hover,
div[role="dialog"] button[aria-label="Close"]:hover {{
    background: rgba(239, 68, 68, 0.3) !important;
    border-color: rgba(239, 68, 68, 0.8) !important;
    color: #ffffff !important;
    transform: rotate(90deg) scale(1.1) !important;
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.5) !important;
}}


/* ============================================================
   ASTRONOMICAL TOP TASKBAR (SLEEK THEMED NAVIGATION HEADER)
   ============================================================ */
.top-taskbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    margin-bottom: 18px;
    border-radius: 20px;
    background: linear-gradient(90deg, rgba(13, 18, 38, 0.95) 0%, rgba(20, 26, 52, 0.85) 50%, rgba(13, 18, 38, 0.95) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(139, 92, 246, 0.15);
    backdrop-filter: blur(16px);
}}

.taskbar-brand {{
    display: flex;
    align-items: center;
    gap: 14px;
}}

.taskbar-logo {{
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: radial-gradient(circle, #8b5cf6 0%, #3b82f6 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.7);
}}

.taskbar-title {{
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    color: #ffffff;
    line-height: 1;
}}

.taskbar-sub {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    color: #94a3b8;
    text-transform: uppercase;
    margin-top: 3px;
}}

.taskbar-metrics {{
    display: flex;
    align-items: center;
    gap: 20px;
}}

.taskbar-chip {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 99px;
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(139, 92, 246, 0.3);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #c084fc;
    text-transform: uppercase;
}}

.taskbar-pulse {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 10px #34d399;
    animation: pulse 1.8s infinite;
}}


/* ============================================================
   INTRO LANDING PAGE STYLES
   ============================================================ */
.intro-hero-wrapper {{
    position: relative;
    width: 100%;
    margin-top: -35px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    z-index: 10;
    pointer-events: none;
}}

.intro-welcome-line {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.35em;
    color: #a78bfa;
    text-transform: uppercase;
    margin-bottom: 8px;
}}

.intro-welcome-line::before, .intro-welcome-line::after {{
    content: "";
    width: 65px;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(167, 139, 250, 0.6));
}}

.intro-welcome-line::after {{
    background: linear-gradient(90deg, rgba(167, 139, 250, 0.6), transparent);
}}

.intro-main-title {{
    font-size: clamp(3.4rem, 8vw, 6.5rem);
    font-weight: 900;
    line-height: 1.02;
    letter-spacing: 0.04em;
    margin: 4px 0 12px 0;
    background: linear-gradient(180deg, #ffffff 25%, #a78bfa 75%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 35px rgba(139, 92, 246, 0.5));
}}

.intro-subtitle {{
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: 0.38em;
    color: #818cf8;
    text-transform: uppercase;
    margin-bottom: 22px;
}}

.intro-description {{
    max-width: 740px;
    font-size: 1.05rem;
    line-height: 1.7;
    color: #cbd5e1;
    margin: 0 auto 26px auto;
    text-shadow: 0 2px 12px rgba(0, 0, 0, 0.9);
}}

/* ============================================================
   BEGIN JOURNEY BUTTON - MATCHING NASA VIBRANT COSMIC LINK CARD EFFECTS
   ============================================================ */
div[data-testid="stColumn"] > div {{
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}}

div[data-testid="stColumn"] div.stButton {{
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}}

.stButton {{
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}}

.stButton > button[key="btn_begin_journey"] {{
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.45) 0%, rgba(59, 130, 246, 0.55) 50%, rgba(139, 92, 246, 0.5) 100%) !important;
    border: 2px solid rgba(56, 189, 248, 0.85) !important;
    border-radius: 999px !important;
    padding: 16px 48px !important;
    height: 64px !important;
    font-size: 1.12rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.22em !important;
    color: #ffffff !important;
    box-shadow: 
        0 0 0 4px rgba(139, 92, 246, 0.25),
        0 0 40px rgba(6, 182, 212, 0.6),
        0 15px 35px rgba(0, 0, 0, 0.7),
        inset 0 0 20px rgba(255, 255, 255, 0.3) !important;
    backdrop-filter: blur(16px) !important;
    margin: 0 auto !important;
    width: 100% !important;
    max-width: 340px !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

.stButton > button[key="btn_begin_journey"]:hover {{
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.8) 0%, rgba(59, 130, 246, 0.85) 50%, rgba(139, 92, 246, 0.8) 100%) !important;
    border-color: #ffffff !important;
    transform: translateY(-5px) scale(1.06) !important;
    box-shadow: 
        0 0 0 6px rgba(56, 189, 248, 0.4),
        0 0 65px rgba(6, 182, 212, 0.95),
        0 20px 45px rgba(0, 0, 0, 0.8),
        inset 0 0 30px rgba(255, 255, 255, 0.6) !important;
}}

.scroll-hint {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    color: #64748b;
    text-transform: uppercase;
    margin-top: 14px;
    pointer-events: auto;
    cursor: pointer;
    transition: color 0.2s ease;
    text-align: center;
}}

.scroll-hint:hover {{
    color: #a78bfa;
}}


/* ============================================================
   ASTRONOMICAL SIDEBAR TASKBAR (ENHANCED THEMATIC DESIGN)
   ============================================================ */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #050818 0%, #080d24 50%, #03050f 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.25) !important;
    box-shadow: 10px 0 30px rgba(0, 0, 0, 0.5) !important;
}}

section[data-testid="stSidebar"] * {{
    color: #cbd5e1;
}}

.sb-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 6px 20px 6px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 12px;
}}

.sb-logo-icon {{
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: radial-gradient(circle, #8b5cf6 0%, #3b82f6 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.6);
}}

.sb-brand-text {{
    display: flex;
    flex-direction: column;
}}

.sb-title {{
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: -0.01em;
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
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    color: #818cf8;
    text-transform: uppercase;
    margin: 22px 0 10px 4px;
}}

.stButton > button {{
    width: 100%;
    border-radius: 12px;
    height: 44px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #cbd5e1 !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

.stButton > button:hover {{
    background: linear-gradient(90deg, rgba(139, 92, 246, 0.25) 0%, rgba(59, 130, 246, 0.2) 100%) !important;
    border-color: rgba(167, 139, 250, 0.6) !important;
    color: #ffffff !important;
    transform: translateX(4px) !important;
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.3) !important;
}}

.sb-status-card {{
    margin-top: 24px;
    padding: 14px 16px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(6, 182, 212, 0.08) 100%);
    border: 1px solid rgba(16, 185, 129, 0.3);
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
}}

.sb-status-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    color: #34d399;
    text-transform: uppercase;
}}

.sb-status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 10px #34d399;
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
    min-height: 290px;
    border-radius: 24px;
    overflow: hidden;
    margin-bottom: 28px;
    border: 1px solid var(--card-border);
    background: 
        linear-gradient(90deg, #05081a 0%, rgba(5, 8, 26, 0.88) 45%, rgba(5, 8, 26, 0.3) 100%),
        url('{hero_bg_b64}') center right / cover no-repeat;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
    transition: transform 0.3s ease;
}}

.hero-container:hover {{
    transform: translateY(-2px);
}}

.hero-content {{
    padding: 42px 48px;
    max-width: 700px;
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
   EXPLORE CELESTIAL OBJECT CARDS (PLANET SHAPED CONTAINERS)
   ============================================================ */
.section-header-title {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 16px;
    text-align: center;
}}

.explore-card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 28px 12px 28px 12px;
    padding: 18px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    text-align: center;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
    backdrop-filter: blur(10px);
}}

.explore-card:hover {{
    transform: translateY(-8px) scale(1.02);
    border-color: var(--card-border-hover);
    box-shadow: 0 20px 40px rgba(139, 92, 246, 0.3);
}}

.explore-img-wrapper {{
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px 0;
}}

.explore-img {{
    width: 170px;
    height: 170px;
    object-fit: cover;
    border-radius: 50%;
    border: 2px solid rgba(167, 139, 250, 0.3);
    box-shadow: 0 0 25px rgba(0, 0, 0, 0.6), 0 0 15px rgba(139, 92, 246, 0.2);
    transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.3s ease, box-shadow 0.3s ease;
}}

.explore-card:hover .explore-img {{
    transform: scale(1.08) rotate(3deg);
    border-color: rgba(167, 139, 250, 0.8);
    box-shadow: 0 0 35px rgba(139, 92, 246, 0.5);
}}

.explore-body {{
    padding: 12px 6px 4px 6px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    width: 100%;
}}

.explore-name {{
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 4px;
}}

.explore-text {{
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.45;
    max-width: 220px;
}}


/* ============================================================
   VIBRANT CELESTIAL HYPERLINK BUTTON CARDS (MOON, NEBULA, PLANET, STAR, GALAXY)
   ============================================================ */
.cosmic-links-wrapper {{
    margin-top: 22px;
    padding-top: 18px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}}

.cosmic-links-header {{
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    color: #a78bfa;
    text-transform: uppercase;
    margin-bottom: 14px;
}}

.cosmic-links-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
    width: 100%;
}}

.cosmic-link-card {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    text-decoration: none !important;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}}

.cosmic-link-card:hover {{
    transform: translateY(-4px) scale(1.03);
    text-decoration: none !important;
}}

.cosmic-link-badge {{
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}}

.cosmic-link-details {{
    display: flex;
    flex-direction: column;
}}

.cosmic-link-title {{
    font-size: 0.92rem;
    font-weight: 800;
    color: #ffffff !important;
    line-height: 1.2;
}}

.cosmic-link-sub {{
    font-size: 0.72rem;
    color: rgba(255, 255, 255, 0.75) !important;
    margin-top: 2px;
}}

/* 🌖 MOON THEME HYPERLINK CARD (Orbital Lunar Silver Pill) */
.cosmic-link-moon {{
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(71, 85, 105, 0.8) 100%);
    border: 1.8px solid rgba(226, 232, 240, 0.6);
    border-radius: 99px;
    box-shadow: 0 0 20px rgba(226, 232, 240, 0.25);
}}

.cosmic-link-moon:hover {{
    background: linear-gradient(135deg, rgba(51, 65, 85, 0.95) 0%, rgba(100, 116, 139, 0.9) 100%);
    border-color: #ffffff;
    box-shadow: 0 0 35px rgba(255, 255, 255, 0.5), inset 0 0 15px rgba(255, 255, 255, 0.3);
}}

.cosmic-link-moon .cosmic-link-badge {{
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #f8fafc 0%, #94a3b8 60%, #475569 100%);
    box-shadow: 0 0 12px rgba(255, 255, 255, 0.8);
}}

/* 🌌 NEBULA THEME HYPERLINK CARD (H-Alpha Wave Capsule) */
.cosmic-link-nebula {{
    background: linear-gradient(135deg, rgba(236, 72, 153, 0.35) 0%, rgba(168, 85, 247, 0.45) 50%, rgba(59, 130, 246, 0.4) 100%);
    border: 1.8px solid rgba(244, 114, 182, 0.7);
    border-radius: 26px 8px 26px 8px;
    box-shadow: 0 0 25px rgba(236, 72, 153, 0.35);
}}

.cosmic-link-nebula:hover {{
    background: linear-gradient(135deg, rgba(236, 72, 153, 0.7) 0%, rgba(168, 85, 247, 0.75) 50%, rgba(59, 130, 246, 0.7) 100%);
    border-color: #f472b6;
    box-shadow: 0 0 45px rgba(236, 72, 153, 0.7);
}}

.cosmic-link-nebula .cosmic-link-badge {{
    border-radius: 20px 6px 20px 6px;
    background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%);
    box-shadow: 0 0 15px #ec4899;
}}

/* 🪐 PLANET THEME HYPERLINK CARD (Saturnian Ring Capsule) */
.cosmic-link-planet {{
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.35) 0%, rgba(217, 119, 6, 0.45) 50%, rgba(180, 83, 9, 0.4) 100%);
    border: 2px solid rgba(251, 191, 36, 0.8);
    border-radius: 999px;
    box-shadow: 0 0 0 4px rgba(245, 158, 11, 0.18), 0 0 25px rgba(245, 158, 11, 0.4);
}}

.cosmic-link-planet:hover {{
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.75) 0%, rgba(217, 119, 6, 0.8) 50%, rgba(180, 83, 9, 0.75) 100%);
    border-color: #fef08a;
    box-shadow: 0 0 0 6px rgba(251, 191, 36, 0.35), 0 0 45px rgba(245, 158, 11, 0.8);
}}

.cosmic-link-planet .cosmic-link-badge {{
    border-radius: 50%;
    background: radial-gradient(circle, #fbbf24 0%, #d97706 70%, #78350f 100%);
    box-shadow: 0 0 14px #fbbf24;
    border: 1.5px solid #fef08a;
}}

/* ⭐ STAR THEME HYPERLINK CARD (Diffraction Glare Pill) */
.cosmic-link-star {{
    background: linear-gradient(135deg, rgba(254, 240, 138, 0.3) 0%, rgba(245, 158, 11, 0.45) 50%, rgba(234, 88, 12, 0.4) 100%);
    border: 2px solid rgba(255, 255, 255, 0.85);
    border-radius: 999px;
    box-shadow: 0 0 25px rgba(251, 191, 36, 0.5), inset 0 0 12px rgba(255, 255, 255, 0.4);
}}

.cosmic-link-star:hover {{
    background: linear-gradient(135deg, rgba(254, 240, 138, 0.75) 0%, rgba(245, 158, 11, 0.85) 50%, rgba(234, 88, 12, 0.8) 100%);
    border-color: #ffffff;
    box-shadow: 0 0 50px rgba(251, 191, 36, 0.95), inset 0 0 20px rgba(255, 255, 255, 0.7);
}}

.cosmic-link-star .cosmic-link-badge {{
    border-radius: 50%;
    background: #ffffff;
    box-shadow: 0 0 20px #ffffff, 0 0 35px #f59e0b;
}}

/* 🌀 GALAXY & TECH HYPERLINK CARD (Spiral Vortex Capsule) */
.cosmic-link-galaxy {{
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.35) 0%, rgba(59, 130, 246, 0.4) 50%, rgba(139, 92, 246, 0.4) 100%);
    border: 1.8px solid rgba(56, 189, 248, 0.75);
    border-radius: 18px 36px 18px 36px;
    box-shadow: 0 0 25px rgba(6, 182, 212, 0.4);
}}

.cosmic-link-galaxy:hover {{
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.75) 0%, rgba(59, 130, 246, 0.8) 50%, rgba(139, 92, 246, 0.75) 100%);
    border-color: #ffffff;
    box-shadow: 0 0 45px rgba(6, 182, 212, 0.85);
}}

.cosmic-link-galaxy .cosmic-link-badge {{
    border-radius: 50%;
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%);
    box-shadow: 0 0 15px #06b6d4;
}}


/* ============================================================
   MAIN PANELS & CARDS (PLANET SHAPED CONTAINERS)
   ============================================================ */
.panel-card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 28px 12px 28px 12px;
    padding: 24px;
    height: 100%;
    transition: border-color 0.3s ease;
    backdrop-filter: blur(10px);
}}

.panel-card:hover {{
    border-color: rgba(139, 92, 246, 0.35);
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
    border: 1.8px dashed rgba(96, 165, 250, 0.4) !important;
    border-radius: 22px !important;
    background: rgba(15, 23, 42, 0.55) !important;
    padding: 22px !important;
    text-align: center;
    transition: all 0.3s ease !important;
}}

[data-testid="stFileUploader"]:hover {{
    border-color: rgba(167, 139, 250, 0.8) !important;
    background: rgba(15, 23, 42, 0.75) !important;
    box-shadow: 0 0 30px rgba(139, 92, 246, 0.25) !important;
}}

.primary-btn > button {{
    width: 100%;
    border-radius: 99px;
    height: 50px;
    font-weight: 700;
    font-size: 0.92rem;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.25s ease !important;
}}

.primary-btn > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.6) !important;
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
    border: 1.5px solid rgba(139, 92, 246, 0.5);
    color: #a78bfa;
    font-size: 0.85rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    z-index: 2;
    box-shadow: 0 0 12px rgba(139, 92, 246, 0.3);
}}

.timeline-step:not(:last-child)::after {{
    content: "";
    position: absolute;
    top: 38px;
    left: 18px;
    width: 1px;
    height: calc(100% - 38px);
    border-left: 2px dashed rgba(139, 92, 246, 0.3);
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
    box-shadow: 0 0 35px rgba(139, 92, 246, 0.35);
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
    border-radius: 16px;
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
    border-radius: 24px;
    padding: 32px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: 
        linear-gradient(90deg, rgba(8, 12, 28, 0.95) 0%, rgba(8, 12, 28, 0.7) 100%),
        url('{cosmic_banner_b64}') center / cover no-repeat;
    border: 1px solid var(--card-border);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
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
# DIALOG POPUPS WITH SHAPED VIBRANT HYPERLINK CARDS
# ============================================================

@st.dialog("Celestial Component Details: Moon")
def show_moon_dialog():
    st.image(str(ASSETS_DIR / "moon.jpg"), caption="3D Lunar Surface Observation", use_container_width=True)
    st.markdown(
        """
        ### Moon (Natural Satellite)
        The Moon is Earth's only natural satellite, featuring cratered highlands, dark volcanic maria, and high-contrast illuminated rims.
        
        **Model Statistics**:
        - **Precision**: 98.02%
        - **Recall**: 94.29%
        - **F1 Score**: 96.12%
        - **Identification Signals**: High edge contrast, impact craters, albedo variance, spherical silhouette.
        
        <div class="cosmic-links-wrapper">
            <div class="cosmic-links-header">EXPLORE MOON RESOURCES</div>
            <div class="cosmic-links-grid">
                <a class="cosmic-link-card cosmic-link-moon" href="https://science.nasa.gov/moon/" target="_blank">
                    <div class="cosmic-link-badge">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2">
                            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                        </svg>
                    </div>
                    <div class="cosmic-link-details">
                        <div class="cosmic-link-title">NASA Moon Science</div>
                        <div class="cosmic-link-sub">Official NASA Lunar Portal</div>
                    </div>
                </a>
                <a class="cosmic-link-card cosmic-link-moon" href="https://en.wikipedia.org/wiki/Moon" target="_blank">
                    <div class="cosmic-link-badge">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="16" x2="12" y2="12"></line>
                            <line x1="12" y1="8" x2="12.01" y2="8"></line>
                        </svg>
                    </div>
                    <div class="cosmic-link-details">
                        <div class="cosmic-link-title">Wikipedia Moon</div>
                        <div class="cosmic-link-sub">Lunar Knowledge Archive</div>
                    </div>
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Celestial Component Details: Nebula")
def show_nebula_dialog():
    st.image(str(ASSETS_DIR / "nebula.jpg"), caption="3D Interstellar Cloud Observation", use_container_width=True)
    st.markdown(
        """
        ### Nebula (Interstellar Gas & Dust)
        Nebulae are diffuse interstellar clouds of hydrogen, helium, and cosmic dust where stars are born.
        
        **Model Statistics**:
        - **Precision**: 97.92%
        - **Recall**: 96.58%
        - **F1 Score**: 97.24%
        - **Identification Signals**: Multi-spectral gas emissions (H-alpha, OIII), diffuse cloud structures, filamentary gas dust.
        
        <div class="cosmic-links-wrapper">
            <div class="cosmic-links-header">EXPLORE NEBULA RESOURCES</div>
            <div class="cosmic-links-grid">
                <a class="cosmic-link-card cosmic-link-nebula" href="https://science.nasa.gov/universe/nebulae/" target="_blank">
                    <div class="cosmic-link-badge">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2">
                            <path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"></path>
                        </svg>
                    </div>
                    <div class="cosmic-link-details">
                        <div class="cosmic-link-title">NASA Nebulae Science</div>
                        <div class="cosmic-link-sub">Interstellar Gas Exploration</div>
                    </div>
                </a>
                <a class="cosmic-link-card cosmic-link-nebula" href="https://esahubble.org/images/archive/category/nebulae/" target="_blank">
                    <div class="cosmic-link-badge">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                            <circle cx="8.5" cy="8.5" r="1.5"></circle>
                            <polyline points="21 15 16 10 5 21"></polyline>
                        </svg>
                    </div>
                    <div class="cosmic-link-details">
                        <div class="cosmic-link-title">ESA Hubble Gallery</div>
                        <div class="cosmic-link-sub">High-Res Telescope Imagery</div>
                    </div>
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Celestial Component Details: Planet")
def show_planet_dialog():
    st.image(str(ASSETS_DIR / "planet.jpg"), caption="3D Planetary Body Observation", use_container_width=True)
    st.markdown(
        """
        ### Planet (Planetary Body)
        Planets are spherical celestial bodies orbiting stars. This class includes gas giants with planetary ring systems (e.g. Saturn, Jupiter) and rocky terrestrial planets.
        
        **Model Statistics**:
        - **Precision**: 97.32%
        - **Recall**: 99.61%
        - **F1 Score**: 98.45%
        - **Identification Signals**: Concentric ring geometries, cloud limb darkening, atmospheric bands.
        
        <div class="cosmic-links-wrapper">
            <div class="cosmic-links-header">EXPLORE PLANET RESOURCES</div>
            <div class="cosmic-links-grid">
                <a class="cosmic-link-card cosmic-link-planet" href="https://science.nasa.gov/planets/" target="_blank">
                    <div class="cosmic-link-badge">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2">
                            <circle cx="12" cy="12" r="7"></circle>
                            <path d="M2.5 12h19"></path>
                        </svg>
                    </div>
                    <div class="cosmic-link-details">
                        <div class="cosmic-link-title">NASA Planets Science</div>
                        <div class="cosmic-link-sub">Planetary System Portal</div>
                    </div>
                </a>
                <a class="cosmic-link-card cosmic-link-planet" href="https://solarsystem.nasa.gov/" target="_blank">
                    <div class="cosmic-link-badge">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                        </svg>
                    </div>
                    <div class="cosmic-link-details">
                        <div class="cosmic-link-title">Solar System 3D</div>
                        <div class="cosmic-link-sub">Interactive Planetary Orbits</div>
                    </div>
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Celestial Component Details: Star")
def show_star_dialog():
    st.image(str(ASSETS_DIR / "star.jpg"), caption="3D Stellar Object Observation", use_container_width=True)
    st.markdown(
        """
        ### Star (Luminous Plasma Sphere)
        Stars are luminous spheres of plasma powered by nuclear fusion. In astronomical imagery, stars present as bright point sources with diffraction spikes.
        
        **Model Statistics**:
        - **Precision**: 100.00%
        - **Recall**: 100.00%
        - **F1 Score**: 100.00%
        - **Identification Signals**: Point-source intensity peak, cross diffraction rays, radial glare halo.
        
        <div class="cosmic-links-wrapper">
            <div class="cosmic-links-header">EXPLORE STAR RESOURCES</div>
            <div class="cosmic-links-grid">
                <a class="cosmic-link-card cosmic-link-star" href="https://science.nasa.gov/universe/stars/" target="_blank">
                    <div class="cosmic-link-badge">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ea580c" stroke-width="2.5">
                            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                        </svg>
                    </div>
                    <div class="cosmic-link-details">
                        <div class="cosmic-link-title">NASA Stars Portal</div>
                        <div class="cosmic-link-sub">Stellar Evolution & Plasma</div>
                    </div>
                </a>
                <a class="cosmic-link-card cosmic-link-star" href="https://www.cosmos.esa.int/web/gaia" target="_blank">
                    <div class="cosmic-link-badge">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ea580c" stroke-width="2.5">
                            <circle cx="12" cy="12" r="3"></circle>
                            <path d="M12 2v3M12 19v3M2 12h3M19 12h3"></path>
                        </svg>
                    </div>
                    <div class="cosmic-link-details">
                        <div class="cosmic-link-title">ESA Gaia Mission</div>
                        <div class="cosmic-link-sub">3D Stellar Mapping Catalog</div>
                    </div>
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Technology Component: MobileNetV3Small")
def show_mobilenet_dialog():
    st.markdown(
        """
        ### MobileNetV3Small Architecture
        MobileNetV3Small is a state-of-the-art lightweight deep neural network architecture designed by Google for efficient computer vision.
        
        **Key Architectural Highlights**:
        - **Depthwise Separable Convolutions**: Drastically reduces parameter count while preserving spatial feature maps.
        - **Hard-Swish Activation**: High-speed, non-linear activation functions tailored for mobile/edge deployment.
        - **Squeeze-and-Excitation (SE)**: Channel-wise attention mechanisms emphasizing key astronomical visual features.
        
        <div class="cosmic-links-wrapper">
            <div class="cosmic-links-header">RESEARCH & ARCHITECTURE DOCUMENTATION</div>
            <div class="cosmic-links-grid">
                <a class="cosmic-link-card cosmic-link-galaxy" href="https://arxiv.org/abs/1905.02244" target="_blank">
                    <div class="cosmic-link-badge">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                        </svg>
                    </div>
                    <div class="cosmic-link-details">
                        <div class="cosmic-link-title">MobileNetV3 Paper</div>
                        <div class="cosmic-link-sub">arXiv Research Publication</div>
                    </div>
                </a>
                <a class="cosmic-link-card cosmic-link-galaxy" href="https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small" target="_blank">
                    <div class="cosmic-link-badge">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2">
                            <polyline points="16 18 22 12 16 6"></polyline>
                            <polyline points="8 6 2 12 8 18"></polyline>
                        </svg>
                    </div>
                    <div class="cosmic-link-details">
                        <div class="cosmic-link-title">TensorFlow Keras API</div>
                        <div class="cosmic-link-sub">Model API Documentation</div>
                    </div>
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Technology Component: Grad-CAM Explainability")
def show_gradcam_dialog():
    st.markdown(
        """
        ### Grad-CAM (Explainable AI)
        Gradient-weighted Class Activation Mapping (Grad-CAM) generates visual explanations for decisions made by convolutional neural networks.
        
        **How it Works**:
        1. Computes gradients of the target class score w.r.t the final convolutional feature maps (`conv_1`).
        2. Applies global average pooling to obtain channel importance weights.
        3. Generates a 2D heatmap overlaying the regions (red/yellow = high attention, blue = low attention).
        
        <div class="cosmic-links-wrapper">
            <div class="cosmic-links-header">EXPLAINABLE AI RESEARCH</div>
            <div class="cosmic-links-grid">
                <a class="cosmic-link-card cosmic-link-galaxy" href="https://arxiv.org/abs/1610.02391" target="_blank">
                    <div class="cosmic-link-badge">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2">
                            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                        </svg>
                    </div>
                    <div class="cosmic-link-details">
                        <div class="cosmic-link-title">Grad-CAM Paper</div>
                        <div class="cosmic-link-sub">arXiv XAI Publication</div>
                    </div>
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ADVANCED PHOTOREALISTIC 3D SOLAR SYSTEM & HUGE SUN ANIMATION
# ============================================================

def render_3d_solar_system():
    """Render interactive photorealistic 3D WebGL solar system with huge radiant Sun and Keplerian planetary orbits."""
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body, html { width: 100%; height: 100%; overflow: hidden; background: #03050d; }
            #webgl-container { width: 100%; height: 600px; position: relative; }
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <div id="webgl-container"></div>
        <script>
            let scene, camera, renderer;
            let sun, coronaLayer1, coronaLayer2, coronaLayer3;
            let planets = [];
            let starField;
            let mouseX = 0, mouseY = 0;

            // HIGH-RESOLUTION PROCEDURAL PHOTOREALISTIC TEXTURE GENERATORS
            function createSunTexture() {
                const canvas = document.createElement('canvas');
                canvas.width = 2048; canvas.height = 1024;
                const ctx = canvas.getContext('2d');
                const grad = ctx.createLinearGradient(0, 0, 0, 1024);
                grad.addColorStop(0, '#fff4cc');
                grad.addColorStop(0.2, '#ffcc00');
                grad.addColorStop(0.5, '#ff6600');
                grad.addColorStop(0.85, '#cc1100');
                grad.addColorStop(1, '#880000');
                ctx.fillStyle = grad; ctx.fillRect(0, 0, 2048, 1024);
                
                // Convective Solar Flare Granulation & Sunspots
                for(let i=0; i<85000; i++) {
                    const alpha = Math.random() * 0.45;
                    ctx.fillStyle = Math.random() > 0.45 ? `rgba(255,255,220,${alpha})` : `rgba(120,10,0,${alpha})`;
                    ctx.fillRect(Math.random()*2048, Math.random()*1024, Math.random()*6+2, Math.random()*6+2);
                }
                return new THREE.CanvasTexture(canvas);
            }

            function createEarthTexture() {
                const canvas = document.createElement('canvas');
                canvas.width = 1024; canvas.height = 512;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = '#0a235c'; ctx.fillRect(0, 0, 1024, 512);
                ctx.fillStyle = '#15803d';
                for(let i=0; i<35; i++) {
                    ctx.beginPath();
                    ctx.ellipse(Math.random()*1024, Math.random()*360+76, Math.random()*140+40, Math.random()*80+25, Math.random()*Math.PI, 0, Math.PI*2);
                    ctx.fill();
                }
                ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
                for(let i=0; i<50; i++) {
                    ctx.beginPath();
                    ctx.ellipse(Math.random()*1024, Math.random()*512, Math.random()*180+50, Math.random()*30+10, Math.random()*0.4, 0, Math.PI*2);
                    ctx.fill();
                }
                return new THREE.CanvasTexture(canvas);
            }

            function createJupiterTexture() {
                const canvas = document.createElement('canvas');
                canvas.width = 1024; canvas.height = 512;
                const ctx = canvas.getContext('2d');
                for(let y=0; y<512; y++) {
                    const noise = Math.sin(y * 0.08) * 0.5 + 0.5;
                    const noise2 = Math.cos(y * 0.04) * 0.5 + 0.5;
                    const r = Math.floor(210 + noise * 45);
                    const g = Math.floor(130 + noise2 * 50);
                    const b = Math.floor(75 + noise * 40);
                    ctx.fillStyle = `rgb(${r},${g},${b})`;
                    ctx.fillRect(0, y, 1024, 1);
                }
                // Great Red Spot
                ctx.fillStyle = '#b91c1c';
                ctx.beginPath();
                ctx.ellipse(720, 320, 75, 45, 0, 0, Math.PI*2);
                ctx.fill();
                return new THREE.CanvasTexture(canvas);
            }

            function createSaturnRingTexture() {
                const canvas = document.createElement('canvas');
                canvas.width = 1024; canvas.height = 1024;
                const ctx = canvas.getContext('2d');
                const grad = ctx.createRadialGradient(512, 512, 100, 512, 512, 500);
                grad.addColorStop(0, 'rgba(0,0,0,0)');
                grad.addColorStop(0.25, 'rgba(217, 119, 6, 0.9)');
                grad.addColorStop(0.48, 'rgba(245, 158, 11, 0.5)');
                grad.addColorStop(0.52, 'rgba(0,0,0,0)'); // Cassini Division Gap
                grad.addColorStop(0.68, 'rgba(251, 191, 36, 0.95)');
                grad.addColorStop(0.85, 'rgba(180, 83, 9, 0.7)');
                grad.addColorStop(1, 'rgba(0,0,0,0)');
                ctx.fillStyle = grad;
                ctx.beginPath(); ctx.arc(512, 512, 500, 0, Math.PI*2); ctx.fill();
                return new THREE.CanvasTexture(canvas);
            }

            function createGenericPlanetTexture(baseHex, bandHex) {
                const canvas = document.createElement('canvas');
                canvas.width = 512; canvas.height = 256;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = baseHex; ctx.fillRect(0, 0, 512, 256);
                ctx.fillStyle = bandHex;
                for(let y=0; y<256; y+=10) {
                    if (Math.random() > 0.35) {
                        ctx.fillRect(0, y, 512, Math.random()*10+2);
                    }
                }
                return new THREE.CanvasTexture(canvas);
            }

            // PLANETARY DATA WITH KEPLERIAN ORBITS & ADVANCED AXIAL TILTS
            const planetData = [
                { name: 'Mercury', radius: 0.55, a: 11.5, b: 10.8, speed: 0.024, tilt: 0.03, inclX: 0.12, base: '#a8a8a8', band: '#787878' },
                { name: 'Venus', radius: 0.85, a: 15.2, b: 14.8, speed: 0.018, tilt: 3.1, inclX: 0.06, base: '#e3bb76', band: '#c49a52' },
                { name: 'Earth', radius: 0.95, a: 19.5, b: 19.0, speed: 0.013, tilt: 0.41, inclX: 0.02, customTex: 'earth', hasMoon: true, hasAtmosphere: true },
                { name: 'Mars', radius: 0.70, a: 24.2, b: 23.5, speed: 0.009, tilt: 0.44, inclX: 0.04, base: '#ef4444', band: '#991b1b' },
                { name: 'Jupiter', radius: 2.1, a: 31.0, b: 29.8, speed: 0.0055, tilt: 0.05, inclX: 0.02, customTex: 'jupiter' },
                { name: 'Saturn', radius: 1.7, a: 38.5, b: 37.0, speed: 0.004, tilt: 0.47, inclX: 0.05, base: '#f59e0b', band: '#d97706', hasRings: true },
                { name: 'Uranus', radius: 1.2, a: 45.0, b: 44.0, speed: 0.0028, tilt: 1.7, inclX: 0.03, base: '#06b6d4', band: '#0891b2' },
                { name: 'Neptune', radius: 1.15, a: 51.5, b: 50.5, speed: 0.002, tilt: 0.49, inclX: 0.03, base: '#2563eb', band: '#1d4ed8' }
            ];

            function init() {
                const container = document.getElementById('webgl-container');
                const width = container.clientWidth;
                const height = container.clientHeight;

                scene = new THREE.Scene();
                scene.fog = new THREE.FogExp2(0x03050d, 0.006);

                camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
                camera.position.set(0, 26, 56);
                camera.lookAt(0, 0, 0);

                renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                renderer.setSize(width, height);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                container.appendChild(renderer.domElement);

                // ILLUMINATION & LIGHTING
                const ambientLight = new THREE.AmbientLight(0x475569, 1.8);
                scene.add(ambientLight);

                const sunLight = new THREE.PointLight(0xfffaed, 5.5, 300);
                scene.add(sunLight);

                // HUGE RADIANT PHOTOREALISTIC SUN (Central Glowing Body)
                const sunRadius = 5.2; // BIGGER DOMINANT SUN
                const sunTex = createSunTexture();
                const sunGeo = new THREE.SphereGeometry(sunRadius, 64, 64);
                const sunMat = new THREE.MeshBasicMaterial({ map: sunTex });
                sun = new THREE.Mesh(sunGeo, sunMat);
                scene.add(sun);

                // Multi-Layer Solar Corona Flares & Atmospheric Wind
                const coronaGeo1 = new THREE.SphereGeometry(sunRadius * 1.15, 32, 32);
                const coronaMat1 = new THREE.MeshBasicMaterial({
                    color: 0xffaa00,
                    transparent: true,
                    opacity: 0.5,
                    side: THREE.BackSide
                });
                coronaLayer1 = new THREE.Mesh(coronaGeo1, coronaMat1);
                sun.add(coronaLayer1);

                const coronaGeo2 = new THREE.SphereGeometry(sunRadius * 1.35, 32, 32);
                const coronaMat2 = new THREE.MeshBasicMaterial({
                    color: 0xff4500,
                    transparent: true,
                    opacity: 0.35,
                    side: THREE.BackSide
                });
                coronaLayer2 = new THREE.Mesh(coronaGeo2, coronaMat2);
                sun.add(coronaLayer2);

                const coronaGeo3 = new THREE.SphereGeometry(sunRadius * 1.65, 32, 32);
                const coronaMat3 = new THREE.MeshBasicMaterial({
                    color: 0x8b5cf6,
                    transparent: true,
                    opacity: 0.22,
                    side: THREE.BackSide
                });
                coronaLayer3 = new THREE.Mesh(coronaGeo3, coronaMat3);
                sun.add(coronaLayer3);

                // PLANETS & KEPLERIAN 3D ORBITS
                planetData.forEach((pd) => {
                    // Generate 3D Elliptical Orbit Curve Line
                    const points = [];
                    for (let i = 0; i <= 128; i++) {
                        const theta = (i / 128) * Math.PI * 2;
                        const x = Math.cos(theta) * pd.a;
                        const z = Math.sin(theta) * pd.b;
                        const y = Math.sin(theta) * pd.a * pd.inclX;
                        points.push(new THREE.Vector3(x, y, z));
                    }
                    const orbitGeo = new THREE.BufferGeometry().setFromPoints(points);
                    const orbitMat = new THREE.LineBasicMaterial({
                        color: 0x818cf8,
                        transparent: true,
                        opacity: 0.35
                    });
                    const orbitLine = new THREE.Line(orbitGeo, orbitMat);
                    scene.add(orbitLine);

                    // Planet Texture Selection
                    let pTex;
                    if (pd.customTex === 'earth') pTex = createEarthTexture();
                    else if (pd.customTex === 'jupiter') pTex = createJupiterTexture();
                    else pTex = createGenericPlanetTexture(pd.base, pd.band);

                    // Planet Mesh with Specular Material
                    const pGeo = new THREE.SphereGeometry(pd.radius, 32, 32);
                    const pMat = new THREE.MeshStandardMaterial({
                        map: pTex,
                        roughness: 0.45,
                        metalness: 0.15
                    });
                    const planet = new THREE.Mesh(pGeo, pMat);
                    planet.rotation.z = pd.tilt;

                    scene.add(planet);

                    // Saturn Rings
                    if (pd.hasRings) {
                        const ringTex = createSaturnRingTexture();
                        const ringGeo = new THREE.RingGeometry(pd.radius * 1.35, pd.radius * 2.8, 64);
                        const ringMat = new THREE.MeshBasicMaterial({
                            map: ringTex,
                            side: THREE.DoubleSide,
                            transparent: true,
                            opacity: 0.85
                        });
                        const ring = new THREE.Mesh(ringGeo, ringMat);
                        ring.rotation.x = Math.PI / 2.3;
                        planet.add(ring);
                    }

                    // Earth Atmospheric Glow Shell
                    if (pd.hasAtmosphere) {
                        const atmosGeo = new THREE.SphereGeometry(pd.radius * 1.08, 32, 32);
                        const atmosMat = new THREE.MeshBasicMaterial({
                            color: 0x38bdf8,
                            transparent: true,
                            opacity: 0.3,
                            side: THREE.BackSide
                        });
                        const atmos = new THREE.Mesh(atmosGeo, atmosMat);
                        planet.add(atmos);
                    }

                    // Earth Moon
                    if (pd.hasMoon) {
                        const moonGeo = new THREE.SphereGeometry(0.25, 16, 16);
                        const moonMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.8 });
                        const moon = new THREE.Mesh(moonGeo, moonMat);
                        moon.position.set(1.6, 0, 0);
                        planet.add(moon);
                        pd.moonMesh = moon;
                    }

                    pd.mesh = planet;
                    pd.angle = Math.random() * Math.PI * 2;
                    planets.push(pd);
                });

                // 3D DEEP SPACE STARFIELD
                const starsGeo = new THREE.BufferGeometry();
                const starsCount = 3500;
                const starPositions = new Float32Array(starsCount * 3);

                for (let i = 0; i < starsCount * 3; i += 3) {
                    starPositions[i] = (Math.random() - 0.5) * 450;
                    starPositions[i + 1] = (Math.random() - 0.5) * 450;
                    starPositions[i + 2] = (Math.random() - 0.5) * 450;
                }

                starsGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
                const starsMat = new THREE.PointsMaterial({
                    color: 0xffffff,
                    size: 0.75,
                    transparent: true,
                    opacity: 0.95
                });
                starField = new THREE.Points(starsGeo, starsMat);
                scene.add(starField);

                // MOUSE TILT CONTROLS
                document.addEventListener('mousemove', (e) => {
                    mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
                    mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
                });

                window.addEventListener('resize', onWindowResize);
                animate();
            }

            function onWindowResize() {
                const container = document.getElementById('webgl-container');
                if (!container) return;
                const width = container.clientWidth;
                const height = container.clientHeight;
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
                renderer.setSize(width, height);
            }

            function animate() {
                requestAnimationFrame(animate);

                // Sun Dynamic Rotation & Corona Flare Pulsations
                if (sun) {
                    sun.rotation.y += 0.003;
                    const pulseTime = Date.now() * 0.0015;
                    coronaLayer1.scale.setScalar(1.0 + Math.sin(pulseTime) * 0.04);
                    coronaLayer2.scale.setScalar(1.0 + Math.cos(pulseTime * 1.3) * 0.06);
                }

                // Planets 3D Keplerian Orbital Motion
                planets.forEach((pd) => {
                    pd.angle += pd.speed * 0.85;
                    pd.mesh.position.x = Math.cos(pd.angle) * pd.a;
                    pd.mesh.position.z = Math.sin(pd.angle) * pd.b;
                    pd.mesh.position.y = Math.sin(pd.angle) * pd.a * pd.inclX;
                    pd.mesh.rotation.y += 0.015;

                    if (pd.moonMesh) {
                        pd.moonMesh.position.x = Math.cos(pd.angle * 4) * 1.6;
                        pd.moonMesh.position.z = Math.sin(pd.angle * 4) * 1.6;
                    }
                });

                if (starField) starField.rotation.y += 0.0003;

                // Interactive 3D Camera Floating Tilt
                camera.position.x += (mouseX * 6.0 - camera.position.x) * 0.04;
                camera.position.y += (-mouseY * 4.5 + 26 - camera.position.y) * 0.04;
                camera.lookAt(0, 0, 0);

                renderer.render(scene, camera);
            }

            window.onload = init;
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=620)


# ============================================================
# RENDER INTRO PAGE
# ============================================================

def render_intro_page():
    """Solar System Animated"""
    # Top Astronomical Taskbar Header
    st.markdown(
        """
        <div class="top-taskbar">
            <div class="taskbar-brand">
                <div class="taskbar-logo">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="9"></circle>
                        <path d="M3.6 9h16.8"></path>
                        <path d="M3.6 15h16.8"></path>
                        <circle cx="12" cy="12" r="3" fill="white"></circle>
                    </svg>
                </div>
                <div>
                    <div class="taskbar-title">NAKSH AI</div>
                    <div class="taskbar-sub">Astronomical Intelligence</div>
                </div>
            </div>
            <div class="taskbar-metrics">
                <div class="taskbar-chip">
                    <span class="taskbar-pulse"></span>
                    3D ORBITAL SIMULATION
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3D Solar System WebGL Canvas
    render_3d_solar_system()

    # Central Text Overlay
    st.markdown(
        """
        <div class="intro-hero-wrapper">
            <div class="intro-welcome-line">WELCOME TO</div>
            <div class="intro-main-title">NAKSH AI</div>
            <div class="intro-subtitle">EXPLORE. ANALYZE. DISCOVER.</div>
            <div class="intro-description">
                An intelligent telescope assistant that uses artificial intelligence to identify celestial objects from astronomical imagery and reveal the regions that influenced its decision.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # DEAD-CENTER BUTTON USING 3 EQUAL STREAMLIT COLUMNS
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        if st.button("BEGIN JOURNEY", key="btn_begin_journey", type="primary"):
            st.session_state["page"] = "dashboard"
            st.rerun()

    # Scroll hint
    st.markdown(
        """
        <div class="scroll-hint">
            SCROLL TO EXPLORE
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RENDER DASHBOARD PAGE (MODEL PREDICTION & EXPLAINABLE AI)
# ============================================================

def render_dashboard_page():
    """Render the Main Naksh AI Dashboard Page."""

    # Model & Data Load
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
    # SIDEBAR TASKBAR (CLEANED & THEMED)
    # ============================================================

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sb-brand">
                <div class="sb-logo-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
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

            <div class="sb-section-title">NAVIGATION TASKBAR</div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Return to Intro", key="btn_sb_nav_intro"):
            st.session_state["page"] = "intro"
            st.rerun()

        st.markdown('<div class="sb-section-title">CELESTIAL CATALOGUE</div>', unsafe_allow_html=True)

        if st.button("Moon (Natural Satellite)", key="btn_sb_moon"):
            show_moon_dialog()
        if st.button("Nebula (Gas & Dust)", key="btn_sb_nebula"):
            show_nebula_dialog()
        if st.button("Planet (Planetary Body)", key="btn_sb_planet"):
            show_planet_dialog()
        if st.button("Star (Luminous Plasma)", key="btn_sb_star"):
            show_star_dialog()

        st.markdown('<div class="sb-section-title">NEURAL ARCHITECTURE</div>', unsafe_allow_html=True)

        if st.button("MobileNetV3Small Network", key="btn_sb_mobilenet"):
            show_mobilenet_dialog()
        if st.button("Grad-CAM Explainability (XAI)", key="btn_sb_gradcam"):
            show_gradcam_dialog()

        st.markdown(
            """
            <div class="sb-status-card">
                <div class="sb-status-header">
                    <span class="sb-status-dot"></span>
                    SYSTEM ONLINE
                </div>
                <div class="sb-status-sub">MobileNetV3 · 98.26% Accuracy</div>
            </div>

            <div class="sb-footer">
                <a href="https://www.python.org/" target="_blank">Python</a> · 
                <a href="https://www.tensorflow.org/" target="_blank">TensorFlow</a> · 
                <a href="https://streamlit.io/" target="_blank">Streamlit</a><br/>
                Naksh AI v2.0 | Astronomical Assistant
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Top Astronomical Taskbar Header for Dashboard Page
    st.markdown(
        """
        <div class="top-taskbar">
            <div class="taskbar-brand">
                <div class="taskbar-logo">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="9"></circle>
                        <path d="M3.6 9h16.8"></path>
                        <path d="M3.6 15h16.8"></path>
                        <circle cx="12" cy="12" r="3" fill="white"></circle>
                    </svg>
                </div>
                <div>
                    <div class="taskbar-title">NAKSH AI DASHBOARD</div>
                    <div class="taskbar-sub">Astronomical Classification & Visual Explainability</div>
                </div>
            </div>
            <div class="taskbar-metrics">
                <div class="taskbar-chip">
                    <span class="taskbar-pulse"></span>
                    98.26% TEST ACCURACY
                </div>
            </div>
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
                    MODEL ONLINE & READY
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # EXPLORE CELESTIAL OBJECTS (PLANET SHAPED CONTAINERS)
    # ============================================================

    st.markdown('<div class="section-header-title">EXPLORE CELESTIAL OBJECTS (CLICK NASA CARDS FOR DETAILS)</div>', unsafe_allow_html=True)

    col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)

    with col_exp1:
        st.markdown(
            f"""
            <div class="explore-card">
                <div class="explore-img-wrapper">
                    <img class="explore-img" src="{moon_b64}" />
                </div>
                <div class="explore-body">
                    <div class="explore-name">Moon</div>
                    <div class="explore-text">Natural satellite orbiting planets.</div>
                    <div style="width: 100%; margin-top: 14px;">
                        <a class="cosmic-link-card cosmic-link-moon" href="https://science.nasa.gov/moon/" target="_blank" style="padding: 10px 14px; gap: 10px;">
                            <div class="cosmic-link-badge" style="width: 26px; height: 26px;">
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2">
                                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                                </svg>
                            </div>
                            <div class="cosmic-link-details">
                                <div class="cosmic-link-title" style="font-size: 0.82rem;">NASA Moon Portal</div>
                            </div>
                        </a>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_exp2:
        st.markdown(
            f"""
            <div class="explore-card">
                <div class="explore-img-wrapper">
                    <img class="explore-img" src="{nebula_b64}" />
                </div>
                <div class="explore-body">
                    <div class="explore-name">Nebula</div>
                    <div class="explore-text">Interstellar clouds of gas and dust.</div>
                    <div style="width: 100%; margin-top: 14px;">
                        <a class="cosmic-link-card cosmic-link-nebula" href="https://science.nasa.gov/universe/nebulae/" target="_blank" style="padding: 10px 14px; gap: 10px;">
                            <div class="cosmic-link-badge" style="width: 26px; height: 26px;">
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2">
                                    <path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"></path>
                                </svg>
                            </div>
                            <div class="cosmic-link-details">
                                <div class="cosmic-link-title" style="font-size: 0.82rem;">NASA Nebulae Portal</div>
                            </div>
                        </a>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_exp3:
        st.markdown(
            f"""
            <div class="explore-card">
                <div class="explore-img-wrapper">
                    <img class="explore-img" src="{planet_b64}" />
                </div>
                <div class="explore-body">
                    <div class="explore-name">Planet</div>
                    <div class="explore-text">Large celestial bodies orbiting stars.</div>
                    <div style="width: 100%; margin-top: 14px;">
                        <a class="cosmic-link-card cosmic-link-planet" href="https://science.nasa.gov/planets/" target="_blank" style="padding: 10px 14px; gap: 10px;">
                            <div class="cosmic-link-badge" style="width: 26px; height: 26px;">
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2">
                                    <circle cx="12" cy="12" r="7"></circle>
                                    <path d="M2.5 12h19"></path>
                                </svg>
                            </div>
                            <div class="cosmic-link-details">
                                <div class="cosmic-link-title" style="font-size: 0.82rem;">NASA Planets Portal</div>
                            </div>
                        </a>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_exp4:
        st.markdown(
            f"""
            <div class="explore-card">
                <div class="explore-img-wrapper">
                    <img class="explore-img" src="{star_b64}" />
                </div>
                <div class="explore-body">
                    <div class="explore-name">Star</div>
                    <div class="explore-text">Luminous spheres of plasma and energy.</div>
                    <div style="width: 100%; margin-top: 14px;">
                        <a class="cosmic-link-card cosmic-link-star" href="https://science.nasa.gov/universe/stars/" target="_blank" style="padding: 10px 14px; gap: 10px;">
                            <div class="cosmic-link-badge" style="width: 26px; height: 26px;">
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#ea580c" stroke-width="2">
                                    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                                </svg>
                            </div>
                            <div class="cosmic-link-details">
                                <div class="cosmic-link-title" style="font-size: 0.82rem; color: #ffffff !important;">NASA Stars Portal</div>
                            </div>
                        </a>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
            <div class="panel-header-title" style="text-align: center;">CLASSIFY ASTRONOMICAL IMAGE</div>
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

            # Center uploaded image preview
            st.markdown('<div style="display: flex; justify-content: center; margin: 12px 0;">', unsafe_allow_html=True)
            st.image(image, caption=f"Uploaded Observation: {uploaded_file.name}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

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
                <div class="panel-header-title" style="margin-top: 15px; text-align: center;">CLASSIFICATION RESULT</div>
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
            <a class="banner-btn" href="https://science.nasa.gov/" target="_blank">LEARN MORE</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN APPLICATION CONTROLLER
# ============================================================

if st.session_state["page"] == "intro":
    render_intro_page()
else:
    render_dashboard_page()
