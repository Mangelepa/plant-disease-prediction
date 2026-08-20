import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Config

MODEL_PATH = "mobilenetv2_plant_disease.keras"
IMG_SIZE = (128, 128)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# Helpers



def format_class_name(raw_name: str) -> tuple[str, str]:
    if "___" in raw_name:
        plant, condition = raw_name.split("___", 1)
    else:
        plant, condition = raw_name, ""
    plant = plant.replace("_", " ")
    condition = condition.replace("_", " ").strip()
    return plant, condition


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.utils.img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    return arr



# Styling


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg: #F3F4EF;
  --panel: #FFFFFF;
  --panel-alt: #FAFAF8;
  --ink: #12140F;
  --ink-soft: #565E51;
  --ink-faint: #94998A;
  --accent: #22C878;
  --accent-deep: #1E6B49;
  --accent-dim: #4C8B6C;
  --alert: #FF6B4A;
  --alert-dim: #C07E6C;
  --line: #DCDFD3;
  --hero-1: #0A1510;
  --hero-2: #0F2318;
  --hero-3: #143726;
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

html { scroll-behavior: smooth; }

.stApp {
  background: var(--bg);
  font-family: 'Inter', -apple-system, sans-serif;
  color: var(--ink);
}

.block-container { max-width: 1040px; padding-top: 0 !important; padding-bottom: 4rem; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(14px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes glow {
  0%, 100% { opacity: 0.55; }
  50% { opacity: 1; }
}
@keyframes drift {
  0% { transform: translate(0,0) rotate(0deg); }
  50% { transform: translate(-2%, 1.5%) rotate(1.5deg); }
  100% { transform: translate(0,0) rotate(0deg); }
}

/* =========================================================
   HERO — full-bleed, photographic-atmosphere, floating cards
   ========================================================= */
.hero-band {
  position: relative;
  left: 50%; right: 50%; margin-left: -50vw; margin-right: -50vw; width: 100vw;
  min-height: 620px;
  background:
    radial-gradient(circle at 18% 24%, rgba(94,196,140,0.30), transparent 42%),
    radial-gradient(circle at 78% 14%, rgba(60,150,100,0.22), transparent 40%),
    radial-gradient(circle at 55% 78%, rgba(30,110,75,0.28), transparent 46%),
    radial-gradient(circle at 92% 70%, rgba(120,210,160,0.14), transparent 38%),
    linear-gradient(165deg, #081410 0%, #0C1F17 45%, #123324 100%);
  overflow: hidden;
  padding: 28px 0 0;
}
.hero-band::before {
  /* soft bokeh grain, mimics depth-of-field light spots on a leaf photo */
  content: "";
  position: absolute; inset: 0;
  background-image:
    radial-gradient(circle, rgba(255,255,255,0.09) 2px, transparent 2.5px),
    radial-gradient(circle, rgba(255,255,255,0.05) 1.5px, transparent 2px);
  background-size: 140px 140px, 70px 70px;
  background-position: 0 0, 35px 60px;
  animation: drift 20s ease-in-out infinite;
  pointer-events: none;
}
.hero-band::after {
  content: "";
  position: absolute; left: 0; right: 0; bottom: 0; height: 90px;
  background: linear-gradient(180deg, transparent, var(--bg));
  pointer-events: none;
}
.hero-inner { max-width: 1120px; margin: 0 auto; padding: 0 2rem 64px; position: relative; z-index: 2; }

/* ---- nav ---- */
.hero-nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 0 44px;
}
.hero-brand { display: flex; align-items: center; gap: 8px; }
.hero-brand-dot {
  width: 9px; height: 9px; border-radius: 50%; background: var(--accent);
  box-shadow: 0 0 10px 2px rgba(34,200,120,0.7);
  animation: glow 2.4s ease-in-out infinite;
}
.hero-brand-text {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 15px;
  letter-spacing: 0.02em; color: #F7F9F5;
}

/* ---- badge ---- */
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.14);
  backdrop-filter: blur(8px); border-radius: 40px; padding: 7px 14px 7px 10px;
  font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; letter-spacing: 0.04em;
  color: #D8E6DA; margin-bottom: 26px;
  animation: fadeInUp 0.6s ease both;
}
.hero-badge .b-icon {
  width: 18px; height: 18px; border-radius: 50%; background: var(--accent);
  display: inline-flex; align-items: center; justify-content: center; color: #06150E;
}
.hero-badge .b-icon svg { width: 11px; height: 11px; }

/* ---- headline ---- */
.hero-grid { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 40px; align-items: center; }
@media (max-width: 900px) { .hero-grid { grid-template-columns: 1fr; } }

.la-title {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: clamp(34px, 4.6vw, 54px); line-height: 1.08; letter-spacing: -0.02em;
  margin: 0 0 18px; color: #F7F9F5;
  animation: fadeInUp 0.7s ease 0.05s both;
}
.la-title .icon-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 0.86em; height: 0.86em; border-radius: 50%;
  background: linear-gradient(135deg, var(--accent), #0F5C3C);
  vertical-align: -0.1em; margin: 0 0.06em; box-shadow: 0 0 0 4px rgba(34,200,120,0.12);
}
.la-title .icon-badge svg { width: 55%; height: 55%; color: #06150E; }
.la-title .accent {
  background: linear-gradient(90deg, #22C878, #A8ECC8);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}

.la-sub {
  font-size: 16px; color: #B9C6BA; max-width: 46ch; margin: 0 0 28px; line-height: 1.65;
  animation: fadeInUp 0.7s ease 0.1s both;
}

.hero-actions { display: flex; align-items: center; gap: 16px; animation: fadeInUp 0.7s ease 0.15s both; }
a.cta-pill, a.cta-pill:link, a.cta-pill:visited, a.cta-pill:hover, a.cta-pill:active {
  display: inline-flex; align-items: center; gap: 9px;
  background: linear-gradient(135deg, #14895A, #0C5C3B) !important;
  color: #F7FBF8 !important; font-family: 'Inter', sans-serif; font-weight: 700; font-size: 14.5px;
  padding: 13px 22px; border-radius: 44px; text-decoration: none !important; border: none; cursor: pointer;
  box-shadow: 0 14px 30px -12px rgba(15,90,60,0.6);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
a.cta-pill:hover { transform: translateY(-2px); box-shadow: 0 18px 34px -10px rgba(15,90,60,0.7); }
.cta-pill svg { width: 15px; height: 15px; }
span.cta-secondary {
  font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #9FAF9F;
  border-bottom: 1px solid rgba(159,175,159,0.4); padding-bottom: 2px;
}

/* ---- floating visual side ---- */
.hero-visual { position: relative; min-height: 360px; }
.orb-frame {
  position: absolute; inset: 6% 10%;
  border-radius: 24px;
  background:
    radial-gradient(circle at 30% 20%, rgba(94,196,140,0.35), transparent 55%),
    radial-gradient(circle at 75% 75%, rgba(20,80,55,0.5), transparent 55%),
    linear-gradient(150deg, rgba(255,255,255,0.06), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.12);
  animation: fadeInUp 0.8s ease 0.2s both;
}
.corner-bracket { position: absolute; width: 22px; height: 22px; border-color: rgba(255,255,255,0.35); opacity: 0.8; }
.cb-tl { top: 2%; left: 4%; border-top: 1.5px solid; border-left: 1.5px solid; border-radius: 4px 0 0 0; }
.cb-tr { top: 2%; right: 4%; border-top: 1.5px solid; border-right: 1.5px solid; border-radius: 0 4px 0 0; }

.float-card {
  position: absolute; right: -4%; bottom: 6%; width: 260px;
  background: rgba(15,30,22,0.72); border: 1px solid rgba(255,255,255,0.14);
  backdrop-filter: blur(14px); border-radius: 14px; padding: 18px;
  box-shadow: 0 30px 60px -24px rgba(0,0,0,0.55);
  animation: fadeInUp 0.8s ease 0.3s both;
}
.fc-status {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.05em;
  text-transform: uppercase; color: #7FE8B4; margin-bottom: 10px;
}
.fc-status svg { width: 13px; height: 13px; }
.fc-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 17px; color: #F7F9F5; margin: 0 0 4px; }
.fc-desc { font-size: 11.5px; color: #A9B8AA; line-height: 1.5; margin: 0 0 14px; }
.fc-conf-row { display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: #8FA090; margin-bottom: 5px; }
.fc-gauge { height: 5px; background: rgba(255,255,255,0.12); border-radius: 3px; overflow: hidden; margin-bottom: 14px; }
.fc-gauge-fill { height: 100%; width: 91%; background: linear-gradient(90deg, #1E6B49, #22C878); border-radius: 3px; }
.fc-actions-label { font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; letter-spacing: 0.08em; text-transform: uppercase; color: #7C8A7C; margin-bottom: 8px; }
.fc-actions { margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 6px; }
.fc-actions li { font-size: 11.5px; color: #D3DDD4; display: flex; align-items: center; gap: 7px; }
.fc-actions li::before { content: ""; width: 4px; height: 4px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }

.float-chip {
  position: absolute; left: 2%; top: 8%;
  background: rgba(15,30,22,0.72); border: 1px solid rgba(255,255,255,0.14);
  backdrop-filter: blur(12px); border-radius: 30px; padding: 9px 16px;
  display: flex; align-items: center; gap: 8px;
  font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #EAF3EC;
  box-shadow: 0 20px 40px -20px rgba(0,0,0,0.5);
  animation: fadeInUp 0.8s ease 0.4s both;
}
.float-chip svg { width: 14px; height: 14px; color: var(--accent); }

@media (max-width: 900px) {
  .hero-visual { min-height: 300px; margin-top: 20px; }
  .float-card { right: 4%; }
}

/* ---- process rail (How it works) ---- */
.process-rail {
  margin-top: 0; padding: 26px 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;
}
.p-step { display: flex; gap: 12px; align-items: flex-start; }
.p-step-num {
  font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--accent-deep);
  border: 1px solid var(--line); border-radius: 50%; width: 24px; height: 24px;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.p-step-text h4 { margin: 0 0 3px; font-size: 14px; font-weight: 600; }
.p-step-text p { margin: 0; font-size: 12.5px; color: var(--ink-faint); line-height: 1.5; }
@media (max-width: 720px) { .process-rail { grid-template-columns: 1fr; gap: 16px; } }

/* =========================================================
   MAIN CONTENT
   ========================================================= */
.panel-label {
  font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--ink-faint); margin: 0 0 10px 2px;
  display: flex; align-items: center; gap: 8px;
}
.panel-label .num {
  width: 18px; height: 18px; border-radius: 50%; background: var(--accent-deep); color: #fff;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 600;
}

/* upload card */
.upload-card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
  padding: 18px; box-shadow: 0 1px 0 rgba(18,20,15,0.03), 0 20px 40px -24px rgba(18,20,15,0.25);
  transition: box-shadow 0.25s ease, transform 0.25s ease;
}
.upload-card:hover { box-shadow: 0 1px 0 rgba(18,20,15,0.03), 0 26px 52px -22px rgba(30,107,73,0.28); }

div[data-testid="stFileUploaderDropzone"] {
  background: linear-gradient(180deg, var(--panel-alt), #F3F5EE) !important;
  border: 1.5px dashed #C7CDBA !important;
  border-radius: 10px !important;
  padding: 8px !important;
  transition: border-color 0.2s ease, background 0.2s ease;
}
div[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--accent-deep) !important;
  background: linear-gradient(180deg, #EFF8F2, #E7F4EC) !important;
}
section[data-testid="stFileUploader"] small { color: var(--ink-faint) !important; }
div[data-testid="stFileUploaderDropzoneInstructions"] svg { color: var(--accent-deep) !important; }

.preview-frame {
  border-radius: 10px; overflow: hidden; border: 1px solid var(--line); margin-top: 12px;
  box-shadow: 0 14px 30px -18px rgba(18,20,15,0.35);
}

/* result card */
.result-card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
  padding: 24px 26px; position: relative; overflow: hidden;
  box-shadow: 0 1px 0 rgba(18,20,15,0.03), 0 20px 40px -24px rgba(18,20,15,0.25);
  animation: fadeInUp 0.5s ease both;
}
.result-card::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--accent-deep), var(--accent));
}
.result-card.alert::before { background: linear-gradient(90deg, #8B3A24, var(--alert)); }

.rc-head {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid var(--line);
}
.class-index { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--ink-faint); letter-spacing: 0.05em; }
.status-pill {
  display: inline-flex; align-items: center; gap: 7px;
  font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase;
  padding: 6px 12px; border-radius: 20px;
}
.status-pill.healthy { color: #0E5C3A; background: rgba(34,200,120,0.14); }
.status-pill.diseased { color: #A63A20; background: rgba(255,107,74,0.14); }
.status-pill svg { width: 13px; height: 13px; }

.rc-plant { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 24px; margin: 0 0 4px; }
.rc-condition { font-size: 14.5px; color: var(--ink-soft); margin: 0 0 22px; }

.confidence-top { display: flex; align-items: baseline; gap: 9px; margin-bottom: 8px; }
.confidence-num {
  font-family: 'IBM Plex Mono', monospace; font-size: 32px; font-weight: 500;
  background: linear-gradient(90deg, var(--accent-deep), var(--accent));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.result-card.alert .confidence-num {
  background: linear-gradient(90deg, #8B3A24, var(--alert));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.confidence-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--ink-faint); letter-spacing: 0.05em; text-transform: uppercase; }
.gauge { height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; margin-bottom: 24px; }
.gauge-fill {
  height: 100%; border-radius: 4px;
  background: linear-gradient(90deg, var(--accent-deep), var(--accent));
  box-shadow: 0 0 12px rgba(34,200,120,0.5);
}
.result-card.alert .gauge-fill {
  background: linear-gradient(90deg, #8B3A24, var(--alert));
  box-shadow: 0 0 12px rgba(255,107,74,0.4);
}

.breakdown-label {
  font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--ink-faint); margin-bottom: 10px;
}
.bar-row { display: grid; grid-template-columns: 22px 1fr auto; align-items: center; gap: 8px; padding: 6px 0; }
.bar-idx {
  font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--ink-faint);
  width: 18px; height: 18px; border-radius: 4px; background: var(--panel-alt);
  display: flex; align-items: center; justify-content: center; border: 1px solid var(--line);
}
.bar-name { font-size: 12.5px; color: var(--ink-soft); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 500; }
.bar-pct { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--ink-faint); min-width: 40px; text-align: right; }
.bar-track { grid-column: 1 / -1; height: 4px; background: var(--line); border-radius: 2px; overflow: hidden; margin: 3px 0 6px; }
.bar-fill { height: 100%; border-radius: 2px; background: var(--accent-dim); }
.bar-fill.first { background: linear-gradient(90deg, var(--accent-deep), var(--accent)); }

.rc-note {
  margin-top: 20px; padding: 12px 14px; border-radius: 8px; background: var(--panel-alt);
  font-size: 11.5px; color: var(--ink-faint); line-height: 1.55;
}

.empty-card {
  border: 1.5px dashed var(--line); border-radius: 14px; padding: 56px 24px;
  text-align: center; color: var(--ink-faint); background: var(--panel-alt);
  display: flex; flex-direction: column; align-items: center; gap: 12px;
}
.empty-card svg { color: #C7CDBA; }
.empty-card p { font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: 0.05em; margin: 0; }
.empty-card span { font-size: 12.5px; color: var(--ink-faint); max-width: 32ch; }

/* sidebar — dark, high-contrast */
section[data-testid="stSidebar"] {
  background: #0A0F0B !important;
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] > div { background: #0A0F0B !important; }

.sb-mark { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.sb-mark-dot {
  width: 8px; height: 8px; border-radius: 50%; background: var(--accent);
  box-shadow: 0 0 8px 1px rgba(34,200,120,0.6);
}
.sb-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 16px; color: #F2F5EF !important; }
.sb-mono { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #9FC9AC !important; line-height: 2.1; }
.sb-mono b { color: #7FE8B4 !important; letter-spacing: 0.04em; }

section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

section[data-testid="stSidebar"] div[data-testid="stExpander"] {
  background: rgba(255,255,255,0.035) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 8px !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
  color: #EAF3EC !important; font-family: 'Inter', sans-serif; font-weight: 500;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] svg { color: #9FC9AC !important; }
section[data-testid="stSidebar"] div[data-testid="stExpander"] p {
  color: #C3D3C6 !important; font-size: 12.5px;
}

/* footer */
.la-footer {
  margin-top: 44px; padding-top: 18px; border-top: 1px solid var(--line);
  font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--ink-faint);
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
}
</style>
"""

ICON_LEAF_OUTLINE = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2C7 6 5 10 5 14a7 7 0 0 0 14 0c0-4-2-8-7-12Z"/><path d="M12 8v12"/></svg>"""
ICON_CHECK = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6 9 17l-5-5"/></svg>"""
ICON_ALERT = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 2.5 17.5A1.8 1.8 0 0 0 4 20h16a1.8 1.8 0 0 0 1.5-2.5L13.7 3.9a1.8 1.8 0 0 0-3.4 0Z"/></svg>"""
ICON_SCAN = """<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 8h8M8 12h8M8 16h5"/></svg>"""
ICON_ARROW = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M13 6l6 6-6 6"/></svg>"""
ICON_BOLT = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2 3 14h7l-1 8 10-12h-7l1-8Z"/></svg>"""


def render_header():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    nav_html = (
        '<div class="hero-nav">'
        '<div class="hero-brand"><span class="hero-brand-dot"></span>'
        '<span class="hero-brand-text">LEAF ANALYZER</span></div>'
        "</div>"
    )

    badge_html = (
        '<div class="hero-badge">'
        f'<span class="b-icon">{ICON_LEAF_OUTLINE}</span>'
        "Computer-vision plant diagnostics</div>"
    )

    headline_html = (
        '<h1 class="la-title">Instant '
        f'<span class="icon-badge">{ICON_LEAF_OUTLINE}</span>'
        ' leaf health check<span class="accent">.</span></h1>'
    )

    sub_html = (
        '<p class="la-sub">Photograph any leaf and get an instant read on plant health — '
        "the model scores it against 38 trained conditions across 14 species and returns "
        "a ranked diagnosis with confidence.</p>"
    )

    actions_html = (
        '<div class="hero-actions">'
        f'<a class="cta-pill" href="#analyzer">Scan a leaf {ICON_ARROW}</a>'
        '<span class="cta-secondary">38 classes listed in the sidebar →</span>'
        "</div>"
    )

    visual_html = (
        '<div class="hero-visual">'
        '<div class="orb-frame"></div>'
        '<span class="corner-bracket cb-tl"></span>'
        '<span class="corner-bracket cb-tr"></span>'
        '<div class="float-chip">'
        f'<span style="display:inline-flex;color:#22C878;">{ICON_BOLT}</span>'
        "38 classes · 14 species</div>"
        '<div class="float-card">'
        f'<div class="fc-status">{ICON_CHECK} Example diagnosis</div>'
        '<h4 class="fc-title">Early Blight</h4>'
        '<p class="fc-desc">Dark concentric spots on lower leaves, spreading upward as the season progresses.</p>'
        '<div class="fc-conf-row"><span>CONFIDENCE</span><span>91%</span></div>'
        '<div class="fc-gauge"><div class="fc-gauge-fill"></div></div>'
        '<p class="fc-actions-label">Recommended actions</p>'
        "<ul class=\"fc-actions\"><li>Remove and destroy affected leaves</li>"
        "<li>Improve airflow around plants</li>"
        "<li>Apply an appropriate fungicide</li></ul>"
        "</div></div>"
    )

    hero_html = (
        '<div class="hero-band"><div class="hero-inner">'
        f"{nav_html}{badge_html}"
        '<div class="hero-grid"><div>'
        f"{headline_html}{sub_html}{actions_html}"
        "</div>"
        f"{visual_html}"
        "</div></div></div>"
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    process_html = (
        '<div id="how-it-works" class="process-rail">'
        '<div class="p-step"><span class="p-step-num">1</span>'
        '<div class="p-step-text"><h4>Capture</h4>'
        "<p>Upload a clear, well-lit photo of a single leaf.</p></div></div>"
        '<div class="p-step"><span class="p-step-num">2</span>'
        '<div class="p-step-text"><h4>Classify</h4>'
        "<p>The model scores the image against every trained class.</p></div></div>"
        '<div class="p-step"><span class="p-step-num">3</span>'
        '<div class="p-step-text"><h4>Review</h4>'
        "<p>Read the top match and confidence, with alternates ranked.</p></div></div>"
        "</div>"
    )
    st.markdown(process_html, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        sidebar_html = (
            '<div class="sb-mark"><span class="sb-mark-dot"></span>'
            '<span class="sb-title">Leaf Analyzer</span></div>'
            '<p class="sb-mono">'
            "<b>ARCHITECTURE</b>&nbsp;&nbsp;MobileNetV2<br/>"
            "<b>INPUT</b>&nbsp;&nbsp;128×128 RGB<br/>"
            "<b>CLASSES</b>&nbsp;&nbsp;38 · 14 species"
            "</p>"
        )
        st.markdown(sidebar_html, unsafe_allow_html=True)
        st.divider()
        st.markdown('<p class="sb-title" style="font-size:13.5px;">Reference classes</p>', unsafe_allow_html=True)
        with st.expander("See all 38 classes"):
            for name in CLASS_NAMES:
                plant, condition = format_class_name(name)
                label = f"{plant} — {condition}" if condition else f"{plant} — Healthy"
                st.write(f"• {label}")


def render_empty_result():
    empty_html = (
        '<div class="empty-card">'
        f"{ICON_SCAN}"
        "<p>NO SAMPLE ANALYZED YET</p>"
        "<span>Upload a leaf photo on the left to run a classification.</span>"
        "</div>"
    )
    st.markdown(empty_html, unsafe_allow_html=True)


def render_result_card(top_idx: int, preds: np.ndarray):
    plant, condition = format_class_name(CLASS_NAMES[top_idx])
    is_healthy = condition.lower() == "healthy" or condition == ""
    confidence = float(preds[top_idx]) * 100

    card_class = "healthy" if is_healthy else "alert"
    pill_class = "healthy" if is_healthy else "diseased"
    pill_icon = ICON_CHECK if is_healthy else ICON_ALERT
    status_text = "Healthy" if is_healthy else "Disease detected"
    condition_text = "No signs of disease detected" if is_healthy else condition

    top5_idx = np.argsort(preds)[::-1][:5]
    bar_rows = ""
    for rank, idx in enumerate(top5_idx, start=1):
        p, c = format_class_name(CLASS_NAMES[idx])
        label = f"{p} — {c}" if c else f"{p} — Healthy"
        pct = float(preds[idx]) * 100
        first_class = "first" if rank == 1 else ""
        bar_rows += (
            '<div class="bar-row">'
            f'<span class="bar-idx">{rank}</span>'
            f'<span class="bar-name">{label}</span>'
            f'<span class="bar-pct">{pct:.1f}%</span>'
            f'<div class="bar-track"><div class="bar-fill {first_class}" style="width:{pct:.1f}%"></div></div>'
            "</div>"
        )

    card_html = (
        f'<div class="result-card {"alert" if not is_healthy else ""}">'
        '<div class="rc-head">'
        f'<span class="class-index">CLASS INDEX {top_idx:02d}/38</span>'
        f'<span class="status-pill {pill_class}">{pill_icon}{status_text}</span>'
        "</div>"
        f'<h3 class="rc-plant">{plant}</h3>'
        f'<p class="rc-condition">{condition_text}</p>'
        '<div class="confidence-top">'
        f'<span class="confidence-num">{confidence:.1f}%</span>'
        '<span class="confidence-label">confidence</span>'
        "</div>"
        f'<div class="gauge"><div class="gauge-fill" style="width:{confidence:.1f}%"></div></div>'
        '<p class="breakdown-label">Ranked alternates</p>'
        f"{bar_rows}"
        '<p class="rc-note">Decision-support only — not a substitute for expert agronomic '
        "diagnosis. Low confidence or unusual lighting can reduce reliability.</p>"
        "</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)



# App


def main():
    st.set_page_config(page_title="Leaf Analyzer", page_icon="🌿", layout="wide")

    render_header()
    render_sidebar()

    try:
        model = load_model()
    except Exception as e:
        st.error(
            f"Could not load the model file `{MODEL_PATH}`. Make sure it's "
            f"in the same folder as this app. Error: {e}"
        )
        st.stop()

    st.markdown('<div id="analyzer" style="padding-top:44px;"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<p class="panel-label"><span class="num">1</span>Input — leaf image</p>', unsafe_allow_html=True)
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.markdown('<div class="preview-frame">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="panel-label"><span class="num">2</span>Output — classification</p>', unsafe_allow_html=True)
        if uploaded_file is not None:
            with st.spinner("Analyzing leaf..."):
                arr = preprocess_image(image)
                preds = model.predict(arr, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            render_result_card(top_idx, preds)
        else:
            render_empty_result()

    footer_html = (
        '<div class="la-footer">'
        "<span>Trained via transfer learning on MobileNetV2. Predictions are probabilistic "
        "and should support, not replace, in-field expertise.</span>"
        "<span>Leaf Analyzer — diagnostic demo</span>"
        "</div>"
    )
    st.markdown(footer_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()