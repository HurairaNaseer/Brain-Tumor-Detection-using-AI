import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import time

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── API URL — Yahan apna API URL daalo ────────────────────────
API_URL = "http://localhost:8000"  # Local ke liye
# API_URL = "https://your-api.ngrok.io"  # Colab ke liye

# ─── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0a0f1e;
    color: #e2e8f0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem; max-width: 1200px; }

.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0f1f3d 50%, #0a1628 100%);
    border-bottom: 1px solid rgba(56, 189, 248, 0.15);
    padding: 3rem 0 2rem 0;
    text-align: center;
    margin: 0 -2rem 2rem -2rem;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
    pointer-events: none;
}

.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    padding: 0.3rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
    margin-bottom: 0.8rem;
}

.hero h1 span {
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    max-width: 480px;
    margin: 0 auto 1.5rem auto;
    line-height: 1.6;
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    flex-wrap: wrap;
}

.stat { text-align: center; }
.stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
}
.stat-label { font-size: 0.75rem; color: #64748b; margin-top: 0.1rem; }

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #cbd5e1;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.result-card {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(56,189,248,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.result-card.success { border-color: rgba(34,197,94,0.3); background: rgba(34,197,94,0.04); }
.result-card.warning { border-color: rgba(251,146,60,0.3); background: rgba(251,146,60,0.04); }
.result-card.danger  { border-color: rgba(239,68,68,0.3);  background: rgba(239,68,68,0.04);  }

.result-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #64748b;
    letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
}
.result-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #f1f5f9; }
.result-value.tumor { color: #f87171; }
.result-value.safe  { color: #4ade80; }

.conf-bar-bg { background: rgba(255,255,255,0.06); border-radius: 100px; height: 8px; margin-top: 0.8rem; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #38bdf8, #818cf8); }

.prob-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.6rem; }
.prob-name { font-size: 0.8rem; color: #94a3b8; width: 90px; flex-shrink: 0; }
.prob-bar-bg { flex: 1; background: rgba(255,255,255,0.06); border-radius: 100px; height: 6px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 100px; }
.prob-pct { font-size: 0.78rem; font-weight: 600; color: #e2e8f0; width: 38px; text-align: right; flex-shrink: 0; }

.info-box { background: rgba(56,189,248,0.05); border: 1px solid rgba(56,189,248,0.15); border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; }
.info-box h4 { font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem; font-weight: 600; color: #38bdf8; margin-bottom: 0.5rem; }
.info-box p { font-size: 0.82rem; color: #94a3b8; line-height: 1.6; }

.api-status { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.75rem; padding: 0.25rem 0.75rem; border-radius: 100px; margin-bottom: 1rem; }
.api-status.online  { background: rgba(34,197,94,0.1);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3);  }
.api-status.offline { background: rgba(239,68,68,0.1);  color: #f87171; border: 1px solid rgba(239,68,68,0.3);  }

.disclaimer { background: rgba(251,191,36,0.05); border: 1px solid rgba(251,191,36,0.2); border-radius: 10px; padding: 1rem 1.2rem; font-size: 0.78rem; color: #fbbf24; line-height: 1.6; margin-top: 1.5rem; }

.stFileUploader > div { background: transparent !important; border: none !important; }
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    width: 100% !important;
}
div[data-testid="stImage"] img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────
CLASS_COLORS = {
    'Glioma':     '#f87171',
    'Meningioma': '#fb923c',
    'No Tumor':   '#4ade80',
    'Pituitary':  '#a78bfa'
}

CLASS_INFO = {
    'Glioma':     {'desc': 'A tumor arising from the glial cells of the brain. The most common brain tumor.', 'severity': 'danger',  'icon': '🔴'},
    'Meningioma': {'desc': 'A tumor originating from the meninges. It is often slow-growing.',          'severity': 'warning', 'icon': '🟠'},
    'No Tumor':   {'desc': 'No tumor was found in the MRI scan. The brain appears normal.',     'severity': 'success', 'icon': '🟢'},
    'Pituitary':  {'desc': 'A tumor occurring on the pituitary gland that affects hormones.',   'severity': 'warning', 'icon': '🟣'}
}

# ─── API Health Check ──────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

# ─── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Medical Imaging</div>
    <h1>Brain Tumor <span>Detection</span></h1>
    <p class="hero-sub">Upload an MRI scan and get instant AI analysis — tumor type, confidence score, and visual explanation.</p>
    <div class="hero-stats">
        <div class="stat"><div class="stat-num">92%</div><div class="stat-label">Model Accuracy</div></div>
        <div class="stat"><div class="stat-num">4</div><div class="stat-label">Tumor Classes</div></div>
        <div class="stat"><div class="stat-num">EfficientNetB3</div><div class="stat-label">Architecture</div></div>
        <div class="stat"><div class="stat-num">12,841</div><div class="stat-label">Training Images</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Layout ────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    st.markdown('<div class="section-title">Upload MRI Scan</div>', unsafe_allow_html=True)

    # API Status
    api_online = check_api()
    status_class = "online" if api_online else "offline"
    status_text  = "● API Online" if api_online else "● API Offline"
    st.markdown(f'<div class="api-status {status_class}">{status_text}</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "MRI Image Upload Karo (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="visible"
    )

    if uploaded:
        img_pil   = Image.open(uploaded).convert('RGB')
        img_array = np.array(img_pil)
        st.image(img_pil, caption="Uploaded MRI", use_container_width=True)
        analyze_btn = st.button("🔍 Analyze MRI")
    else:
        analyze_btn = False

    st.markdown("""
    <div class="info-box" style="margin-top:1rem;">
        <h4>Detectable Types</h4>
        <p>🔴 Glioma &nbsp;&nbsp; 🟠 Meningioma<br>🟣 Pituitary &nbsp;&nbsp; 🟢 No Tumor</p>
    </div>
    <div class="info-box">
        <h4>How it Works</h4>
        <p>Image goes to the FastAPI backend → EfficientNetB3 analyzes it → Result is returned.</p>
    </div>
    """, unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; color:#334155;">
            <div style="font-size:4rem; opacity:0.4; margin-bottom:1rem;">🔬</div>
            <div style="font-size:0.9rem; opacity:0.6;">MRI upload karo aur Analyze karo</div>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded and not analyze_btn:
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem;">
            <div style="font-size:4rem; margin-bottom:1rem;">⬅️</div>
            <div style="font-size:0.9rem; color:#64748b;">"Analyze MRI" button dabao</div>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded and analyze_btn:
        if not api_online:
            st.error("❌ API offline hai! Pehle api.py run karo.")
        else:
            with st.spinner("AI analyze kar raha hai..."):
                time.sleep(0.5)
                uploaded.seek(0)
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, uploaded.read(), uploaded.type)}
                )
                result = response.json()

            if result.get("success"):
                pred_class = result["prediction"]
                confidence = result["confidence"]
                all_probs  = result["all_classes"]
                info       = CLASS_INFO[pred_class]
                severity   = info["severity"]

                # Main result card
                safe_class = 'safe' if pred_class == 'No Tumor' else 'tumor'
                st.markdown(f"""
                <div class="result-card {severity}">
                    <div class="result-label">Detection Result</div>
                    <div class="result-value {safe_class}">{info['icon']} {pred_class}</div>
                    <div style="font-size:0.82rem; color:#94a3b8; margin-top:0.5rem;">{info['desc']}</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{confidence:.0f}%"></div>
                    </div>
                    <div style="font-size:0.78rem; color:#64748b; margin-top:0.4rem;">
                        Confidence: <strong style="color:#e2e8f0;">{confidence:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probabilities
                st.markdown('<div class="section-title" style="margin-top:1rem;">All Class Probabilities</div>', unsafe_allow_html=True)

                prob_html = ""
                for cname, pct in all_probs.items():
                    color = CLASS_COLORS[cname]
                    bold  = "font-weight:700; color:#f1f5f9;" if cname == pred_class else ""
                    prob_html += f"""
                    <div class="prob-row">
                        <div class="prob-name" style="{bold}">{cname}</div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
                        </div>
                        <div class="prob-pct">{pct:.1f}%</div>
                    </div>
                    """
                st.markdown(prob_html, unsafe_allow_html=True)

                # MRI View
                st.markdown('<div class="section-title" style="margin-top:1rem;">MRI Visual</div>', unsafe_allow_html=True)

                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                fig.patch.set_facecolor('#0a0f1e')

                axes[0].imshow(img_array)
                axes[0].set_title('Original MRI', color='#94a3b8', fontsize=11, pad=10)
                axes[0].axis('off')

                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                img_eq   = cv2.equalizeHist(img_gray)
                axes[1].imshow(img_eq, cmap='bone')
                axes[1].set_title('Enhanced View', color='#94a3b8', fontsize=11, pad=10)
                axes[1].axis('off')
                axes[1].text(0.5, 0.04, f"{pred_class} — {confidence:.1f}%",
                    transform=axes[1].transAxes, ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='#38bdf8',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0f1e', alpha=0.8, edgecolor='#38bdf8'))

                for ax in axes:
                    ax.set_facecolor('#0a0f1e')

                plt.tight_layout(pad=1.5)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0a0f1e')
                buf.seek(0)
                st.image(buf, use_container_width=True)
                plt.close()

                st.markdown("""
                <div class="disclaimer">
                    ⚠️ This tool is for educational and research purposes only. Please consult a qualified doctor for any medical decisions.
                </div>
                """, unsafe_allow_html=True)

            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")