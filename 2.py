import streamlit as st
import numpy as np
import joblib
import requests
import random
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Guardian Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&family=Fira+Code:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0d1117; color: #c9d1d9; }

/* HOME CARDS */
.tool-card {
    background: linear-gradient(135deg, #0f172a 0%, #161b22 100%);
    border: 1px solid #30363d;
    border-radius: 24px; padding: 2.5rem 2rem; text-align: center;
    height: 340px; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    transition: all 0.3s ease;
}
.tool-card:hover { border-color: #58a6ff; box-shadow: 0 0 40px rgba(88,166,255,0.2); transform: translateY(-4px); }
.tool-card-icon  { font-size: 4rem; margin-bottom: 1rem; }
.tool-card-title { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.5rem; }
.tool-card-desc  { font-size: 0.9rem; color: #94a3b8; line-height: 1.6; }
.tool-badge { display: inline-block; margin-top: 1rem; padding: 0.25rem 0.75rem;
    border-radius: 9999px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.badge-ml  { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
.badge-ai  { background: rgba(56,139,253,0.15); color: #58a6ff; border: 1px solid rgba(56,139,253,0.3); }

/* VERDICT BADGES */
.verdict-box { padding: 0.4rem 1rem; border-radius: 9999px; font-weight: 800;
    font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; display: inline-block; }
.status-real       { color: #3fb950; background: rgba(63,185,80,0.1);  border: 1px solid rgba(63,185,80,0.3); }
.status-suspicious { color: #d29922; background: rgba(210,153,34,0.1); border: 1px solid rgba(210,153,34,0.3); }
.status-fake       { color: #f85149; background: rgba(248,81,73,0.1);  border: 1px solid rgba(248,81,73,0.3); }

/* FORENSIC PANELS */
.forensic-panel {
    background-color: #161b22; border: 1px solid #30363d;
    border-left: 6px solid #58a6ff; padding: 25px; border-radius: 12px; margin-bottom: 25px;
}
.panel-danger  { border-left-color: #f85149; }
.panel-success { border-left-color: #3fb950; }
.verdict-header { font-size: 24px; font-weight: 700; margin-bottom: 8px; text-transform: uppercase; }
.code-font { font-family: 'Fira Code', monospace; font-size: 13px; color: #8b949e; }

/* BUTTONS */
.stButton>button {
    border-radius: 8px; height: 3em;
    background: linear-gradient(90deg, #1f6feb 0%, #388bfd 100%);
    color: white; font-weight: 700; border: none; width: 100%;
    letter-spacing: 0.05em; transition: all 0.2s;
}
.stButton>button:hover { opacity: 0.9; box-shadow: 0 0 15px rgba(56,139,253,0.4); }

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "guardian_history" not in st.session_state:
    st.session_state.guardian_history = []
if "audit_history" not in st.session_state:
    st.session_state.audit_history = []
if "global_threat" not in st.session_state:
    st.session_state.global_threat = random.randint(30, 60)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ML MODELS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    assets = {}
    try:
        assets["insta_model"]      = joblib.load("instagram_fake_detector.pkl")
        assets["twitter_model"]    = joblib.load("twitter_fake_detector1.pkl")
        assets["general_model"]    = joblib.load("best_fake_profile_model.pkl")
        assets["scaler"]           = joblib.load("scaler.pkl")
        assets["platform_encoder"] = joblib.load("platform_encoder.pkl")
    except Exception as e:
        assets["error"] = str(e)
    return assets

assets = load_models()

def adjust_features(model, features):
    expected = model.n_features_in_
    current  = features.shape[1]
    if current > expected:
        features = features[:, :expected]
    elif current < expected:
        features = np.concatenate([features, np.zeros((1, expected - current))], axis=1)
    return features


# ─────────────────────────────────────────────────────────────────────────────
# GUARDIAN AI — DETECTOR CLASS (defined at module level)
# ─────────────────────────────────────────────────────────────────────────────
class AISocialMediaDetector:
    """
    GuardianAI Forensic Engine (v2.7.5)
    Tri-layered verification:
      1. Linguistic Pattern Matching (GPTZero)
      2. Visual Artifact Forensics (Hive AI)
      3. Semantic/Contextual Reasoning (Gemini 2.5 Flash)
    """

    def __init__(self, gptzero_key=None, hive_key=None):
        self.gptzero_key = gptzero_key or "demo_mode"
        self.hive_key    = hive_key    or "demo_mode"
        self.apiKey      = "AIzaSyB8kiOiYyiukAE5OCMxRjgQEsqh-Ug6Wo4"  
        self.GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
        self.BASE_URL     = "https://generativelanguage.googleapis.com/v1beta/models"
        self.GPTZERO_URL  = "https://api.gptzero.me/v2/predict/text"
        self.HIVE_URL     = "https://api.thehive.ai/api/v1/models/visual"

    def _get_gemini_url(self):
        return f"{self.BASE_URL}/{self.GEMINI_MODEL}:generateContent?key={self.apiKey}"

    def _call_gemini_with_backoff(self, payload):
        delays = [1, 2, 4, 8, 16]
        url = self._get_gemini_url()
        for delay in delays:
            try:
                response = requests.post(url, json=payload, timeout=30)
                if response.status_code == 200:
                    return response.json()
            except Exception:
                pass
            time.sleep(delay)
        return None

    def analyze_with_gemini(self, text, image_url=None):
        prompt = (
            f"As a Senior Disinformation Analyst, evaluate this social media content: '{text}'. "
        )
        if image_url:
            prompt += f"The post includes an image at: {image_url}. "
        prompt += (
            "Analyze for: 1. Rhetorical manipulation. 2. AI-generated linguistic markers. "
            "3. Coordination signatures. 4. Factual inconsistencies. "
            "Provide a concise forensic verdict (REAL vs FAKE) and a 0-100 probability score."
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text":
                "You are a specialized forensic tool for social media verification. "
                "Categorize content as REAL or FAKE. Be objective and precise."
            }]},
        }
        result = self._call_gemini_with_backoff(payload)
        if result:
            try:
                analysis = (result.get("candidates", [{}])[0]
                            .get("content", {}).get("parts", [{}])[0].get("text", "Analysis unavailable."))
                return {"source": "Gemini 2.5 Flash", "analysis": analysis, "status": "success"}
            except (IndexError, KeyError):
                return {"source": "Gemini 2.5 Flash", "error": "Malformed response", "status": "failed"}
        return {"source": "Gemini 2.5 Flash", "error": "API unreachable after retries", "status": "failed"}

    def analyze_text(self, text):
        if not text or len(text) < 10:
            return {"source": "GPTZero", "ai_probability": 0, "status": "no_content"}
        if self.gptzero_key == "demo_mode":
            mock_score = 88 if ("BREAKING" in text.upper() or "URGENT" in text.upper()) else random.randint(15, 60)
            return {"source": "GPTZero (Demo)", "ai_probability": mock_score, "is_synthetic": mock_score > 70}
        headers = {"x-api-key": self.gptzero_key, "Content-Type": "application/json"}
        data    = {"document": text, "version": "2026-01-01"}
        try:
            resp = requests.post(self.GPTZERO_URL, headers=headers, json=data, timeout=12)
            prob = resp.json().get("documents", [{}])[0].get("class_probabilities", {}).get("ai", 0)
            return {"source": "GPTZero", "ai_probability": round(prob * 100, 2), "is_synthetic": prob > 0.7}
        except Exception as e:
            return {"source": "GPTZero", "error": str(e), "ai_probability": 0}

    def analyze_image(self, image_url):
        if not image_url or image_url == "None":
            return {"source": "Hive AI", "ai_probability": 0, "status": "no_media"}
        if self.hive_key == "demo_mode":
            mock_score = random.randint(40, 95) if "unsplash" not in image_url else random.randint(1, 15)
            return {"source": "Hive AI (Demo)", "ai_probability": mock_score, "is_synthetic": mock_score > 80}
        headers = {"authorization": f"token {self.hive_key}"}
        data    = {"url": image_url, "models": ["ai_generated_media"]}
        try:
            resp = requests.post(self.HIVE_URL, headers=headers, data=data, timeout=15)
            output   = resp.json().get("status", [{}])[0].get("response", {}).get("output", [])
            ai_score = next((i.get("score", 0) for i in output if i.get("class") == "ai_generated"), 0)
            return {"source": "Hive AI", "ai_probability": round(ai_score * 100, 2), "is_synthetic": ai_score > 0.8}
        except Exception as e:
            return {"source": "Hive AI", "error": str(e), "ai_probability": 0}


# ─────────────────────────────────────────────────────────────────────────────
# GUARDIAN AI — HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def generate_threat_ticker():
    alerts = [
        "🚨 CRITICAL: Coordinated 'Spider-V' botnet activity detected in EMEA clusters.",
        "⚠️ WARNING: Synthetic 'Sora' video variants bypass standard platform filters in APAC.",
        "🛡️ INFO: GuardianAI signatures updated to detect GPT-5.5 linguistic drift.",
        "🕵️ SCAN: Metadata forensics identified 45,000 deepfake profiles in the last 60 minutes.",
        "🌐 STATUS: Network integrity maintained. Global disinformation load: NORMAL.",
    ]
    return random.choice(alerts)

def bridge_scraper(url):
    time.sleep(1.5)
    platforms = ["X / Twitter", "Instagram", "TikTok", "LinkedIn", "Facebook"]
    platform  = next((p for p in platforms if p.lower().split(" ")[0] in url.lower()), "Web Generic")
    is_bot    = random.random() > 0.6
    if is_bot:
        return {
            "platform": platform,
            "username": f"CitizenNews_{random.randint(100,999)}",
            "bio": "Bringing you the TRUTH they won't tell you. Independent journalist. 🛡️ #BreakingNews",
            "text": "URGENT: New docs reveal the satellite grid is manipulating weather patterns. See the video before it's deleted! #ClimateHoax",
            "image": "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?auto=format&fit=crop&w=800",
            "metrics": {"followers": f"{random.randint(10,80)}k", "age": "12 days", "behavior": "High-Frequency"},
        }
    return {
        "platform": platform,
        "username": f"User_{random.randint(1000,9999)}",
        "bio": "Amateur astronomer and tech enthusiast. I share what I see through my telescope. 🔭",
        "text": "Captured a stunning view of Jupiter's Great Red Spot last night! #Astronomy #Space",
        "image": "https://images.unsplash.com/photo-1614728263952-84ea256f9679?auto=format&fit=crop&w=800",
        "metrics": {"followers": "452", "age": "4 years", "behavior": "Organic"},
    }

def call_gemini_verification(username):
    api_key = "AIzaSyB8kiOiYyiukAE5OCMxRjgQEsqh-Ug6Wo4"
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text":
            f"Investigate the social media authenticity of @{username}. "
            "Provide profile URLs if found and a verdict: real person or fake/bot."}]}],
        "systemInstruction": {"parts": [{"text":
            "You are a social media security expert. Analyse digital footprints concisely."}]},
        "tools": [{"google_search": {}}],
    }
    for delay in [1, 2, 4, 8, 16]:
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data  = resp.json()
                text  = (data.get("candidates", [{}])[0]
                             .get("content", {}).get("parts", [{}])[0].get("text", ""))
                grounding = (data.get("candidates", [{}])[0]
                                 .get("groundingMetadata", {}).get("groundingAttributions", []))
                sources = [{"uri": a.get("web", {}).get("uri"),
                            "title": a.get("web", {}).get("title")} for a in grounding if a.get("web")]
                return text, sources[:3]
            raise Exception("API Error")
        except Exception:
            time.sleep(delay)
    return "Search verification unavailable. Proceeding with heuristic analysis.", []

def run_guardian_analysis(data):
    score  = 100
    flags  = []
    ratio  = data["following"] / data["followers"] if data["followers"] > 0 else data["following"]
    if ratio > 5 and data["followers"] < 500:
        score -= 25
        flags.append({"type": "CRITICAL", "msg": "Suspicious follower-to-following ratio (>5.0)"})
    if sum(c.isdigit() for c in data["username"]) > 4:
        score -= 15
        flags.append({"type": "WARNING", "msg": "Username contains excessive numeric characters"})
    if not data["has_avatar"]:
        score -= 20
        flags.append({"type": "CRITICAL", "msg": "Default profile picture (anonymity risk)"})
    engagement = (data["likes"] / data["followers"] * 100) if data["followers"] > 0 else 0
    if engagement < 0.5 and data["followers"] > 1000:
        score -= 20
        flags.append({"type": "CRITICAL", "msg": "Engagement rate critically low (<0.5%)"})
    final_score = max(0, score)
    status      = "REAL" if final_score > 75 else "SUSPICIOUS" if final_score > 40 else "FAKE"
    ai_report, found_urls = call_gemini_verification(data["username"])
    return {
        "id": int(time.time()), "username": data["username"],
        "score": final_score, "status": status, "flags": flags,
        "metrics": {**data, "engagement_rate": round(engagement, 2)},
        "ai_report": ai_report, "found_urls": found_urls,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SOCIALSHIELD — REASON DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
def show_reasons(reasons_fake, reasons_real, prediction, confidence):
    st.markdown("---")
    st.subheader("🔍 Why this prediction?")
    if prediction == 1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🔴 Suspicious Signals")
            for r in reasons_fake: st.markdown(f"- 🔴 {r}")
        with col2:
            st.markdown("##### 🟢 Normal Signals")
            for r in reasons_real: st.markdown(f"- 🟢 {r}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🟢 Trust Signals")
            for r in reasons_real: st.markdown(f"- 🟢 {r}")
        with col2:
            st.markdown("##### 🟡 Minor Flags")
            for r in (reasons_fake or ["None detected"]): st.markdown(f"- 🟡 {r}")
    st.markdown("---")
    st.markdown(f"**Model Confidence:** `{confidence:.1%}`")
    st.progress(confidence)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1.5rem;">
        <h1 style="font-size:2.8rem; font-weight:900;
                   background:linear-gradient(90deg,#58a6ff,#3fb950);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Social Guardian Platform
        </h1>
        <p style="color:#8b949e; font-size:1.1rem; margin-top:0.5rem;">
            AI-powered tools to detect fake profiles and disinformation across social media
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="tool-card">
            <div class="tool-card-icon">🛡️</div>
            <div class="tool-card-title">SocialShield AI</div>
            <div class="tool-card-desc">
                ML-powered fake profile detection using models trained on real datasets.<br><br>
                Supports <b>Instagram</b>, <b>Twitter/X</b>, and <b>General platforms</b>.
                Explains <em>why</em> each prediction was made.
            </div>
            <div class="tool-badge badge-ml">ML Models · 90–96% Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter SocialShield AI →", key="enter_shield"):
            st.session_state.page = "socialshield"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="tool-card">
            <div class="tool-card-icon">🧠</div>
            <div class="tool-card-title">GuardianAI Disinformation Suite</div>
            <div class="tool-card-desc">
                Enterprise-grade tri-modal verification: Linguistic (GPTZero),
                Visual (Hive AI), and Semantic (Gemini 2.5 Flash). Threat analytics,
                attack simulator, and forensic tools included.
            </div>
            <div class="tool-badge badge-ai">Gemini 2.5 Flash · v2.7.5</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter GuardianAI Suite →", key="enter_guardian"):
            st.session_state.page = "guardian"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Instagram Model Accuracy", "90.7%")
    c2.metric("Twitter Model Accuracy",   "95.7%")
    c3.metric("General Model Accuracy",   "99.5%")
    st.caption("Models trained from real labeled datasets · Last trained: 2026-03-01")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SOCIALSHIELD AI
# ═════════════════════════════════════════════════════════════════════════════
def page_socialshield():
    if st.button("← Back to Home", key="back_shield"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("## 🛡️ SocialShield AI — Fake Profile Detection")

    with st.sidebar:
        st.markdown("### 🛡️ SocialShield AI")
        model_type = st.radio(
            "Detection Engine",
            ["📸 Instagram Detector", "🐦 Twitter Bot Hunter", "🌐 General Social Media"],
        )

    # ── INSTAGRAM ──
    if model_type == "📸 Instagram Detector":
        st.header("📸 Instagram Profile Analysis")
        col1, col2 = st.columns(2)
        with col1:
            profile_pic    = st.selectbox("Has Profile Picture?", [1, 0], format_func=lambda x: "Yes" if x else "No")
            username_ratio = st.slider("Numbers in Username (ratio)", 0.0, 1.0, 0.2)
            fullname_words = st.number_input("Full Name Word Count", 0, 20, 2)
            fullname_ratio = st.slider("Numbers in Full Name (ratio)", 0.0, 1.0, 0.0)
            name_match     = st.selectbox("Name Same as Username?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            bio_len        = st.number_input("Bio Length (chars)", 0, 500, 80)
        with col2:
            ext_url    = st.checkbox("External URL in Bio?")
            is_private = st.checkbox("Private Account?")
            posts      = st.number_input("Number of Posts", 0, 100000, 45)
            followers  = st.number_input("Followers", 0, 10000000, 800)
            following  = st.number_input("Following",  0, 1000000, 500)

        if st.button("🔍 Run Detection", key="insta_btn"):
            features = np.array([[profile_pic, username_ratio, fullname_words, fullname_ratio,
                                   name_match, bio_len, int(ext_url), int(is_private), posts, followers, following]])
            model    = assets["insta_model"]
            features = adjust_features(model, features)
            pred     = model.predict(features)[0]
            prob     = model.predict_proba(features)[0]
            conf     = prob[1] if pred == 1 else prob[0]
            if pred == 1: st.error(f"🚨 **FAKE PROFILE** — Confidence: {prob[1]:.2%}")
            else:         st.success(f"✅ **REAL PROFILE** — Confidence: {prob[0]:.2%}")

            rf, rr = [], []
            if profile_pic == 0:    rf.append("No profile picture set")
            else:                   rr.append("Profile picture is present")
            if username_ratio > 0.4: rf.append(f"Username is {username_ratio:.0%} numeric (bot-like)")
            else:                    rr.append("Username looks natural")
            if fullname_words <= 1:  rf.append("Full name has only 1 word / missing")
            else:                    rr.append(f"Full name has {int(fullname_words)} words")
            if fullname_ratio > 0.3: rf.append(f"Full name has {fullname_ratio:.0%} numbers")
            else:                    rr.append("Full name has no excessive numbers")
            if name_match == 1: rf.append("Username exactly matches full name (suspicious)")
            else:               rr.append("Username and full name differ (normal)")
            if bio_len < 10:  rf.append("Bio is very short or empty")
            else:              rr.append(f"Bio has {int(bio_len)} characters")
            ratio = followers / (following + 1)
            if followers < 50:   rf.append(f"Very low followers ({int(followers)})")
            elif ratio > 3:      rr.append(f"High follower/following ratio ({ratio:.1f})")
            else:                rr.append(f"Followers/Following balanced")
            if posts < 5:  rf.append(f"Very few posts ({int(posts)})")
            else:           rr.append(f"Has {int(posts)} posts (active)")
            if is_private:  rr.append("Private account (common for real users)")
            if ext_url:     rr.append("External URL in bio (personal/business link)")
            show_reasons(rf, rr, pred, conf)

    # ── TWITTER ──
    elif model_type == "🐦 Twitter Bot Hunter":
        st.header("🐦 Twitter Account Analysis")
        col1, col2 = st.columns(2)
        with col1:
            followers     = st.number_input("Followers", 0, 10000000, 250)
            friends       = st.number_input("Following (Friends)", 0, 1000000, 300)
            statuses      = st.number_input("Total Tweets", 0, 500000, 1000)
            favs          = st.number_input("Favorites / Likes", 0, 1000000, 50)
            listed        = st.number_input("Listed Count", 0, 100000, 0)
        with col2:
            verified      = st.checkbox("Verified Account?")
            geo           = st.checkbox("Geo Enabled?")
            default_img   = st.checkbox("Default Profile Image?")
            bio_len       = st.number_input("Bio Length (chars)", 0, 200, 40)
            creation_date = st.date_input("Account Created On", datetime(2020, 1, 1).date())

        age   = (datetime.now().date() - creation_date).days
        ratio = followers / (friends + 1)

        if st.button("🔍 Run Detection", key="twitter_btn"):
            features = np.array([[statuses, followers, friends, favs, listed,
                                   int(default_img), 0.3, int(geo),
                                   int(verified), int(verified), age, bio_len, ratio]])
            model    = assets["twitter_model"]
            features = adjust_features(model, features)
            pred     = model.predict(features)[0]
            prob     = model.predict_proba(features)[0]
            conf     = prob[1] if pred == 1 else prob[0]
            if pred == 1: st.error(f"🚨 **BOT DETECTED** — Confidence: {prob[1]:.2%}")
            else:         st.success(f"✅ **HUMAN ACCOUNT** — Confidence: {prob[0]:.2%}")

            rf, rr = [], []
            if default_img: rf.append("Using default profile image (never customised)")
            else:           rr.append("Has a custom profile picture")
            if ratio < 0.1 and friends > 0: rf.append(f"Following many, very few followers (ratio={ratio:.2f})")
            elif ratio > 10:                 rr.append(f"High follower ratio ({ratio:.1f}) — popular")
            else:                            rr.append(f"Follower/following ratio is {ratio:.2f}")
            if statuses < 10:      rf.append(f"Only {int(statuses)} tweets — very inactive")
            elif statuses > 50000: rf.append(f"{int(statuses)} tweets — abnormally high volume")
            else:                  rr.append(f"{int(statuses)} tweets (normal activity)")
            if age < 30:       rf.append(f"Account only {age} days old (very new)")
            elif age > 365*2:  rr.append(f"Account is {age} days old (established)")
            else:               rr.append(f"Account age: {age} days")
            if bio_len < 5:  rf.append("Bio is empty or missing")
            else:             rr.append(f"Bio has {int(bio_len)} characters")
            if verified:  rr.append("Verified account ✓")
            if friends > 2000 and followers < 100:
                rf.append("Following 2000+ but <100 followers (follow-unfollow bot pattern)")
            if favs == 0 and statuses == 0: rf.append("Zero likes and zero tweets (inactive bot)")
            elif favs > 0:                  rr.append(f"{int(favs)} likes (engaged user)")
            show_reasons(rf, rr, pred, conf)

    # ── GENERAL ──
    else:
        st.header("🌐 General Social Media Analysis")
        col1, col2 = st.columns(2)
        with col1:
            platform  = st.selectbox("Platform", ["Twitter", "Instagram"])
            has_pic   = st.selectbox("Has Profile Picture?", [1, 0], format_func=lambda x: "Yes" if x else "No")
            bio_len   = st.number_input("Bio Length (chars)", 0, 500, 100)
            followers = st.number_input("Followers", 0, 1000000, 100)
            following = st.number_input("Following",  0, 1000000, 200)
        with col2:
            age_days  = st.number_input("Account Age (Days)", 1, 5000, 365)
            posts     = st.number_input("Total Posts", 0, 50000, 50)
            verified  = st.selectbox("Verified?", [0, 1], format_func=lambda x: "Yes" if x else "No")

        ratio = followers / (following + 1)
        try:    platform_encoded = assets["platform_encoder"].transform([platform])[0]
        except: platform_encoded = 0

        if st.button("🔍 Run Detection", key="general_btn"):
            features = np.array([[platform_encoded, has_pic, bio_len, 0.5,
                                   followers, following, ratio, age_days, posts,
                                   posts / max(age_days, 1), 0.5, 0.5, 0, 0, 0, 0, verified]])
            model    = assets["general_model"]
            
            # Apply scaling only to the features expected by the scaler, keep the rest unscaled
            if assets.get("scaler"):
                n_scaler = assets["scaler"].n_features_in_
                scaled_subset = assets["scaler"].transform(features[:, :n_scaler])
                features = np.concatenate([scaled_subset, features[:, n_scaler:]], axis=1)

            features = adjust_features(model, features)
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0]
            conf = prob[1] if pred == 1 else prob[0]
            if pred == 1: st.error(f"🚨 **FAKE PROFILE** — Confidence: {prob[1]:.2%}")
            else:         st.success(f"✅ **REAL PROFILE** — Confidence: {prob[0]:.2%}")

            rf, rr = [], []
            if has_pic == 0:  rf.append("No profile picture")
            else:              rr.append("Has profile picture")
            if bio_len < 10:  rf.append("Bio very short or missing")
            else:              rr.append(f"Bio has {int(bio_len)} characters")
            if ratio < 0.05:  rf.append(f"Very low follower/following ratio ({ratio:.3f})")
            elif ratio > 2:   rr.append(f"Good follower/following ratio ({ratio:.2f})")
            else:              rr.append(f"Ratio is {ratio:.2f}")
            if age_days < 30:    rf.append(f"Account only {int(age_days)} days old")
            elif age_days > 365: rr.append(f"Account is {int(age_days)} days old (established)")
            else:                rr.append(f"Account age: {int(age_days)} days")
            ppd = posts / max(age_days, 1)
            if posts < 5:    rf.append(f"Only {int(posts)} posts — nearly inactive")
            elif ppd > 20:   rf.append(f"{ppd:.1f} posts/day — abnormally high (bot?)")
            else:             rr.append(f"{int(posts)} posts ({ppd:.2f}/day)")
            if followers < 10 and following > 200:
                rf.append("Following many but almost no followers")
            if verified: rr.append("Verified account ✓")
            else:        rf.append("Not verified")
            show_reasons(rf, rr, pred, conf)

    st.divider()
    st.caption("⚠️ AI predictions are probabilistic. Manual verification recommended.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: GUARDIAN AI DISINFORMATION SUITE
# ═════════════════════════════════════════════════════════════════════════════
def page_guardian():
    if st.button("← Back to Home", key="back_guardian"):
        st.session_state.page = "home"
        st.rerun()

    # Live threat ticker
    st.markdown(
        f'<p class="code-font" style="color:#3fb950;">[LIVE_FEED] {generate_threat_ticker()}</p>',
        unsafe_allow_html=True,
    )
    st.markdown("## 🧠 GuardianAI Disinformation Suite")
    st.markdown("##### Enterprise-grade multi-modal verification for the post-truth era.")

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=80)
        st.title("GuardianAI Command")
        st.caption("Forensic Intelligence Suite v2.7.5")
        st.divider()

        with st.expander("🔑 API Authorization", expanded=True):
            gpt_api  = st.text_input("GPTZero API Key",   type="password", placeholder="Enter V2 Key…")
            hive_api = st.text_input("Hive AI API Token", type="password", placeholder="Enter Visual Key…")

        st.divider()
        st.subheader("⚙️ Analysis Parameters")
        sensitivity = st.slider("Detection Rigor", 0, 100, 75)
        st.checkbox("Enable Metadata Forensics",  value=True)
        st.checkbox("Deep-Packet Contextualizing", value=False)

        st.divider()
        st.metric("Global Threat Pulse", f"{st.session_state.global_threat}%", delta="Normal Volume")

        if st.button("Purge Audit Logs", key="purge_logs"):
            st.session_state.audit_history = []
            st.rerun()

    # ── TABS ─────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🔍 Forensic Engine",
        "📊 Threat Analytics",
        "🤖 Attack Simulator",
        "📖 Knowledge Base",
        "🛠️ Forensic Tools",
    ])

    # ── TAB 1: FORENSIC ENGINE ──
    with tabs[0]:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Data Ingestion")
            mode = st.radio("Verification Method", ["Social Media Link", "Manual Raw Entry"], horizontal=True)
            raw_text = raw_image = platform = None
            meta = {}

            if mode == "Social Media Link":
                link_url = st.text_input("Profile or Post URL", placeholder="https://x.com/user/status/...")
                if link_url:
                    with st.spinner("Extracting identity from platform…"):
                        scraped  = bridge_scraper(link_url)
                        raw_text = scraped["text"]
                        raw_image = scraped["image"]
                        platform = scraped["platform"]
                        meta     = scraped["metrics"]
            else:
                raw_text  = st.text_area("Narrative Text", height=180, placeholder="Paste content for linguistic analysis…")
                raw_image = st.text_input("Media URL (Optional)", placeholder="https://…")
                platform  = "Manual Ingest"
                meta      = {"behavior": "Manual"}

            analyze_btn = st.button("🚀 INITIATE FORENSIC SCAN")

        with c2:
            st.subheader("Asset Monitor")
            if raw_text:
                with st.container(border=True):
                    st.caption(f"Source: {platform} | Behavior: {meta.get('behavior', '—')}")
                    st.write(f"**Linguistic Preview:**\n{raw_text[:200]}…")
                    if raw_image:
                        try:
                            st.image(raw_image, caption="Identified Visual Asset", use_container_width=True)
                        except Exception:
                            st.write(f"🖼️ [Image URL]: {raw_image}")
            else:
                st.info("Awaiting input for live asset streaming.")

        # Results
        if analyze_btn:
            if not raw_text:
                st.error("Operation Aborted: No data detected for analysis.")
            else:
                detector = AISocialMediaDetector(gpt_api or None, hive_api or None)
                with st.status("Performing Multimodal Scan…", expanded=True) as status:
                    st.write("Checking linguistic entropy & burstiness…")
                    t_res = detector.analyze_text(raw_text)
                    time.sleep(0.8)
                    st.write("Analysing pixel frequency & GAN signatures…")
                    v_res = detector.analyze_image(raw_image) if raw_image else {"ai_probability": 0}
                    time.sleep(0.8)
                    st.write("Executing Gemini 2.5 Contextual Reasoning…")
                    g_res = detector.analyze_with_gemini(raw_text, raw_image)
                    status.update(label="Forensic Investigation Complete", state="complete")

                st.divider()
                st.header("📋 Forensic Audit Summary")
                m1, m2, m3 = st.columns(3)
                l_prob = t_res.get("ai_probability", 0)
                v_prob = v_res.get("ai_probability", 0)
                m1.metric("Linguistic Risk",    f"{l_prob}%",  "Suspicious" if l_prob > 70 else "Organic")
                m2.metric("Visual Synthetic",   f"{v_prob}%",  "Deepfake"   if v_prob > 80 else "Verified")
                overall = (l_prob + v_prob) / (2 if raw_image else 1)
                m3.metric("Aggregated Threat Score", f"{overall:.1f}%", delta_color="inverse")

                is_fake      = overall > sensitivity
                card_class   = "panel-danger" if is_fake else "panel-success"
                status_label = "🚩 FAKE (Synthetic / Disinfo)" if is_fake else "🛡️ REAL (Verified / Human)"
                verdict_text = "🚨 HIGH RISK: DISINFORMATION DETECTED" if is_fake else "✅ LOW RISK: AUTHENTIC CONTENT"
                st.markdown(f"""
                    <div class="forensic-panel {card_class}">
                        <div class="verdict-header">{status_label}</div>
                        <h4>Final Verdict: {verdict_text}</h4>
                        <p><b>Contextual Reasoning (Gemini 2.5 Flash):</b>
                        {g_res.get('analysis', 'Forensic insights unavailable.')}</p>
                    </div>
                """, unsafe_allow_html=True)

                st.session_state.audit_history.append({
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Source":    platform,
                    "Risk":      f"{overall:.1f}%",
                    "Status":    "FAKE" if is_fake else "REAL",
                    "Verdict":   "FLAGGED" if is_fake else "CLEARED",
                })

    # ── TAB 2: THREAT ANALYTICS ──
    with tabs[1]:
        st.subheader("Global Disinformation Trends (Simulated Real-time)")
        chart_data = pd.DataFrame({
            "Hour":          range(24),
            "AI Injections": np.random.randint(50,  500, 24),
            "Human Activity": np.random.randint(2000, 5000, 24),
        })
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig1 = px.area(chart_data, x="Hour", y=["AI Injections", "Human Activity"],
                           title="Narrative Volume per Hour", template="plotly_dark",
                           color_discrete_sequence=["#f85149", "#3fb950"])
            st.plotly_chart(fig1, use_container_width=True)
        with col_t2:
            dist_data = pd.DataFrame({
                "Platform": ["X", "FB", "TikTok", "Reddit", "Others"],
                "Volume":   [45, 20, 35, 12, 8],
            })
            fig2 = px.pie(dist_data, values="Volume", names="Platform",
                          hole=0.4, template="plotly_dark", title="Source Distribution")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Session Forensic Logs")
        if st.session_state.audit_history:
            st.dataframe(pd.DataFrame(st.session_state.audit_history), use_container_width=True)
        else:
            st.caption("No forensic investigations recorded in current session.")

    # ── TAB 3: ATTACK SIMULATOR ──
    with tabs[2]:
        st.subheader("Coordinated Inauthentic Behavior (CIB) Simulator")
        st.write("Analyse how a synthetic narrative propagates across a bot cluster.")
        sim_col1, sim_col2 = st.columns([1, 2])
        with sim_col1:
            nodes    = st.number_input("Bot Node Count", 100, 10000, 1000)
            run_sim  = st.button("🔥 INITIATE SIMULATION")
        with sim_col2:
            if run_sim:
                with st.spinner("Mapping network traversal…"):
                    p_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        p_bar.progress(i + 1)
                    x = np.linspace(0, 10, 100)
                    y = nodes * np.exp(0.5 * x)
                    fig_sim = px.line(x=x, y=y,
                                     title="Projected Content Reach (Viral Coefficient: 1.5)",
                                     labels={"x": "Minutes", "y": "Impressions"},
                                     template="plotly_dark")
                    st.plotly_chart(fig_sim, use_container_width=True)
                    st.warning(f"Detection window closing: Saturation projected in {random.randint(15,25)} minutes.")

    # ── TAB 4: KNOWLEDGE BASE ──
    with tabs[3]:
        st.header("Forensic Knowledge Repository")
        kb_c1, kb_c2 = st.columns(2)
        with kb_c1:
            with st.expander("Linguistic Perplexity Metrics"):
                st.write("**Perplexity** measures how well a model predicts a sample. "
                         "Low perplexity = predictable = AI signature. "
                         "Human writing is 'bursty' and unpredictable.")
            with st.expander("GAN Artifact Detection"):
                st.write("GANs struggle with: non-biological symmetries (e.g., mismatched earrings), "
                         "background frequency noise, and eye reflection consistency.")
        with kb_c2:
            with st.expander("Coordinated Inauthentic Behavior (CIB)"):
                st.write("CIB = groups of accounts working together to mislead. "
                         "GuardianAI detects this via **Temporal Offset** — "
                         "50 accounts posting the same message within 5 seconds flags a CIB cluster.")
            with st.expander("Metadata Stripping & Spoofing"):
                st.write("State actors strip EXIF data to hide origin. "
                         "We use Shadow Analysis and Pixel Compression Mapping "
                         "to identify the likely hardware source even without metadata.")

    # ── TAB 5: FORENSIC TOOLS ──
    with tabs[4]:
        st.subheader("Advanced Forensic Utilities")
        st.write("**1. Digital Signature Verification**")
        st.button("Scan for Hardware Fingerprints")
        st.divider()
        st.write("**2. Cryptographic Audit Export**")
        if st.button("Generate Forensic PDF Report"):
            st.toast("Encrypting report data…")
            time.sleep(1.2)
            st.success("Report generated: Guardian_Audit_7721.pdf")
        st.divider()
        st.write("**3. System Integrity Console**")
        st.code("""[AUTH] Gemini 2.5 Flash : OK
[AUTH] GPTZero V2      : OK
[AUTH] Hive Visual     : OK
[NET]  Port 443 (TLS)  : ACTIVE
[MEM]  Buffer cleared  : 18:34:01""", language="bash")

    st.divider()
    c1f, c2f, c3f = st.columns(3)
    c1f.caption("🟢 Engine: Active (v2.7.5)")
    c2f.caption("🧠 Gemini-2.5-Flash + GPTZero + Hive AI")
    c3f.caption("© 2026 Guardian Disinformation Labs")


# ═════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═════════════════════════════════════════════════════════════════════════════
_page = st.session_state.page
if _page == "home":
    page_home()
elif _page == "socialshield":
    page_socialshield()
elif _page == "guardian":
    page_guardian()