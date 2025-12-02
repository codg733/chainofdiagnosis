import streamlit as st
from streamlit_mic_recorder import speech_to_text
from googletrans import Translator
import pandas as pd
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Chain-of-Diagnosis (CoD)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- MODERN CLINICAL UI STYLES ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Global Settings */
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .stApp {
        background-color: #f1f5f9; /* softer background */
    }

    /* Cards */
    .clinical-card {
        background: white;
        border-radius: 14px;
        padding: 1.25rem;
        box-shadow: 0 8px 20px rgba(16,24,40,0.06);
        border: 1px solid rgba(15,23,42,0.04);
        margin-bottom: 1.1rem;
    }

    /* Headers */
    h1 { color: #0f172a; letter-spacing: -0.02em; margin-bottom: 0.1rem; }
    h2, h3 { color: #111827; font-weight: 700; }
    .subtle { color: #64748b; }

    .section-label {
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 0.5rem;
        display: block;
    }

    /* Diagnosis Hero Card */
    .diagnosis-hero {
        background: linear-gradient(135deg, #ecfeff 0%, #eff6ff 100%);
        border-left: 6px solid #2563eb;
        border-radius: 10px;
        padding: 1rem;
        color: #0f172a;
        margin-bottom: 1rem;
    }
    .diagnosis-score {
        background: #0f172a;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
    }

    /* Transcript Box */
    .transcript-box {
        background-color: #f8fafc;
        border-left: 4px solid #c7d2fe;
        padding: 0.9rem;
        border-radius: 0 10px 10px 0;
        font-size: 0.95rem;
        color: #334155;
        margin-top: 0.6rem;
    }

    /* Reasoning Steps */
    .step-box {
        display: flex;
        align-items: flex-start;
        margin-bottom: 0.9rem;
    }
    .step-number {
        background: #eef2ff;
        color: #0f172a;
        min-width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 12px;
        margin-top: 2px;
    }

    /* Make text area look nicer */
    textarea[role="textbox"] {
        border-radius: 10px !important;
        box-shadow: inset 0 2px 6px rgba(2,6,23,0.03) !important;
        border: 1px solid rgba(2,6,23,0.06) !important;
        padding: 12px !important;
    }

    /* Custom Button Styling override */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 16px;
        transition: all 0.14s;
        box-shadow: 0 6px 16px rgba(16,24,40,0.06);
    }

    /* Small helpers */
    .small-muted { font-size:0.9rem; color:#6b7280; margin-top:6px; }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "stt_key" not in st.session_state:
    st.session_state.stt_key = f"stt_{int(time.time())}"
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "last_translation" not in st.session_state:
    st.session_state.last_translation = ""
if "patient_text" not in st.session_state:
    # This will store symptoms (entered by text or appended from voice).
    st.session_state.patient_text = ""

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("<div style='display:flex; align-items:center; gap:8px;'><h3 style='margin:0;'>🩺 CoD Settings</h3></div>", unsafe_allow_html=True)
    
    st.markdown("<span class='section-label'>AI Model Configuration</span>", unsafe_allow_html=True)
    model = st.selectbox("Model", ["Llama-3-8b", "Llama-3-70b", "GPT-4o-mini", "Med-PaLM 2"], index=0)
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.3, 0.05)
    max_steps = st.slider("Reasoning Depth", 3, 15, 7)
    
    st.divider()
    
    st.markdown("<span class='section-label'>Session History</span>", unsafe_allow_html=True)
    if st.session_state.chat_history:
        for i, case in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Case {len(st.session_state.chat_history) - i + 1}"):
                st.caption(case['response']['final'])
    else:
        st.info("No cases analyzed yet.")

    st.divider()
    st.caption("v1.2.0 • Research Prototype")

# ---------- MAIN CONTENT ----------
# Header (ensure this is visible and not pushed away)
st.markdown("<h1>🩺 Chain-of-Diagnosis <span style='color:#64748b; font-weight:300; font-size:1rem;'>Assistant</span></h1>", unsafe_allow_html=True)
st.markdown("<p class='subtle' style='margin-top:-8px;'>AI-powered clinical reasoning and differential diagnosis generator.</p>", unsafe_allow_html=True)
st.write("")  # spacer

# --- SECTION 1: CLINICAL INPUT ---
with st.container():
    st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)

    # Top row: large text area (left) + audio & actions (right)
    left_col, right_col = st.columns([1.8, 1])

    with left_col:
        st.markdown("<span class='section-label'>📋 Symptoms (text input only) — keep concise</span>", unsafe_allow_html=True)
        placeholder = "Enter symptoms only (example: 'Progressive exertional dyspnea x 6 weeks, intermittent chest tightness, fatigue, orthopnea, 2 pillow')"

        patient_text = st.text_area(
            "Symptoms (text or append from voice)",
            value=st.session_state.patient_text,
            height=200,
            placeholder=placeholder,
            label_visibility="collapsed",
            key="patient_text_area"
        )
        st.session_state.patient_text = patient_text

    with right_col:
        st.markdown("<span class='section-label'>🎙️ Dictate Symptoms</span>", unsafe_allow_html=True)

        # Language Selector
        lang_map = {
            "en": "🇺🇸 English", "hi": "🇮🇳 Hindi", "ta": "🇮🇳 Tamil",
            "te": "🇮🇳 Telugu", "es": "🇪🇸 Spanish", "fr": "🇫🇷 French",
            "de": "🇩🇪 German", "ar": "🇸🇦 Arabic", "zh-cn": "🇨🇳 Chinese"
        }

        lc1, lc2 = st.columns([2, 1])
        with lc1:
            stt_lang = st.selectbox("Language", list(lang_map.keys()), format_func=lambda x: lang_map[x], label_visibility="collapsed")
        with lc2:
            auto_translate = st.checkbox("Translate", value=True)

        # Recording Interface
        col_mic, col_reset = st.columns([3, 1])
        with col_mic:
            transcript = speech_to_text(
                language=stt_lang,
                start_prompt="Start Recording",
                stop_prompt="Stop Recording",
                just_once=True,
                use_container_width=True,
                key=st.session_state.stt_key
            )
        with col_reset:
            if st.button("🔄 Reset"):
                st.session_state.stt_key = f"stt_{int(time.time())}"
                st.session_state.last_transcript = ""
                st.session_state.last_translation = ""
                st.rerun()

        # Transcript Handling
        if transcript:
            st.session_state.last_transcript = transcript

        if st.session_state.last_transcript:
            display_text = st.session_state.last_transcript

            # Translation Logic
            if auto_translate and stt_lang != "en":
                try:
                    translator = Translator()
                    tr = translator.translate(st.session_state.last_transcript, src=stt_lang, dest="en")
                    st.session_state.last_translation = tr.text
                    display_text = tr.text
                except Exception:
                    st.warning("⚠️ Translation unavailable.")
            elif stt_lang == "en":
                st.session_state.last_translation = st.session_state.last_transcript

            st.markdown(f"<div class='transcript-box'><b>Transcript:</b> {display_text}</div>", unsafe_allow_html=True)

            # Append transcript to the Symptoms text area (patient_text)
            if st.button("📎 Append to Symptoms", use_container_width=True):
                if st.session_state.patient_text and not st.session_state.patient_text.endswith("\n"):
                    st.session_state.patient_text += "\n"
                st.session_state.patient_text += f"[Audio Note]: {display_text}"
                st.success("Appended transcript to symptoms.")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # End Card

    # --- Inline action row: keep Analyze button beside the text area for a compact layout ---
    with st.container():
        col_left_hint, col_button = st.columns([1.8, 0.6])
        with col_left_hint:
            st.markdown("<div class='small-muted'>When ready, click <strong>Analyze Case</strong>. Symptoms will be used as the primary input — demographics/labs may be appended via notes.</div>", unsafe_allow_html=True)
        with col_button:
            run_btn = st.button("🚀 Analyze Case", type="primary", use_container_width=True)

# --- SECTION 3: RESULTS ---
if run_btn and st.session_state.patient_text.strip():
    st.divider()
    with st.spinner("👩‍⚕️ Analyzing clinical data & synthesizing differential diagnosis..."):
        time.sleep(1.2)  # Simulated latency for UX

        # --- MOCK RESPONSE (Replace with your LLM call) ---
        response = {
            "steps": [
                "Analyzed symptom chronicity: Subacute dyspnea + fatigue suggests cardiac/metabolic causes.",
                "If BNP or exam available, use them to triage for heart failure vs pulmonary causes.",
                "Check for arrhythmia evidence—AF can precipitate decompensation.",
                "Consider thyroid dysfunction as reversible cause of AF and tachyarrhythmia.",
                "Synthesis: probable heart failure phenotype; confirm with echo and targeted labs."
            ],
            "differential": [
                {"Disease": "HFpEF w/ New AF", "Confidence": 0.68, "Key Evidence": "Dyspnea, orthopnea, irregular pulse, elevated BNP (if present)"},
                {"Disease": "Interstitial Lung Disease (Early)", "Confidence": 0.15, "Key Evidence": "CXR interstitial markings, progressive dyspnea"},
                {"Disease": "Thyroid Disease (Hyper/ Hypo effect on rhythm)", "Confidence": 0.12, "Key Evidence": "AF, fatigue, abnormal TSH if present"},
                {"Disease": "Acute Coronary Syndrome", "Confidence": 0.05, "Key Evidence": "Chest pain history, ischemic ECG/enzyme changes (if present)"}
            ],
            "final": "Heart Failure (probable HFpEF) possibly precipitated by new-onset AF",
            "confidence": 0.68,
            "next_steps": [
                "Transthoracic Echocardiogram (TTE) to assess structure and EF.",
                "ECG/cardiac enzymes to evaluate ischemia; start rate control for AF.",
                "Labs: BNP, TSH, renal function, electrolytes; start diuretics if volume overloaded.",
                "Consider anticoagulation risk assessment (CHA2DS2-VASc) for AF."
            ]
        }

        st.session_state.chat_history.append({"input": st.session_state.patient_text, "response": response})

        # --- DISPLAY RESULTS ---
        # 1. Final Diagnosis Hero
        st.markdown(f"""
        <div class='diagnosis-hero'>
            <div style='display:flex; justify-content:space-between; align-items:center; gap:16px;'>
                <div>
                    <span style='font-size:0.82rem; text-transform:uppercase; letter-spacing:1px; opacity:0.8;'>Primary Diagnosis</span>
                    <h2 style='margin:0; color:#0f172a;'>{response['final']}</h2>
                </div>
                <div class='diagnosis-score'>
                    {int(response['confidence']*100)}% Confidence
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Split Layout: Reasoning (Left) vs Differential Table (Right)
        r1, r2 = st.columns([1, 1.15])

        with r1:
            st.markdown("<div class='clinical-card' style='height:100%'>", unsafe_allow_html=True)
            st.markdown("### 🧠 Diagnostic Reasoning")
            for i, step in enumerate(response["steps"], 1):
                st.markdown(f"""
                <div class='step-box'>
                    <div class='step-number'>{i}</div>
                    <div style='color:#111827; font-size:0.96rem;'>{step}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with r2:
            st.markdown("<div class='clinical-card' style='height:100%'>", unsafe_allow_html=True)
            st.markdown("### 📊 Differential Diagnosis")

            df = pd.DataFrame(response["differential"])
            # st.column_config.* exists in newer Streamlit; fallback handled.
            try:
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Disease": st.column_config.TextColumn("Condition", width="medium"),
                        "Confidence": st.column_config.ProgressColumn(
                            "Probability",
                            format="%.0f%%",
                            min_value=0,
                            max_value=1,
                            width="small"
                        ),
                        "Key Evidence": st.column_config.TextColumn("Supporting Evidence")
                    }
                )
            except Exception:
                df_display = df.copy()
                df_display["Confidence"] = (df_display["Confidence"] * 100).round().astype(int).astype(str) + "%"
                st.dataframe(df_display, use_container_width=True, hide_index=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # 3. Next Steps (Full Width)
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("### ✅ Recommended Investigations & Management")

        ns_cols = st.columns(len(response['next_steps']))
        for idx, step in enumerate(response["next_steps"]):
            with ns_cols[idx]:
                st.info(step)

        st.markdown("</div>", unsafe_allow_html=True)

# Helpful small controls
st.markdown("<div style='position:fixed; bottom:12px; right:18px; font-size:12px; color:#64748b;'>CoD v1.2.0 • Research Prototype</div>", unsafe_allow_html=True)
