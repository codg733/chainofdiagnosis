import streamlit as st
from streamlit_mic_recorder import speech_to_text
from googletrans import Translator
import pandas as pd
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="🩺 Chain-of-Diagnosis (CoD)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- STYLES ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }
h1, h2, h3 { color:#1e293b; font-weight:600; }

.stApp {
    background: linear-gradient(135deg,#f1f5f9 0%,#e2e8f0 100%);
}

.transcript-box {
    background: linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);
    border: 2px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.10);
    min-height: 120px;
    overflow-y: auto;
    font-size: 1.02rem;
    line-height: 1.6;
    color:#334155;
    margin-bottom: 1.2rem;
}

.diagnosis-card {
    background: linear-gradient(135deg,#dbeafe 0%,#bfdbfe 100%);
    border: 2px solid #3b82f6;
    border-radius: 16px;
    padding: 1.7rem;
    box-shadow: 0 10px 40px rgba(59,130,246,0.2);
}

.section-header {
    color:#1e40af;
    font-weight:600;
    font-size:1.1rem;
    margin:0.7rem 0 0.6rem 0;
}

.stButton > button {
    background: linear-gradient(135deg,#3b82f6 0%,#1d4ed8 100%);
    color:white;
    border-radius:12px;
    border:none;
    padding:0.5rem 1.5rem;
    font-weight:500;
    font-size:0.95rem;
    box-shadow:0 4px 14px rgba(59,130,246,0.4);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow:0 7px 20px rgba(59,130,246,0.5);
}
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "stt_key" not in st.session_state:
    # key for the speech_to_text component; will change each new recording
    st.session_state.stt_key = f"stt_{int(time.time())}"
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "last_translation" not in st.session_state:
    st.session_state.last_translation = ""
if "patient_text" not in st.session_state:
    st.session_state.patient_text = ""

# ---------- HEADER ----------
st.title("🩺 Chain-of-Diagnosis (CoD)")
st.markdown("""
<div style='text-align:center;color:#64748b;margin-bottom:1.8rem;'>
  <h3 style='color:#1e293b;margin-bottom:0.3rem;'>Interpretable Step-by-Step Medical Reasoning</h3>
  <p style='margin:0;'>Audio input • Multilingual transcription • Transparent differential diagnosis</p>
  <div style='color:#94a3b8;font-size:0.85rem;margin-top:0.2rem;'>
    Research prototype – not for real clinical use.
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- LAYOUT ----------
left, right = st.columns([2, 1])

# =====================================================================
# LEFT COLUMN – PATIENT INFO + AUDIO
# =====================================================================
with left:
    st.markdown("<div class='section-header'>📋 Patient Information</div>", unsafe_allow_html=True)

    default_text = """Age: 62 years
Sex: Female
Chief complaint: Progressive dyspnea on exertion, intermittent chest pain, fatigue for 6 weeks.

Vitals: Temp 37.2°C, HR 88 bpm (irregular), RR 18, BP 135/85, SpO2 92%

Physical: JVD+, bibasilar crackles, 1+ edema, irregular rhythm
Labs: Hb 11.2, Cr 1.8, BNP 450, TSH 6.2
ECG: New AF, CXR: Cardiomegaly + interstitial markings"""

    if not st.session_state.patient_text:
        st.session_state.patient_text = default_text

    patient_text = st.text_area(
        "Patient summary",
        value=st.session_state.patient_text,
        height=170,
        key="patient_text_area"
    )

    # keep in state
    st.session_state.patient_text = patient_text

    # ---- AUDIO BLOCK ----
    st.markdown("<div class='section-header'>🎙️ Audio to Text</div>", unsafe_allow_html=True)

    lang_map = {
        "en": "🇺🇸 English",
        "hi": "🇮🇳 Hindi",
        "ta": "🇮🇳 Tamil",
        "te": "🇮🇳 Telugu",
        "es": "🇪🇸 Spanish",
        "fr": "🇫🇷 French",
        "de": "🇩🇪 German",
        "ar": "🇸🇦 Arabic",
        "zh-cn": "🇨🇳 Chinese"
    }
    col_l1, col_l2 = st.columns([3, 2])
    with col_l1:
        stt_lang = st.selectbox(
            "Speech language",
            options=list(lang_map.keys()),
            format_func=lambda x: lang_map[x],
            index=0
        )
    with col_l2:
        auto_translate = st.checkbox("Translate to English", value=True)

    # control buttons
    start_new = st.button("🎤 New Recording (fix if stuck)")
    if start_new:
        # change key so component re-mounts -> fixes “only first time works”
        st.session_state.stt_key = f"stt_{int(time.time())}"
        st.session_state.last_transcript = ""
        st.session_state.last_translation = ""
        st.experimental_rerun()

    # mic component – single run, but we remount with new key when needed
    transcript = speech_to_text(
        language=stt_lang,
        start_prompt="Start recording",
        stop_prompt="Stop",
        just_once=True,
        use_container_width=True,
        key=st.session_state.stt_key
    )

    if transcript:
        st.session_state.last_transcript = transcript

    if st.session_state.last_transcript:
        st.markdown(
            f"<div class='transcript-box'><b>Original ({lang_map[stt_lang]}):</b><br>{st.session_state.last_transcript}</div>",
            unsafe_allow_html=True
        )

        # translation
        if auto_translate and stt_lang != "en":
            try:
                translator = Translator()
                tr = translator.translate(st.session_state.last_transcript,
                                          src=stt_lang, dest="en")
                st.session_state.last_translation = tr.text
                st.markdown(
                    f"<div class='transcript-box'><b>English translation:</b><br>{tr.text}</div>",
                    unsafe_allow_html=True
                )
            except Exception:
                st.warning("Translation service error; please try again.")
        elif stt_lang == "en":
            st.session_state.last_translation = st.session_state.last_transcript

        # merge transcript into patient box
        if st.button("🔗 Add transcript to patient summary"):
            extra = st.session_state.last_translation or st.session_state.last_transcript
            st.session_state.patient_text = (
                st.session_state.patient_text + "\n\n📢 Audio notes: " + extra
            )
            st.success("Transcript merged into patient summary.")
            st.experimental_rerun()

    # clinical task
    st.markdown("<div class='section-header'>🎯 Clinical Task</div>", unsafe_allow_html=True)
    clinical_task = st.text_area(
        "Instruction for the AI",
        value="Perform step-by-step differential diagnosis with confidence scores and recommended next investigations.",
        height=80
    )

    run_btn = st.button("🚀 Run Chain-of-Diagnosis", use_container_width=True)

# =====================================================================
# RIGHT COLUMN – MODEL SETTINGS
# =====================================================================
with right:
    st.markdown("<div class='section-header'>⚙️ Model Settings</div>", unsafe_allow_html=True)

    model = st.selectbox("Model", ["llama3-8b", "llama3-70b", "gpt-4o-mini", "codellama"])
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.05)
    max_steps = st.slider("Max reasoning steps", 3, 15, 7)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#94a3b8;font-size:0.85rem;'>"
        "Audio issues usually fix by clicking <b>New Recording</b> to reset the mic component."
        "</div>",
        unsafe_allow_html=True
    )

# =====================================================================
# RESULTS
# =====================================================================
if run_btn and st.session_state.patient_text.strip():
    with st.spinner("Analyzing case with Chain-of-Diagnosis..."):
        # MOCK RESULT – replace with your LangChain / LLM call
        response = {
            "steps": [
                "Identify key symptoms and chronicity (subacute dyspnea, chest pain, fatigue).",
                "Rule out acute coronary syndrome (normal troponin, no ischemic ECG changes).",
                "Elevated BNP, CXR congestion, JVD and edema support heart failure with preserved EF.",
                "New atrial fibrillation likely precipitating decompensation (rate‑related, loss of atrial kick).",
                "Mild renal dysfunction suggests cardiorenal physiology; subclinical hypothyroidism may worsen HF."
            ],
            "differential": [
                {"disease": "HFpEF with new-onset AF", "confidence": 0.68,
                 "evidence": "BNP↑, congestion on CXR, JVD, edema, AF"},
                {"disease": "Early connective-tissue disease ILD", "confidence": 0.15,
                 "evidence": "Interstitial markings, family history of autoimmune disease"},
                {"disease": "Thyroid-related cardiomyopathy", "confidence": 0.12,
                 "evidence": "TSH mildly ↑, fatigue"},
                {"disease": "Post-viral myocarditis", "confidence": 0.05,
                 "evidence": "Recent URI, non‑specific symptoms"}
            ],
            "final": "Heart failure with preserved ejection fraction precipitated by new-onset atrial fibrillation",
            "confidence": 0.68,
            "next_steps": [
                "Transthoracic echocardiography to assess EF, diastolic function and valves.",
                "Rate control and anticoagulation strategy for atrial fibrillation.",
                "Renal function monitoring and cautious diuresis.",
                "Autoimmune panel (ANA, RF, etc.) if ILD suspicion persists.",
                "Repeat thyroid panel (TSH, free T4) and manage subclinical hypothyroidism as per guidelines."
            ]
        }

        st.session_state.chat_history.append(
            {"input": st.session_state.patient_text, "task": clinical_task, "response": response}
        )

        st.markdown("---")
        st.markdown("<div class='section-header'>🧠 Reasoning Chain</div>", unsafe_allow_html=True)
        for i, step in enumerate(response["steps"], 1):
            st.markdown(f"**Step {i}.** {step}")

        st.markdown("<div class='section-header'>📊 Differential Diagnosis</div>", unsafe_allow_html=True)
        df = pd.DataFrame(response["differential"])
        df["Confidence %"] = (df["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(df[["disease", "Confidence %", "evidence"]],
                     use_container_width=True, hide_index=True)

        st.markdown("<div class='section-header'>✅ Final Diagnosis</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='diagnosis-card'>"
            f"<h3 style='margin:0 0 0.5rem 0;'>Most likely:</h3>"
            f"<h2 style='margin:0 0 0.7rem 0;'>{response['final']}</h2>"
            f"<p style='margin:0;'>Confidence: <b>{int(response['confidence']*100)}%</b></p>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='section-header'>📋 Suggested Next Steps</div>", unsafe_allow_html=True)
        for s in response["next_steps"]:
            st.markdown(f"- {s}")

# =====================================================================
# HISTORY
# =====================================================================
with st.expander("📚 Session case history"):
    if st.session_state.chat_history:
        for i, case in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"**Case {i}:** {case['response']['final']}")
    else:
        st.write("No previous cases yet.")
