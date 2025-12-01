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
        background-color: #f8fafc; /* Slate-50 */
    }

    /* Cards */
    .clinical-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    /* Headers */
    h1 { color: #0f172a; letter-spacing: -0.025em; }
    h2, h3 { color: #334155; font-weight: 600; }
    
    .section-label {
        text-transform: uppercase;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.5rem;
        display: block;
    }

    /* Diagnosis Hero Card */
    .diagnosis-hero {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); /* Blue-50 to Blue-100 */
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 1.5rem;
        color: #1e3a8a;
        margin-bottom: 1.5rem;
    }
    .diagnosis-score {
        background: #2563eb;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Transcript Box */
    .transcript-box {
        background-color: #f1f5f9;
        border-left: 4px solid #cbd5e1;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
        color: #475569;
        margin-top: 0.5rem;
    }

    /* Reasoning Steps */
    .step-box {
        display: flex;
        align-items: flex-start;
        margin-bottom: 0.8rem;
    }
    .step-number {
        background: #e2e8f0;
        color: #475569;
        min-width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 12px;
        margin-top: 2px;
    }

    /* Custom Button Styling override */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
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
    st.session_state.patient_text = ""

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("🩺 CoD Settings")
    
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

# Header
st.markdown("<h1>🩺 Chain-of-Diagnosis <span style='color:#94a3b8; font-weight:300; font-size:1.5rem;'>Assistant</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748b; margin-top:-15px;'>AI-powered clinical reasoning and differential diagnosis generator.</p>", unsafe_allow_html=True)

# Layout: 2 Columns for Input (Left) and Results (Right) could be too crowded. 
# Better: Top Section for Input, Bottom Section for Results.

# --- SECTION 1: CLINICAL INPUT ---
with st.container():
    st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1.5, 1])
    
    # Left: Text Input
    with c1:
        st.markdown("<span class='section-label'>📋 Patient Information (Vitals, Labs, Symptoms)</span>", unsafe_allow_html=True)
        
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
            "Patient Summary", 
            value=st.session_state.patient_text,
            height=200,
            label_visibility="collapsed",
            key="patient_text_area"
        )
        st.session_state.patient_text = patient_text

    # Right: Audio Input
    with c2:
        st.markdown("<span class='section-label'>🎙️ Dictate Clinical Notes</span>", unsafe_allow_html=True)
        
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
            
            if st.button("📎 Append to Patient Info", use_container_width=True):
                st.session_state.patient_text += f"\n\n[Audio Note]: {display_text}"
                st.success("Added to notes.")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True) # End Card

# --- SECTION 2: TASK & ACTION ---
st.markdown("<span class='section-label'>Instruction for AI</span>", unsafe_allow_html=True)
col_task, col_btn = st.columns([3, 1])

with col_task:
    clinical_task = st.text_input(
        "Task", 
        value="Perform step-by-step differential diagnosis with confidence scores and recommended next investigations.",
        label_visibility="collapsed"
    )

with col_btn:
    run_btn = st.button("🚀 Analyze Case", type="primary", use_container_width=True)

# --- SECTION 3: RESULTS ---
if run_btn and st.session_state.patient_text.strip():
    st.divider()
    with st.spinner("👩‍⚕️ Analyzing clinical data & synthesizing differential diagnosis..."):
        time.sleep(1.5) # Simulated latency for effect
        
        # --- MOCK RESPONSE (Replace with LLM Call) ---
        response = {
            "steps": [
                "Analyzed symptom chronicity: Subacute dyspnea (6wks) + fatigue points to cardiac/metabolic causes.",
                "Evaluated labs: High BNP (450) and JVD strongly suggest volume overload (Heart Failure).",
                "ECG correlation: New Atrial Fibrillation explains irregular HR and possible HF decompensation.",
                "Ruled out ACS: No acute chest pain description or mention of ischemic ECG changes (though enzyme check needed).",
                "Synthesis: Clinical picture matches HFpEF exacerbated by new-onset AF."
            ],
            "differential": [
                {"Disease": "HFpEF w/ New AF", "Confidence": 0.68, "Key Evidence": "BNP↑, JVD, Crackles, AF"},
                {"Disease": "ILD (Early)", "Confidence": 0.15, "Key Evidence": "Interstitial markings on CXR"},
                {"Disease": "Thyrotoxicosis", "Confidence": 0.12, "Key Evidence": "TSH abnormal, AF, Fatigue"},
                {"Disease": "Viral Myocarditis", "Confidence": 0.05, "Key Evidence": "Subacute onset"}
            ],
            "final": "Heart Failure (HFpEF) precipitated by Atrial Fibrillation",
            "confidence": 0.68,
            "next_steps": [
                "Transthoracic Echocardiogram (TTE) to assess ejection fraction.",
                "Initiate rate control (Beta-blockers/CCB) and anticoagulation.",
                "Diuretics for volume overload (Furosemide).",
                "Repeat Thyroid panel."
            ]
        }
        
        st.session_state.chat_history.append({"input": st.session_state.patient_text, "response": response})

        # --- DISPLAY RESULTS ---
        
        # 1. Final Diagnosis Hero
        st.markdown(f"""
        <div class='diagnosis-hero'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <span style='font-size:0.9rem; text-transform:uppercase; letter-spacing:1px; opacity:0.8;'>Primary Diagnosis</span>
                    <h2 style='margin:0; color:#1e3a8a;'>{response['final']}</h2>
                </div>
                <div class='diagnosis-score'>
                    {int(response['confidence']*100)}% Confidence
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        




        # 2. Split Layout: Reasoning (Left) vs Differential Table (Right)
        r1, r2 = st.columns([1, 1.2])
        
        with r1:
            st.markdown("<div class='clinical-card' style='height:100%'>", unsafe_allow_html=True)
            st.markdown("### 🧠 Diagnostic Reasoning")
            for i, step in enumerate(response["steps"], 1):
                st.markdown(f"""
                <div class='step-box'>
                    <div class='step-number'>{i}</div>
                    <div style='color:#334155; font-size:0.95rem;'>{step}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with r2:
            st.markdown("<div class='clinical-card' style='height:100%'>", unsafe_allow_html=True)
            st.markdown("### 📊 Differential Diagnosis")
            
            df = pd.DataFrame(response["differential"])
            
            # Using Streamlit's new column config for Progress Bars
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
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. Next Steps (Full Width)
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("### ✅ Recommended Investigations & Management")
        
        ns_cols = st.columns(len(response['next_steps']))
        for idx, step in enumerate(response["next_steps"]):
            with ns_cols[idx]:
                st.info(step)
        
        st.markdown("</div>", unsafe_allow_html=True)
