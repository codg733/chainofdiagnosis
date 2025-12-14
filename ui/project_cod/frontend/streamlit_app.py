# frontend/streamlit_app.py
# ============================================================
# MedPath AI — Clinical Dashboard UI (Refactored)
# ============================================================

import os
import time
import re
import sys
import json
import streamlit as st
import pandas as pd

# ------------------ Setup Paths & Imports ------------------
# Add project root to sys.path to find config and backend
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
import importlib
importlib.reload(config)

from backend import database
importlib.reload(database)

from backend.model_loader import hybrid_retrieve, ensure_disease_map
from backend import diagnosis_engine
importlib.reload(diagnosis_engine)

from backend.diagnosis_engine import (
    score_candidates,
    candidate_symptom_pool,
    choose_best_followup,
    build_final_report,
    is_incompatible,
)

# ------------------ Voice Recording & STT ------------------
try:
    from streamlit_mic_recorder import speech_to_text
    _HAS_VOICE_RECORDER = True
except ImportError:
    _HAS_VOICE_RECORDER = False

# ------------------ Input & Translation ------------------
try:
    from deep_translator import GoogleTranslator
    _HAS_TRANSLATOR = True
except ImportError:
    _HAS_TRANSLATOR = False

def translate_text(text, target_lang="en"):
    """
    Translates text to target_lang using deep_translator.
    Retries once on failure.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return text
        
    if not _HAS_TRANSLATOR:
        return text

    # Quick check: if text is pure ASCII and target is 'en', skip
    if target_lang == "en" and all(ord(c) < 128 for c in text):
        return text

    try:
        # Simplify lang code
        lang_code = target_lang.split("-")[0]
        return GoogleTranslator(source='auto', target=lang_code).translate(text)
    except Exception:
        # Retry once
        try:
             time.sleep(0.5)
             lang_code = target_lang.split("-")[0]
             return GoogleTranslator(source='auto', target=lang_code).translate(text)
        except Exception:
             return text


# Ensure KB loaded
ensure_disease_map()

# ------------------ Page Config ------------------
st.set_page_config(page_title="MedPath AI — Clinical Dashboard", layout="wide")

# ------------------ CSS Styling ------------------
st.markdown(
    r"""
<style>
/* --- card palette (dark) --- */
:root {
  --card-bg: #0f172a;        /* dark navy */
  --card-bg-2: #111827;      /* slightly different dark */
  --muted: #94a3b8;
  --accent: #2563eb;
  --panel-border: rgba(255,255,255,0.04);
  --text: #e6eef8;           /* light text for readability on dark */
  --subtle: #c7d2fe;
}

/* General card */
.card, .followup-card, .patient-card, .report-card {
    background: var(--card-bg);
    color: var(--text) !important;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid var(--panel-border);
    box-shadow: 0 8px 20px rgba(2,6,23,0.35);
}

/* Header / brand */
.header { display:flex; align-items:center; gap:14px; color:var(--text); }
.brand { width:48px; height:48px; background:var(--accent); color:white; font-weight:700; display:flex; align-items:center; justify-content:center; border-radius:8px; }

/* Small text */
.small-muted { color: var(--muted); font-size:0.95rem; }
.section-title { font-weight:700; margin-bottom:8px; color:var(--text); }

/* Buttons (override) */
.stButton>button {
    background: linear-gradient(180deg, #1f2937, #0b1220) !important;
    color: var(--text) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    padding: 8px 12px !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 16px rgba(2,6,23,0.35) !important;
    font-weight:700;
}
.stButton>button:hover {
    transform: translateY(-1px);
}

/* Input widgets in dark cards */
.stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div>div {
    background: #0b1220 !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
}

/* Followup card gradient */
.followup-card { background: linear-gradient(180deg,#0b1220,#0f172a); border:1px solid rgba(255,255,255,0.03); }

/* Diagnosis score pill */
.diagnosis-score { background: #064e3b; color: #e6fdf6; padding:6px 10px; border-radius:999px; font-weight:700; }

/* Make sure markdown inside cards uses card text color */
.card * { color: var(--text) !important; }

/* Narrow screen */
@media (max-width:900px){
  .two-col { flex-direction:column; gap:12px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------ Session State Handling ------------------
DEFAULTS = {
    "name": "",
    "age": "",
    "gender": "M",
    "initial_symptoms": "",
    "symptoms": [],
    "asked": set(),
    "finished": False,
    "rounds": 0,
    "last_voice_transcript": "",
    "session_language": "en", # User interface language
    "show_history": False,
    "stt_key": f"stt_{int(time.time())}",
    "stt_key_q": f"stt_q_{int(time.time())}",
    "last_transcript": "",
    "last_translation": "",
    "current_symptoms_text": "", # For English logic
    "negatives": set(), # Track symptoms user said NO to
    "consultation_started": False
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_session():
    st.session_state["name"] = ""
    st.session_state["age"] = ""
    st.session_state["gender"] = "M"
    st.session_state["initial_symptoms"] = ""
    st.session_state["current_symptoms_text"] = ""
    st.session_state["symptoms"] = []
    st.session_state["asked"] = set()
    st.session_state["finished"] = False
    st.session_state["rounds"] = 0
    st.session_state["negatives"] = set()
    st.session_state["last_voice_transcript"] = ""
    st.session_state["stt_key"] = f"stt_{int(time.time())}"
    st.session_state["stt_key_q"] = f"stt_q_{int(time.time())}"
    st.session_state["last_transcript"] = ""
    st.session_state["last_translation"] = ""
    st.session_state["consultation_started"] = False
    # Keep language and history toggle


    
# ------------------ Heuristic Parser ------------------
def parse_personal_info_from_text(text: str):
    """Heuristic parser to extract name, age, gender, and symptoms."""
    res = {"name": "", "age": "", "gender": "", "symptoms_text": "", "raw": text}
    if not text: return res
    
    txt = text.strip()
    low = txt.lower()

    # Simple regex extractions
    age_match = re.search(r"\b(\d{1,3})\s*(years|yrs|yr|old)\b", low)
    if age_match: res["age"] = age_match.group(1)

    if re.search(r"\b(male|man|boy)\b", low): res["gender"] = "M"
    elif re.search(r"\b(female|woman|girl)\b", low): res["gender"] = "F"

    # Name is hard to extract reliably without NER, skipping for now or simple heuristic
    name_match = re.search(r"(my name is|this is|i am)\s+([A-Z][a-z]+)", txt)
    if name_match: res["name"] = name_match.group(2)

    # Symptoms: assume everything else is symptoms if not Name/Age/Gender
    # For now, just return the whole text as symptoms if short
    res["symptoms_text"] = txt 
    return res

# ------------------ UI Layout ------------------

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    # Language Selector
    lang_opts = list(config.SUPPORTED_LANGUAGES.keys())
    lang_names = list(config.SUPPORTED_LANGUAGES.values())
    curr = 0
    if st.session_state["session_language"] in lang_opts:
        curr = lang_opts.index(st.session_state["session_language"])
    
    # Dynamic Label using current session language
    lang_label_txt = "Language"
    if st.session_state["session_language"] != "en":
        lang_label_txt += " / " + translate_text("Language", st.session_state["session_language"])

    sel = st.selectbox(lang_label_txt, lang_names, index=curr)
    
    # Map back to code
    # Find code where value == sel
    sel_code = [k for k,v in config.SUPPORTED_LANGUAGES.items() if v == sel][0]
    st.session_state["session_language"] = sel_code

    st.markdown("---")
    if st.button("New Consultation"):
        reset_session()
        st.rerun()

    if st.button("View History"):
        st.session_state["show_history"] = not st.session_state["show_history"]

# Header
col_h1, col_h2 = st.columns([0.85, 0.15])
with col_h1:
    st.markdown("<div class='header'><div class='brand'>CoD</div><div><h2 style='margin:0'>MedPath AI</h2><div class='small-muted'>Advanced Diagnostic Assistant</div></div></div>", unsafe_allow_html=True)

st.markdown("---")

# ------------------ HISTORY VIEW ------------------
if st.session_state["show_history"]:
    st.subheader("Consultation History")
    rows = database.get_sessions_summary()
    
    if rows:
        # rows: (id, timestamp, name, symptoms, report)
        for row in rows:
            sid, ts, pname, syms, rpt = row
            
            # Create a card-like container
            with st.container():
                st.markdown(f"""
                <div style='background: #1e293b; padding: 12px; border-radius: 8px; margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.1);'>
                    <div style='display: flex; justify-content: space-between; align-items: start;'>
                        <div>
                            <div style='color: #94a3b8; font-size: 0.85em;'>{ts}</div>
                            <div style='color: #e2e8f0; font-weight: bold; font-size: 1.1em;'>{pname or 'Unknown'}</div>
                            <div style='color: #cbd5e1; margin-top: 4px;'>{syms}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Full Report Expander
                with st.expander("View Full Consultation Details"):
                    st.markdown(rpt)
                
                # Action Buttons
                c1, c2 = st.columns([0.85, 0.15])

                with c2:
                    if st.button("Delete", key=f"del_{sid}"):
                        database.delete_session(sid)
                        st.toast("Consultation Deleted")
                        time.sleep(0.5)
                        st.rerun()
                st.markdown("---")

    else:
        st.info("No history found.")
        
    if st.button("Close History"):
        st.session_state["show_history"] = False
        st.rerun()
    st.stop() # Stop rendering main app if history is open

# ------------------ MAIN APP ------------------
left_col, right_col = st.columns([0.45, 0.55])

# --- LEFT COLUMN: INPUT ---
with left_col:
    st.markdown("<div class='card patient-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Patient Details</div>", unsafe_allow_html=True)
    
    # Voice Input
    voice_label_txt = "Voice Input (Auto-Translate enabled)"
    if st.session_state["session_language"] != "en":
        # Translate a generic instruction
        voice_label_txt += " / " + translate_text("Record symptoms in voice", st.session_state["session_language"])
        
    st.markdown(f"<div class='small-muted'>{voice_label_txt}</div>", unsafe_allow_html=True)
    
    if _HAS_VOICE_RECORDER:
        stt_lang = st.session_state.get("session_language", "en")
        
        # Dynamic prompts for initial input
        start_txt_i = "Start Recording 🎙️"
        stop_txt_i = "Stop 🛑 (Listening...)"
        
        if stt_lang != "en":
            start_txt_i = translate_text("Start Recording", stt_lang) + " 🎙️"
            stop_txt_i = translate_text("Stop Listening", stt_lang) + " 🛑"

        transcript = speech_to_text(
            language=stt_lang,
            start_prompt=start_txt_i,
            stop_prompt=stop_txt_i,
            just_once=True,
            use_container_width=True,
            key=st.session_state["stt_key"]
        )

        if transcript:
            # Auto-append Logic
            display_en = translate_text(transcript, "en")
            
            # Append with a space or newline separator
            current_sym = st.session_state["initial_symptoms"]
            if current_sym:
                if not current_sym.endswith(" ") and not current_sym.endswith("\n"):
                    st.session_state["initial_symptoms"] += " "
            
            st.session_state["initial_symptoms"] += display_en
            st.toast(f"Voice Added: {display_en}")
            
            # Reset key immediately to consume the input and prevent stale data
            st.session_state["stt_key"] = f"stt_{int(time.time())}"
            st.rerun()
            
    else:
        st.warning("Voice recorder not available.")



    
    c1, c2, c3 = st.columns(3)
    with c1:
        # Binding directly to session state keys updates them automatically on blur/enter
        st.text_input("Name (Optional)", key="name")
    with c2:
        st.text_input("Age (Optional)", key="age")
    with c3:
        st.selectbox("Gender", ["M", "F", "Other"], key="gender")
        
    symptoms_input = st.text_area("Symptoms (Any Language)", key="initial_symptoms", height=100)
    
    start_btn = st.button("Start Consultation", type="primary", use_container_width=True)
    
    if start_btn:
        # Translate to English for processing
        en_sym = ""
        if st.session_state["initial_symptoms"]:
             en_sym = translate_text(st.session_state["initial_symptoms"], "en")
             # Store English version for logic, keep UI original
             st.session_state["current_symptoms_text"] = en_sym
        
        # Reset diag state
        st.session_state["symptoms"] = []
        st.session_state["asked"] = set()
        st.session_state["negatives"] = set()
        st.session_state["rounds"] = 1  # Start loop
        st.session_state["finished"] = False
        st.session_state["consultation_started"] = True # Explicit flag
        
        # Initial Retrieve using English text
        init = hybrid_retrieve(en_sym, k=6)
        # Filter incompatible
        filtered = [d for d in init if not is_incompatible(d, st.session_state["age"], st.session_state["gender"])] or init
        
        # Extract keywords
        pool = candidate_symptom_pool(filtered)
        
        detected_symptoms = []
        for token in config.SYMPTOM_TOKENS:
            if token in en_sym.lower():
                detected_symptoms.append(token)
        
        st.session_state["symptoms"] = list(set(detected_symptoms))
        
        st.success("Consultation Started.")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# --- RIGHT COLUMN: DIAGNOSIS & FOLLOWUP ---
with right_col:
    st.markdown("<div class='card report-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Diagnostic Engine</div>", unsafe_allow_html=True)
    
    if not st.session_state.get("consultation_started"):
        st.info("👈 Please enter details and click 'Start Consultation'.")
        st.markdown(
            """
            <div style='text-align: center; margin-top: 40px; opacity: 0.6;'>
                <h3>Waiting to start...</h3>
                <p>Enter symptoms and click the button.</p>
            </div>
            """, unsafe_allow_html=True
        )
    
    elif st.session_state["finished"]:
        # --- FINAL REPORT ---
        st.success("Consultation Complete")
        
        # Use the English text we stored; fallback to initial if missing (though expectation is it's set)
        query_text = st.session_state.get("current_symptoms_text", "") or st.session_state["initial_symptoms"]
        
        # Retrieve final candidates
        # We append the explicit symptom tokens + the raw text for best context
        full_query = ", ".join(st.session_state["symptoms"]) + " " + query_text
        
        candidates = hybrid_retrieve(full_query, k=8)
        candidates = [c for c in candidates if not is_incompatible(c, st.session_state["age"], st.session_state["gender"])] or candidates
        
        candidates = [c for c in candidates if not is_incompatible(c, st.session_state["age"], st.session_state["gender"])] or candidates
        
        probs = score_candidates(
            candidates, 
            st.session_state["symptoms"], 
            st.session_state["age"], 
            st.session_state["gender"],
            negatives=list(st.session_state.get("negatives", []))
        )
        
        # Build Report using the backend engine (includes Next Steps)
        report_md = build_final_report(
            st.session_state["name"],
            st.session_state["age"],
            st.session_state["gender"],
            st.session_state["symptoms"],
            candidates,
            probs
        )

        
        # Translate Report if needed
        if st.session_state["session_language"] != "en":
            report_display = translate_text(report_md, st.session_state["session_language"])
        else:
            report_display = report_md
            
        st.markdown(report_display)
        
        # Save to DB
        # We need to save ONLY ONCE. Check if we already saved? 
        # A simple flag in session state could work, but for now we just save.
        # But wait, reruns happen. Let's add a saved flag.
        if not st.session_state.get("saved_to_db", False):
            try:
                database.save_session(
                    st.session_state["name"],
                    st.session_state["age"],
                    st.session_state["gender"],
                    st.session_state["symptoms"],
                    probs,
                    report_md,
                    st.session_state.get("last_voice_transcript", "")
                )
                st.session_state["saved_to_db"] = True
                st.toast("Session saved to database.")
            except Exception as e:
                st.error(f"Failed to save to DB: {e}")



    else:
        # --- ACTIVE CONSULTATION ---
        
        # Scoring
        candidates = hybrid_retrieve(", ".join(st.session_state["symptoms"]) + " " + st.session_state["initial_symptoms"], k=8)
        probs = score_candidates(
            candidates, 
            st.session_state["symptoms"], 
            st.session_state["age"], 
            st.session_state["gender"],
            negatives=list(st.session_state.get("negatives", []))
        )
        
        if not probs:
             st.warning("No clear diagnosis found yet.")
        else:
            top_d = max(probs, key=probs.get)
            top_p = probs[top_d]
            
            st.markdown(f"**Current Lead:** {top_d} ({top_p:.1%})")
            
            # Stop Conditions
            # 1. High Confidence AND Min Rounds met
            cond_confidence = (top_p >= config.CONFIDENCE_THRESHOLD_STOP and st.session_state["rounds"] >= config.MIN_FOLLOWUP_QUESTIONS)
            # 2. Max Rounds reached
            cond_max_rounds = (st.session_state["rounds"] >= config.MAX_FOLLOWUP_QUESTIONS)
            
            if cond_confidence or cond_max_rounds:
                st.session_state["finished"] = True
                st.rerun()

            # Next Question
            next_q = choose_best_followup(candidates, st.session_state["symptoms"], st.session_state["asked"], min_questions=config.MIN_FOLLOWUP_QUESTIONS)
            
            if next_q:
                # Translate Question
                q_display = next_q
                if st.session_state["session_language"] != "en":
                    q_display = translate_text(f"Do you have {next_q}?", st.session_state["session_language"])
                else:
                    q_display = f"Do you have {next_q}?"
                    
                st.info(q_display)
                
                # Voice Input Option (Replaced with Custom Recorder)
                if _HAS_VOICE_RECORDER:
                     stt_lang = st.session_state.get("session_language", "en")
                     # Dynamic key per round to reset state automatically
                     # Dynamic prompts
                     start_txt = "Record Answer"
                     stop_txt = "Stop"
                     
                     if stt_lang != "en":
                         start_txt = translate_text(start_txt, stt_lang)
                         stop_txt = translate_text(stop_txt, stt_lang)

                     transcript_q = speech_to_text(
                        language=stt_lang,
                        start_prompt=start_txt,
                        stop_prompt=stop_txt,
                        just_once=True,
                        use_container_width=True,
                        key=st.session_state["stt_key_q"]
                     )
                     
                     if transcript_q:
                         # Auto-append logic for answers
                         current_ans = st.session_state["last_voice_transcript"]
                         if current_ans:
                             if not current_ans.endswith(" ") and not current_ans.endswith("\n"):
                                 st.session_state["last_voice_transcript"] += " "
                         
                         st.session_state["last_voice_transcript"] += transcript_q
                         
                         # Consume input
                         st.session_state["stt_key_q"] = f"stt_q_{int(time.time())}"
                         st.rerun()
                
                with st.form(key=f"ans_form_{st.session_state['rounds']}"):

                    # Dynamic Label based on selected language
                    label_txt = "Your Answer"
                    if st.session_state["session_language"] != "en":
                        label_txt += " / " + translate_text("Your Answer", st.session_state["session_language"])
                    
                    # Pre-fill with voice transcript if available
                    default_val = st.session_state.get("last_voice_transcript", "")
                    
                    user_ans = st.text_input(label_txt, value=default_val, placeholder="e.g. Yes / No / I have fever")
                    
                    # Clear transcript from session after using it (so it doesn't stick for next round)
                    # We can't clear it immediately inside form render, but we can overwite logic
                    
                    submit_ans = st.form_submit_button("Submit Answer")
                
                if submit_ans:
                    # Clear the transcript buffer so next question starts clean (unless they record again)
                    st.session_state["last_voice_transcript"] = "" 
                    
                    if user_ans:
                        # 1. Translate to English
                        ans_en = translate_text(user_ans, "en").lower()

                    
                    # 2. Extract Yes/No for the specific question
                    # Simple heuristics
                    is_yes = any(w in ans_en for w in ["yes", "yeah", "have", "sure", "correct", "yep", "positive"])
                    is_no = any(w in ans_en for w in ["no", "nah", "don't", "not", "negative"])
                    
                    # Conflict resolution (e.g. "No I don't have it"): "no" wins if present usually, unless "no doubt I have it"
                    # Let's prioritize explicit No, but if they say "I have x", it's a Yes for x.
                    # For the *current* question `next_q`:
                    # If they say "Yes", we add `next_q`.
                    # If they simply describe *other* symptoms, we might assume No for this one unless they say Yes?
                    # Let's assume implied No if they don't say Yes, OR just track "asked" and don't add it.
                    
                    if is_yes and not is_no:
                        st.session_state["symptoms"].append(next_q)
                        st.toast(f"Note: Reported {next_q}")
                    
                    elif is_no:
                         st.session_state["negatives"].add(next_q)
                         st.toast(f"Note: Ruled out {next_q}")

                    # 3. Extract *other* symptoms from text
                    # (e.g. "No, but I have fever")
                    found_others = []
                    for token in config.SYMPTOM_TOKENS:
                         # Check strict token presence to avoid partial matches like 'pain' in 'paint' (though tokens are simple)
                         # Simple check:
                         if token in ans_en and token != next_q.lower() and token not in st.session_state["symptoms"]:
                             st.session_state["symptoms"].append(token)
                             found_others.append(token)
                    
                    if found_others:
                        st.toast(f"Also noted: {', '.join(found_others)}")

                    st.session_state["asked"].add(next_q.lower())
                    st.session_state["rounds"] += 1
                    st.rerun()
                elif submit_ans and not user_ans:
                    st.warning("Please enter an answer.")

            else:
                st.session_state["finished"] = True
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
