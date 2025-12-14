import re
import math
import numpy as np
try:
    import config
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

from backend.model_loader import hybrid_retrieve, ensure_disease_map

# Load disease map
disease_symptom_map = ensure_disease_map()


# ============================================================
# AGE / GENDER FILTER
# ============================================================
def is_incompatible(disease_name, age, gender):
    d = (disease_name or "").lower()

    try:
        age_i = int(age)
    except:
        age_i = None

    g = (gender or "").lower()

    # Basic gating rules
    if age_i is not None and age_i < 40:
        if any(k in d for k in ["elderly", "geriatric", "senile"]):
            return True

    if age_i is not None and age_i >= 18:
        if any(k in d for k in ["infant", "child", "pediatric", "neonate"]):
            return True

    if age_i is not None and age_i < 25:
        if any(k in d for k in ["stenosis", "aneurysm", "heart failure"]):
            return True

    if age_i is not None and age_i < 30:
        if any(k in d for k in ["coronar", "angina", "ischemi"]):
            return True

    if g.startswith("m"):
        if any(k in d for k in ["pregnan", "ovarian", "uterine", "breast"]):
            return True

    if g.startswith("f"):
        if any(k in d for k in ["prostate", "testicular"]):
            return True

    return False


# ============================================================
# SCORING ENGINE
# ============================================================
def score_candidates(candidates, symptoms, age=None, gender=None, negatives=None):

    scores = {}
    for i, d in enumerate(candidates):

        if is_incompatible(d, age, gender):
            scores[d] = 1e-9
            continue

        kb = (disease_symptom_map.get(d, "") or "").lower()

        exact = sum(1 for s in symptoms if s.lower() in kb)

        partial = 0
        for s in symptoms:
            toks = re.findall(r"\w+", s.lower())
            hits = sum(1 for t in toks if t in kb)
            if hits >= max(1, len(toks) // 2):
                partial += 1

        # Tuned weights: Higher weight for exact matches to boost confidence
        score = (
            1
            + config.SCORE_WEIGHT_EXACT * exact
            + config.SCORE_WEIGHT_PARTIAL * partial
            + config.SCORE_WEIGHT_RANK_DECAY * (len(candidates) - i)
        )

        s_txt = " ".join(symptoms).lower()
        if "acute" in s_txt and "chronic" in (kb + d):
            score *= config.SCORE_PENALTY_ACUTE_CHRONIC_MISMATCH
        if "chronic" in s_txt and "acute" in (kb + d):
            score *= config.SCORE_PENALTY_ACUTE_CHRONIC_MISMATCH * 1.1 # slightly higher penalty

        if any(t in d for t in ["tumor", "carcinoma", "neoplasm"]):
            score *= config.SCORE_PENALTY_TUMOR_KEYWORDS

        # Negative Symptom Penalty
        if negatives:
            for neg in negatives:
                if neg.lower() in kb:
                    score *= config.SCORE_PENALTY_NEGATIVE_MATCH

        scores[d] = score

    arr = np.array(list(scores.values()))
    if len(arr) > 0:
        arr -= arr.max()
        # Lower temperature creates sharper probability distribution (higher confidence)
        ex = np.exp(arr / config.TEMPERATURE_SCALING) 
        ex /= ex.sum() + 1e-12
        result = {d: float(p) for d, p in zip(scores.keys(), ex)}
    else:
        result = {}

    # ----- Prevent perfect certainty -----
    max_conf = max(result.values()) if result else 0
    if max_conf >= config.MAX_CONFIDENCE_CAP: # Allow higher confidence cap
        sf = config.MAX_CONFIDENCE_CAP / max_conf
        result = {d: min(p * sf, config.MAX_CONFIDENCE_CAP) for d, p in result.items()}

    # ----- Assign minimum floor confidence -----
    MIN_PROB = config.MIN_PROBABILITY_FLOOR # Lower floor to distinguish low probability items better
    result = {d: max(v, MIN_PROB) for d, v in result.items()}

    # Re-normalize
    total = sum(result.values())
    if total > 0:
        result = {d: v / total for d, v in result.items()}

    return result


# ============================================================
# FOLLOW-UP SYMPTOM EXTRACTION
# ============================================================
def candidate_symptom_pool(candidates, max_per=6, max_total=20):

    pool = []
    for d in candidates:
        txt = (disease_symptom_map.get(d, "") or "")
        parts = re.split(r"[;,.\n•\-/()]+", txt)

        count = 0
        for p in parts:
            p = p.strip()
            if 1 <= len(p.split()) <= 10:
                pool.append(p)
                count += 1
            if count >= max_per:
                break

    out = []
    seen = set()
    for p in pool:
        if p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
        if len(out) >= max_total:
            break

    return out


# ============================================================
# BEST FOLLOW-UP QUESTION (INFO GAIN)
# ============================================================
def choose_best_followup(candidates, symptoms, asked, min_questions=3):

    base = score_candidates(candidates, symptoms)
    H0 = -sum(p * math.log(p + 1e-9) for p in base.values())

    pool = candidate_symptom_pool(candidates)
    pool = [p for p in pool if p.lower() not in asked]

    best = None
    best_delta = 0

    for p in pool:
        new = score_candidates(candidates, symptoms + [p])
        H1 = -sum(q * math.log(q + 1e-9) for q in new.values())
        d = H0 - H1

        if d > best_delta:
            best_delta = d
            best = p

    if best is None and len(asked) < min_questions and pool:
        return pool[0]

    return best


# ============================================================
# DIAGNOSTIC REASONING REPORT (CLEANED + EXPANDED)
# ============================================================
def build_final_report(name, age, gender, symptoms, candidates, probs):

    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return "Not enough data to generate a report."
        
    top_d, top_p = ranked[0]

    def snip(d):
        return (disease_symptom_map.get(d, "") or "").split(".")[0][:260]

    report = []

    # ---------------- HEADER ----------------
    report.append("### 🧠 Diagnostic Reasoning Report")
    report.append(f"**Patient**: {name or 'N/A'}, **Age**: {age or 'N/A'}, **Gender**: {gender or 'N/A'}")
    report.append(f"Reported symptoms: **{', '.join(symptoms)}**\n")

    # ---------------- SYMPTOM ANALYSIS ----------------
    report.append("#### 🔍 Analyzing patient symptoms")
    report.append("The system identified the following relevant symptoms:")
    for s in symptoms:
        report.append(f"- {s}")

    # ---------------- LIKELY DIAGNOSES ----------------
    report.append("\n#### 📌 Likely diagnoses based on current evidence")
    for d, p in ranked[:6]:
        report.append(f"- **{d}** — {snip(d)} *(confidence {p:.2f})*")

    # ---------------- DIAGNOSTIC REASONING ----------------
    report.append("\n#### 🧠 Diagnostic Analysis\n")
    
    # Dynamic templates to reduce repetition
    high_conf_templates = [
        "**{d}** stands out as the primary candidate. The presence of **{symptoms}** is highly consistent with this condition.",
        "Clinical evidence points strongly towards **{d}**, supported by **{symptoms}**.",
        "**{d}** is the most probable diagnosis given the combination of **{symptoms}**."
    ]
    
    partial_conf_templates = [
        "**{d}** is a possibility, though the picture is incomplete. Symptoms like **{symptoms}** overlap, but other key features may be missing.",
        "Consider **{d}** as a differential. While **{symptoms}** match, confidence is moderate.",
        "There are indicators for **{d}** (**{symptoms}**), but it remains a secondary consideration."
    ]
    
    low_conf_templates = [
        "**{d}** cannot be ruled out but is less likely. Only minor overlaps ({symptoms}) were noted.",
        "**{d}** has low probability, with only isolated symptoms ({symptoms}) matching.",
        "Evidence for **{d}** is weak, primarily limited to {symptoms}."
    ]

    import random

    for i, (d, p) in enumerate(ranked[:5]):
        kb = (disease_symptom_map.get(d, "") or "").lower()
        matching_symptoms = [s for s in symptoms if s.lower() in kb]
        
        if not matching_symptoms:
            continue
            
        pretty_symptoms = ", ".join(matching_symptoms)
        
        if p > 0.5:
            tmpl = high_conf_templates[i % len(high_conf_templates)]
        elif p > 0.2:
            tmpl = partial_conf_templates[i % len(partial_conf_templates)]
        else:
            tmpl = low_conf_templates[i % len(low_conf_templates)]
            
        reasoning = tmpl.format(d=d, symptoms=pretty_symptoms)
        report.append(f"- {reasoning} *(Confidence: {p:.1%})*")

    # ---------------- FINAL IMPRESSION ----------------
    report.append("\n#### ✅ Final diagnostic impression")
    report.append(
        f"Based on the current evidence, **{top_d}** is the most likely diagnosis, "
        f"with an estimated confidence of **{top_p:.2f}**."
    )

    # ---------------- NEXT STEPS ----------------
    report.append("\n#### 🩺 Recommended next steps")
    next_steps = []

    bl = top_d.lower()
    s_txt = " ".join(symptoms).lower()

    if "infect" in bl or "fever" in s_txt:
        next_steps.append("Order CBC + CRP/ESR to evaluate infection or inflammation.")

    if "lung" in bl or "cough" in s_txt:
        next_steps.append("Consider chest X-ray to assess respiratory involvement.")

    if not next_steps:
        next_steps.append("Perform a focused physical exam and targeted laboratory testing.")

    for ns in next_steps:
        report.append(f"- {ns}")

    # ---------------- OPTIONAL: NEXT QUESTION ----------------
    if top_p < 0.60:
        pool = candidate_symptom_pool(candidates)
        next_q = pool[0] if pool else "any other symptoms"
        report.append("\n#### ❓ Additional symptom check")
        report.append(f"To refine the diagnosis further, do you have **{next_q}**?")

    return "\n".join(report)


# ============================================================
# PIPELINE TESTER
# ============================================================
def predict_from_text(text, name="N/A", age=None, gender=None, k=6):

    init = hybrid_retrieve(text, k=k)
    filtered = [d for d in init if not is_incompatible(d, age, gender)] or init

    seed = candidate_symptom_pool(filtered)[:3]
    asked = set()
    current = seed.copy()

    max_q = config.MAX_FOLLOWUP_QUESTIONS
    min_q = config.MIN_FOLLOWUP_QUESTIONS
    
    for i in range(max_q):
        cand = hybrid_retrieve(", ".join(current), k=k)
        cand = [d for d in cand if not is_incompatible(d, age, gender)] or cand

        # Check confidence for early stopping (only after min questions)
        if i >= min_q:
            temp_probs = score_candidates(cand, current, age, gender)
            top_p = max(temp_probs.values()) if temp_probs else 0.0
            if top_p >= config.CONFIDENCE_THRESHOLD_STOP:
                break

        nxt = choose_best_followup(cand, current, asked)
        if not nxt:
            break
            
        current.append(nxt)
        asked.add(nxt.lower())

    final = hybrid_retrieve(", ".join(current), k=k)
    final = [d for d in final if not is_incompatible(d, age, gender)] or final

    probs = score_candidates(final, current, age, gender)
    report = build_final_report(name, age, gender, current, final, probs)

    return {
        "symptoms": current,
        "candidates": final,
        "probabilities": probs,
        "report": report
    }
