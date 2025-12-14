import sys
import os
import types
from unittest.mock import MagicMock

# --- MOCKING START ---
# We need to mock backend.model_loader BEFORE importing backend.diagnosis_engine
# because diagnosis_engine calls ensure_disease_map() at the top level.

mock_loader = types.ModuleType("backend.model_loader")
sys.modules["backend.model_loader"] = mock_loader

# Dummy disease map
dummy_map = {
    "Influenza": "fever, cough, body aches, fatigue, headache",
    "Common Cold": "cough, runny nose, sneezing, sore throat, mild fever",
    "COVID-19": "fever, cough, shortness of breath, loss of taste or smell, fatigue",
    "Asthma": "shortness of breath, wheezing, chest tightness, cough",
    "Migraine": "headache, nausea, sensitivity to light, dizziness"
}

def mock_ensure_disease_map():
    return dummy_map

def mock_hybrid_retrieve(symptoms, k=6):
    return list(dummy_map.keys())

# Attach mocks
mock_loader.ensure_disease_map = mock_ensure_disease_map
mock_loader.hybrid_retrieve = mock_hybrid_retrieve

# Now we can safely import diagnosis_engine
# Now we can safely import diagnosis_engine
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
from backend.diagnosis_engine import score_candidates, choose_best_followup

# Override the map inside diagnosis_engine just in case
import backend.diagnosis_engine
backend.diagnosis_engine.disease_symptom_map = dummy_map

# --- MOCKING END ---

def test_scenario():
    print("\n--- Testing Confidence Scoring (Mocked) ---")
    symptoms = ["fever", "cough"]
    age = "25"
    gender = "M"
    candidates = ["Influenza", "Common Cold", "COVID-19", "Asthma", "Migraine"]
    
    probs = score_candidates(candidates, symptoms, age, gender)
    print("Probabilities:")
    for d, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {d}: {p:.4f}")
    
    # Check if confidence is sharp (one dominates)
    top_d = max(probs, key=probs.get)
    top_p = probs[top_d]
    print(f"Max Confidence: {top_p:.4f} ({top_d})")
    
    if top_p < 0.2:
         print("WARNING: Confidence is still very low.")
    else:
         print(f"Confidence looks acceptable (Formula: 1 + 4*exact). Expected 'Influenza' or 'COVID-19' to be high since 'fever' and 'cough' match.")

    print("\n--- Testing Follow-up Loop (Visual verification of repetition) ---")
    asked = set()
    current_symptoms = list(symptoms)
    
    # Run loop
    for i in range(6):
        print(f"\nRound {i+1}")
        next_q = choose_best_followup(candidates, current_symptoms, asked, min_questions=3)
        
        if not next_q:
            print("No more questions from backend (Correct behavior if pool exhausted).")
            break
            
        print(f"Suggested Question: {next_q}")
        
        if next_q.lower() in asked:
            print(f"FAILED: Repeated question suggested: {next_q}")
            sys.exit(1)
            
        asked.add(next_q.lower())
        
        # Simulate positive answer for one, negative for others to vary flow
        if i == 0:
            print(f"  User answers YES to {next_q}")
            current_symptoms.append(next_q)
        else:
            print(f"  User answers NO to {next_q}")
            
    print("\nSUCCESS: No repetitions detected.")

if __name__ == "__main__":
    test_scenario()
