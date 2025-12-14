import sys
import os
import re

# Mock environment
# Mock environment
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from backend.diagnosis_engine import score_candidates, choose_best_followup, is_incompatible
from backend.model_loader import ensure_disease_map

ensure_disease_map()

# Define dummy candidates for testing without full retriever
# In a real run, hybrid_retrieve would fetch these.
# We will use keys from disease_map if possible, or just mock them if we knew them.
# Let's inspect variable disease_symptom_map to get some real keys.
from backend.diagnosis_engine import disease_symptom_map

# Pick a few diseases for a test scenario: "Flu", "Common Cold", "COVID-19"
# We need to find closer keys in the real map.
keys = list(disease_symptom_map.keys())
candidates = [k for k in keys if "influenza" in k.lower() or "cold" in k.lower() or "covid" in k.lower() or "asthma" in k.lower()][:5]

print(f"Test Candidates: {candidates}")

def test_scenario():
    print("\n--- Testing Confidence Scoring ---")
    symptoms = ["fever", "cough"]
    age = "25"
    gender = "M"
    
    probs = score_candidates(candidates, symptoms, age, gender)
    print("Probabilities:")
    for d, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {d}: {p:.4f}")
    
    # Check if confidence is high enough (should differ significantly)
    top_p = max(probs.values()) if probs else 0
    print(f"Max Confidence: {top_p:.4f}")
    
    if top_p < 0.2:
         print("WARNING: Confidence is still very low.")
    else:
         print("Confidence looks improved.")

    print("\n--- Testing Follow-up Loop (Max 4 Check handled in frontend, but testing repetition) ---")
    asked = set()
    current_symptoms = list(symptoms)
    
    for i in range(6): # Try to go beyond 4 to see if backend keeps generating unique q's
        print(f"\nRound {i+1}")
        next_q = choose_best_followup(candidates, current_symptoms, asked, min_questions=3)
        
        if not next_q:
            print("No more questions from backend.")
            break
            
        if next_q.lower() in asked:
            print(f"ERROR: Repeated question suggested: {next_q}")
            break
            
        print(f"Suggested Question: {next_q}")
        asked.add(next_q.lower())
        # Simulate answering "no" to keep searching
        
    print("\nTest Complete.")

if __name__ == "__main__":
    test_scenario()
