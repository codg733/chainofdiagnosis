import sys
import os
import config

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.diagnosis_engine import score_candidates

def test_negatives():
    print(f"Config Penalty: {config.SCORE_PENALTY_NEGATIVE_MATCH} (Expected 0.5)")
    
    # Mock candidates
    # Assume 'Flu' has 'fever' in its symptom map (we know this from general knowledge data)
    candidates = ["Influenza", "Common Cold"]
    
    # Text Mock - Just relying on engine to look up map
    # We won't load the real map here to avoid heavy startup if possible, 
    # but score_candidates imports it. We need to respect that.
    
    print("\n--- Test 1: Baseline (Positive Fever) ---")
    symptoms = ["fever", "cough"]
    scores_base = score_candidates(candidates, symptoms)
    print(f"Flu Score (Base): {scores_base.get('Influenza', 0):.4f}")
    
    print("\n--- Test 2: Negative Penalty (No Fever) ---")
    # Scenario: User has cough, but explicitly NO fever
    symptoms_neg = ["cough"] 
    negatives = ["fever"]
    scores_neg = score_candidates(candidates, symptoms_neg, negatives=negatives)
    print(f"Flu Score (with No Fever): {scores_neg.get('Influenza', 0):.4f}")
    
    if scores_neg.get('Influenza', 0) < scores_base.get('Influenza', 0):
        print("\nSUCCESS: Score decreased significantly.")
    else:
        print("\nFAIL: Score did not decrease.")

if __name__ == "__main__":
    test_negatives()
