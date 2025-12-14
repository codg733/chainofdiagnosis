import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.diagnosis_engine import predict_from_text
import config

def test_engine():
    print(f"Config Check:")
    print(f"  MAX_FOLLOWUP_QUESTIONS: {config.MAX_FOLLOWUP_QUESTIONS} (Expected 8)")
    print(f"  CONFIDENCE_THRESHOLD_STOP: {config.CONFIDENCE_THRESHOLD_STOP} (Expected 0.75)")
    print("-" * 50)

    # Case 1: Vague symptoms -> Should ask more questions
    print("\n--- Test Case 1: Vague Symptoms (chest pain) ---")
    res1 = predict_from_text("chest pain", k=10)
    symptoms1 = res1["symptoms"]
    top_cand1 = list(res1["probabilities"].items())[0] if res1["probabilities"] else ("None", 0)
    
    print(f"Total Symptoms collected: {len(symptoms1)}")
    print(f"Final Diagnosis: {top_cand1[0]} ({top_cand1[1]:.2f})")
    print(f"Questions asked (approx): {len(symptoms1) - 3}") # Subtract initial seed

    # Case 2: Specific Symptoms -> Should stop early
    print("\n--- Test Case 2: Specific Symptoms (Heart Attack) ---")
    txt = "chest pain, radiating to left arm, sweating, nausea, shortness of breath, crushing pain"
    res2 = predict_from_text(txt, k=10)
    symptoms2 = res2["symptoms"]
    top_cand2 = list(res2["probabilities"].items())[0] if res2["probabilities"] else ("None", 0)

    print(f"Total Symptoms collected: {len(symptoms2)}")
    print(f"Final Diagnosis: {top_cand2[0]} ({top_cand2[1]:.2f})")
    print(f"Questions asked (approx): {len(symptoms2) - 3}") 

if __name__ == "__main__":
    test_engine()
