import sys
import os
import config

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.diagnosis_engine import predict_from_text

def test_report():
    print("Generating Sample Report...")
    # Mock a clear case
    txt = "chest pain, sweating, radiating to arm"
    res = predict_from_text(txt, name="John Doe", age=55, gender="M")
    
    print("\n" + "="*50)
    print(res["report"])
    print("="*50 + "\n")

if __name__ == "__main__":
    test_report()
