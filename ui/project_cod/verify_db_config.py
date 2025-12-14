
import sys
import os
import json

# Add project root to sys.path (current dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from backend import database

def test_config():
    print("Testing Config...")
    assert config.BASE_MODEL_NAME == "Qwen/Qwen2.5-7B-Instruct"
    assert config.SCORE_WEIGHT_EXACT == 4.0
    print("✅ Config loaded successfully.")

def test_db():
    print("Testing Database...")
    database.init_db()
    
    sid = database.save_session(
        name="Test User",
        age="30",
        gender="M",
        symptoms=["fever", "cough"],
        diagnosis_json={"Flu": 0.8},
        report="Test Report",
        transcript="I have a fever"
    )
    print(f"Session saved with ID: {sid}")
    
    rows = database.get_sessions_summary()
    assert len(rows) > 0
    print(f"Found {len(rows)} sessions.")
    print("✅ Database works.")

if __name__ == "__main__":
    test_config()
    test_db()
