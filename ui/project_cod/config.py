import os

# ---------------------------------------
# PATHS
# ---------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Model Paths
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ADAPTER_DIR = os.path.join(BACKEND_DIR, "diagnosis_gpt_v3_3_model")
RETRIEVER_CACHE_DIR = os.path.join(BACKEND_DIR, "retriever_cache")
DATABASE_PATH = os.path.join(BACKEND_DIR, "patients.db")

# ---------------------------------------
# DIAGNOSIS ENGINE SETTINGS
# ---------------------------------------
# Scoring Weights
SCORE_WEIGHT_EXACT = 4.0
SCORE_WEIGHT_PARTIAL = 1.2
SCORE_WEIGHT_RANK_DECAY = 0.1
SCORE_PENALTY_ACUTE_CHRONIC_MISMATCH = 0.7
SCORE_PENALTY_TUMOR_KEYWORDS = 0.7
SCORE_PENALTY_NEGATIVE_MATCH = 0.5 # Halve the score if a required symptom is missing (user said No)

# Confidence & Loop Control
CONFIDENCE_THRESHOLD_STOP = 0.75  # Higher confidence required to auto-stop
MIN_PROBABILITY_FLOOR = 0.05
MAX_CONFIDENCE_CAP = 0.95
TEMPERATURE_SCALING = 0.6

# Questions Limits
MIN_FOLLOWUP_QUESTIONS = 3
MAX_FOLLOWUP_QUESTIONS = 8 # Allow more questions in "Free Mode" until confidence is met

# ---------------------------------------
# MULTILINGUAL SUPPORT
# ---------------------------------------
SUPPORTED_LANGUAGES = {
    "en": "English",
    "te": "Telugu",
    "ta": "Tamil",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "es": "Spanish",
    "fr": "French"
}

# ---------------------------------------
# SYMPTOM TOKENS (For heuristic parsing)
# ---------------------------------------
SYMPTOM_TOKENS = {
    "pain", "fever", "cough", "headache", "nausea", "vomit", "vomiting", 
    "diarrhea", "dizzy", "fatigue", "weakness", "rash", "bleeding", 
    "breath", "cold", "sneeze", "runny", "stomach", "chest", "throat"
}
