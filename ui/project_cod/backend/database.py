import sqlite3
import json
import uuid
from datetime import datetime
try:
    import config
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

def init_db():
    conn = sqlite3.connect(config.DATABASE_PATH)
    c = conn.cursor()
    
    # Patients table (optional if we want to reuse patients)
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    age TEXT,
                    gender TEXT,
                    created_at TIMESTAMP
                )''')

    # Sessions table
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    symptoms TEXT,
                    diagnosis_result TEXT, -- JSON
                    final_report TEXT,
                    transcript TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY(patient_id) REFERENCES patients(id)
                )''')
    
    conn.commit()
    conn.close()

def save_session(name, age, gender, symptoms, diagnosis_json, report, transcript=""):
    init_db()
    conn = sqlite3.connect(config.DATABASE_PATH)
    c = conn.cursor()

    # Check if patient exists (simple fuzzy check or just create new for now to avoid complexity)
    # Ideally we'd match on name+age, but let's just create a new entry for every session 
    # unless we want to implement a lookup. Given the requirements, let's keep it simple.
    
    patient_id = str(uuid.uuid4())
    c.execute("INSERT INTO patients (id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?)",
              (patient_id, name, age, gender, datetime.now()))

    session_id = str(uuid.uuid4())
    # diagnosis_json needs to be string
    if isinstance(diagnosis_json, dict):
        diag_str = json.dumps(diagnosis_json)
    else:
        diag_str = str(diagnosis_json)
        
    # symptoms list to string
    if isinstance(symptoms, list):
        sym_str = ", ".join(symptoms)
    else:
        sym_str = str(symptoms)

    c.execute("INSERT INTO sessions (id, patient_id, symptoms, diagnosis_result, final_report, transcript, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (session_id, patient_id, sym_str, diag_str, report, transcript, datetime.now()))

    conn.commit()
    conn.close()
    return session_id

def get_sessions_summary():
    init_db()
    conn = sqlite3.connect(config.DATABASE_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT s.id, s.timestamp, p.name, s.symptoms, s.final_report 
        FROM sessions s 
        JOIN patients p ON s.patient_id = p.id 
        ORDER BY s.timestamp DESC
    """)
    rows = c.fetchall()
    conn.close()
    return rows

def delete_session(session_id):
    init_db()
    conn = sqlite3.connect(config.DATABASE_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

