from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import torch
import yaml
import joblib
import numpy as np
import sqlite3
import os
import json
from datetime import datetime
from typing import List

from models.encoders.log_encoder import LogEncoder
from models.encoders.text_encoder import TextEncoder
from models.encoders.behavior_encoder import BehaviorEncoder
from models.fusion_model import UXFusionModel

from pipelines.text_preprocessing import encode_text

app = FastAPI(title="UX Agent Autonomous & Diagnostic API")

# ------------------------------------------------
# 🔧 Enhanced Persistence & History Management
# ------------------------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DB_PATH = config['paths']['db_path']
DECAY = config['autonomy']['decay_factor']

def init_db():
    """Initializes the database with summary and detailed session history."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Table for the North Star (Global Score)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ux_state (
            id INTEGER PRIMARY KEY,
            global_score REAL,
            total_sessions INTEGER
        )
    ''')
    
    # 2. Table for Diagnostic Drill-Down (Session History)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            session_score REAL,
            raw_logs TEXT,
            raw_behavior TEXT,
            review_text TEXT
        )
    ''')

    cursor.execute('SELECT COUNT(*) FROM ux_state')
    if cursor.fetchone()[0] == 0:
        cursor.execute('INSERT INTO ux_state (global_score, total_sessions) VALUES (5.0, 0)')
    
    conn.commit()
    conn.close()

def log_session_to_db(score, logs, behavior, text):
    """Saves detailed metrics for the Dashboard's Diagnostic Drill-Down."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO sessions (timestamp, session_score, raw_logs, raw_behavior, review_text)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.now(), score, json.dumps(logs), json.dumps(behavior), text))
    conn.commit()
    conn.close()

def get_global_state():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT global_score, total_sessions FROM ux_state WHERE id = 1')
    state = cursor.fetchone()
    conn.close()
    return {"global_score": state[0], "total_sessions": state[1]}

def update_global_state(new_session_score):
    state = get_global_state()
    current_global = state['global_score']
    current_count = state['total_sessions']

    updated_global = (new_session_score * DECAY) + (current_global * (1 - DECAY))
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE ux_state 
        SET global_score = ?, total_sessions = ? 
        WHERE id = 1
    ''', (updated_global, current_count + 1))
    conn.commit()
    conn.close()
    
    if updated_global < current_global - 0.5:
        print(f"⚠️ PROACTIVE ALERT: Significant UX trend decline detected! (-{current_global - updated_global:.2f})")
    
    return updated_global

init_db()

# ------------------------------------------------
# 🔧 Load Scalers and Models
# ------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    scalers = joblib.load(config['paths']['scaler'])
    log_scaler = scalers['log_scaler']
    beh_scaler = scalers['beh_scaler']
    
    log_encoder = LogEncoder().to(device)
    text_encoder = TextEncoder().to(device)
    beh_encoder = BehaviorEncoder().to(device)
    fusion_model = UXFusionModel().to(device)

    log_encoder.load_state_dict(torch.load("models/log_encoder.pt", map_location=device))
    text_encoder.load_state_dict(torch.load("models/text_encoder.pt", map_location=device))
    beh_encoder.load_state_dict(torch.load("models/behavior_encoder.pt", map_location=device))
    fusion_model.load_state_dict(torch.load("models/fusion_model.pt", map_location=device))
    
    log_encoder.eval()
    text_encoder.eval()
    beh_encoder.eval()
    fusion_model.eval()
    print("✔ Autonomous & Diagnostic Agent Ready.")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

# ------------------------------------------------
# Schemas
# ------------------------------------------------
class UXRequest(BaseModel):
    logs: list
    behavior: list
    review_text: str

class UXResponse(BaseModel):
    session_score: float
    global_ux_score: float
    total_sessions_analyzed: int
    alert_status: str
    explanation: dict

# ------------------------------------------------
# Routes
# ------------------------------------------------

@app.get("/status")
def get_status():
    return get_global_state()

@app.get("/history")
def get_history(page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100)):
    offset = (page - 1) * limit
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT * FROM sessions 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        
        rows = cursor.fetchall()
        history = []
        for row in rows:
            item = dict(row)
            item['raw_logs'] = json.loads(item['raw_logs'])
            item['raw_behavior'] = json.loads(item['raw_behavior'])
            history.append(item)
            
        return {"page": page, "limit": limit, "data": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/predict", response_model=UXResponse)
def predict(data: UXRequest):
    try:
        # 1. Scaling
        log_input = np.array(data.logs).reshape(1, -1)
        beh_input = np.array(data.behavior).reshape(1, -1)
        log_tensor = torch.tensor(log_scaler.transform(log_input), dtype=torch.float).to(device)
        beh_tensor = torch.tensor(beh_scaler.transform(beh_input), dtype=torch.float).to(device)

        # 2. NLP Preprocessing
        enc = encode_text(data.review_text)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # 3. Inference
        with torch.no_grad():
            log_emb = log_encoder(log_tensor)
            beh_emb = beh_encoder(beh_tensor)
            text_emb = text_encoder(input_ids, attention_mask)
            raw_score_tensor = fusion_model(log_emb, text_emb, beh_emb)

        # 🔧 4. Inference Amplifier (Phase 3 of Calibration Plan)
        # This pushes scores > 5.5 higher and scores < 4.5 lower to fix neutral bias.
        raw_val = float(raw_score_tensor.item())
        if raw_val > 5.5:
            amplified_score = raw_val + (raw_val - 5.0) * 0.8  # Push positive scores higher
        elif raw_val < 4.5:
            amplified_score = raw_val - (5.0 - raw_val) * 1.2  # Pull negative scores lower
        else:
            amplified_score = raw_val

        session_score = float(np.clip(amplified_score, 0, 10))

        # 5. 🧠 Diagnostic Persistence & Global Update
        log_session_to_db(session_score, data.logs, data.behavior, data.review_text)
        new_global_score = update_global_state(session_score)
        
        status = "CRITICAL" if session_score < 4.0 else "HEALTHY"
        
        # requested debug print
        print(f"DEBUG: Text: {data.review_text[:30]}... | Raw: {raw_val:.4f} | Final: {session_score:.2f}")

        return UXResponse(
            session_score=session_score,
            global_ux_score=new_global_score,
            total_sessions_analyzed=get_global_state()['total_sessions'],
            alert_status=status,
            explanation={
                "logs": data.logs,
                "behavior": data.behavior,
                "text": data.review_text
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))