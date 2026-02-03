from fastapi import FastAPI
from pydantic import BaseModel
import torch
import yaml
import joblib
import numpy as np

from models.encoders.log_encoder import LogEncoder
from models.encoders.text_encoder import TextEncoder
from models.encoders.behavior_encoder import BehaviorEncoder
from models.fusion_model import UXFusionModel

from pipelines.text_preprocessing import encode_text

app = FastAPI(title="UX Agent API")

# ------------------------------------------------
# 🔧 Load Configuration and Scalers
# ------------------------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

try:
    scalers = joblib.load(config['paths']['scaler'])
    log_scaler = scalers['log_scaler']
    beh_scaler = scalers['beh_scaler']
    print("✔ Loaded numerical scalers successfully.")
except Exception as e:
    print("❌ ERROR: Could not load scalers. Ensure you have run train_ux_agent.py first.")
    print(e)

# ------------------------------------------------
# Load device
# ------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------
# Load Models (patched encoder sizes)
# ------------------------------------------------
log_encoder = LogEncoder().to(device)
text_encoder = TextEncoder().to(device)       # 128-dim output now
beh_encoder = BehaviorEncoder().to(device)
fusion_model = UXFusionModel().to(device)

# ------------------------------------------------
# Load trained weights (IMPORTANT)
# ------------------------------------------------
try:
    log_encoder.load_state_dict(torch.load("models/log_encoder.pt", map_location=device))
    text_encoder.load_state_dict(torch.load("models/text_encoder.pt", map_location=device))
    beh_encoder.load_state_dict(torch.load("models/behavior_encoder.pt", map_location=device))
    fusion_model.load_state_dict(torch.load("models/fusion_model.pt", map_location=device))
    print("✔ Loaded model weights successfully.")
except Exception as e:
    print("❌ ERROR: Could not load model weights.")
    print(e)

log_encoder.eval()
text_encoder.eval()
beh_encoder.eval()
fusion_model.eval()


# ------------------------------------------------
# Request Schema
# ------------------------------------------------
class UXRequest(BaseModel):
    logs: list
    behavior: list
    review_text: str


# ------------------------------------------------
# Response Schema
# ------------------------------------------------
class UXResponse(BaseModel):
    ux_score: float
    explanation: dict


# ------------------------------------------------
# Inference Route
# ------------------------------------------------
@app.post("/predict", response_model=UXResponse)
def predict(data: UXRequest):

    # 🔧 FIX: Scale numerical inputs before tensor conversion
    # This aligns the live input with the training distribution
    log_input = np.array(data.logs).reshape(1, -1)
    beh_input = np.array(data.behavior).reshape(1, -1)
    
    log_scaled = log_scaler.transform(log_input)
    beh_scaled = beh_scaler.transform(beh_input)

    # Convert to tensors
    log_tensor = torch.tensor(log_scaled, dtype=torch.float).to(device)
    beh_tensor = torch.tensor(beh_scaled, dtype=torch.float).to(device)

    # Encode text → returns dict with input_ids + attention_mask
    enc = encode_text(data.review_text)

    # 🔧 FIX: Ensure all tensors are moved to the active device
    input_ids = enc["input_ids"].to(device)          # shape: (1, seq_len)
    attention_mask = enc["attention_mask"].to(device)

    # Forward pass (no gradient)
    with torch.no_grad():
        log_emb = log_encoder(log_tensor)                    # (1, 32)
        beh_emb = beh_encoder(beh_tensor)                    # (1, 32)
        text_emb = text_encoder(input_ids, attention_mask)   # (1, 128)
        raw_score = fusion_model(log_emb, text_emb, beh_emb) # (1, 1)

    # 🔧 FIX: Scale output using linear clamping instead of sigmoid
    ux_score = float(raw_score.squeeze().clamp(0, 10))

    # Explanation dictionary
    explanation = {
        "log_features_used": data.logs,
        "behavior_features_used": data.behavior,
        "text": data.review_text
    }

    return UXResponse(
        ux_score=ux_score,
        explanation=explanation
    )