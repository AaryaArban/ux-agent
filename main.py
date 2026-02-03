import torch
import os
import yaml
import joblib
import numpy as np
from models.encoders.log_encoder import LogEncoder
from models.encoders.text_encoder import TextEncoder
from models.encoders.behavior_encoder import BehaviorEncoder
from models.fusion_model import UXFusionModel
from pipelines.text_preprocessing import encode_text

# ----------------------------------------------------
# 🔧 Load Configuration and Scalers
# ----------------------------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load Scalers to normalize incoming test data
try:
    scalers = joblib.load(config['paths']['scaler'])
    log_scaler = scalers['log_scaler']
    beh_scaler = scalers['beh_scaler']
    print("✅ Numerical scalers loaded successfully.")
except Exception as e:
    print("⚠️ Warning: Could not load scalers. Predictions may be inaccurate.")
    log_scaler, beh_scaler = None, None

# ----------------------------------------------------
# 🔧 GLOBAL DEVICE SETTING
# ----------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------
# LOAD TRAINED MODELS (ONE TIME ONLY)
# ----------------------------------------------------
def load_models():
    """
    Loads trained model weights ONCE and keeps them in memory.
    """
    print(f"🔧 Loading trained UX Agent models on {DEVICE}...")

    # Ensure encoders match the dimensions set in feature_engineering/log_preprocessing
    log_encoder = LogEncoder().to(DEVICE)
    text_encoder = TextEncoder().to(DEVICE)
    beh_encoder = BehaviorEncoder().to(DEVICE)
    fusion_model = UXFusionModel().to(DEVICE)

    # ---- Load Trained Weights ----
    try:
        log_encoder.load_state_dict(torch.load("models/log_encoder.pt", map_location=DEVICE))
        text_encoder.load_state_dict(torch.load("models/text_encoder.pt", map_location=DEVICE))
        beh_encoder.load_state_dict(torch.load("models/behavior_encoder.pt", map_location=DEVICE))
        fusion_model.load_state_dict(torch.load("models/fusion_model.pt", map_location=DEVICE))
        print("✅ Models loaded successfully!")
    except Exception as e:
        print("⚠️ Warning: Could not load weights. Please ensure models are re-trained after code changes.")
        print(f"Details: {e}")

    # ---- Switch to eval mode ----
    log_encoder.eval()
    text_encoder.eval()
    beh_encoder.eval()
    fusion_model.eval()

    return log_encoder, text_encoder, beh_encoder, fusion_model

# Load models ONCE when script starts
LOG_ENCODER, TEXT_ENCODER, BEH_ENCODER, FUSION_MODEL = load_models()

# ----------------------------------------------------
# PYTHON-LEVEL PREDICTION (NO API)
# ----------------------------------------------------
def predict_ux(log_vec, beh_vec, review_text):
    """
    Direct inference from Python code.
    Updated to use numerical scaling and linear clamping.
    """

    # 🔧 FIX: Scale numerical inputs before tensor conversion
    # This ensures "large" values are normalized to the training range
    log_input = np.array(log_vec).reshape(1, -1)
    beh_input = np.array(beh_vec).reshape(1, -1)
    
    if log_scaler and beh_scaler:
        log_scaled = log_scaler.transform(log_input)
        beh_scaled = beh_scaler.transform(beh_input)
    else:
        log_scaled, beh_scaled = log_input, beh_input

    # Convert inputs to tensors
    log_tensor = torch.tensor(log_scaled, dtype=torch.float).to(DEVICE)
    beh_tensor = torch.tensor(beh_scaled, dtype=torch.float).to(DEVICE)

    # Process text inputs
    enc = encode_text(review_text)
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        log_emb = LOG_ENCODER(log_tensor)
        beh_emb = BEH_ENCODER(beh_tensor)
        text_emb = TEXT_ENCODER(input_ids, attention_mask)
        raw_score = FUSION_MODEL(log_emb, text_emb, beh_emb)

    # 🔧 FIX: Linear Clamping instead of Sigmoid
    # This matches the model's regression training and preserves result granularity.
    final_score = float(raw_score.squeeze().clamp(0, 10))

    return final_score

# ----------------------------------------------------
# RUN FASTAPI IF CALLING DIRECTLY
# ----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # Start the API server
    uvicorn.run("ux_agent_api:app", host="0.0.0.0", port=8000, reload=True)