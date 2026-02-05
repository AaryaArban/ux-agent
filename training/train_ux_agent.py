import torch
from torch.utils.data import DataLoader
import yaml
import joblib
import os
import sys
import sqlite3
from sklearn.preprocessing import StandardScaler

# 🔧 Path Setup: Ensure the project root is accessible for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.encoders.log_encoder import LogEncoder
from models.encoders.text_encoder import TextEncoder
from models.encoders.behavior_encoder import BehaviorEncoder
from models.fusion_model import UXFusionModel

from training.dataset_loader import UXDataset
from training.loss_functions import UXLoss


def init_autonomous_baseline(db_path):
    """
    Initializes or resets the global score baseline in the database.
    Ensures the autonomous system starts from a neutral state (5.0) 
    after a fresh model training session.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ux_state (
            id INTEGER PRIMARY KEY,
            global_score REAL,
            total_sessions INTEGER
        )
    ''')
    
    # Check if record exists
    cursor.execute('SELECT COUNT(*) FROM ux_state')
    if cursor.fetchone()[0] == 0:
        cursor.execute('INSERT INTO ux_state (global_score, total_sessions) VALUES (5.0, 0)')
    else:
        # Reset baseline for the new model
        cursor.execute('UPDATE ux_state SET global_score = 5.0, total_sessions = 0 WHERE id = 1')
    
    conn.commit()
    conn.close()
    print(f"🧠 Autonomous memory initialized/reset at {db_path}")


def train_ux_agent(
    logs, behaviors, text_encodings, labels,
    batch_size=16, epochs=5, lr=1e-4
):
    """
    Full UX Agent training loop optimized for amplified labels.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Training on: {device}")

    dataset = UXDataset(logs, behaviors, text_encodings, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    log_encoder = LogEncoder().to(device)
    text_encoder = TextEncoder().to(device)
    beh_encoder = BehaviorEncoder().to(device)
    fusion_model = UXFusionModel().to(device)

    # 🔧 Optimization: Combined parameters for all encoders and the fusion layer
    params = (
        list(log_encoder.parameters()) +
        list(text_encoder.parameters()) +
        list(beh_encoder.parameters()) +
        list(fusion_model.parameters())
    )

    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = UXLoss()

    print(f"\n🚀 Beginning Training for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        # Set all components to training mode
        fusion_model.train(); text_encoder.train()
        log_encoder.train(); beh_encoder.train()

        for batch in loader:
            log_vec = batch['log'].to(device)
            beh_vec = batch['beh'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lbl = batch['label'].to(device)

            optimizer.zero_grad()

            # Forward pass across all encoders
            log_emb = log_encoder(log_vec)
            beh_emb = beh_encoder(beh_vec)
            text_emb = text_encoder(input_ids, attention_mask)

            # Fusion and prediction
            preds = fusion_model(log_emb, text_emb, beh_emb)
            loss = criterion(preds, lbl)
            
            # Backward pass
            loss.backward()

            # 🔧 Gradient Clipping: Prevents exploding gradients with amplified labels
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    # Save weights to the models directory
    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(log_encoder.state_dict(), os.path.join(model_dir, "log_encoder.pt"))
    torch.save(text_encoder.state_dict(), os.path.join(model_dir, "text_encoder.pt"))
    torch.save(beh_encoder.state_dict(), os.path.join(model_dir, "behavior_encoder.pt"))
    torch.save(fusion_model.state_dict(), os.path.join(model_dir, "fusion_model.pt"))

    print(f"💾 ✔ Weights saved to {model_dir}")
    return log_encoder, text_encoder, beh_encoder, fusion_model


if __name__ == "__main__":
    print("🚀 Initializing Autonomous UX Agent Training...")

    # Load configuration
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load Tensors using processed data path
    data_dir = os.path.join(PROJECT_ROOT, config['paths']['processed_data'])
    try:
        logs = torch.load(os.path.join(data_dir, "log_vectors.pt"))
        behaviors = torch.load(os.path.join(data_dir, "behavior_vectors.pt"))
        text_encodings = torch.load(os.path.join(data_dir, "text_encodings.pt"))
        labels = torch.load(os.path.join(data_dir, "labels.pt"))
    except Exception as e:
        print(f"❌ ERROR: Data not found in {data_dir}. Run your generator first.")
        raise SystemExit

    # Numerical Scaling
    print("⚖️ Fitting StandardScalers...")
    log_scaler = StandardScaler()
    beh_scaler = StandardScaler()
    logs_scaled = torch.from_numpy(log_scaler.fit_transform(logs.numpy())).float()
    behaviors_scaled = torch.from_numpy(beh_scaler.fit_transform(behaviors.numpy())).float()

    # Save Scaler for the Autonomous API
    scaler_save_path = os.path.join(PROJECT_ROOT, config['paths']['scaler'])
    joblib.dump({'log_scaler': log_scaler, 'beh_scaler': beh_scaler}, scaler_save_path)

    # Run Training loop
    train_ux_agent(
        logs_scaled, behaviors_scaled, text_encodings, labels,
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        lr=config['training']['learning_rate']
    )

    # 🧠 Sync Autonomous Memory
    db_abs_path = os.path.join(PROJECT_ROOT, config['paths']['db_path'])
    init_autonomous_baseline(db_abs_path)
    
    print("\n✅ Training and Autonomous Sync Complete!")