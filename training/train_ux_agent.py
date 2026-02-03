import torch
from torch.utils.data import DataLoader
import yaml
import joblib
from sklearn.preprocessing import StandardScaler

from models.encoders.log_encoder import LogEncoder
from models.encoders.text_encoder import TextEncoder
from models.encoders.behavior_encoder import BehaviorEncoder
from models.fusion_model import UXFusionModel

from training.dataset_loader import UXDataset
from training.loss_functions import UXLoss


def train_ux_agent(
    logs, behaviors, text_encodings, labels,
    batch_size=16, epochs=5, lr=1e-4
):
    """
    Full UX Agent training loop:
    - BERT fine‑tuning
    - Log + Behavior encoders
    - Fusion model
    - Gradient clipping
    - Model saving
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Using device: {device}")

    # Dataset + Dataloader
    dataset = UXDataset(logs, behaviors, text_encodings, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    log_encoder = LogEncoder().to(device)
    text_encoder = TextEncoder().to(device)       # 128‑dim output
    beh_encoder = BehaviorEncoder().to(device)
    fusion_model = UXFusionModel().to(device)

    # Collect trainable parameters
    params = (
        list(log_encoder.parameters()) +
        list(text_encoder.parameters()) +    # NOW fine‑tunable BERT
        list(beh_encoder.parameters()) +
        list(fusion_model.parameters())
    )

    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = UXLoss()

    print(f"\n🚀 Beginning Training for {epochs} epochs...\n")

    # Training Loop
    for epoch in range(epochs):
        total_loss = 0
        fusion_model.train()
        text_encoder.train()
        log_encoder.train()
        beh_encoder.train()

        for batch in loader:
            log_vec = batch['log'].to(device)
            beh_vec = batch['beh'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lbl = batch['label'].to(device)

            optimizer.zero_grad()

            # Forward passes
            log_emb = log_encoder(log_vec)                # (B, 32)
            beh_emb = beh_encoder(beh_vec)                # (B, 32)
            text_emb = text_encoder(input_ids, attention_mask)  # (B, 128)

            preds = fusion_model(log_emb, text_emb, beh_emb)

            # Loss
            loss = criterion(preds, lbl)
            loss.backward()

            # 🔥 Gradient Clipping (important for BERT)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    # ------------------------------------------------
    # SAVE ALL TRAINED WEIGHTS
    # ------------------------------------------------
    torch.save(log_encoder.state_dict(), "models/log_encoder.pt")
    torch.save(text_encoder.state_dict(), "models/text_encoder.pt")
    torch.save(beh_encoder.state_dict(), "models/behavior_encoder.pt")
    torch.save(fusion_model.state_dict(), "models/fusion_model.pt")

    print("\n💾 ✔ Models saved successfully in /models/\n")

    return log_encoder, text_encoder, beh_encoder, fusion_model


# ----------------------------------------------------------
# MAIN EXECUTION ENTRY POINT
# ----------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting UX Agent Training...")

    # Load Config [cite: 1]
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    try:
        logs = torch.load("data/processed/log_vectors.pt")
        behaviors = torch.load("data/processed/behavior_vectors.pt")
        text_encodings = torch.load("data/processed/text_encodings.pt")
        labels = torch.load("data/processed/labels.pt")
    except Exception as e:
        print("❌ ERROR: Unable to load training data from data/processed/")
        print(e)
        raise SystemExit

    # 🔧 1. Initialize Scalers 
    log_scaler = StandardScaler()
    beh_scaler = StandardScaler()

    # 🔧 2. Fit and Transform Numerical Data 
    # We convert tensors to numpy for sklearn, then back to tensors for training
    print("⚖️ Normalizing numerical features...")
    logs_scaled = torch.from_numpy(log_scaler.fit_transform(logs.numpy())).float()
    behaviors_scaled = torch.from_numpy(beh_scaler.fit_transform(behaviors.numpy())).float()

    # 🔧 3. Save Scalers for API Inference 
    scaler_data = {
        'log_scaler': log_scaler,
        'beh_scaler': beh_scaler
    }
    joblib.dump(scaler_data, config['paths']['scaler'])
    print(f"💾 Scalers saved to {config['paths']['scaler']}")

    # Start training with config values
    train_ux_agent(
        logs_scaled, 
        behaviors_scaled, 
        text_encodings, 
        labels,
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        lr=config['training']['learning_rate']
    )