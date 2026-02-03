import torch
import torch.nn as nn

class UXFusionModel(nn.Module):
    """
    Combines log, text, and behavior embeddings into a unified UX Score.
    Updates:
    - Softmax normalization for modality weights to prevent text dominance
    - Maintains original architecture for regression head stability
    - Preserves Xavier initialization for linear layers
    """

    def __init__(self, log_dim=32, text_dim=128, beh_dim=32):
        super(UXFusionModel, self).__init__()

        # 🔥 Learnable modality weights
        # Initialized to equal values; Softmax will handle the relative scaling
        self.alpha_log = nn.Parameter(torch.tensor(1.0))
        self.alpha_text = nn.Parameter(torch.tensor(1.0))
        self.alpha_beh = nn.Parameter(torch.tensor(1.0))

        # Final fused embedding size
        fusion_dim = log_dim + text_dim + beh_dim

        # 🔧 Fusion normalization & dropout
        self.layernorm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(0.2)

        # 🔥 Strong regression head
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, log_emb, text_emb, beh_emb):
        # 🔧 FIX: Apply Softmax to ensure weights are relative and sum to 1.0
        # This prevents any one modality from drowning out the others mathematically
        weights = torch.softmax(torch.stack([self.alpha_log, self.alpha_text, self.alpha_beh]), dim=0)

        # Apply learnable modality weights
        fused = torch.cat([
            weights[0] * log_emb,
            weights[1] * text_emb,
            weights[2] * beh_emb
        ], dim=1)

        # Normalize + dropout for stability
        fused = self.layernorm(fused)
        fused = self.dropout(fused)

        # Predict final UX score (Raw logit for regression)
        score = self.regressor(fused)
        return score