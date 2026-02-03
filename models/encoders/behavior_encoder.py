import torch
import torch.nn as nn

class BehaviorEncoder(nn.Module):
    """
    Encodes user interaction behavior features.
    Updates:
    - Removed input LayerNorm to preserve distinct behavioral signals
    - Maintains 32-dim output for fusion consistency
    - Xavier initialization for stable training
    """

    def __init__(self, input_dim=5, hidden_dim=32, output_dim=32):
        super(BehaviorEncoder, self).__init__()

        # 🔧 FIX: Removed self.norm = nn.LayerNorm(input_dim)
        # Reason: LayerNorm across different interaction metrics (e.g., loops vs taps) 
        # destroys the individual signal.
        # Normalization is now handled by the StandardScaler in the training pipeline.

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 🔧 FIX: Removed x = self.norm(x)
        # Numerical features are now pre-scaled externally
        x = self.encoder(x)       # Encode to 32‑dim vector
        return x