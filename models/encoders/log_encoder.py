import torch
import torch.nn as nn

class LogEncoder(nn.Module):
    """
    Encodes numerical UX log features into a stable dense vector.
    Updates:
    - Removed input LayerNorm to preserve feature individuality
    - Maintains 32-dim output for fusion consistency
    - Xavier initialization for faster convergence
    """

    def __init__(self, input_dim=10, hidden_dim=32, output_dim=32):
        super(LogEncoder, self).__init__()

        # 🔧 FIX: Removed self.norm = nn.LayerNorm(input_dim)
        # Reason: LayerNorm across different metrics (seconds vs counts) destroys 
        # the individual feature signal.
        # We now handle normalization externally via StandardScaler.

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 🔧 Initialize weights properly for regression stability
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 🔧 FIX: Removed x = self.norm(x)
        # Numerical features are now pre-scaled by the pipeline
        x = self.encoder(x)            # Encode into 32‑dim vector
        return x