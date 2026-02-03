import torch
import torch.nn as nn
from transformers import DistilBertModel


class TextEncoder(nn.Module):
    """
    Encodes user review text using DistilBERT.
    Updates:
    - Maintains UNFROZEN BERT layers for UX-specific fine-tuning
    - Strict 128-dim projection for fusion compatibility
    - Optimized stabilization head
    """

    def __init__(self, model_name="distilbert-base-uncased"):
        super(TextEncoder, self).__init__()

        # Load DistilBERT
        self.bert = DistilBertModel.from_pretrained(model_name)

        # 🔓 UNFREEZE BERT PARAMETERS (Fine-tuning is essential for sentiment)
        for param in self.bert.parameters():
            param.requires_grad = True

        # 🔽 Projection layer (768 → 128)
        self.proj = nn.Linear(768, 128)

        # ⚖️ Stabilization for the projection
        # We keep LayerNorm here because BERT hidden states require scaling 
        # before fusion with numerical embeddings.
        self.layernorm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(0.2)

        self._init_weights()

    def _init_weights(self):
        """Initializes the projection layer weights."""
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, input_ids, attention_mask):
        # BERT forward pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # CLS token embedding represents the entire sequence (768‑dim)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # 🔽 Project → Normalize → Dropout
        proj = self.proj(cls_embedding)       # 768 → 128
        proj = self.layernorm(proj)
        proj = self.dropout(proj)

        # Final text embedding (128‑dim)
        return proj