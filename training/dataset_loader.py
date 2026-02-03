import torch
from torch.utils.data import Dataset

class UXDataset(Dataset):
    """
    Combines log, behavior, and pre-tokenized text features
    into a single dataset for the UX Agent training loop.
    Updates:
    - Optimized tensor handling for pre-scaled numerical data
    - Aligned shapes for multi-modal fusion compatibility
    - Preserved long dtype for BERT-specific token inputs
    """

    def __init__(self, log_data, behavior_data, text_encodings, labels):
        # Data is now passed as pre-processed tensors
        self.log_data = log_data
        self.behavior_data = behavior_data
        self.text_encodings = text_encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # ------------------------
        # Numerical Features (already scaled by StandardScaler)
        # ------------------------
        # Using .clone().detach() is safer for tensors originating from numpy/sklearn
        log_vec = self.log_data[idx].clone().detach().float()
        beh_vec = self.behavior_data[idx].clone().detach().float()

        # ------------------------
        # Text Features (already tokenized)
        # Must be long dtype for embedding lookup in DistilBERT
        # ------------------------
        text_input_ids = self.text_encodings["input_ids"][idx].clone().detach().long()
        text_attention = self.text_encodings["attention_mask"][idx].clone().detach().long()

        # ------------------------
        # Labels — ensure shape (1,) for regression compatibility
        # ------------------------
        label = self.labels[idx].clone().detach().float().reshape(1)

        # ------------------------
        # Return batch dict for the training loop
        # ------------------------
        return {
            "log": log_vec,                  # (10,) → Input for LogEncoder
            "beh": beh_vec,                  # (5,)  → Input for BehaviorEncoder
            "input_ids": text_input_ids,     # (max_length,) → Input for TextEncoder
            "attention_mask": text_attention,
            "label": label                   # (1,) → Target for UXLoss
        }