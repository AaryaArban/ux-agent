import shap
import torch
import numpy as np

class UXSHAPExplainer:
    """
    Generates SHAP explanations for UX Agent predictions.
    Updates:
    - Fixed model_forward to correctly slice concatenated feature batches
    - [cite_start]Aligned slicing indices with model embedding dimensions (32, 128, 32) [cite: 1]
    - Optimized tensor conversion for the forward pass
    """

    def __init__(self, fusion_model, background_data):
        """
        background_data: numpy array or tensor of representative samples (N x 192)
        """
        self.model = fusion_model
        self.background = background_data

        # Use Kernel SHAP for model-agnostic explanation
        self.explainer = shap.KernelExplainer(
            self.model_forward,
            self.background
        )

    def model_forward(self, input_data):
        """
        SHAP passes a batch of samples (N, 192). 
        We must slice them into the three modalities.
        """
        # 🔧 FIX: Slice the batch of data by column indices to match modality sizes
        # [cite_start]log_dim=32, text_dim=128, beh_dim=32 [cite: 1]
        log_emb = input_data[:, 0:32]
        text_emb = input_data[:, 32:160]
        beh_emb = input_data[:, 160:192]

        with torch.no_grad():
            # Pass individual modality tensors to the fusion model
            outputs = self.model(
                torch.tensor(log_emb, dtype=torch.float),
                torch.tensor(text_emb, dtype=torch.float),
                torch.tensor(beh_emb, dtype=torch.float)
            )

        return outputs.cpu().numpy()

    def explain(self, log_vec, text_vec, beh_vec):
        """
        Returns SHAP values for a single prediction.
        """
        # 🔧 FIX: Concatenate into a single feature vector to match explainer's expected input
        instance = np.concatenate([
            log_vec.reshape(1, -1), 
            text_vec.reshape(1, -1), 
            beh_vec.reshape(1, -1)
        ], axis=-1)

        shap_values = self.explainer.shap_values(instance)

        return shap_values