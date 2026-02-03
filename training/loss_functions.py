import torch
import torch.nn as nn

class UXLoss(nn.Module):
    """
    Custom regression loss for UX score prediction.
    Updates:
    - Enforced strict (Batch, 1) shape to prevent broadcasting errors
    - Uses SmoothL1Loss (Huber Loss) for outlier robustness
    - Maintains weight parameter for modality-specific balancing
    """

    def __init__(self, weight=1.0):
        super(UXLoss, self).__init__()
        # SmoothL1Loss is less sensitive to outliers than MSE
        self.loss_fn = nn.SmoothL1Loss()
        self.weight = weight

    def forward(self, predictions, targets):
        """
        Calculates loss between UX score predictions and ground truth labels.
        """
        # 🔧 FIX: Ensure both are 2D tensors of shape (Batch, 1)
        # This prevents the common "size mismatch" or "broadcasting" errors in PyTorch regression.
        if predictions.dim() == 1:
            predictions = predictions.view(-1, 1)
        
        if targets.dim() == 1:
            targets = targets.view(-1, 1)

        # Ensure target is float32 to match model output
        targets = targets.float()

        # Calculate base loss
        loss = self.loss_fn(predictions, targets)

        # Apply weighting (useful if balancing different data sources)
        return self.weight * loss