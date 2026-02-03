import numpy as np

def scale_score(raw_value):
    """
    Converts raw model output → 0–10 UX score.
    Update: Removed sigmoid scaling to prevent score distortion.
    The model is trained on a linear scale, so we use linear clamping.
    """
    # 🔧 FIX: Replace sigmoid with linear clipping to match model training logic
    return float(np.clip(raw_value, 0, 10))


def format_explanation(shap_values, feature_names):
    """
    Converts SHAP values into a readable dictionary.
    """
    explanation = {}

    for name, val in zip(feature_names, shap_values):
        explanation[name] = round(float(val), 4)

    return explanation


def combine_modalities(log_vec, text_vec, beh_vec):
    """
    Utility for combining model inputs for SHAP/kernel explainers.
    """
    # Ensure inputs are treated as numpy arrays for concatenation
    return np.concatenate([
        np.array(log_vec).reshape(1, -1), 
        np.array(text_vec).reshape(1, -1), 
        np.array(beh_vec).reshape(1, -1)
    ], axis=-1)