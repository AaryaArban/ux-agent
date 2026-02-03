import re
import torch
import yaml
from transformers import AutoTokenizer

# 🔧 Load Configuration to ensure parameter consistency
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    MODEL_NAME = config['model']['text_model_name']
    MAX_LENGTH = config['model']['max_length']
except Exception:
    # Fallbacks if config is missing during standalone testing
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 128

# Load tokenizer once for efficiency
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def clean_text(text: str) -> str:
    """
    Cleans UX review text:
    - Lowercase
    - Remove non-alphanumeric characters
    - Normalize whitespace
    """

    if not isinstance(text, str):
        return "no text provided"

    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 🔧 FIX: Avoid empty string as BERT requires at least one token
    if not text:
        text = "no text provided"

    return text


def encode_text(text: str, max_length: int = None) -> dict:
    """
    Tokenizes text for DistilBERT model.
    Updates:
    - Uses global MAX_LENGTH from config (128)
    - Returns CPU tensors (API/Model handles device movement)
    """

    if max_length is None:
        max_length = MAX_LENGTH

    cleaned = clean_text(text)

    # Tokenization with fixed padding and truncation
    encoding = tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Ensure returned tensors are CPU for initial loading
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"]
    }