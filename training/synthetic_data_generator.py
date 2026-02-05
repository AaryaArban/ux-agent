import torch
from transformers import DistilBertTokenizer
import random
import numpy as np
import os
import yaml

def generate_synthetic_training_data(n=10000):
    """
    Generates an amplified synthetic dataset to fix model calibration bias.
    Updates:
    - 🚀 Label Stretching: Pushes Good to 10.0 and Bad to 0.0 to fix mid-range clustering.
    - 🚀 Authentic Text: Expanded to 15+ descriptive reviews per class for better NLP training.
    - 🚀 Extreme Logs: Widened friction gaps to ensure technical logs correlate with scores.
    """

    # 1. Load Configuration
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        max_len = config['model']['max_length']
        model_name = config['model']['text_model_name']
        processed_path = config['paths']['processed_data']
    except Exception:
        max_len = 128
        model_name = "distilbert-base-uncased"
        processed_path = "data/processed"

    os.makedirs(processed_path, exist_ok=True)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # -------------------------
    # 🚀 Authentic Text Pools (15+ per class)
    # -------------------------
    good_texts = [
        "This app has completely changed how I manage my tasks; the UI is incredibly intuitive and snappy.",
        "I'm really impressed with the latest update, the navigation feels much smoother than before.",
        "Absolutely love the dark mode implementation! It's clear the developers put a lot of thought into UX.",
        "Five stars! Feature-rich and easy to navigate without a tutorial. Exceptional work.",
        "The performance is rock solid. I haven't experienced a single lag spike since I started using it.",
        "Great job on the new checkout flow, it's much faster now and doesn't require unnecessary clicks.",
        "Best app in its category. The transitions are fluid and the design is very modern.",
        "I love how the app anticipates what I need next; the predictive features are actually helpful.",
        "Everything works exactly as expected. It's refreshing to use software that just works perfectly.",
        "The haptic feedback and loading animations make the whole experience feel premium and high-end.",
        "Finally an app that doesn't drain my battery while providing such a smooth interface!",
        "Super satisfied with the speed. It saves me at least twenty minutes of work every single day.",
        "The integration with my other services is seamless. I didn't hit a single roadblock during setup.",
        "I've recommended this to all my colleagues. The reliability and clean aesthetic are unmatched.",
        "Brilliant execution on the user interface; it's very accessible and everything is where it should be."
    ]

    neutral_texts = [
        "The app is decent for basic use, but I find the menu layout a bit confusing at times.",
        "It gets the job done, though I noticed some minor stuttering when scrolling through long lists.",
        "An average experience overall. It works well enough but isn't particularly special.",
        "I like the features, but the loading times could definitely be improved in the next update.",
        "The app is okay, but I've seen better designs elsewhere. Functional but lacks that wow factor.",
        "It's a solid app, but it crashed once when I was trying to upload a large file earlier today.",
        "I'm satisfied for now, but I hope they fix the occasional freezing on the home screen.",
        "The interface is fine, but some buttons are a bit too small for easy tapping.",
        "It's useful for what I need, but the navigation feels a little dated compared to modern apps.",
        "A decent effort, but I feel like some advanced features are hidden too deep in settings.",
        "The app performs okay most of the time, though it occasionally feels sluggish under heavy use.",
        "I have mixed feelings; the core functionality is great but secondary features feel half-baked.",
        "It's fine for occasional use, but I wouldn't rely on it for anything mission-critical.",
        "The recent update fixed some bugs, but it also introduced a bit of lag that wasn't there.",
        "Good app overall, but the notification system is a bit overwhelming and hard to customize."
    ]

    bad_texts = [
        "Extremely disappointed. The app keeps crashing every time I try to open the main dashboard.",
        "This is unusable! The lag is so bad that it takes five seconds to register a single button press.",
        "I've lost my data twice now because the app closed unexpectedly. Terrible experience.",
        "The UI is a disaster. I can't find anything and constant loading spinners are driving me crazy.",
        "Worst update ever. Everything was working fine until yesterday, now it's just a buggy mess.",
        "Tired of the 'Something went wrong' errors. Why can't this app maintain a stable connection?",
        "The navigation is a labyrinth. I spent ten minutes just trying to find the settings page.",
        "Avoid this app at all costs. It's slow, bloated, and crashes at least three times every hour.",
        "I keep clicking buttons and nothing happens. The app isn't even registering my inputs.",
        "The performance is abysmal. It feels like I'm running this on a ten-year-old phone.",
        "Total waste of time. The app froze during a critical task and I had to restart my phone.",
        "I hate how complicated the login process is. It's like they want to make it hard to get in.",
        "The constant stuttering and visual glitches make the app feel very cheap and poorly developed.",
        "Nothing works! I've tried reinstalling three times but the crashing issue still persists.",
        "I'm uninstalling this immediately. The level of frustration this app causes is not worth it."
    ]

    log_vectors = []
    behavior_vectors = []
    labels = []
    all_texts = []

    print(f"📊 Generating {n} samples with amplified label separation...")
    for _ in range(n):
        rand_choice = random.random()

        if rand_choice > 0.66: # 🟢 GOOD (Score 8.5 – 10.0) - Amplified
            text = random.choice(good_texts)
            label = 8.5 + random.random() * 1.5
            log_vec = torch.rand(10) * 0.8  # Ultra-low friction
            beh_vec = torch.rand(5) * 0.8
        
        elif rand_choice > 0.33: # 🟡 NEUTRAL (Score 4.5 – 6.5)
            text = random.choice(neutral_texts)
            label = 4.5 + random.random() * 2.0
            log_vec = torch.rand(10) * 3.0
            beh_vec = torch.rand(5) * 3.0
        
        else: # 🔴 BAD (Score 0.0 – 2.5) - Amplified
            text = random.choice(bad_texts)
            label = 0.0 + random.random() * 2.5
            log_vec = 6.0 + torch.rand(10) * 4.0 # High friction floor
            beh_vec = 6.0 + torch.rand(5) * 4.0

        log_vectors.append(log_vec)
        behavior_vectors.append(beh_vec)
        labels.append(torch.tensor(label, dtype=torch.float))
        all_texts.append(text)

    # Batch tokenize
    encoding = tokenizer(
        all_texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    # Save
    torch.save(torch.stack(log_vectors), os.path.join(processed_path, "log_vectors.pt"))
    torch.save(torch.stack(behavior_vectors), os.path.join(processed_path, "behavior_vectors.pt"))
    torch.save({
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"]
    }, os.path.join(processed_path, "text_encodings.pt"))
    torch.save(torch.stack(labels), os.path.join(processed_path, "labels.pt"))

    print(f"✔ Amplified training data generated! ({n} samples)")
    print(f"📂 Saved to {processed_path}")

if __name__ == "__main__":
    generate_synthetic_training_data(n=10000)