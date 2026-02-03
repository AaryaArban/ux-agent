import pandas as pd
import numpy as np

def engineer_features(logs_df, behavior_df):
    """
    Combines aggregated logs + behavior into derived UX metrics.
    Ensures exactly 10 features for LogEncoder input dimension.
    """
    
    # 1. Align data by merging on common session keys
    # This prevents the page loops of one user from being attributed to another's latency.
    # We assume behavior_df has columns: user_id, session_id, and 5 behavior metrics.
    combined = logs_df.merge(behavior_df, on=['user_id', 'session_id'], how='inner')

    # 2. Derive UX Metrics (Totaling 10 columns for the model)
    
    # Feature 7: Friction Score (Weighted interaction penalty)
    combined['friction_score'] = (
        combined['glitches'] * 0.4 +
        combined['crashes'] * 0.3 +
        combined['latency_avg'] * 0.3
    )

    # Feature 8: Navigation Complexity (Ratio of time to events)
    combined['navigation_complexity'] = (
        combined['navigation_time'] / (combined['total_events'] + 1e-5)
    )

    # Feature 9: Error Severity (Weighted crash impact)
    combined['error_severity'] = (
        combined['crashes'] * 3 +
        combined['max_latency'] * 0.5
    )
    
    # Feature 10: Event Density (Events per second of navigation)
    combined['event_density'] = combined['total_events'] / (combined['navigation_time'] + 1e-5)

    # 3. Final Selection (Must match log_input_dim: 10 in config.yaml)
    # These match the order the model was trained on.
    final_features = [
        'latency_avg',          # 1: From preprocessing
        'glitches',             # 2: From preprocessing
        'crashes',              # 3: From preprocessing
        'navigation_time',       # 4: From preprocessing
        'total_events',          # 5: From preprocessing
        'max_latency',           # 6: From preprocessing
        'friction_score',        # 7: Engineered
        'navigation_complexity', # 8: Engineered
        'error_severity',        # 9: Engineered
        'event_density'          # 10: Engineered
    ]

    return combined[final_features]