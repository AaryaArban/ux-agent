import pandas as pd

def preprocess_logs(df):
    """
    Cleans and aggregates raw UX logs into session-level features.
    Updated to handle long-format JSON schema.
    """

    # 1. Convert timestamps and sort for chronological consistency
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values(['user_id', 'session_id', 'timestamp'])

    # 2. 🔧 FIX: Pivot the event_type rows into columns
    # This transforms 'glitch', 'latency', etc., into actual columns we can aggregate
    df_pivoted = df.pivot_table(
        index=['user_id', 'session_id', 'timestamp'], 
        columns='event_type', 
        values='value',
        aggfunc='first'
    ).reset_index()

    # 3. Ensure all expected columns exist even if missing from raw data
    # Schema keys: glitch | latency | crash | navigation
    expected_cols = ['latency', 'glitch', 'crash', 'navigation']
    for col in expected_cols:
        if col not in df_pivoted.columns:
            df_pivoted[col] = 0

    # 4. Handle missing numerical data
    df_pivoted = df_pivoted.fillna(0)

    # 5. Aggregate into session-level features
    # We produce 6 base features here; the remaining 4 will be created in feature_engineering.py
    session_features = df_pivoted.groupby(['user_id', 'session_id']).agg(
        latency_avg=('latency', 'mean'),      # Feature 1
        glitches=('glitch', 'sum'),           # Feature 2
        crashes=('crash', 'sum'),             # Feature 3
        navigation_time=('navigation', 'sum'),# Feature 4
        total_events=('navigation', 'count'), # Feature 5: Event density
        max_latency=('latency', 'max')        # Feature 6: Severity peak
    ).reset_index()

    return session_features