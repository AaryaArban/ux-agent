import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import yaml
import json
import numpy as np
import time
from datetime import datetime

# 🔧 Load Configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DB_PATH = config['paths']['db_path']
DEFAULT_PAGE_SIZE = config['ui']['dashboard_page_size']
MAX_DIAGNOSTIC = config['ui']['max_diagnostic_buffer']

def load_data(limit=100, offset=0, search_query=None, sentiment_filter=None):
    """
    🔧 Updated: Supports dynamic limits, offsets, and SQL-level filtering
    for pagination and diagnostic drill-downs.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Build dynamic query
        query = "SELECT * FROM sessions"
        conditions = []
        params = []

        if search_query:
            conditions.append("review_text LIKE ?")
            params.append(f"%{search_query}%")
        
        if sentiment_filter == "Positive":
            conditions.append("session_score > 7")
        elif sentiment_filter == "Neutral":
            conditions.append("session_score BETWEEN 4.5 AND 7")
        elif sentiment_filter == "Negative":
            conditions.append("session_score < 4.5")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += f" ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()

        if df.empty:
            return df

        # Labels for expansion
        log_labels = ['lat_avg', 'glitches', 'crashes', 'nav_time', 'events', 'max_lat', 'friction', 'complexity', 'severity', 'density']
        beh_labels = ['rage_taps', 'page_loops', 'misclicks', 'dead_clicks', 'u_turns']

        def expand_json(row, column, labels):
            vals = json.loads(row[column])
            return pd.Series(vals, index=labels)

        log_df = df.apply(expand_json, column='raw_logs', labels=log_labels, axis=1)
        beh_df = df.apply(expand_json, column='raw_behavior', labels=beh_labels, axis=1)
        
        return pd.concat([df, log_df, beh_df], axis=1)
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()

# --- PAGE CONFIG ---
st.set_page_config(page_title="UX Agent", layout="wide", page_icon="🚀")

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🛠️ Dashboard Controls")
history_depth = st.sidebar.slider("History Depth (Sessions for Analysis)", 50, 500, MAX_DIAGNOSTIC)
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)

if st.sidebar.button('🔄 Force Refresh'):
    st.rerun()

# --- 1. DATA LOADING & OVERVIEW ---
st.title("🚀 UX Agent ")
# Load the primary dataset for Overview & Diagnostics
df_main = load_data(limit=history_depth)

if df_main.empty:
    st.warning("📡 Waiting for autonomous data stream... Ensure log_watcher.py and streamer.py are running.")
    st.stop()

# Calculate Rolling Stats
current_score = df_main['session_score'].iloc[0]
avg_score = df_main['session_score'].mean()
prev_avg = df_main['session_score'].tail(len(df_main)//2).mean()
delta = avg_score - prev_avg

col1, col2, col3 = st.columns(3)
col1.metric("Current Session Score", f"{current_score:.2f}")
col2.metric("Rolling Global UX Score", f"{avg_score:.2f}", delta=f"{delta:.2f}")
col3.metric("System Status", "CRITICAL" if avg_score < 4.5 else "HEALTHY", 
          delta_color="inverse" if avg_score < 4.5 else "normal")

# Trend Chart
st.subheader(f"📈 Overall UX Score Trend (Last {history_depth} Sessions)")
fig = px.line(df_main, x='timestamp', y='session_score', 
              template="plotly_dark")
fig.update_traces(line_color='#00ffcc')
st.plotly_chart(fig, use_container_width=True)

# --- 2. DIAGNOSTIC DRILL-DOWN ---
st.markdown("---")
st.subheader("🔍 Diagnostic Drill-Down")

tab1, tab2, tab3 = st.tabs(["📊 Metric Correlation", "💬 Review Feed", "📉 Technical Spikes"])

with tab1:
    st.write("Identify metrics driving the score down:")
    metrics = ['lat_avg', 'glitches', 'crashes', 'rage_taps', 'dead_clicks']
    corr = df_main[metrics + ['session_score']].corr()['session_score'].drop('session_score').sort_values()
    
    fig_corr = px.bar(corr, orientation='h', 
                      title="Metric Correlation with UX Score",
                      color=corr, color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    # 🔧 Pagination & Search Logic
    col_search, col_filt = st.columns([2, 1])
    with col_search:
        search_q = st.text_input("🔍 Search Reviews by Keyword", "")
    with col_filt:
        sentiment_f = st.selectbox("Filter Sentiment", ["All", "Positive", "Neutral", "Negative"])
    
    # Session state for pagination offset
    if 'page_offset' not in st.session_state:
        st.session_state.page_offset = 0

    # Load paginated data
    df_feed = load_data(limit=DEFAULT_PAGE_SIZE, offset=st.session_state.page_offset, 
                       search_query=search_q, sentiment_filter=sentiment_f)

    if not df_feed.empty:
        display_df = df_feed[['timestamp', 'session_score', 'review_text']].copy()
        display_df['Sentiment'] = display_df['session_score'].apply(
            lambda x: "🟢 Positive" if x > 7 else ("🟡 Neutral" if x > 4.5 else "🔴 Negative")
        )
        st.table(display_df)
        
        # Pagination Buttons
        col_prev, col_page, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("⬅️ Previous") and st.session_state.page_offset >= DEFAULT_PAGE_SIZE:
                st.session_state.page_offset -= DEFAULT_PAGE_SIZE
                st.rerun()
        with col_next:
            if st.button("Next ➡️"):
                st.session_state.page_offset += DEFAULT_PAGE_SIZE
                st.rerun()
    else:
        st.info("No reviews match your filters.")

with tab3:
    st.write("Latency vs. Rage Taps analysis")
    fig_spike = px.scatter(df_main, x='lat_avg', y='rage_taps', color='session_score',
                           size='crashes', hover_data=['review_text'], template="plotly_dark")
    st.plotly_chart(fig_spike, use_container_width=True)

# --- 3. ACTIONABLE INSIGHTS ---
st.markdown("---")
col_ins, col_alt = st.columns(2)

with col_ins:
    st.subheader("💡 Actionable Insights")
    top_issue = corr.idxmin()
    if avg_score < 6.0:
        st.error(f"🚨 **Root Cause:** Most highly correlated with **{top_issue.upper()}**.")
    else:
        st.success("✅ **Insight:** Performance is stable across all tracked metrics.")

with col_alt:
    st.subheader("⚠️ Predictive Alerts")
    last_5 = df_main['session_score'].head(5).values
    if len(last_5) >= 5 and np.all(np.diff(last_5) < 0):
        st.warning("🔥 **ALERT:** Negative 5-session trend detected.")
    else:
        st.info("📡 No immediate threats detected in stream.")

# Auto-refresh
if auto_refresh:
    time.sleep(5)
    st.rerun()