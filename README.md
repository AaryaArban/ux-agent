# 🚀 UX Agent

The **UX Agent** is an autonomous machine learning system designed to monitor, analyze, and predict User Experience (UX) quality in real-time. By fusing technical logs, user behavioral patterns, and natural language reviews, the agent provides a holistic "Global UX Score" and offers deep-dive diagnostic insights into system performance.

---

## 🛠️ Tech Stack & Dependencies

### **Core Technologies**

* **Intelligence**: PyTorch (DistilBERT for NLP, Multi-Layer Perceptrons for Log/Behavior Encoders).
* **API Layer**: FastAPI & Uvicorn.
* **Visuals**: Streamlit & Plotly.
* **Persistence**: SQLite3 (Historical Session Memory).
* **Automation**: Watchdog (Event-driven file monitoring).

### **Installation**

Ensure you have Python 3.10+ installed. Run the following command to install all required libraries:

```bash
pip install fastapi uvicorn watchdog requests scikit-learn joblib pyyaml transformers torch streamlit plotly pandas numpy

```

---

## 📂 Project Structure

```text
BE-MPR/
├── api/
│   └── ux_agent_api.py      # The "Brain" (Inference & Database Management)
├── dashboard/
│   └── app.py               # The "View" (Real-time Command Center)
├── deployment/
│   └── log_watcher.py       # The "Eyes" (Monitors incoming telemetry)
├── scripts/
│   └── autonomous_streamer.py # The "User" (Simulates continuous traffic)
├── models/                  # Trained weights (.pt) and Scalers (.pkl)
├── data/
│   ├── ux_history.db        # Persistent SQL database
│   └── incoming_telemetry/  # Buffer for real-time JSON logs
└── config.yaml              # Global project configuration

```

---

## ⚙️ Execution Steps

### **Phase 1: Preparation**

Before launching the autonomous loop, you must prepare the model:

1. **Generate Data**: `python synthetic_data_generator.py` (Creates amplified training signals).
2. **Train Model**: `python -m training.train_ux_agent` (Saves weights and initializes DB).

### **Phase 2: Launching the Ecosystem**

You can use the provided `run_ux_agent.bat` file or open four terminals manually:

1. **Terminal 1 (API)**:
`uvicorn api.ux_agent_api:app --reload`
2. **Terminal 2 (Watcher)**:
`python deployment/log_watcher.py`
3. **Terminal 3 (Streamer)**:
`python scripts/autonomous_streamer.py`
4. **Terminal 4 (Dashboard)**:
`streamlit run dashboard/app.py`

---

## 🧠 Detailed Working of the UX Agent

### **1. The Data Fusion Process**

The agent processes three distinct data streams simultaneously:

* **Log Stream**: 10 technical metrics (Latency, Crashes, Load times).
* **Behavior Stream**: 5 patterns (Rage clicks, Page loops, Misclicks).
* **Sentiment Stream**: Raw user review text analyzed via DistilBERT.

### **2. Autonomous Memory & Scoring**

* **EMA Scoring**: Every new session updates a "Global UX Score" using an Exponential Moving Average (EMA). This prevents the score from jumping erratically while still reacting to trends.
* **Decay Factor**: A `0.1` decay factor ensures the system retains a "memory" of previous performance while prioritizing recent data.

### **3. Diagnostic Drill-Down & Alerting**

* **Metric Correlation**: The system mathematically calculates which technical metric has the strongest negative correlation with the current UX score.
* **Predictive Alerts**: If the system detects five consecutive sessions with declining scores, it triggers a **Predictive Alert** in the dashboard to warn developers before a system failure occurs.
* **Pagination & Filter**: The dashboard supports SQL-level pagination and sentiment filtering, allowing developers to audit thousands of historical reviews without performance lag.

---
