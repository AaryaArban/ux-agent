@echo off
title UX Agent Master Control
echo 🚀 Launching UX Agent Autonomous Ecosystem...

:: --- TERMINAL 1: THE BRAIN (API) ---
echo 🧠 Starting Core API...
start cmd /k "set PYTHONPATH=%%PYTHONPATH%%;. && uvicorn api.ux_agent_api:app --host 0.0.0.0 --port 8000 --reload"

:: Wait a few seconds for the API to initialize before starting the others
timeout /t 5 /nobreak > nul

:: --- TERMINAL 2: THE EYES (WATCHER) ---
echo 🕵️ Starting Log Watcher...
start cmd /k "python deployment/log_watcher.py"

:: --- TERMINAL 3: THE USER (STREAMER) ---
echo 📡 Starting Autonomous User Streamer...
start cmd /k "python scripts/autonomous_streamer.py"

:: --- TERMINAL 4: THE VIEW (DASHBOARD) ---
echo 📈 Starting Command Center Dashboard...
start cmd /k "streamlit run dashboard/app.py"

echo.
echo ✅ All systems are live! 
echo Check your browser at http://localhost:8501 for the Dashboard.
echo ------------------------------------------------------------
pause