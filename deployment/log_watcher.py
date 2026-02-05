import os
import time
import json
import yaml
import requests
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 🔧 Load Configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MONITOR_DIR = config['autonomy']['monitor_dir']
ARCHIVE_DIR = os.path.join(MONITOR_DIR, "processed_archive")
API_URL = "http://localhost:8000/predict"

# Ensure directories exist
os.makedirs(MONITOR_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

class UXLogHandler(FileSystemEventHandler):
    """
    Watches for new telemetry files and automatically triggers the UX Agent.
    """
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".json"):
            print(f"📂 New telemetry detected: {os.path.basename(event.src_path)}")
            self.process_file(event.src_path)

    def process_file(self, file_path):
        try:
            # 1. Read the new autonomous data
            with open(file_path, "r") as f:
                data = json.load(f)

            # 2. Automatically send to the UX Agent API
            response = requests.post(API_URL, json=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Autonomous Update Success!")
                print(f"   - Session Score: {result['session_score']:.2f}")
                print(f"   - New Global UX Health: {result['global_ux_score']:.2f}")
                
                # 3. Archive the file so it isn't processed again
                dest_path = os.path.join(ARCHIVE_DIR, os.path.basename(file_path))
                shutil.move(file_path, dest_path)
            else:
                print(f"❌ API Error: {response.text}")

        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")

if __name__ == "__main__":
    event_handler = UXLogHandler()
    observer = Observer()
    observer.schedule(event_handler, MONITOR_DIR, recursive=False)
    
    print(f"🕵️ UX Watcher is active and monitoring: {MONITOR_DIR}")
    print("Drop a .json file there to see the magic happen...")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()