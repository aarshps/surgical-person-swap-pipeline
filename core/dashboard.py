from flask import Flask, render_template, jsonify
import os
import psutil
import subprocess

# Configurations
PROJECT_ROOT = "/root/hora-odiyan"
app = Flask(__name__, template_folder=os.path.join(PROJECT_ROOT, 'templates'))
PROCESSOR_SCRIPT = "core/pipeline/odiyan_processor.py"
DAEMON_LOG = os.path.join(PROJECT_ROOT, "odiyan_daemon.log")

def is_running(cmd_part):
    for proc in psutil.process_iter(['cmdline']):
        try:
            if proc.info['cmdline'] and any(cmd_part in s for s in proc.info['cmdline']):
                return proc.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

import json

def get_system_status():
    processor_pid = is_running("odiyan_processor.py")
    sd_server_pid = is_running("sd-server")
    
    # Get last 10 log lines
    logs = []
    if os.path.exists(DAEMON_LOG):
        try:
            output = subprocess.check_output(["tail", "-n", "10", DAEMON_LOG])
            logs = output.decode('utf-8').splitlines()
        except Exception as e:
            logs = [f"Error reading logs: {str(e)}"]

    # Counts
    target_dir = os.path.join(PROJECT_ROOT, "target_pics")
    sample_dir = os.path.join(PROJECT_ROOT, "samples/odiyan_swaps")
    
    targets = len(os.listdir(target_dir)) if os.path.exists(target_dir) else 0
    samples = len(os.listdir(sample_dir)) if os.path.exists(sample_dir) else 0

    # Task Status
    task_status = {"phase": "IDLE", "progress": 0, "details": "Waiting..."}
    status_file = os.path.join(PROJECT_ROOT, "data/task_status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                task_status = json.load(f)
        except: pass

    return {
        "processor": {"status": "RUNNING" if processor_pid else "STOPPED", "pid": processor_pid},
        "sd_server": {"status": "RUNNING" if sd_server_pid else "STOPPED", "pid": sd_server_pid},
        "queue": targets,
        "archive": samples,
        "logs": logs,
        "task": task_status
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    status_data = get_system_status()
    
    # Granular Counts
    target_dir = os.path.join(PROJECT_ROOT, "target_pics")
    sample_dir = os.path.join(PROJECT_ROOT, "samples/odiyan_swaps")
    targets = [f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(target_dir) else []
    samples = [f for f in os.listdir(sample_dir) if f.startswith("surgical_")] if os.path.exists(sample_dir) else []
    
    pending_count = len(targets) - len(samples)
    status_data["counts"] = {
        "pending": pending_count,
        "in_progress": 1 if status_data["processor"]["status"] == "RUNNING" and (pending_count > 0) else 0,
        "completed": len(samples)
    }
    
    # Estimated duration for the current phase (in seconds)
    # Phase 2 (Flux 1024x1024) is the bottleneck (~1080s)
    phase_durations = {
        "Phase 1: Anchoring Identity": 30,
        "Phase 2: Flux Refinement": 1080, # 18 mins
        "Phase 3: Harmonization": 15,
        "Phase 4: Integration": 20
    }
    
    current_phase = status_data.get("task", {}).get("phase", "IDLE")
    status_data["task"]["estimated_duration"] = phase_durations.get(current_phase, 60)
    
    # Total ETA (Pending count * ~20 mins)
    total_eta_seconds = pending_count * 1200 
    mins, secs = divmod(total_eta_seconds, 60)
    hours, mins = divmod(mins, 60)
    status_data["eta"] = f"{hours:02d}:{mins:02d}:{secs:02d}"
    
    return jsonify(status_data)

if __name__ == '__main__':
    # Listen on all interfaces so it's accessible externally
    app.run(host='0.0.0.0', port=5001)
