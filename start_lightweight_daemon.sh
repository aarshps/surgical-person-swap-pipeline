#!/bin/bash
# start_lightweight_daemon.sh - SPRINT v2.1 CPU DAEMON
# Starts the lightweight, ONNX/MediaPipe-based face swap daemon in the background.

PROJECT_ROOT="/root/hora-odiyan"
DAEMON_LOG="$PROJECT_ROOT/lightweight_daemon.log"
PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
PROCESSOR_PY="$PROJECT_ROOT/core/pipeline/lightweight_processor.py"

cd "$PROJECT_ROOT"

# Ensure directories exist
mkdir -p target_pics samples/odiyan_swaps data/failed data/profiles odiyan_refs

# Ensure models are downloaded
if [ ! -f "models/inswapper_128.onnx" ] || [ ! -f "models/face_landmarker.task" ]; then
    echo "Required models missing. Automatically downloading..."
    "$PYTHON_BIN" models/download_models.py
fi

# Cleanup existing logs if they exceed 50MB (prevents disk fill)
if [ -f "$DAEMON_LOG" ] && [ $(stat -c%s "$DAEMON_LOG") -gt 52428800 ]; then
    echo "$(date) - Rotating log file" > "$DAEMON_LOG"
fi

# Kill any existing stray lightweight processes to ensure a clean start
pkill -f "lightweight_processor.py"

echo "--- Starting Lightweight CPU Daemon ---"
echo "Log: $DAEMON_LOG"

# Launch the processor as a watchdog daemon background process
nohup "$PYTHON_BIN" -u "$PROCESSOR_PY" --daemon >> "$DAEMON_LOG" 2>&1 &

PID=$!
echo "Lightweight CPU Daemon started with PID: $PID"
echo "System is active. Drop target images in 'target_pics/' and references in 'odiyan_refs/'."
