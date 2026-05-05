#!/bin/bash
# start_daemon.sh - PRODUCTION HARDENED
# Starts the Odiyan image processing daemon in the background and ensures uptime.

PROJECT_ROOT="/root/hora-odiyan"
DAEMON_LOG="$PROJECT_ROOT/odiyan_daemon.log"
PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
PROCESSOR_PY="$PROJECT_ROOT/core/pipeline/odiyan_processor.py"

cd "$PROJECT_ROOT"

# Ensure directories exist
mkdir -p target_pics samples/odiyan_swaps data/failed data/profiles

# Cleanup existing logs if they exceed 50MB (prevents disk fill)
if [ -f "$DAEMON_LOG" ] && [ $(stat -c%s "$DAEMON_LOG") -gt 52428800 ]; then
    echo "$(date) - Rotating log file" > "$DAEMON_LOG"
fi

# Kill any existing stray processes to ensure a clean start
pkill -f "odiyan_processor.py"
# We don't pkill sd-server here as it might be loading models (48GB take time)

echo "--- Starting Odiyan Production Daemon ---"
echo "Log: $DAEMON_LOG"

# Launch the processor as a headless background process
nohup "$PYTHON_BIN" -u "$PROCESSOR_PY" >> "$DAEMON_LOG" 2>&1 &

PID=$!
echo "Daemon started with PID: $PID"
echo "System is now autonomous. Simply drop images in 'target_pics/' and collect from 'samples/odiyan_swaps/'."
