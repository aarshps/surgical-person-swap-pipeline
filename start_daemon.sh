#!/bin/bash
# start_daemon.sh
# Starts the Odiyan image processing daemon in the background using nohup.

DAEMON_LOG="odiyan_daemon.log"

echo "Starting Odiyan Daemon..."
nohup ./venv/bin/python core/pipeline/odiyan_processor.py > "$DAEMON_LOG" 2>&1 &
PID=$!
echo "Daemon started with PID: $PID"
echo "Logs are being written to $DAEMON_LOG"
echo "The daemon will automatically process any new images placed in 'target_pics/'."