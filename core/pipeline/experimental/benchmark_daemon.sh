#!/bin/bash
# Benchmark Daemon: Run multiple swapping engines in parallel
# Usage: ./benchmark_daemon.sh <target_image_path>

TARGET=$1
if [ -z "$TARGET" ]; then
    echo "Usage: ./benchmark_daemon.sh <target_image_path>"
    exit 1
fi

PYTHON_BIN="/root/hora-odiyan/venv/bin/python"
PROCESSOR_PY="/root/hora-odiyan/core/pipeline/odiyan_processor.py"

echo "Starting benchmark pipeline for $TARGET"

# 1. Standard Pipeline (InsightFace)
$PYTHON_BIN $PROCESSOR_PY --engine insightface --output "samples/benchmark/insightface/" &

# 2. Experimental Pipeline (FaceFusion)
$PYTHON_BIN $PROCESSOR_PY --engine facefusion --output "samples/benchmark/facefusion/" &

# 3. Experimental Pipeline (DreamID)
$PYTHON_BIN $PROCESSOR_PY --engine dreamid --output "samples/benchmark/dreamid/" &

echo "Pipelines launched in parallel."
