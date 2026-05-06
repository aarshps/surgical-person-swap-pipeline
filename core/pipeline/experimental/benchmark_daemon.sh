#!/bin/bash
# Benchmark Daemon: Run multiple swapping engines in parallel for the same target
# Usage: ./benchmark_daemon.sh <target_image_path>

TARGET=$1
if [ -z "$TARGET" ]; then
    echo "Usage: ./benchmark_daemon.sh <target_image_path>"
    exit 1
fi

echo "Starting benchmark pipeline for $TARGET"

# 1. Standard Pipeline (InsightFace)
./start_daemon.sh --process "$TARGET" --engine "insightface" --output "samples/benchmark/insightface/" &

# 2. Experimental Pipeline (FaceFusion)
./start_daemon.sh --process "$TARGET" --engine "facefusion" --output "samples/benchmark/facefusion/" &

# 3. Experimental Pipeline (DreamID)
./start_daemon.sh --process "$TARGET" --engine "dreamid" --output "samples/benchmark/dreamid/" &

echo "Pipelines launched in parallel."
