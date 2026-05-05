#!/bin/bash
# start_server.sh - Flux.1-schnell Engine (High Fidelity)
./stable-diffusion.cpp/build/bin/sd-server \
  --diffusion-model models/flux1-schnell-q8_0.gguf \
  --vae models/ae.safetensors \
  --clip_l models/clip_l.safetensors \
  --t5xxl models/t5xxl_fp16.safetensors \
  --listen-ip 127.0.0.1 \
  --listen-port 1234 \
  --threads 12 \
  --steps 4 \
  --cfg-scale 1.0 \
  --sampling-method euler
