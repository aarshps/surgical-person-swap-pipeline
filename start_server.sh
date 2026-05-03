#!/bin/bash
# Start Stable Diffusion Server
./stable-diffusion.cpp/build/bin/sd-server \
  -m models/dreamshaper_8_q8_0.gguf \
  --listen-ip 127.0.0.1 \
  --listen-port 1234 \
  --threads 2