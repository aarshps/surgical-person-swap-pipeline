# README - Odiyan (by Hora)

## Overview
Odiyan is an autonomous AI pipeline for high-fidelity identity replacement. It utilizes Flux.1-schnell (Bleeding-Edge) for texture generation and Laplacian multi-band blending for surgical integration.

## Quick Start
1. **Ensure Daemon is running**: `./start_daemon.sh`
2. **Drop Targets**: Place images in `target_pics/`.
3. **Collect**: Swaps appear in `samples/odiyan_swaps/`.

## Architecture (Flux.1)
- **Identity Learning**: Learns features from `odiyan_refs/`.
- **Refinement**: Flux.1-schnell (via `--diffusion-model`) produces hyper-realistic 1024x1024+ textures.
- **Blending**: Frequency Harmonization and Laplacian Pyramids ensure zero sharpness mismatch between face and body.

## Model Dependencies (External Assets)
The pipeline requires the following external model assets to be placed in `models/`. These are not committed to source control for security and size reasons.

| Asset | Source | Expected Filename | Purpose |
| :--- | :--- | :--- | :--- |
| **Flux.1-schnell** | [city96/FLUX.1-schnell-gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf) | `flux1-schnell-q8_0.gguf` | Diffusion Model (Q8_0) |
| **T5XXL Encoder** | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) | `t5xxl_fp16.safetensors` | Text Encoding |
| **CLIP-L Encoder** | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) | `clip_l.safetensors` | Text Encoding |
| **VAE** | [flux-safetensors/flux-safetensors](https://huggingface.co/flux-safetensors/flux-safetensors) | `ae.safetensors` | Image Decoder |

## Developer Standards
- **Clean Harness**: Never commit user data (`data/`, `target_pics/`, `odiyan_refs/`).
- **Data Hygiene**: Use `.gitignore` to maintain a sterile harness.
- **Security**: This repository is a harness and MUST NOT contain sensitive user content.
