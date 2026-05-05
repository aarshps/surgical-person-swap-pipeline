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

## Developer Standards
- **Clean Harness**: Never commit user data (`data/`, `target_pics/`, `odiyan_refs/`).
- **Data Hygiene**: Use `.gitignore` to maintain a sterile harness.
- **Security**: This repository is a harness and MUST NOT contain sensitive user content.
