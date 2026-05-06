# System Agents

This project utilizes a distributed agent-based model to manage the complexities of the AI inference pipeline. Each agent is responsible for a specific domain of the "Odiyan" transformation.

## 1. Pipeline Orchestrator (Autonomous Daemon)
- **Strategic Goal:** Ensure process continuity, preventing terminal timeouts while processing high-res files in low-resource environments.
- **Actions:** 
  - Runs strictly as a background daemon (`start_daemon.sh`), polling `target_pics/` for new workloads.
  - Manages the sequential hand-off between Python scripts and the C++ SD server.
  - Monitors system memory (RAM) implicitly by isolating processes and prevents foreground execution hang-ups.

## 2. Identity Architect
- **Strategic Goal:** Create a mathematical "Soul" of the target subject.
- **Actions:**
  - Ingests reference portraits from `odiyan_refs/`.
  - Normalizes and averages face embeddings.
  - Manages persistent storage of facial profiles for fast sub-second retrieval.

## 3. High-Fidelity Surgical Inpainter
- **Strategic Goal:** Perform "Head Replacement" at ultra-high 1024x1024 resolutions without altering the environment.
- **Actions:**
  - Calculates the optimal bounding box for head-crops based on 106-point 2D facial landmarks.
  - Communicates with the Stable Diffusion API to "bake" ultra-realistic, 8k resolution textures into the crop.
  - Generates high-frequency masks for feathered re-integration.

## 4. Security & Operational Standards (Session Handoff)
- **Clean Harness Principle**: This repository is strictly the code harness. 
  - **NEVER** commit `data/`, `odiyan_refs/`, `target_pics/`, `samples/`, or `*.npy` files.
  - **Data Hygiene**: Always use `git status` to ensure only source code is tracked. Staging binaries or user data is a security violation.
- **GitHub Identity**: All GitHub-related actions for this repository MUST use the Optacbook GitHub App identity. Do not rely on a personal or global GitHub session for fetches, pushes, issue/PR work, or connector/API actions. Keep this as a repo-level rule for `hora-odiyan`; do not promote it into global machine instructions.
- **Local Machine Execution Ban**: On the `/Users/aps/Source/aarshps/hora-odiyan` checkout, agents MUST NOT run project code, tests, scripts, daemons, inference, downloads, model servers, or repo-local automation. Limit work on this machine to source/doc edits plus git inspection, commit, and direct `main` pushes under the Optacbook identity.
- **Model Engine**: Transitioned to Flux.1-schnell (Bleeding-Edge). All future generation MUST utilize the `Flux` architecture via `--diffusion-model`.
- **Architectural Standards**: Prioritize Frequency Harmonization and Laplacian Blending over Gaussian blurring for seamless integration.
- **Agent Resumption**: Future agents MUST check `ARCHITECTURE.md` and `SKILLS.md`. Ensure that `start_server.sh` is configured to use the Flux weight mapping correctly.
- **Transparency**: If new automation is added, commit ONLY the logic, never the generated assets.
