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

## 5. Session Hand-off & Best Practices
- **Clean Harness Principle**: This repo remains a code harness. NEVER commit `data/`, `odiyan_refs/`, `target_pics/`, `samples/`, or `*.npy` files.
- **Reference Updates**: If a user updates identity photos, download them into `odiyan_refs/` using the CLI methods documented in the session history (do not commit these).
- **Agent Resumption**: Future agents MUST check `ARCHITECTURE.md` and the existing skill definitions in `SKILLS.md`.
- **Transparency**: Always ensure that local automation scripts (like the ones used to download images) are generalized. If an agent adds new scripts, commit only the automation logic, NEVER the data it produces.