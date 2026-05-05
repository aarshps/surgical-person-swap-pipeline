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

## 4. Texture Validator
- **Strategic Goal:** Maintain eye structural integrity and break the "AI look".
- **Actions:**
  - Implements Laplacian pyramid blending for grain matching.
  - Disables over-aggressive GFPGAN facial restoration, preferring the natural 1024x1024 SD output to preserve perfect eye reflections.
  - Re-anchors the identity using InsightFace on the final composite.