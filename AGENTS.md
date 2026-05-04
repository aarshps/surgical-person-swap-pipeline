# System Agents

This project utilizes a distributed agent-based model to manage the complexities of the AI inference pipeline. Each agent is responsible for a specific domain of the "Odiyan" transformation.

## 1. Pipeline Orchestrator
- **Strategic Goal:** Ensure process continuity in low-resource environments.
- **Actions:** 
  - Manages the sequential hand-off between Python scripts and the C++ SD server.
  - Monitors system memory (RAM) and triggers process kills to prevent kernel panic or OOM freezes.
  - Handles the temporary storage of intermediate latents and image crops.

## 2. Identity Architect
- **Strategic Goal:** Create a mathematical "Soul" of the target subject.
- **Actions:**
  - Scrapes or ingests large datasets of reference portraits.
  - Normalizes and averages face embeddings to eliminate "outlier" facial expressions or lighting.
  - Manages the persistent storage of `.npy` facial profiles for sub-second retrieval.

## 3. Surgical Inpainter
- **Strategic Goal:** Perform "Head Replacement" without altering the environment.
- **Actions:**
  - Calculates the optimal bounding box for head-crops based on 106-point 2D facial landmarks.
  - Communicates with the Stable Diffusion API to "bake" textures into the crop.
  - Generates high-frequency masks for feathered re-integration.

## 4. Texture Validator
- **Strategic Goal:** Break the "AI look" through realism synchronization.
- **Actions:**
  - Implements Laplacian pyramid blending for grain matching.
  - Performs Contrast Limited Adaptive Histogram Equalization (CLAHE) for lighting depth.
  - Runs final GFPGAN restoration to bring back high-res details like pores and eye glints.
