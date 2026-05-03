# Agent Skills

Detailed capabilities and workflows for the agents powering the Surgical Swap Pipeline.

## Skill: 3D Facial Profiling (Identity Anchor)
- **Description:** Building a robust identity model that is immune to single-photo lighting bias.
- **Workflow:**
  1. Initialize `insightface.app.FaceAnalysis` with the `buffalo_l` model.
  2. Ingest a directory of reference portraits.
  3. Extract the `normed_embedding` for the primary face in each image.
  4. Perform a vector-mean calculation: `avg_emb = np.mean(embeddings, axis=0)`.
  5. Normalize the resulting vector: `avg_emb / np.linalg.norm(avg_emb)`.
  6. Save as a persistent `.npy` file.

## Skill: Surgical Head Extraction & Inpainting
- **Description:** The core logic for isolating the head without losing context.
- **Workflow:**
  1. Detect the 106 2D facial landmarks on the target image.
  2. Calculate the "Head Box" by padding the face bounding box:
     - Top Padding: 100% of face height (to capture hair).
     - Bottom Padding: 40% of face height (to capture neck/jaw).
     - Side Padding: 60% of face width.
  3. Extract the crop at high resolution.
  4. Communicate with the local `sd-server` via the `sdapi/v1/img2img` endpoint.
  5. Use a low `denoising_strength` (0.2 - 0.3) to "bake" textures while preserving the skeletal identity established in Phase 1.

## Skill: Resource-Constrained Server Management
- **Description:** Optimizing AI inference for 8GB RAM CPU-only environments.
- **Workflow:**
  1. Configure `stable-diffusion.cpp` with quantized weights (`q8_0`) to reduce memory footprint.
  2. Set thread count dynamically (typically 2-4) to balance OS stability and rendering speed.
  3. Implement a "Sequential Handoff" where heavy models (InsightFace vs. Stable Diffusion) never reside in memory at the same time.
  4. Monitor process exit codes and trigger automatic server restarts on connection timeouts.

## Skill: Realism Synchronization (Post-Processing)
- **Description:** Final visual blending to eliminate the "AI Paste" effect.
- **Workflow:**
  1. **Color Matching:** Match source face histogram to target neck/body lighting.
  2. **Texture Transfer:** Extract high-frequency noise from target skin via Laplacian filter and overlay it onto the swapped face.
  3. **Seamless Blending:** Use a 31px - 51px Gaussian blurred alpha mask for the surgical reintegration.
  4. **Restoration:** Apply `GFPGAN` to the final composite to sharpen eyes, lips, and pores.
