# Odiyan System Skills

These skills define the agentic logic for the `OdiyanOrchestrator` agent.

## Skill: Odiyan (State-Aware)
- **Objective:** Swap a subject's head onto a target while maintaining 100% background fidelity.
- **Agent Instructions:**
  1. Initialize `InsightFace` (buffalo_l) and `GFPGAN`.
  2. Load cached facial profile from `data/profiles/`.
  3. Execute **Anchor Phase**: Generate base swap, perform histogram color matching, and calculate square head-crop metadata.
  4. Execute **Baking Phase**: Start `stable-diffusion.cpp`, resize head-crop to 512x512, perform img2img (10 steps, 0.25 denoising), and save head crop.
  5. Execute **Integration Phase**: Re-pad the crop to original resolution, generate dynamic elliptical mask to hide chest/neck artifacts, and overlay refined head.
  6. Execute **Identity Lock**: Re-run GFPGAN restoration (inner-only mask) and apply Laplacian grain overlay for seamless texture.

## Skill: Memory-Efficient Inference
- **Objective:** Maintain stability on limited (8GB) RAM.
- **Agent Instructions:**
  1. Use quantized GGUF models (`dreamshaper_8_q8_0.gguf`).
  2. Implement sequential hand-off: Free GPU/CPU memory (via process cleanup) between InsightFace, Stable Diffusion, and GFPGAN tasks.
  3. Use isolated subprocess calls for external binary tasks (SD server) to prevent heap fragmentation.
  4. Monitor `dmesg` and process logs for OOM signals and implement exponential backoff on retry.

## Skill: Identity Profiling
- **Objective:** Maintain perfect subject likeness across arbitrary angles.
- **Agent Instructions:**
  1. Ingest large reference datasets (50+ images).
  2. Extract normed embeddings for all detected faces.
  3. Calculate vector mean: `avg_emb = mean(embeddings)`.
  4. Store result as a persistent `.npy` file.
  5. On subsequent runs, bypass inference entirely by loading the cached profile.
