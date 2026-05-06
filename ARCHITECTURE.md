# Odiyan (by Hora) Architecture

This pipeline is designed for ultra-high-fidelity, surgical-grade identity replacement. It operates as an autonomous background daemon, optimized for high-performance CPU execution (48GB RAM, 12 Cores), achieving maximum clarity via 1024x1024+ Stable Diffusion baking.

## Parallel Benchmarking Architecture

To improve face-swapping fidelity, we are transitioning to a multi-engine benchmark harness.

- **`core/pipeline/experimental/benchmark_daemon.sh`**: The orchestrator for parallel execution. It routes the same input target to multiple swapping engines.
- **Engine-Agnostic Interface**: The pipeline will be refactored to support:
  - `insightface`: The current baseline.
  - `facefusion`: Modular, high-fidelity swap and enhancement.
  - `dreamid`: Diffusion Transformer-based identity preservation.

- **Benchmarking Workflow**:
  1. Drop target in `target_pics/`.
  2. Invoke `benchmark_daemon.sh <target>`.
  3. Analyze outputs in `samples/benchmark/<engine>/` to compare fidelity, lighting, and seams.


## Core Components

- **`Odiyan Daemon (start_daemon.sh)`**: The central autonomous background service. Leverages 48GB RAM to keep all models (InsightFace, Flux.1) memory-resident for instant inference.
- **`InsightFace Inswapper`**: Handles the foundational identity transfer.
- **`Flux.1-schnell (1024x1024)`**: The bleeding-edge replacement for Stable Diffusion. Used to bake hyper-realistic skin and texture without the artificial contrast and artifacts of older models. With 48GB RAM, we utilize the Q8_0 quantization for maximum fidelity.
- **`Frequency Harmonization & Grain Matching`**: The "Surgical" core. It extracts high-frequency noise and texture (grain, pores) from the target body and injects it into the Flux-generated face to eliminate the "blurry face/sharp body" mismatch.
- **`Laplacian Blending`**: Multi-band blending that integrates images at multiple frequency levels, ensuring a seamless transition.

## Workflow

1.  **Identity Learning**: The Daemon calculates the average facial embedding from reference photos.
2.  **Odiyan Daemon Loop**:
    - Polls `target_pics/` for unprocessed images.
    - InsightFace performs an initial swap to set the base identity.
    - The face is surgically cropped and sent to the **Flux Engine** for high-res detail generation (4 steps, 1.0 CFG, Euler sampler).
    - **Frequency Harmonization**: The system analyzes the sharpness and grain profile of the target image and forces the Flux output to match these metrics.
    - **Integration**: Laplacian pyramids are used to blend the refined face back into the target.
3.  **Output**: Refined images are exported automatically to `samples/odiyan_swaps/`.