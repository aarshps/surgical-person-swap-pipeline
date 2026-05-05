# Odiyan (by Hora) Architecture

This pipeline is designed for ultra-high-fidelity, surgical-grade identity replacement. It operates as an autonomous background daemon, optimized for high-performance CPU execution (48GB RAM, 12 Cores), achieving maximum clarity via 1024x1024+ Stable Diffusion baking.

## Architectural Flow

```mermaid
graph TD
    A[Source Images (odiyan_refs)] -->|Embedding Learning| B(Identity Profile: odiyan.npy)
    C[Target Image (target_pics)] -->|Detection & Swap| D[InsightFace Inswapper]
    D -->|Ultra-Res Baking| E[Stable Diffusion 1024x1024+]
    E -->|Frequency Separation| F[Detail & Grain Matching]
    F -->|Laplacian Blending| G[Final Composite]
    B -->|Identity Lock| D
    G -->|Identity Lock Validation| H[Output File (samples/odiyan_swaps)]
```

## Core Components

- **`Odiyan Daemon (start_daemon.sh)`**: The central autonomous background service. Leverages 48GB RAM to keep all models (InsightFace, SD) memory-resident for instant inference.
- **`InsightFace Inswapper`**: Handles the foundational identity transfer.
- **`Stable Diffusion (1024x1024+)`**: Used to bake highly detailed skin and texture.
- **`Frequency Separation & Grain Matching`**: The "Surgical" core. It extracts high-frequency noise and texture (grain, pores) from the target body and injects it into the AI-generated face to eliminate the "blurry face/sharp body" mismatch.
- **`Laplacian Blending`**: Multi-band blending that integrates images at multiple frequency levels, ensuring a seamless transition without the "soft halo" effect of standard Gaussian masks.

## Workflow

1.  **Identity Learning**: The Daemon calculates the average facial embedding from reference photos.
2.  **Odiyan Daemon Loop**:
    - Polls `target_pics/` for unprocessed images.
    - InsightFace performs an initial swap.
    - The face is surgically cropped and sent to the local SD Server for high-res detail generation.
    - **Frequency Harmonization**: The system analyzes the sharpness (Laplacian variance) and grain profile of the target image and forces the refined face to match these metrics exactly.
    - **Integration**: Laplacian pyramids are used to blend the refined face back into the target.
3.  **Output**: Refined images are exported automatically to `samples/odiyan_swaps/`.