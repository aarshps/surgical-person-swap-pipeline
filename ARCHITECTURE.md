# Odiyan (by Hora) Architecture

This pipeline is designed for ultra-high-fidelity, surgical-grade identity replacement. It operates as an autonomous background daemon, optimizing for local CPU execution while achieving maximum clarity via 1024x1024 Stable Diffusion baking.

## Architectural Flow

```mermaid
graph TD
    A[Source Images (odiyan_refs)] -->|Embedding Learning| B(Identity Profile: odiyan.npy)
    C[Target Image (target_pics)] -->|Detection & Swap| D[InsightFace Inswapper]
    D -->|Ultra-Res Baking| E[Stable Diffusion 1024x1024]
    E -->|Surgical Masking & Blending| F[Final Composite]
    B -->|Identity Lock| D
    F -->|Identity Lock Validation| G[Output File (samples/odiyan_swaps)]
```

## Core Components

- **`Odiyan Daemon (start_daemon.sh)`**: The central autonomous background service. Never run manually in the foreground. Watches `target_pics/` and orchestrates the multi-phase integration pipeline automatically.
- **`InsightFace Inswapper`**: Handles the foundational identity transfer.
- **`Stable Diffusion (1024x1024)`**: Used to bake highly detailed skin, sharp focus, and ultra-realistic texture back into the surgical crop.
- **`Surgical Masking`**: A proprietary landmark-based mask generation technique that ensures zero distortion of the target's original jaw, neck, hair, or surroundings. GFPGAN restoration is explicitly disabled during this phase to preserve the structural integrity and realism of the eyes.

## Workflow

1.  **Identity Learning**: The Daemon calculates the average facial embedding from reference photos in `odiyan_refs/` and saves it as a persistent NumPy array.
2.  **Odiyan Daemon Loop**:
    - Polls `target_pics/` for unprocessed images.
    - InsightFace performs an initial swap to set the identity.
    - The face is surgically cropped and sent to the local SD Server for an ultra-high-resolution (1024x1024, 25 steps, 8.0 cfg) refinement pass.
    - The 1024x1024 refined head is feathered back onto the original body using a dynamic 106-point landmark mask.
3.  **Output**: Refined images are exported automatically to `samples/odiyan_swaps/`.