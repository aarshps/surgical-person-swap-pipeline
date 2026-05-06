# Gemini Agent Instructions for Odiyan (by Hora)

This file contains the core context, architectural constraints, and operational guidelines for any future AI agent interacting with this repository. It serves as the permanent tracker and handoff document.

## 1. Branding & Naming
- **Application Name**: Odiyan
- **Organization Name**: Hora
- **Context**: Never refer to the project using legacy names (e.g., "Surgical Person Swap", "Aya", or "Mrunal"). The term "surgical" is only used as an adjective for technical precision (e.g., "surgical crop", "surgical integration").

## 2. Core Architecture
- The pipeline is an autonomous background daemon, operating in a high-resource CPU environment (48GB RAM, 12 Cores).
- **Daemon Execution**: NEVER run the processing scripts directly from the CLI. Always use the background daemon script (`start_daemon.sh`) which polls for new target images.
- **Phase 1 (Anchor)**: Identity anchoring using InsightFace.
- **Phase 2 (Bake)**: Ultra-high-resolution (1024x1024+) texture baking via local Stable Diffusion (`stable-diffusion.cpp` img2img). Flux.1-schnell (Q8_0 GGUF) is the mandated engine for its superior identity retention. Inference is CPU-based (12 cores, ~20 mins per image).
- **Phase 3 (Integrate)**: Frequency harmonization and Laplacian blending. Grain matching from the target body is mandatory to avoid "sharp body/blurry face" mismatch.
- **Dashboard**: Real-time monitoring web interface (Flask) running on port 5001. Features cubic-bezier animations and a live ETA timer.
- Detailed flow is maintained in `ARCHITECTURE.md`.

## 3. Current Critical Challenges (Handoff Note)
- **Active Task**: See [GitHub Issue #1](https://github.com/aarshps/hora-odiyan/issues/1) for a detailed breakdown of current quality issues.
- **Contrast Issue**: Latest Flux outputs are reported as "too contrasty". Prompts must favor soft lighting and balanced dynamics.
- **Neck Seams**: Persistent visible lines around the neck area. Blending masks need refinement (better landmark adherence, non-linear feathering).
- **Identity Fidelity**: Continuous refinement of the "Aya" profile is needed. Current logic uses weighted averaging of 112+ frames with outlier removal.

## 3. Strict Security & Data Policies
- **Clean Harness Principle**: This repository is strictly the code harness.
- **NEVER** commit user data, target images, reference images, or generated numpy profiles (`.npy`).
- The `.gitignore` must aggressively exclude data directories: `data/references/`, `data/targets/`, `data/profiles/`, `odiyan_refs/`, `target_pics/`, `samples/`, and model weights (`*.gguf`, `*.onnx`, `*.pth`).

## 4. Agent Skills & Documentation Standard
- Agent behavioral logic is defined in `AGENTS.md` and `SKILLS.md`.
- **Constraint**: Every individual skill defined in `SKILLS.md` MUST remain under 100 lines of text. Do not bloat skill definitions. Keep them actionable and concise.
- When updating workflows, ensure corresponding documentation (`README.md`, `ARCHITECTURE.md`) is accurately synchronized.

## 5. Development Guidelines
- Utilize the 48GB RAM to keep models resident (avoid sequential unloading).
- Parallelize independent processing tasks across available CPU cores.
- Rely on vanilla Python OpenCV (`cv2`) and NumPy (`numpy`) for image compositing. Avoid heavy image processing frameworks if simple matrix operations suffice.
