# Odiyan System Skills

These skills define the agentic logic for the `OdiyanOrchestrator` agent. Keep all skills granular, focused, and under 100 lines.

## Skill: Autonomous High-Res Odiyan Daemon
- **Objective:** Run the pipeline as a background service, swapping heads continuously while maintaining 1024x1024 high-fidelity texture and preserving eye integrity.
- **Agent Instructions:**
  1. Never run `odiyan_processor.py` directly in the foreground. Always use `./start_daemon.sh` or run it via `nohup` in the background.
  2. The daemon watches `target_pics/` continuously. When a new file is detected, it processes it and outputs to `samples/odiyan_swaps/`.
  3. **Anchor Phase**: Load identity from `odiyan_refs/`, generate base swap, and calculate metadata.
  4. **Baking Phase**: Start SD Server. Resize head-crop to 1024x1024. Use 25 steps, 8.0 cfg, and "ultra-realistic, 8k resolution, highly detailed skin, sharp focus" prompts.
  5. **Integration Phase**: Dynamically mask and seamlessly blend the 1024x1024 crop back to the original body.
  6. **Identity Lock**: Re-run identity anchoring if needed, but **skip GFPGAN restoration** to preserve the natural structural integrity of the eyes. Rely on high-res SD baking for clarity.

## Skill: Memory-Efficient Inference
- **Objective:** Maintain stability on limited (8GB) RAM.
- **Agent Instructions:**
  1. Use quantized GGUF models (`dreamshaper_8_q8_0.gguf`).
  2. Ensure the SD server is isolated from the main Python thread.
  3. Allow the background daemon to handle heavy loads, preventing terminal timeouts.

## Skill: Identity Profiling
- **Objective:** Maintain perfect subject likeness across arbitrary angles.
- **Agent Instructions:**
  1. Load references from `odiyan_refs/`.
  2. Extract normed embeddings and calculate the vector mean.
  3. Store result persistently. Bypass redundant learning if profile exists.

## Skill: Environment & Dependency Management
- **Objective:** Ensure the Python virtual environment correctly supports the complex AI model matrix (GFPGAN, InsightFace, Torch).
- **Agent Instructions:**
  1. **Strict Dependencies:** The stack is strictly incompatible with NumPy 2.x. Ensure `numpy<2` (e.g. 1.26.4). OpenCV (`opencv-python` and `opencv-python-headless`) must be `<4.9`, and `tifffile` must be `<2024`.
  2. **Basicsr Patch:** GFPGAN's `basicsr` dependency breaks on newer `torchvision`. You MUST patch it: `sed -i 's/functional_tensor/functional/' venv/lib/python*/site-packages/basicsr/data/degradations.py`.
  3. **Missing Models:** Ensure `inswapper_128.onnx` is present in the project root. If missing, download it via `wget -O inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx`.
  4. **System GL Libs:** Ensure `libgl1` and `libglib2.0-0` are installed on the host OS.
  5. Always verify `start_daemon.sh` points explicitly to the `venv` python path to prevent global system conflicts.