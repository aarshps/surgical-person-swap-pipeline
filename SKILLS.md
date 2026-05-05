# Odiyan System Skills

These skills define the agentic logic for the `OdiyanOrchestrator` agent. Keep all skills granular, focused, and under 100 lines.

## Skill: Autonomous High-Res Odiyan Daemon
- **Objective:** Run the pipeline as a background service, swapping heads continuously while maintaining 1024x1024+ high-fidelity texture and preserving eye integrity.
- **Agent Instructions:**
  1. Never run `odiyan_processor.py` directly in the foreground. Always use `./start_daemon.sh` or run it via `nohup` in the background.
  2. The daemon watches `target_pics/` continuously. When a new file is detected, it processes it and outputs to `samples/odiyan_swaps/`.
  3. **Anchor Phase**: Load identity from `odiyan_refs/`, generate base swap, and match colors.
  4. **Baking Phase**: Start SD Server if needed. Resize head-crop to 1024x1024. Use the Flux.1-schnell configuration (4 steps, 1.0 CFG, Euler sampler) with identity-retention prompts.
  5. **Integration Phase**: Use Frequency Harmonization to match the target body's grain/noise profile, then dynamically mask and use Laplacian Pyramids to seamlessly blend the crop back to the original body.

## Skill: High-Resource Inference
- **Objective:** Maintain stability and maximize performance on the 48GB RAM, 12 Core system.
- **Agent Instructions:**
  1. Utilize the `Flux.1-schnell` model with Q8_0 quantization alongside T5XXL and CLIP-L encoders.
  2. Ensure the `stable-diffusion.cpp` server is booted with the `--diffusion-model` flag and utilizes all available CPU cores (`--threads 12`).
  3. Models are kept memory-resident to leverage the 48GB RAM for high-speed continuous processing.

## Skill: Identity Profiling
- **Objective:** Maintain perfect subject likeness across arbitrary angles.
- **Agent Instructions:**
  1. Load references from `odiyan_refs/`.
  2. Extract normed embeddings and calculate the vector mean.
  3. Store result persistently in `data/profiles/odiyan.npy`. Bypass redundant learning if the profile already exists.

## Skill: Environment & Dependency Management
- **Objective:** Ensure the Python virtual environment correctly supports the complex AI model matrix (InsightFace, Torch, ONNXRuntime).
- **Agent Instructions:**
  1. **Strict Dependencies:** Ensure `requirements.txt` is strictly followed. The stack is incompatible with NumPy 2.x. Ensure `numpy<2` (e.g., 1.26.4).
  2. **Missing Models:** Ensure `inswapper_128.onnx` is present in the project root. If missing, download it via `wget -O inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx`.
  3. **System GL Libs:** Ensure `libgl1` and `libglib2.0-0` are installed on the host OS for OpenCV.
  4. Always verify `start_daemon.sh` points explicitly to the `venv` python path to prevent global system conflicts.