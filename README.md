# Odiyan (by Hora)

A professional-grade, autonomous AI pipeline for identity replacement.

## ⚠️ Critical Security Policy (Clean Harness Principle)
This repository is strictly a **CODE HARNESS**.
- **NEVER** commit user data, target images, reference images, or generated profiles (`.npy`).
- The `.gitignore` is configured to ignore all data and output folders.
- Future agents MUST NOT deviate from this policy.

## 🚀 Architecture
The pipeline now operates as an **Autonomous Background Daemon**.
- **Daemon Execution**: Use `./start_daemon.sh` (which runs `odiyan_processor.py` in the background).
- **Polling**: The daemon continuously watches `target_pics/` for new images.
- **Processing**: Automatically transforms targets into surgical swaps at 768x768 resolution using the learned identity profile.
- **Output**: Results appear in `samples/odiyan_swaps/`.

## 📖 Usage
1. **Train Identity**: Place reference photos in `odiyan_refs/`.
2. **Start Daemon**: Run `./start_daemon.sh`.
3. **Queue Targets**: Drop target photos into `target_pics/`.
4. **Check Status**: Use `./venv/bin/python status.py` for a quick snapshot.

## 📜 Documentation
- [System Agents](AGENTS.md)
- [Agent Skills](SKILLS.md)
- [Architecture Details](ARCHITECTURE.md)
- [Core Directives](GEMINI.md)
