# Odiyan Setup Guide (Step-by-Step)

This guide provides instructions to replicate the Odiyan production environment from scratch on a Linux-based high-resource system (recommended: 48GB+ RAM, 12+ CPU Cores).

## 1. System Requirements
- OS: Linux (Ubuntu 22.04+ recommended)
- RAM: 48GB+
- CPU: 12 Cores+ (physical cores preferred)
- Disk Space: 50GB+ (for model weights and virtual environment)
- NVIDIA Drivers: Installed (optional, currently using CPU-optimized GGUF/Inference)

## 2. Environment Initialization
```bash
# 1. Clone the repository
git clone <repository-url>
cd hora-odiyan

# 2. Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python Dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Model Dependencies
Flux.1-schnell requires large external model assets. Create the `models/` directory and download the following:

```bash
mkdir -p models

# Download models manually or via scripts/fetch_models.sh
# Source links are defined in README.md under "Model Dependencies"
```

## 4. Build Configuration
Ensure `stable-diffusion.cpp` is built correctly for your architecture:
```bash
cd stable-diffusion.cpp
mkdir build
cd build
cmake -DSD_FLASH_ATTN=ON ..
make -j12
```

## 5. Daemon Configuration
1. **Directories**: Ensure `target_pics/`, `samples/`, `data/`, and `odiyan_refs/` exist.
2. **References**: Place your identity reference photos in `odiyan_refs/`.
3. **Daemon**: Execute `chmod +x start_daemon.sh start_server.sh` and run `./start_daemon.sh`.

## 6. Verification
- Monitor `odiyan_daemon.log`: `tail -f odiyan_daemon.log`
- Check `samples/odiyan_swaps/` for generated output.
- Check system status: `./venv/bin/python status.py`
