# Surgical Person Swap Pipeline

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

A professional-grade, surgical AI pipeline for high-fidelity person swapping. Unlike traditional face swappers that replace a simple facial mask, this pipeline "bakes" the identity, hair, and skin textures into the target image while preserving 100% of the original background and body resolution.

## 🚀 Key Features

- **Surgical Head Inpainting:** Precisely targets the head and hair area using landmark detection, preventing background distortion and resolution loss.
- **Identity Lock (50+ Image Profiling):** Builds a robust 3D facial profile by mathematically averaging embeddings from a large dataset of the source subject.
- **Sequential Execution:** Memory-optimized workflow designed to run on 8GB RAM CPU environments by isolating high-load AI models.
- **Texture Synchronization:** Uses Laplacian grain transfer and adaptive histogram matching to ensure facial skin feels "one" with the target body.
- **Multi-Phase Pipeline:** Established workflow covering identity anchoring, high-res baking, and final restoration.

## 🛠 Architecture

The pipeline is split into three distinct phases for maximum stability and quality. For a detailed breakdown of the internal surgical integration and orchestration, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aarshps/surgical-person-swap-pipeline.git
   cd surgical-person-swap-pipeline
   ```

2. **Initialize Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Model Placement:**
   Ensure models are placed in the `models/` directory (ignored by git):
   - `dreamshaper_8_q8_0.gguf`
   - `inswapper_128.onnx`

## 📖 Usage

1. **Train the Profile:** Drop 20-50 photos of your subject into `aya_refs/`.
2. **Set Target:** Place the image you want to swap into `target_pics/`.
3. **Execute:**
   ```bash
   ./venv/bin/python aya_processor.py
   ```
4. **Results:** Found in `samples/aya_swaps/`.

## 📜 Documentation
- [System Agents](AGENTS.md)
- [Agent Skills](SKILLS.md)
