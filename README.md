# Odiyan (by Hora) — Autonomous High-Fidelity Identity Transfer Pipeline

Odiyan is an autonomous, high-fidelity identity transfer and face replacement platform. The repository is structured into a multi-tier architecture offering both high-end Stable Diffusion (Flux) texture-baking pipelines and an ultra-fast, pure-CPU lightweight pipeline (Sprint v2.1 Certified).

---

## ⚠️ Important Quality & Performance Notice

The existing legacy methods in this repository did **NOT** successfully generate swapped faces or personas to the required quality expectations. Through comprehensive testing, the following bottlenecks were identified in the legacy engines:
1. **InsightFaceEngine / FaceFusionEngine:** Crude statistical channel-wise BGR color scaling resulted in major skin tone hue rotations (shifting face skin towards yellow or green under mixed lighting). Furthermore, standard rigid square (128x128) warp masking caused severe boundary pixel bleed, leaving harsh rectangular seams on target frames.
2. **DreamIDEngine:** Remained a mocked, incomplete implementation.

To resolve these issues, the codebase has been hardened with the **Sprint v2.1 Certified Lightweight InSwapper Pipeline**, integrated as a first-class modular engine (`LightweightInSwapperEngine`). This engine resolves these quality errors via perceptually uniform **CIE LAB Reinhard color transfer** and inward-feathered **C1-continuous smoothstep oval masking**, achieving bit-perfect background preservation and seamless color matching.

---

## Repository Structure

```
hora-odiyan/
├── core/
│   ├── pipeline/
│   │   ├── swap_engines.py           # Face swap engine drivers (includes LightweightInSwapperEngine)
│   │   ├── lightweight_processor.py   # Standing CPU-only v2.1 pipeline and daemon watcher
│   │   ├── odiyan_processor.py       # Stable Diffusion (Flux) heavy pipeline daemon
│   │   └── ... (modular phases)
│   ├── validators/                   # Full Quantitative Verification Suite
│   │   ├── validator.py              # Master validator runner
│   │   ├── likeness_validator.py     # Cosine identity similarity
│   │   ├── blending_validator.py     # Boundary transition gradients
│   │   ├── spatial_integrity_validator.py # Bit-perfect background checker
│   │   ├── specularity_validator.py  # Face-to-neck luminance and specularity
│   │   └── realism_validator.py      # Laplacian texture variance and sharpness
│   └── ...
├── models/                           # Model assets and automatic downloader
│   ├── download_models.py            # Automated downloader utility
│   └── README.md                     # Model storage specification
├── start_daemon.sh                   # Launches the heavy GPU/Flux daemon
├── start_lightweight_daemon.sh       # Launches the lightweight, pure-CPU Sprint v2.1 daemon
├── test_lightweight_pipeline.py      # Push-button integrated verification script
├── requirements.txt                  # Python dependencies
└── README.md                         # Platform documentation
```

---

## Quick Start (Lightweight CPU Pipeline)

### 1. Installation
Ensure dependencies are installed and fetch the required ONNX/MediaPipe model files:
```bash
pip install -r requirements.txt
python models/download_models.py
```

### 2. Run Verification Harness
Verify that your local environment passes all quantitative validator suites (Likeness, Blending, Spatial, Specularity, and Realism):
```bash
python test_lightweight_pipeline.py
```

### 3. Start the Background Daemon
To run the lightweight CPU daemon which watches for targets:
```bash
./start_lightweight_daemon.sh
```
- Drop target images into `target_pics/`.
- Add source references in `odiyan_refs/`.
- Collect finished outputs in `samples/odiyan_swaps/`.

---

## Quick Start (Heavy Flux Pipeline)

For high-resolution detail baking using the Stable Diffusion Flux backend:
1. **Ensure the SD Server is running**:
   ```bash
   ./start_server.sh
   ```
2. **Launch the Heavy Daemon**:
   ```bash
   ./start_daemon.sh
   ```
3. **Usage**:
   - Place target images in `target_pics/`.
   - Place references in `odiyan_refs/`.
   - Swaps appear in `samples/odiyan_swaps/`.

---

## Quantitative Quality Standards (Sprint v2.1)

All certified production outputs must meet the following strict validation checks enforced by the validator suite under `core/validators/`:

| Metric | Threshold | Purpose |
| :--- | :--- | :--- |
| **Likeness Similarity** | Average Cosine Sim > 0.40, Max > 0.50 | Ensures strong resemblance to the reference identity. |
| **Spatial Integrity** | Background Pixels Altered < 0.05% | Guarantees bit-perfect, zero-bleed background preservation. |
| **Blending Seamlessness** | Border Gradient Ratio < 2.20 | Ensures seamless, transition-free boundaries. |
| **Specularity** | Face-to-Neck Luminance Ratio < 1.15 | Prevents waxy/shiny face highlights relative to neck. |

---

## Developer Standards
- **Clean Harness**: Never commit large model files (`*.onnx`, `*.pth`, `*.gguf`) or user references/targets.
- **Git Discipline**: Ensure new assets and directories are cataloged in `.gitignore`.
