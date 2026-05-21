# Archive: image_pipeline v2.1 (Local Development Workspace)

This directory contains the complete source code snapshot from the local `image_pipeline/` development workspace that produced the Sprint v2.1 Certified lightweight face-swap pipeline.

> **NOTE:** This code was the original development environment. The production-ready versions have been integrated into the main repository under `core/pipeline/` and `core/validators/`.

## Contents

### Source Code (`src/`)

| File | Description |
|:---|:---|
| `processor.py` | The original standalone InSwapper CPU processor with MediaPipe landmarks, CIE LAB Reinhard transfer, and C1-continuous smoothstep blending |
| `pipeline.py` | Watchdog-based folder watcher that orchestrates the processor |
| `hybrid_processor.py` | Intermediate hybrid processor (pure InSwapper per AA001 directive) |
| `diffusion_processor.py` | Full Stable Diffusion Inpainting + IP-Adapter-FaceID processor (decommissioned in v2) |
| `precompute_identity.py` | Elite identity embedding pre-computation and caching utility |
| `download_models.py` | Model downloader for ONNX/MediaPipe assets |
| `debug.py` | Debug/diagnostic utilities |
| `inspect_diff.py` | Image diff inspection tool |
| `stress_test.py` | Pipeline stress test harness |
| `test_processor.py` | Processor unit test harness |
| `validator.py` | Original master validation orchestrator |
| `likeness_validator.py` | Cosine identity similarity validator |
| `blending_validator.py` | Boundary seam gradient validator |
| `spatial_integrity_validator.py` | Background preservation validator |
| `specularity_validator.py` | Face-to-neck luminance validator |
| `realism_validator.py` | Sharpness texture variance validator |

### Documentation

| File | Description |
|:---|:---|
| `README.md` (original) | Original image_pipeline project documentation (overwritten by this file) |
| `HANDOVER_REPORT.md` | Complete project handover report with all validation results, bug fixes, architecture decisions, AA001 rulings, and Sprint v2.1 certification metrics |

### Environment

| File | Description |
|:---|:---|
| `requirements_frozen.txt` | Exact `pip freeze` of the working Python 3.14 virtual environment |

## Models (Not Included — Too Large for Git)

The following models were used and can be re-downloaded via `python models/download_models.py`:

| Model | Size | Purpose |
|:---|:---|:---|
| `inswapper_128.onnx` | 554 MB | Face swap ONNX model |
| `arcface_w600k_r50.onnx` | 174 MB | Face embedding extraction |
| `face_landmarker.task` | 3.7 MB | MediaPipe 468-point face mesh |
| `scrfd_10g_bnkps.onnx` | 16.9 MB | SCRFD face detection (legacy) |

## Reproduction

To recreate this environment on a fresh machine:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_frozen.txt
python models/download_models.py
python src/pipeline.py
```
