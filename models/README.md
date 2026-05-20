# Model Assets Directory

This directory contains the machine learning models utilized by the photorealistic face swap pipelines. 

To prevent heavy binary clutter, these large model files are excluded from git tracking via `.gitignore`. You can automatically download or place them here using the provided utility scripts.

## Automated Download

To fetch all the model files required for the lightweight CPU-only pipeline, run the following script from the repository root:

```bash
python models/download_models.py
```

## Model Information

1. **`inswapper_128.onnx`** (554 MB)
   - *Purpose*: Pre-trained high-fidelity face swapping model.
   - *Description*: Takes target aligned face ($128 \times 128$) and source identity embeddings, outputs swapped face.

2. **`arcface_w600k_r50.onnx`** (174 MB)
   - *Purpose*: High-fidelity face embedding extraction.
   - *Description*: Extracts a 512-dimensional normalized face representation vector used as the identity anchor.

3. **`face_landmarker.task`** (3.7 MB)
   - *Purpose*: 468-point 3D facial landmark detection.
   - *Description*: MediaPipe face mesh solution used for spatial alignment and $C^1$-continuous distance-transform smoothstep blending.
