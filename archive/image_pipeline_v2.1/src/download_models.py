import os
import requests

def download_file(url, path):
    if os.path.exists(path):
        print(f"File already exists: {path}")
        return
    print(f"Downloading {url} to {path}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, stream=True, headers=headers)
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

def main():
    models_dir = "image_pipeline/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Models URLs (Community mirrors or direct links)
    # inswapper_128.onnx is the standard for face swapping
    models = {
        "inswapper_128.onnx": "https://huggingface.co/ApacheOne/insightface/resolve/main/inswapper_128.onnx",
        "scrfd_10g_bnkps.onnx": "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/antelopev2/scrfd_10g_bnkps.onnx",
        "arcface_w600k_r50.onnx": "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/antelopev2/arcface_w600k_r50.onnx"
    }
    
    for name, url in models.items():
        download_file(url, os.path.join(models_dir, name))

if __name__ == "__main__":
    main()
