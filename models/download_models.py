import os
import requests

def download_file(url, path):
    if os.path.exists(path):
        print(f"File already exists: {path}")
        return
    print(f"Downloading {url} to {path}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, stream=True, headers=headers)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        raise RuntimeError(f"Failed to download model: HTTP status {response.status_code}")

def main():
    # Make sure we are in the project root or models/ directory
    # Find model directory path relative to repository root
    models_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(models_dir, exist_ok=True)
    
    # Models URLs (Hugging Face community mirrors and official tasks)
    models = {
        "inswapper_128.onnx": "https://huggingface.co/ApacheOne/insightface/resolve/main/inswapper_128.onnx",
        "arcface_w600k_r50.onnx": "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/antelopev2/arcface_w600k_r50.onnx",
        "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    }
    
    for name, url in models.items():
        dest_path = os.path.join(models_dir, name)
        try:
            download_file(url, dest_path)
        except Exception as e:
            print(f"Error downloading {name}: {e}")

if __name__ == "__main__":
    main()
