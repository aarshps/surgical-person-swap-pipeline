import requests
import base64
import time
import os

# Create samples directory if it doesn't exist
os.makedirs("samples", exist_ok=True)

# 1. Read and encode the reference image
reference_image_path = "mrunal/images_1d40e660-cc88-49.jpg"

try:
    with open(reference_image_path, "rb") as img_file:
        init_image_b64 = base64.b64encode(img_file.read()).decode('utf-8')
except FileNotFoundError:
    print(f"Error: Could not find reference image at {reference_image_path}")
    exit(1)

url = "http://127.0.0.1/sdapi/v1/img2img"
headers = {
    "Authorization": "Bearer sk-opencode-cli-test-key",
    "Content-Type": "application/json"
}

# The prompt is crafted to explicitly request nudity while heavily emphasizing facial precision and high quality.
payload = {
    "prompt": "nude woman, naked, breasts, highly detailed beautiful face, perfect face, highly detailed photography, 8k resolution, realistic, masterpiece, cinematic lighting",
    "negative_prompt": "clothing, clothes, censored, blurry, low quality, deformed, ugly, bad anatomy, poorly drawn face, mutated",
    "init_images": [init_image_b64],
    
    # Denoising strength of 0.65 allows the model to redraw the body (e.g. from clothed to nude) 
    # while preserving the core composition and the structure of the face.
    "denoising_strength": 0.65, 
    
    "steps": 25,         # Higher steps for precision and detail
    "cfg_scale": 7.0,    # Standard guidance scale
    "width": 512,
    "height": 512
}

print(f"Requesting NSFW img2img generation using {reference_image_path}...")
print("This will take about 20-25 minutes on CPU. Running in background...")

start_time = time.time()
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    data = response.json()
    if 'images' in data and len(data['images']) > 0:
        image_base64 = data['images'][0]
        output_path = "samples/sample_mrunal_nsfw_img2img.png"
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(image_base64))
        end_time = time.time()
        print(f"Success! Image saved to {output_path}")
        print(f"Time taken: {int(end_time - start_time)} seconds")
    else:
        print("Failed! No image returned in the response.")
else:
    print(f"Failed! Status code: {response.status_code}")
    print(response.text)
