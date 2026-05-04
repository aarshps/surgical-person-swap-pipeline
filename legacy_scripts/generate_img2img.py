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
        # Stable diffusion API expects base64 encoded string for images
        init_image_b64 = base64.b64encode(img_file.read()).decode('utf-8')
except FileNotFoundError:
    print(f"Error: Could not find reference image at {reference_image_path}")
    exit(1)

# 2. Configure the API Request
# Notice we are using the /img2img endpoint now instead of /txt2img
url = "http://127.0.0.1/sdapi/v1/img2img"
headers = {
    "Authorization": "Bearer sk-opencode-cli-test-key",
    "Content-Type": "application/json"
}

payload = {
    "prompt": "a beautiful woman, highly detailed portrait, 4k, realistic, masterpiece, cinematic lighting",
    "negative_prompt": "blurry, low quality, deformed, ugly, bad anatomy",
    "init_images": [init_image_b64],
    
    # denoising_strength controls how much the AI changes the original image.
    # 0.0 = exact copy of the original image
    # 1.0 = completely ignores the original image and draws something new
    # 0.4 to 0.6 is usually the sweet spot for keeping the structure but enhancing/stylizing it.
    "denoising_strength": 0.5, 
    
    "steps": 25,         # Increased steps for higher quality
    "cfg_scale": 7.0,    # How strictly the AI follows your prompt vs doing its own thing (7.0 is standard)
    "width": 512,        # Target resolution
    "height": 512
}

print(f"Requesting img2img generation using {reference_image_path}...")
print("This will take about 20-25 minutes on CPU...")

start_time = time.time()
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    data = response.json()
    if 'images' in data and len(data['images']) > 0:
        image_base64 = data['images'][0]
        output_path = "samples/sample_mrunal_img2img.png"
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
