import requests
import json
import base64
import time
import os

# Create samples directory if it doesn't exist
os.makedirs("samples", exist_ok=True)

url = "http://127.0.0.1/sdapi/v1/txt2img"
headers = {
    "Authorization": "Bearer sk-opencode-cli-test-key",
    "Content-Type": "application/json"
}
payload = {
    "prompt": "a majestic golden eagle soaring over snowy mountains, highly detailed, 4k",
    "negative_prompt": "blurry, low quality, deformed",
    "steps": 20,  # 20 steps for high quality
    "width": 256, # Kept at 256x256 to fit within the 5-minute timeout window
    "height": 256
}

print("Requesting 256x256 image with 20 steps (this will take ~3.5 minutes on CPU)...")
start_time = time.time()
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    data = response.json()
    if 'images' in data and len(data['images']) > 0:
        image_base64 = data['images'][0]
        output_path = "samples/sample_eagle_medium.png"
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
