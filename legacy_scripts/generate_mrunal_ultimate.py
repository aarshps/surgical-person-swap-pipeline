import requests
import json
import base64
import time
import os
import cv2
import insightface
import sys

# 1. Setup Configuration
os.makedirs("samples", exist_ok=True)
url = "http://127.0.0.1/sdapi/v1/txt2img"
headers = {
    "Authorization": "Bearer sk-opencode-cli-test-key",
    "Content-Type": "application/json"
}

# The prompt is crafted for extreme photorealism
payload = {
    "prompt": "nude woman, naked, bare breasts, raw photo, ultra realistic, highly detailed photography, Fujifilm XT4, 85mm lens, natural daylight, sharp focus, beautiful lighting, skin pores, slight imperfections, masterpiece, best quality, cinematic, looking at viewer",
    "negative_prompt": "clothing, clothes, underwear, censored, cartoon, illustration, painting, anime, 3d, render, blurry, deformed, mutated, bad anatomy, artificial, plastic skin, airbrushed, oversaturated",
    "steps": 25,          # 25 steps provides a very high level of detail
    "cfg_scale": 6.5,     # 6.5 gives the model slightly more freedom for realism
    "width": 512,         # Keeping it 512x512 so it finishes in ~20 minutes on CPU
    "height": 512
}

reference_image_path = "mrunal/images_1d40e660-cc88-49.jpg"
base_output_path = "samples/ultimate_base_nsfw.png"
final_output_path = "samples/ultimate_mrunal_nsfw.png"

# 2. Phase 1: Generate the Base Image
print(f"Step 1: Generating highly realistic base body (takes ~20-25 minutes)...")
start_time = time.time()
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    data = response.json()
    if 'images' in data and len(data['images']) > 0:
        image_base64 = data['images'][0]
        with open(base_output_path, "wb") as f:
            f.write(base64.b64decode(image_base64))
        print(f"Phase 1 Complete! Base image generated in {int(time.time() - start_time)} seconds.")
    else:
        print("Failed to get image data.")
        sys.exit(1)
else:
    print(f"API Error {response.status_code}: {response.text}")
    sys.exit(1)

# 3. Phase 2: Face Swap for Exact Identity
print("Step 2: Initializing Face Swap models...")
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

print("Applying Mrunal's face to the realistic generation...")
source_img = cv2.imread(reference_image_path)
target_img = cv2.imread(base_output_path)

if source_img is None or target_img is None:
    print("Error reading images for face swap.")
    sys.exit(1)

source_faces = app.get(source_img)
target_faces = app.get(target_img)

if len(source_faces) > 0 and len(target_faces) > 0:
    # We use the first face detected in both images
    result_img = swapper.get(target_img, target_faces[0], source_faces[0], paste_back=True)
    cv2.imwrite(final_output_path, result_img)
    print(f"Success! The ultimate realistic NSFW image of Mrunal has been saved to: {final_output_path}")
else:
    print("Face swapping failed! Could not detect a face in either the source or generated image.")
