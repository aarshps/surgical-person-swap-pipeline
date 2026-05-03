import requests
import base64
import json
import os

def run_sd_phase():
    url = "http://127.0.0.1:1234/sdapi/v1/img2img"
    import cv2
    img = cv2.imread("temp_head_crop.png")
    img = cv2.resize(img, (512, 512))
    _, img_encoded = cv2.imencode('.png', img)
    img_b64 = base64.b64encode(img_encoded).decode('utf-8')
    
    payload = {
        "prompt": "Aya, beautiful woman, long dark wavy hair, elegant, detailed skin, photorealistic",
        "negative_prompt": "cartoon, painting, illustration, blurry, deformed, bad anatomy",
        "init_images": [img_b64],
        "denoising_strength": 0.25,
        "steps": 10,
        "width": 512,
        "height": 512
    }
    
    print("Sending Phase 2 request to SD server...")
    response = requests.post(url, json=payload, timeout=1200)
    if response.status_code == 200:
        img_data = base64.b64decode(response.json()['images'][0])
        with open("temp_refined_head.png", "wb") as f:
            f.write(img_data)
        print("Phase 2 Complete. Refined head saved.")
    else:
        print(f"Phase 2 Failed: {response.status_code} {response.text}")

if __name__ == "__main__":
    run_sd_phase()
