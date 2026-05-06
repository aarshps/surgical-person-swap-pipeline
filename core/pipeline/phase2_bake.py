import requests
import base64

FLUX_WIDTH = 1024
FLUX_HEIGHT = 1024
FLUX_STEPS = 4
FLUX_CFG_SCALE = 1.0
FLUX_DENOISING_STRENGTH = 0.32
FLUX_TIMEOUT_SECONDS = 2400

def run_sd_phase():
    url = "http://127.0.0.1:1234/sdapi/v1/img2img"
    import cv2
    img = cv2.imread("temp_head_crop.png")
    img = cv2.resize(img, (FLUX_WIDTH, FLUX_HEIGHT))
    _, img_encoded = cv2.imencode('.png', img)
    img_b64 = base64.b64encode(img_encoded).decode('utf-8')
    
    payload = {
        "prompt": (
            "Odiyan, beautiful woman, long dark wavy hair, exact same facial geometry "
            "and expression as source, natural skin texture, balanced exposure, "
            "soft real-world lighting, visible pores, same camera perspective"
        ),
        "negative_prompt": (
            "cartoon, painting, illustration, blurry, deformed, bad anatomy, "
            "overprocessed, high contrast, oversaturated, blown highlights, "
            "crushed shadows, plastic skin, waxy, airbrushed, halo, seam, watermark"
        ),
        "init_images": [img_b64],
        "denoising_strength": FLUX_DENOISING_STRENGTH,
        "steps": FLUX_STEPS,
        "cfg_scale": FLUX_CFG_SCALE,
        "sampler_name": "Euler",
        "width": FLUX_WIDTH,
        "height": FLUX_HEIGHT
    }
    
    print("Sending Phase 2 request to SD server...")
    response = requests.post(url, json=payload, timeout=FLUX_TIMEOUT_SECONDS)
    if response.status_code == 200:
        img_data = base64.b64decode(response.json()['images'][0])
        with open("temp_refined_head.png", "wb") as f:
            f.write(img_data)
        print("Phase 2 Complete. Refined head saved.")
    else:
        print(f"Phase 2 Failed: {response.status_code} {response.text}")

if __name__ == "__main__":
    run_sd_phase()
