import requests
import base64
import time
import os

def refine_image(input_path, output_name, prompt, denoising=0.2):
    url = "http://127.0.0.1:1234/sdapi/v1/img2img" # Port from start_server.sh
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, "rb") as img_file:
        init_image_b64 = base64.b64encode(img_file.read()).decode('utf-8')

    payload = {
        "prompt": prompt,
        "negative_prompt": "cartoon, painting, illustration, (worst quality, low quality, normal quality:2), (blurry:2), deformed, digital artifacts",
        "init_images": [init_image_b64],
        "denoising_strength": denoising,
        "steps": 20,
        "cfg_scale": 7.0,
        "width": 512,
        "height": 512
    }

    print(f"Refining {input_path} with denoising {denoising}...")
    try:
        response = requests.post(url, json=payload, timeout=600)
        if response.status_code == 200:
            data = response.json()
            image_base64 = data['images'][0]
            output_path = f"samples/{output_name}.png"
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(image_base64))
            print(f"Refinement complete! Saved to: {output_path}")
        else:
            print(f"Failed! Status: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error during request: {e}")

if __name__ == "__main__":
    # The hyper-realistic swap we just did
    input_img = "samples/mrunal_mia_hyper_real.png"
    
    # Prompt focusing on skin texture and realistic details
    refinement_prompt = "photorealistic portrait of a woman, highly detailed skin texture, pores, fine hairs, soft cinematic lighting, 8k resolution, masterpiece, (realism:1.5)"
    
    # Run refinement
    refine_image(input_img, "mrunal_mia_final_refined", refinement_prompt, denoising=0.15)
