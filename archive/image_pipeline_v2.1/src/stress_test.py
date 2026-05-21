import cv2
import os
import numpy as np

def create_lighting_variations(input_path, output_dir):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Dim Light
    dim = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
    cv2.imwrite(os.path.join(output_dir, "gabbie_dim.jpg"), dim)
    
    # 2. Bright Light
    bright = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
    cv2.imwrite(os.path.join(output_dir, "gabbie_bright.jpg"), bright)
    
    # 3. Warm Light (Shift towards Yellow/Red)
    warm = img.astype(np.float32)
    warm[:, :, 2] *= 1.2 # Red
    warm[:, :, 1] *= 1.1 # Green
    warm = np.clip(warm, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "gabbie_warm.jpg"), warm)
    
    print(f"Created 3 lighting variations in {output_dir}")

if __name__ == "__main__":
    create_lighting_variations("image_pipeline/input/gabbie_1.jpg", "image_pipeline/input")
