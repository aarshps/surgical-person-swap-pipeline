import os
import cv2
import numpy as np
import sys
sys.path.append('image_pipeline/src')
from processor import ImageProcessor

def get_texture_variance(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_bool = mask > 127
    inv_mask_bool = mask < 127
    
    if not np.any(mask_bool) or not np.any(inv_mask_bool):
        return 0, 0
        
    face_pixels = gray[mask_bool].flatten()
    bg_pixels = gray[inv_mask_bool].flatten()
    
    face_var = np.var(cv2.Laplacian(face_pixels, cv2.CV_64F))
    bg_var = np.var(cv2.Laplacian(bg_pixels, cv2.CV_64F))
    
    return face_var, bg_var

def main():
    processor = ImageProcessor()
    output_dir = "image_pipeline/output"
    
    outputs = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not outputs:
        print("No outputs found.")
        return

    print("-" * 80)
    print(f"{'Output File':<25} | {'Face Sharpness':<15} | {'BG Sharpness':<15} | {'Ratio'}")
    print("-" * 80)

    for out_name in outputs:
        out_path = os.path.join(output_dir, out_name)
        img = cv2.imread(out_path)
        if img is None: continue
        
        data = processor.get_face_data(img)
        if not data: continue
        
        _, mask = data
        face_var, bg_var = get_texture_variance(img, mask)
        ratio = face_var / bg_var if bg_var > 0 else 0
        
        print(f"{out_name:<25} | {face_var:>15.2f} | {bg_var:>15.2f} | {ratio:.4f}")

if __name__ == '__main__':
    main()
