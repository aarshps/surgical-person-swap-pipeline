import os
import cv2
import numpy as np
import sys
sys.path.append('image_pipeline/src')
from processor import ImageProcessor

def main():
    processor = ImageProcessor()
    
    output_dir = "image_pipeline/output"
    input_dir = "image_pipeline/input"
    
    outputs = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not outputs:
        print("No outputs found.")
        return
        
    print("-" * 110)
    print(f"{'Output File':<30} | {'% Altered':<15} | {'BG MSE':<15} | {'Max Diff':<10} | {'Verdict'}")
    print("-" * 110)

    for out_name in outputs:
        base_name = out_name.replace("transformed_", "").replace(".png", ".jpg")
        input_path = os.path.join(input_dir, base_name)
        
        # Fallbacks for stress tests if needed
        if not os.path.exists(input_path):
            if "bright" in out_name: input_path = os.path.join(input_dir, "gabbie_bright.jpg")
            elif "dim" in out_name: input_path = os.path.join(input_dir, "gabbie_dim.jpg")
            elif "warm" in out_name: input_path = os.path.join(input_dir, "gabbie_warm.jpg")
            elif "grace_hopper" in out_name: input_path = os.path.join(input_dir, "grace_hopper.jpg")
            else: input_path = os.path.join(input_dir, "gabbie_1.jpg")
            
        out_path = os.path.join(output_dir, out_name)
        
        in_img = cv2.imread(input_path)
        out_img = cv2.imread(out_path)
        
        if in_img is None or out_img is None:
            print(f"{out_name:<30} | {'ERROR':<15} | {'ERROR':<15} | {'ERROR':<10} | {'NO IMAGE'}")
            continue
            
        if in_img.shape != out_img.shape:
            out_img = cv2.resize(out_img, (in_img.shape[1], in_img.shape[0]))

        # Get face mask
        data = processor.get_face_data(in_img)
        if not data:
            print(f"{out_name:<30} | {'ERROR':<15} | {'ERROR':<15} | {'ERROR':<10} | {'NO FACE'}")
            continue
            
        _, face_mask = data
        
        # Non-face region mask (Background/Body)
        bg_mask = cv2.bitwise_not(face_mask)
        
        # Calculate difference
        diff = cv2.absdiff(in_img, out_img)
        
        # Mask the difference to only look at background
        bg_diff = cv2.bitwise_and(diff, diff, mask=bg_mask)
        
        # Quantitative Analysis
        total_bg_pixels = np.sum(bg_mask > 0)
        changed_pixels = np.sum(np.any(bg_diff > 0, axis=2))
        
        mse = np.mean(bg_diff**2)
        max_diff = np.max(bg_diff)
        
        verdict = "PASS" if changed_pixels == 0 else "FAIL"
        pct_altered = (changed_pixels / total_bg_pixels) * 100 if total_bg_pixels > 0 else 0
        
        print(f"{out_name:<30} | {pct_altered:>14.4f}% | {mse:>15.4f} | {max_diff:>10} | {verdict}")

if __name__ == '__main__':
    main()
