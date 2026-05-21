import os
import cv2
import numpy as np
import sys
sys.path.append('image_pipeline/src')
from processor import ImageProcessor

def get_blending_score(img, mask):
    # Dilate and erode mask to get boundary region
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=3)
    eroded = cv2.erode(mask, kernel, iterations=3)
    
    # Boundary is between dilated and eroded
    boundary = cv2.bitwise_xor(dilated, eroded)
    
    # Get image gradient (edges)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    
    # Average gradient on the boundary
    boundary_pixels = gradient[boundary > 127]
    if len(boundary_pixels) == 0:
        return 0
    boundary_grad = np.mean(boundary_pixels)
    
    # Average gradient inside and outside (Context)
    inside = cv2.bitwise_and(eroded, eroded)
    outside = cv2.bitwise_not(dilated)
    
    inside_pixels = gradient[inside > 127]
    outside_pixels = gradient[outside > 127]
    
    inside_grad = np.mean(inside_pixels) if len(inside_pixels) > 0 else 0
    outside_grad = np.mean(outside_pixels) if len(outside_pixels) > 0 else 0
    
    # Context gradient
    context_grad = (inside_grad + outside_grad) / 2
    
    # Blending ratio (higher means harsher boundary compared to surroundings)
    ratio = boundary_grad / (context_grad + 1e-5)
    return ratio

def main():
    processor = ImageProcessor()
    output_dir = "image_pipeline/output"
    
    outputs = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not outputs:
        print("No outputs found.")
        return

    print("-" * 80)
    print(f"{'Output File':<30} | {'Blending Ratio':<15} | {'Verdict'}")
    print("-" * 80)

    for out_name in outputs:
        out_path = os.path.join(output_dir, out_name)
        img = cv2.imread(out_path)
        if img is None: continue
        
        data = processor.get_face_data(img)
        if not data:
            print(f"{out_name:<30} | {'ERROR':<15} | {'NO FACE'}")
            continue
            
        _, mask = data
        ratio = get_blending_score(img, mask)
        
        # Target: Ratio should be < 1.5 (Boundary should seamlessly blend with context)
        verdict = "PASS" if ratio < 1.5 else "FAIL (HARSH CUT)"
        
        print(f"{out_name:<30} | {ratio:>15.4f} | {verdict}")

if __name__ == '__main__':
    main()
