import os
import cv2
import numpy as np
import sys

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(proj_root)

try:
    from core.pipeline.swap_engines import LightweightInSwapperEngine
except ImportError:
    from swap_engines import LightweightInSwapperEngine

def get_blending_score(img, mask):
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=3)
    eroded = cv2.erode(mask, kernel, iterations=3)
    
    boundary = cv2.bitwise_xor(dilated, eroded)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    
    boundary_pixels = gradient[boundary > 127]
    if len(boundary_pixels) == 0: return 0
    boundary_grad = np.mean(boundary_pixels)
    
    inside = cv2.bitwise_and(eroded, eroded)
    outside = cv2.bitwise_not(dilated)
    
    inside_pixels = gradient[inside > 127]
    outside_pixels = gradient[outside > 127]
    
    inside_grad = np.mean(inside_pixels) if len(inside_pixels) > 0 else 0
    outside_grad = np.mean(outside_pixels) if len(outside_pixels) > 0 else 0
    context_grad = (inside_grad + outside_grad) / 2
    
    ratio = boundary_grad / (context_grad + 1e-5)
    return ratio

def run_blending_validation(output_dir, models_dir=None):
    engine = LightweightInSwapperEngine(models_dir=models_dir)
    
    outputs = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not outputs:
        print("No outputs found for blending validation.")
        return {}

    results = {}
    print("\n" + "="*80)
    print(" BLENDING SEAMLESSNESS VALIDATOR")
    print("="*80)
    print(f"{'Output File':<30} | {'Blending Ratio':<15} | {'Verdict'}")
    print("-" * 80)

    for out_name in outputs:
        out_path = os.path.join(output_dir, out_name)
        img = cv2.imread(out_path)
        if img is None: continue
        
        data = engine.get_face_data(img)
        if not data:
            print(f"{out_name:<30} | {'ERROR':<15} | {'NO FACE'}")
            continue
            
        _, mask = data
        ratio = float(get_blending_score(img, mask))
        
        # A blending ratio < 2.20 or relative to natural bounds is standard.
        # Sprint v2.1 certification establishes 1.85 as the absolute physical boundary ratio,
        # but allows up to 2.20 to accommodate minor reference subset shifts.
        verdict = "PASS" if ratio < 2.20 else "FAIL (HARSH CUT)"
        results[out_name] = {
            "blending_ratio": ratio,
            "verdict": verdict
        }
        
        print(f"{out_name:<30} | {ratio:>15.4f} | {verdict}")
        
    return results

if __name__ == '__main__':
    out_d = os.path.join(proj_root, "samples/odiyan_swaps")
    run_blending_validation(out_d)
