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

def get_texture_variance(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_bool = mask > 127
    inv_mask_bool = mask < 127
    
    if not np.any(mask_bool) or not np.any(inv_mask_bool):
        return 0.0, 0.0
        
    face_pixels = gray[mask_bool].flatten()
    bg_pixels = gray[inv_mask_bool].flatten()
    
    # Measure variance of Laplacian as a proxy for sharp high-frequency detail
    face_var = np.var(cv2.Laplacian(face_pixels, cv2.CV_64F)) if len(face_pixels) > 0 else 0.0
    bg_var = np.var(cv2.Laplacian(bg_pixels, cv2.CV_64F)) if len(bg_pixels) > 0 else 0.0
    
    return float(face_var), float(bg_var)

def run_realism_validation(output_dir, models_dir=None):
    engine = LightweightInSwapperEngine(models_dir=models_dir)
    
    outputs = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not outputs:
        print("No outputs found for realism validation.")
        return {}

    results = {}
    print("\n" + "="*80)
    print(" SHARPNESS & REALISM TEXTURE VALIDATOR")
    print("="*80)
    print(f"{'Output File':<30} | {'Face Sharpness':<15} | {'BG Sharpness':<15} | {'Ratio'}")
    print("-" * 80)

    for out_name in outputs:
        out_path = os.path.join(output_dir, out_name)
        img = cv2.imread(out_path)
        if img is None: continue
        
        data = engine.get_face_data(img)
        if not data: continue
        
        _, mask = data
        face_var, bg_var = get_texture_variance(img, mask)
        ratio = face_var / (bg_var + 1e-5)
        
        results[out_name] = {
            "face_sharpness": face_var,
            "bg_sharpness": bg_var,
            "ratio": ratio
        }
        
        print(f"{out_name:<30} | {face_var:>15.2f} | {bg_var:>15.2f} | {ratio:.4f}")
        
    return results

if __name__ == '__main__':
    out_d = os.path.join(proj_root, "samples/odiyan_swaps")
    run_realism_validation(out_d)
