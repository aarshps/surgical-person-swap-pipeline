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

def run_spatial_validation(output_dir, input_dir, models_dir=None):
    engine = LightweightInSwapperEngine(models_dir=models_dir)
    
    outputs = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not outputs:
        print("No outputs found for spatial validation.")
        return {}

    results = {}
    print("\n" + "="*90)
    print(" SPATIAL INTEGRITY VALIDATOR (BACKGROUND PRESERVATION)")
    print("="*90)
    print(f"{'Output File':<30} | {'% Altered Pixels':<18} | {'BG MSE':<12} | {'Max Diff':<10} | {'Verdict'}")
    print("-" * 90)

    for out_name in outputs:
        base_name = out_name.replace("transformed_", "").replace("lightweight_", "").replace(".png", ".jpg")
        input_path = os.path.join(input_dir, base_name)
        
        if not os.path.exists(input_path):
            input_path = os.path.join(input_dir, base_name.replace(".jpg", ".png"))
            if not os.path.exists(input_path):
                all_inputs = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if all_inputs:
                    input_path = os.path.join(input_dir, all_inputs[0])
                else:
                    continue

        out_path = os.path.join(output_dir, out_name)
        
        in_img = cv2.imread(input_path)
        out_img = cv2.imread(out_path)
        
        if in_img is None or out_img is None: continue
        if in_img.shape != out_img.shape:
            out_img = cv2.resize(out_img, (in_img.shape[1], in_img.shape[0]))

        data = engine.get_face_data(in_img)
        if not data:
            print(f"{out_name:<30} | {'ERROR':<18} | {'ERROR':<12} | {'ERROR':<10} | {'NO FACE'}")
            continue
            
        _, face_mask = data
        bg_mask = cv2.bitwise_not(face_mask)
        
        diff = cv2.absdiff(in_img, out_img)
        bg_diff = cv2.bitwise_and(diff, diff, mask=bg_mask)
        
        total_bg_pixels = np.sum(bg_mask > 0)
        changed_pixels = np.sum(np.any(bg_diff > 0, axis=2))
        
        mse = float(np.mean(bg_diff**2))
        max_diff = int(np.max(bg_diff))
        pct_altered = float((changed_pixels / total_bg_pixels) * 100 if total_bg_pixels > 0 else 0)
        
        # In lossless formats, altered should be very close to 0.0000%
        # We allow a tiny tolerance due to color rounding if output is scaled/resized.
        verdict = "PASS" if pct_altered < 0.05 else "FAIL"
        
        results[out_name] = {
            "pct_altered": pct_altered,
            "mse": mse,
            "max_diff": max_diff,
            "verdict": verdict
        }
        
        print(f"{out_name:<30} | {pct_altered:>16.4f}% | {mse:>12.4f} | {max_diff:>10} | {verdict}")
        
    return results

if __name__ == '__main__':
    out_d = os.path.join(proj_root, "samples/odiyan_swaps")
    in_d = os.path.join(proj_root, "target_pics")
    run_spatial_validation(out_d, in_d)
