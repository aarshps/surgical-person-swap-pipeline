import os
import cv2
import numpy as np
import sys
import mediapipe as mp

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(proj_root)

try:
    from core.pipeline.swap_engines import LightweightInSwapperEngine
except ImportError:
    from swap_engines import LightweightInSwapperEngine

def get_specularity_metrics(img, engine):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    detection_result = engine.landmarker.detect(mp_image)
    
    if not detection_result.face_landmarks: return None
    landmarks = detection_result.face_landmarks[0]
    h, w, _ = img.shape
    
    face_data = engine.get_face_data(img)
    if face_data is None: return None
    _, face_mask = face_data
    
    # Landmark 152 is the chin bottom
    chin_x = int(landmarks[152].x * w)
    chin_y = int(landmarks[152].y * h)
    
    # Sample neck skin region
    sample_h, sample_w = 30, 50
    y1 = min(h-1, chin_y + 15)
    y2 = min(h-1, y1 + sample_h)
    x1 = max(0, chin_x - sample_w // 2)
    x2 = min(w-1, x1 + sample_w)
    
    neck_mask = np.zeros((h, w), dtype=np.uint8)
    if y2 > y1 and x2 > x1:
        neck_mask[y1:y2, x1:x2] = 255
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    face_l = l_channel[face_mask > 0]
    neck_l = l_channel[neck_mask > 0]
    
    if face_l.size == 0 or neck_l.size == 0: return None
    
    face_peak = np.percentile(face_l, 95)
    neck_peak = np.percentile(neck_l, 95)
    
    face_mean = np.mean(face_l)
    neck_mean = np.mean(neck_l)
    
    spec_ratio = face_peak / (neck_peak + 1e-5)
    tone_ratio = face_mean / (neck_mean + 1e-5)
    
    return {
        'face_peak': float(face_peak),
        'neck_peak': float(neck_peak),
        'spec_ratio': float(spec_ratio),
        'face_mean': float(face_mean),
        'neck_mean': float(neck_mean),
        'tone_ratio': float(tone_ratio)
    }

def run_specularity_validation(output_dir, models_dir=None):
    engine = LightweightInSwapperEngine(models_dir=models_dir)
    
    outputs = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not outputs:
        print("No outputs found for specularity validation.")
        return {}

    results = {}
    print("\n" + "="*95)
    print(" SKIN SPECULARITY & TONE MATCHING VALIDATOR")
    print("="*95)
    print(f"{'Output File':<25} | {'Face Peak L':<12} | {'Neck Peak L':<12} | {'Spec Ratio':<12} | {'Tone Ratio':<12} | {'Verdict'}")
    print("-" * 95)

    for out_name in outputs:
        out_path = os.path.join(output_dir, out_name)
        img = cv2.imread(out_path)
        if img is None: continue
        
        metrics = get_specularity_metrics(img, engine)
        if not metrics:
            print(f"{out_name:<25} | {'ERROR':<12} | {'ERROR':<12} | {'ERROR':<12} | {'ERROR':<12} | {'NO FACE/NECK'}")
            continue
            
        # Target: Spec Ratio < 1.15
        verdict = "PASS" if metrics['spec_ratio'] < 1.15 else "FAIL (OILY HIGHLIGHTS)"
        results[out_name] = {
            "spec_ratio": metrics['spec_ratio'],
            "tone_ratio": metrics['tone_ratio'],
            "verdict": verdict
        }
        
        print(f"{out_name:<25} | {metrics['face_peak']:>12.2f} | {metrics['neck_peak']:>12.2f} | {metrics['spec_ratio']:>12.4f} | {metrics['tone_ratio']:>12.4f} | {verdict}")
        
    return results

if __name__ == '__main__':
    out_d = os.path.join(proj_root, "samples/odiyan_swaps")
    run_specularity_validation(out_d)
