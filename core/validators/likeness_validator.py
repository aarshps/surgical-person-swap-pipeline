import os
import cv2
import numpy as np
import sys

# Add project root to path
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(proj_root)

try:
    from core.pipeline.swap_engines import LightweightInSwapperEngine
except ImportError:
    from swap_engines import LightweightInSwapperEngine

def calculate_cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10)

def get_landmarks_dist(pts1, pts2):
    return np.mean(np.linalg.norm(pts1 - pts2, axis=1))

def run_likeness_validation(output_dir, input_dir, reference_dir, models_dir=None):
    engine = LightweightInSwapperEngine(models_dir=models_dir)
    
    outputs = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not outputs:
        print("No outputs found for likeness validation.")
        return {}

    all_refs = sorted([f for f in os.listdir(reference_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not all_refs:
        print("No references found for likeness validation.")
        return {}
        
    # Pick 5 diverse references
    step = max(1, len(all_refs) // 5)
    selected_refs = [all_refs[i * step] for i in range(min(5, len(all_refs)))]
    
    ref_embs = []
    for ref_name in selected_refs:
        ref_path = os.path.join(reference_dir, ref_name)
        img = cv2.imread(ref_path)
        data = engine.get_face_data(img)
        if data:
            pts, _ = data
            align, _ = engine.align_face(img, pts)
            emb = engine.get_face_embedding(align)
            if emb is not None:
                ref_embs.append(emb)

    if not ref_embs:
        print("Error: Could not extract reference embeddings.")
        return {}

    results = {}
    print("\n" + "="*80)
    print(" LIKENESS SIMILARITY VALIDATOR")
    print("="*80)
    print(f"{'Output File':<30} | {'Avg Ref Sim':<12} | {'Max Sim':<10} | {'Struct Dev':<10} | {'Verdict'}")
    print("-" * 80)

    for out_name in outputs:
        # Match output to input
        base_name = out_name.replace("transformed_", "").replace("lightweight_", "").replace(".png", ".jpg")
        input_path = os.path.join(input_dir, base_name)
        
        # Fallbacks
        if not os.path.exists(input_path):
            # Try png fallback
            input_path = os.path.join(input_dir, base_name.replace(".jpg", ".png"))
            if not os.path.exists(input_path):
                # Use first input file found
                all_inputs = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if all_inputs:
                    input_path = os.path.join(input_dir, all_inputs[0])
                else:
                    continue

        out_path = os.path.join(output_dir, out_name)
        out_img = cv2.imread(out_path)
        in_img = cv2.imread(input_path)
        
        if out_img is None or in_img is None: continue
        
        out_data = engine.get_face_data(out_img)
        in_data = engine.get_face_data(in_img)
        
        if not out_data or not in_data: continue
        
        out_pts, _ = out_data
        in_pts, _ = in_data
        
        face_size = np.linalg.norm(in_pts[0] - in_pts[1])
        struct_dev = get_landmarks_dist(out_pts, in_pts) / (face_size + 1e-10)
        
        out_align, _ = engine.align_face(out_img, out_pts)
        out_emb = engine.get_face_embedding(out_align)
        if out_emb is None: continue
        
        sims = [calculate_cosine_similarity(out_emb, r_emb) for r_emb in ref_embs]
        avg_sim = float(np.mean(sims))
        max_sim = float(np.max(sims))
        
        verdict = "PASS" if avg_sim > 0.40 and max_sim > 0.50 else "FAIL"
        results[out_name] = {
            "avg_sim": avg_sim,
            "max_sim": max_sim,
            "struct_dev": float(struct_dev),
            "verdict": verdict
        }
        
        print(f"{out_name:<30} | {avg_sim:.4f}      | {max_sim:.4f}   | {struct_dev:.4f}     | {verdict}")
        
    return results

if __name__ == '__main__':
    # Default local dirs for quick standalone run
    out_d = os.path.join(proj_root, "samples/odiyan_swaps")
    in_d = os.path.join(proj_root, "target_pics")
    ref_d = os.path.join(proj_root, "odiyan_refs")
    run_likeness_validation(out_d, in_d, ref_d)
