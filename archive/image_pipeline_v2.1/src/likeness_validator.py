import os
import cv2
import numpy as np
import torch
from processor import ImageProcessor

def calculate_cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def get_landmarks_dist(pts1, pts2):
    return np.mean(np.linalg.norm(pts1 - pts2, axis=1))

def main():
    processor = ImageProcessor()
    
    output_dir = "image_pipeline/output"
    input_dir = "image_pipeline/input"
    reference_dir = "image_pipeline/reference"
    
    outputs = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not outputs:
        print("No outputs found.")
        return

    # Select 5 diverse references for individual comparison
    all_refs = sorted([f for f in os.listdir(reference_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not all_refs:
        print("No references found.")
        return
    step = max(1, len(all_refs) // 5)
    selected_refs = [all_refs[i * step] for i in range(min(5, len(all_refs)))]
    
    ref_embs = []
    for ref_name in selected_refs:
        ref_path = os.path.join(reference_dir, ref_name)
        img = cv2.imread(ref_path)
        data = processor.get_face_data(img)
        if data:
            pts, _ = data
            align, _ = processor.align_face(img, pts)
            emb = processor.get_face_embedding(align)
            ref_embs.append(emb)

    print("-" * 100)
    print(f"{'Output File':<25} | {'Avg Ref Sim':<12} | {'Max Indiv Sim':<15} | {'Min Indiv Sim':<15} | {'Struct Dev'}")
    print("-" * 100)

    for out_name in outputs:
        # Match output to input
        base_name = out_name.replace("transformed_", "").replace(".png", ".jpg")
        input_path = os.path.join(input_dir, base_name)
        
        # Fallback for stress tests if needed
        if not os.path.exists(input_path):
            if "bright" in out_name: input_path = os.path.join(input_dir, "gabbie_bright.jpg")
            elif "dim" in out_name: input_path = os.path.join(input_dir, "gabbie_dim.jpg")
            elif "warm" in out_name: input_path = os.path.join(input_dir, "gabbie_warm.jpg")
            elif "grace_hopper" in out_name: input_path = os.path.join(input_dir, "grace_hopper.jpg")
            else: input_path = os.path.join(input_dir, "gabbie_1.jpg")

        out_path = os.path.join(output_dir, out_name)
        out_img = cv2.imread(out_path)
        in_img = cv2.imread(input_path)
        
        if out_img is None or in_img is None: continue
        
        out_data = processor.get_face_data(out_img)
        in_data = processor.get_face_data(in_img)
        
        if not out_data or not in_data: continue
        
        out_pts, _ = out_data
        in_pts, _ = in_data
        
        # Measure structural deviation (normalized by face size)
        face_size = np.linalg.norm(in_pts[0] - in_pts[1]) # distance between eyes as proxy
        struct_dev = get_landmarks_dist(out_pts, in_pts) / face_size
        
        # Measure likeness
        out_align, _ = processor.align_face(out_img, out_pts)
        out_emb = processor.get_face_embedding(out_align)
        
        sims = [calculate_cosine_similarity(out_emb, r_emb) for r_emb in ref_embs]
        avg_sim = np.mean(sims)
        max_sim = np.max(sims)
        min_sim = np.min(sims)
        
        print(f"{out_name:<25} | {avg_sim:.4f}      | {max_sim:.4f}        | {min_sim:.4f}        | {struct_dev:.4f}")

if __name__ == '__main__':
    main()
