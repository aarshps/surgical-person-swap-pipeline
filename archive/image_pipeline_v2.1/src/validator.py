import os
import cv2
import numpy as np
from processor import ImageProcessor

def calculate_cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def main():
    processor = ImageProcessor()
    
    output_dir = "image_pipeline/output"
    reference_dir = "image_pipeline/reference"
    
    # Get reference
    references = [f for f in os.listdir(reference_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not references:
        print("No references found for validation.")
        return
        
    ref_path = os.path.join(reference_dir, references[0])
    ref_img = cv2.imread(ref_path)
    ref_data = processor.get_face_data(ref_img)
    if not ref_data:
        print("No face found in reference image.")
        return
        
    ref_pts, _ = ref_data
    ref_align, _ = processor.align_face(ref_img, ref_pts)
    ref_emb = processor.get_face_embedding(ref_align)
    
    print(f"Validation against reference: {references[0]}")
    print("-" * 50)
    
    # Validate outputs
    outputs = [f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not outputs:
        print("No output images to validate.")
        return
        
    for output_file in outputs:
        out_path = os.path.join(output_dir, output_file)
        out_img = cv2.imread(out_path)
        
        if out_img is None:
            print(f"FAILED: {output_file} - Could not read image.")
            continue
            
        out_data = processor.get_face_data(out_img)
        if not out_data:
            print(f"FAILED: {output_file} - No face found in output image.")
            continue
            
        out_pts, _ = out_data
        out_align, _ = processor.align_face(out_img, out_pts)
        out_emb = processor.get_face_embedding(out_align)
        
        similarity = calculate_cosine_similarity(ref_emb, out_emb)
        
        status = "PASSED" if similarity > 0.4 else "FAILED"
        print(f"[{status}] {output_file} - Face Similarity Score: {similarity:.4f}")

if __name__ == '__main__':
    main()
