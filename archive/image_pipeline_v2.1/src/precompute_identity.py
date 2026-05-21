import os
import cv2
import numpy as np
import torch
from processor import ImageProcessor

def main():
    processor = ImageProcessor()
    ref_dir = "image_pipeline/reference"
    references = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not references:
        print("No reference images found.")
        return
        
    print(f"Pre-computing identity from {len(references)} images using lightweight ImageProcessor...")
    
    all_embeds = []
    for ref_path in references:
        img = cv2.imread(ref_path)
        if img is None:
            continue
        face_data = processor.get_face_data(img)
        if face_data is None:
            continue
        pts, _ = face_data
        aligned, _ = processor.align_face(img, pts, is_inswapper=False)
        if aligned is None:
            continue
        embedding = processor.get_face_embedding(aligned)
        if embedding is not None:
            all_embeds.append(embedding)
            
    if not all_embeds:
        print("Error: Could not extract any face embeddings from the reference images.")
        return
        
    all_embs_np = np.vstack(all_embeds)
    centroid = np.mean(all_embs_np, axis=0)
    dists = np.linalg.norm(all_embs_np - centroid, axis=1)
    
    # Curate the top 5 highest-quality embeddings (closest to centroid)
    elite_indices = np.argsort(dists)[:5]
    elite_mean = np.mean(all_embs_np[elite_indices], axis=0)
    
    # Normalize the average embedding
    norm = np.linalg.norm(elite_mean)
    if norm > 0:
        elite_mean = elite_mean / norm
        
    # Shape of embedding in saved dict: torch.Tensor of shape (1, 1, 512)
    face_embeds = torch.from_numpy(elite_mean).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    
    cache_dir = "image_pipeline/models"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "reference_identity.pt")
    
    torch.save({"embedding": face_embeds, "metadata": {"count": len(references)}}, cache_path)
    print(f"Pre-computation and caching complete. Saved elite embedding to {cache_path}")

if __name__ == "__main__":
    main()

