import os
import cv2
import numpy as np
import insightface
from skimage.exposure import match_histograms
import glob

def match_colors(source, target):
    return match_histograms(source, target, channel_axis=-1)

class Phase1:
    def __init__(self, profile_path="data/profiles/odiyan.npy"):
        print("Loading InsightFace for Phase 1...")
        self.app = insightface.app.FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
        self.profile_path = profile_path

    def run(self, target_path):
        emb = np.load(self.profile_path)
        class MockFace:
            def __init__(self, embedding):
                self.embedding = embedding
                self.normed_embedding = embedding
        odiyan_face = MockFace(emb)
        
        target_img = cv2.imread(target_path)
        faces = self.app.get(target_img)
        if not faces: return None
        target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        
        # Identity Anchor
        swapped_base = self.swapper.get(target_img, target_face, odiyan_face, paste_back=True)
        swapped_base = match_colors(swapped_base, target_img)
        
        # Surgical Crop (Strictly Square to prevent distortion)
        bbox = target_face.bbox.astype(int)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        
        # We need a large enough box to cover the head and hair
        cx = bbox[0] + w // 2
        cy = bbox[1] + h // 2
        
        # Calculate the dimension of the square crop
        crop_size = int(max(w, h) * 2.2)
        half_size = crop_size // 2
        
        y1 = max(0, cy - half_size)
        y2 = min(target_img.shape[0], cy + half_size)
        x1 = max(0, cx - half_size)
        x2 = min(target_img.shape[1], cx + half_size)
        
        head_crop = swapped_base[y1:y2, x1:x2]
        
        # Pad if it's not a perfect square (happens at image edges)
        pad_bottom = crop_size - (y2 - y1)
        pad_right = crop_size - (x2 - x1)
        
        if pad_bottom > 0 or pad_right > 0:
            head_crop = cv2.copyMakeBorder(head_crop, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)
            
        cv2.imwrite("temp_head_crop.png", head_crop)
        
        # Save metadata for phase 3: coordinates and the original crop dimensions
        with open("temp_meta.txt", "w") as f:
            f.write(f"{y1},{y2},{x1},{x2},{crop_size},{pad_bottom},{pad_right}")
        print("Phase 1 Complete. Square crop saved.")

if __name__ == "__main__":
    import sys
    p1 = Phase1()
    if len(sys.argv) > 1:
        p1.run(sys.argv[1])
    else:
        targets = glob.glob("data/targets/mixed/*.jpg")
        if targets:
            p1.run(targets[0]) # Process one at a time
