import os
import cv2
import numpy as np
import insightface
from skimage.exposure import match_histograms
import glob

def match_colors(source, target):
    return match_histograms(source, target, channel_axis=-1)

class Phase1:
    def __init__(self, profile_path="profiles/aya.npy"):
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
        aya_face = MockFace(emb)
        
        target_img = cv2.imread(target_path)
        faces = self.app.get(target_img)
        if not faces: return None
        target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        
        # Identity Anchor
        swapped_base = self.swapper.get(target_img, target_face, aya_face, paste_back=True)
        swapped_base = match_colors(swapped_base, target_img)
        
        # Surgical Crop
        bbox = target_face.bbox.astype(int)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        y1, y2 = max(0, bbox[1]-int(h*1.0)), min(target_img.shape[0], bbox[3]+int(h*0.4))
        x1, x2 = max(0, bbox[0]-int(w*0.6)), min(target_img.shape[1], bbox[2]+int(w*0.6))
        
        head_crop = swapped_base[y1:y2, x1:x2]
        cv2.imwrite("temp_head_crop.png", head_crop)
        # Save metadata for phase 3
        with open("temp_meta.txt", "w") as f:
            f.write(f"{y1},{y2},{x1},{x2}")
        print("Phase 1 Complete. Crop saved.")

if __name__ == "__main__":
    p1 = Phase1()
    targets = glob.glob("target_pics/*.jpg")
    if targets:
        p1.run(targets[0]) # Process one at a time
