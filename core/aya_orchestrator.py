import os
import cv2
import numpy as np
import insightface
from gfpgan import GFPGANer
from skimage.exposure import match_histograms
import glob

class AyaOrchestrator:
    def __init__(self):
        print("Initializing Advanced Face-Replace Pipeline...")
        self.app = insightface.app.FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
        self.restorer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None
        )
        self.profile_path = "data/profiles/aya.npy"
        self.aya_face = None

    def execute_swap(self, target_path):
        """Advanced Face Replace: No SD, pure high-fidelity face blending."""
        if not self.aya_face: self._load_profile()
        target_img = cv2.imread(target_path)
        if target_img is None: return
        
        faces = self.app.get(target_img)
        if not faces: return
        target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        
        print(f"[Agent] Executing advanced face replace for: {target_path}")
        
        # 1. High-Fidelity Swap
        swapped = self.swapper.get(target_img, target_face, self.aya_face, paste_back=True)
        
        # 3. GFPGAN Restoration (Restore High-Freq details)
        _, _, restored = self.restorer.enhance(swapped, has_aligned=False, only_center_face=False, paste_back=True)
        
        # 4. Surgical Integration (Blend GFPGAN inner face over swapped face and add grain)
        final = self._integrate_face(target_img, swapped, restored, target_face)
        
        out_path = os.path.join("output/samples/aya_swaps", "advanced_" + os.path.basename(target_path))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, final)
        print(f"[Agent] Success: {out_path}")

    def _integrate_face(self, original, swapped, restored, target_face):
        """Creates a highly feathered mask of the inner face to blend GFPGAN restored face onto the InsightFace swapped body."""
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        
        # 106-point landmarks; points 33+ are inner facial features (eyes, nose, mouth)
        # This explicitly avoids pasting over the target's original jawline, neck, or hair.
        points = target_face.landmark_2d_106.astype(np.int32)[33:]
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Heavy feathering to make the edge invisible
        mask = cv2.GaussianBlur(mask, (31, 31), 15)
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        
        # Blend GFPGAN (restored) over InsightFace (swapped)
        blended = (restored * mask_3ch + swapped * (1.0 - mask_3ch)).astype(np.float32)
            
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _load_profile(self):
        if os.path.exists(self.profile_path):
            print(f"[Agent] Loading persistent profile from {self.profile_path}...")
            emb = np.load(self.profile_path)
            class MockFace:
                def __init__(self, embedding):
                    self.embedding = embedding
                    self.normed_embedding = embedding
            self.aya_face = MockFace(emb)
            return True
        return False

    def learn_aya(self, ref_dir="data/references/aya"):
        print(f"[Agent] Learning Aya's features from {ref_dir}...")
        ref_paths = glob.glob(os.path.join(ref_dir, "*"))
        embeddings = []
        for path in ref_paths:
            img = cv2.imread(path)
            if img is None: continue
            faces = self.app.get(img)
            if len(faces) > 0:
                face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                embeddings.append(face.normed_embedding)
        if not embeddings: return False
        avg_emb = np.mean(embeddings, axis=0)
        normed_emb = avg_emb / np.linalg.norm(avg_emb)
        class MockFace:
            def __init__(self, embedding):
                self.embedding = embedding
                self.normed_embedding = embedding
        self.aya_face = MockFace(normed_emb)
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        np.save(self.profile_path, self.aya_face.embedding)
        return True

if __name__ == "__main__":
    orch = AyaOrchestrator()
    for t in glob.glob("data/targets/mixed/*.jpg"):
        orch.execute_swap(t)

