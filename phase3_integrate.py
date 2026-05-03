import os
import cv2
import numpy as np
import insightface
from gfpgan import GFPGANer
import glob

def create_face_mask(img, landmarks):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    points = landmarks.astype(np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.GaussianBlur(mask, (31, 31), 15)
    return mask

def match_texture_and_blend(swapped_img, target_img, mask):
    mask_blur = cv2.GaussianBlur(mask, (31, 31), 15)
    mask_3ch = cv2.merge([mask_blur, mask_blur, mask_blur]) / 255.0
    blended = (swapped_img * mask_3ch + target_img * (1.0 - mask_3ch)).astype(np.uint8)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    target_high_freq = cv2.Laplacian(target_gray, cv2.CV_64F)
    grain = np.clip(target_high_freq, -10, 10)
    final_img = blended.astype(np.float32)
    for i in range(3):
        final_img[:,:,i] += grain * (mask_blur / 255.0) * 0.8
    return np.clip(final_img, 0, 255).astype(np.uint8)

class Phase3:
    def __init__(self, profile_path="profiles/aya.npy"):
        print("Loading Restoration Models for Phase 3...")
        self.app = insightface.app.FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
        self.restorer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None
        )
        self.profile_path = profile_path

    def run(self, target_path, output_path):
        emb = np.load(self.profile_path)
        class MockFace:
            def __init__(self, embedding):
                self.embedding = embedding
                self.normed_embedding = embedding
        aya_face = MockFace(emb)
        
        target_img = cv2.imread(target_path)
        with open("temp_meta.txt", "r") as f:
            y1, y2, x1, x2 = map(int, f.read().split(","))
        
        refined_head = cv2.imread("temp_refined_head.png")
        if refined_head is None:
            print("No refined head found, using raw crop")
            refined_head = cv2.imread("temp_head_crop.png")
            
        refined_head = cv2.resize(refined_head, (x2-x1, y2-y1))
        
        # Integration
        crop_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        cv2.rectangle(crop_mask, (10, 10), (x2-x1-10, y2-y1-10), 255, -1)
        crop_mask = cv2.GaussianBlur(crop_mask, (51, 51), 25)
        crop_mask_3ch = cv2.merge([crop_mask, crop_mask, crop_mask]) / 255.0
        
        composite = target_img.copy()
        composite[y1:y2, x1:x2] = (refined_head * crop_mask_3ch + target_img[y1:y2, x1:x2] * (1.0 - crop_mask_3ch)).astype(np.uint8)
        
        # Identity Lock
        faces = self.app.get(composite)
        if faces:
            f_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
            swapped_final = self.swapper.get(composite, f_face, aya_face, paste_back=True)
            _, _, restored = self.restorer.enhance(swapped_final, has_aligned=False, only_center_face=False, paste_back=True)
            mask = create_face_mask(composite, f_face.landmark_2d_106)
            final = match_texture_and_blend(restored, composite, mask)
        else: final = composite
        
        cv2.imwrite(output_path, final)
        print(f"Final Success: {output_path}")

if __name__ == "__main__":
    p3 = Phase3()
    targets = glob.glob("target_pics/*.jpg")
    if targets:
        out_path = os.path.join("samples/aya_swaps", "final_sequential_" + os.path.basename(targets[0]))
        p3.run(targets[0], out_path)
