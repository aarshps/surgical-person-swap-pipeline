import os
import cv2
import numpy as np
import insightface
from gfpgan import GFPGANer
import glob

def create_face_mask(img, landmarks, inner_only=False):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    points = landmarks.astype(np.int32)
    if inner_only:
        # Exclude the jaw/contour points (first 33 points in 106-point landmarks)
        # This restricts the mask to eyebrows, eyes, nose, and mouth.
        points = points[33:]
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    
    blur_amount = 15 if inner_only else 31
    mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), blur_amount//2)
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
    def __init__(self, profile_path="data/profiles/aya.npy"):
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
            y1, y2, x1, x2, crop_size, pad_bottom, pad_right = map(int, f.read().split(","))
        
        refined_head = cv2.imread("temp_refined_head.png")
        if refined_head is None:
            print("No refined head found, using raw crop")
            refined_head = cv2.imread("temp_head_crop.png")
            
        # 1. Resize back to the full square size
        refined_head = cv2.resize(refined_head, (crop_size, crop_size))
        
        # 2. Crop out the padding to match the original bounding box (x2-x1, y2-y1)
        # If we padded bottom and right, the original image is at [0 : crop_size-pad_bottom, 0 : crop_size-pad_right]
        orig_h = crop_size - pad_bottom
        orig_w = crop_size - pad_right
        
        refined_head_unpadded = refined_head[0:orig_h, 0:orig_w]
        
        # Integration (Dynamic Organic Masking)
        crop_faces = self.app.get(refined_head_unpadded)
        crop_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        if crop_faces:
            c_face = sorted(crop_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
            bbox = c_face.bbox.astype(int)
            w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
            cx, cy = bbox[0] + w//2, bbox[1] + h//2
            
            # Draw an ellipse that covers hair (top and sides) but avoids the lower chest area
            axes = (int(w * 1.1), int(h * 1.4))
            # Shift center UP to protect the chest
            center = (cx, int(cy - h * 0.15))
            cv2.ellipse(crop_mask, center, axes, 0, 0, 360, 255, -1)
        else:
            # Fallback if no face detected in crop
            cx, cy = orig_w // 2, int(orig_h * 0.4)
            axes = (int(orig_w * 0.4), int(orig_h * 0.5))
            cv2.ellipse(crop_mask, (cx, cy), axes, 0, 0, 360, 255, -1)
            
        # Heavy blur for seamless feathering
        crop_mask = cv2.GaussianBlur(crop_mask, (81, 81), 40)
        crop_mask_3ch = cv2.merge([crop_mask, crop_mask, crop_mask]) / 255.0
        
        composite = target_img.copy()
        composite[y1:y2, x1:x2] = (refined_head_unpadded * crop_mask_3ch + target_img[y1:y2, x1:x2] * (1.0 - crop_mask_3ch)).astype(np.uint8)
        
        # Identity Lock
        faces = self.app.get(composite)
        if faces:
            f_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
            swapped_final = self.swapper.get(composite, f_face, aya_face, paste_back=True)
            _, _, restored = self.restorer.enhance(swapped_final, has_aligned=False, only_center_face=False, paste_back=True)
            
            # Use inner_only=True so we only paste back GFPGAN's eyes/nose/mouth.
            # This preserves the photorealistic skin/lighting generated by Stable Diffusion.
            mask = create_face_mask(composite, f_face.landmark_2d_106, inner_only=True)
            final = match_texture_and_blend(restored, composite, mask)
        else: final = composite
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, final)
        print(f"Final Success: {output_path}")

if __name__ == "__main__":
    import sys
    p3 = Phase3()
    if len(sys.argv) > 2:
        p3.run(sys.argv[1], sys.argv[2])
    else:
        targets = glob.glob("data/targets/mixed/*.jpg")
        if targets:
            out_path = os.path.join("output/samples/aya_swaps", "final_sequential_" + os.path.basename(targets[0]))
            p3.run(targets[0], out_path)
