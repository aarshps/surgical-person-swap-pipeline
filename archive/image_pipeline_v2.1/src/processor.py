import cv2
import numpy as np
import onnxruntime as ort
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, models_dir="image_pipeline/models"):
        self.models_dir = models_dir
        self.face_swapper = ort.InferenceSession(os.path.join(models_dir, "inswapper_128.onnx"), providers=['CPUExecutionProvider'])
        self.face_recognizer = ort.InferenceSession(os.path.join(models_dir, "arcface_w600k_r50.onnx"), providers=['CPUExecutionProvider'])
        
        base_options = python.BaseOptions(model_asset_path=os.path.join(models_dir, "face_landmarker.task"))
        options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, num_faces=1)
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.arcface_dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        self.inswapper_dst = self.arcface_dst + 8.0

    def get_face_data(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detection_result = self.landmarker.detect(mp_image)
        if not detection_result.face_landmarks: return None
        landmarks = detection_result.face_landmarks[0]
        h, w, _ = image.shape
        indices = [473, 468, 1, 291, 61]
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices], dtype=np.float32)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        oval_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in oval_indices], dtype=np.int32)
        cv2.fillPoly(mask, [oval_pts], 255)
        return pts, mask

    def align_face(self, img, landmarks, is_inswapper=False):
        dst = self.inswapper_dst if is_inswapper else self.arcface_dst
        tform = cv2.estimateAffinePartial2D(landmarks, dst)[0]
        if tform is None: return None, None
        size = (128, 128) if is_inswapper else (112, 112)
        warped = cv2.warpAffine(img, tform, size)
        return warped, tform

    def get_face_embedding(self, aligned_face_112):
        if aligned_face_112 is None: return None
        face_img = cv2.cvtColor(aligned_face_112, cv2.COLOR_BGR2RGB)
        face_img = face_img.transpose((2, 0, 1)).astype(np.float32) / 127.5 - 1.0
        face_img = np.expand_dims(face_img, axis=0)
        embedding = self.face_recognizer.run(None, {self.face_recognizer.get_inputs()[0].name: face_img})[0]
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0: embedding /= norm
        return embedding

    def swap_face(self, target_img, source_embedding, target_pts):
        target_face_align, tform = self.align_face(target_img, target_pts, is_inswapper=True)
        if tform is None: return None, None
        
        # InSwapper usually expects BGR [0, 255] or [0, 1]?
        # Typical insightface usage: img = img.astype(np.float32)
        face_input = target_face_align.astype(np.float32) / 255.0
        face_input = face_input.transpose((2, 0, 1))
        face_input = np.expand_dims(face_input, axis=0)
        source_emb = np.expand_dims(source_embedding, axis=0)
        
        swapped_face = self.face_swapper.run(None, {
            self.face_swapper.get_inputs()[0].name: face_input,
            self.face_swapper.get_inputs()[1].name: source_emb
        })[0][0]
        
        swapped_face = swapped_face.transpose((1, 2, 0)) * 255.0
        swapped_face = np.clip(swapped_face, 0, 255).astype(np.uint8)
        
        # Color transfer in CIE LAB space to match the skin tone and lighting of the target face
        mask_128 = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(mask_128, (64, 64), 45, 255, -1)
        mask_bool = mask_128 > 0
        
        swapped_face_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_face_align_lab = cv2.cvtColor(target_face_align, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        src_mean = np.mean(swapped_face_lab[mask_bool], axis=0)
        src_std = np.std(swapped_face_lab[mask_bool], axis=0)
        tgt_mean = np.mean(target_face_align_lab[mask_bool], axis=0)
        tgt_std = np.std(target_face_align_lab[mask_bool], axis=0)
        
        matched_face_lab = swapped_face_lab.copy()
        for i in range(3):
            std_ratio = tgt_std[i] / (src_std[i] + 1e-5)
            std_ratio = np.clip(std_ratio, 0.1, 10.0)
            matched_face_lab[:, :, i] = (matched_face_lab[:, :, i] - src_mean[i]) * std_ratio + tgt_mean[i]
            
        matched_face_lab = np.clip(matched_face_lab, 0, 255).astype(np.uint8)
        swapped_face_matched = cv2.cvtColor(matched_face_lab, cv2.COLOR_LAB2BGR)
        
        return swapped_face_matched, tform

    def process_person(self, target_path, reference_paths, output_path):
        import torch
        logger.info(f"Architecture v2.1 (InSwapper + LAB Reinhard + Cons Mask): Processing {os.path.basename(target_path)}")
        target_img_cv = cv2.imread(target_path)
        if target_img_cv is None: return
        
        cache_path = "image_pipeline/models/reference_identity.pt"
        if os.path.exists(cache_path):
            cached_data = torch.load(cache_path, weights_only=False)
            elite_mean = cached_data["embedding"].detach().cpu().numpy().flatten()
            elite_mean = elite_mean / np.linalg.norm(elite_mean)
        else:
            logger.error("No reference_identity.pt found.")
            return
            
        target_data = self.get_face_data(target_img_cv)
        if target_data is None: return
        target_pts, face_mask = target_data
        
        swapped_face_matched, tform = self.swap_face(target_img_cv, elite_mean, target_pts)
        if tform is None:
            logger.error("Face swapping failed due to alignment error.")
            return
            
        inv_tform = cv2.invertAffineTransform(tform)
        h, w, _ = target_img_cv.shape
        swapped_img = cv2.warpAffine(swapped_face_matched, inv_tform, (w, h))
        
        # Inward-feathered soft blend (perfect background preservation & seamless blending)
        dist_transform = cv2.distanceTransform(face_mask, cv2.DIST_L2, 5)
        feather_width = 15.0
        x = np.clip(dist_transform / feather_width, 0.0, 1.0)
        soft_blend_mask_2d = 3 * (x ** 2) - 2 * (x ** 3)
        soft_blend_mask = np.expand_dims(soft_blend_mask_2d, axis=-1)
        output = (swapped_img * soft_blend_mask + target_img_cv * (1.0 - soft_blend_mask)).astype(np.uint8)

        cv2.imwrite(output_path, output)
        logger.info(f"Saved photorealistic result (Iteration 60 - InSwapper v2.1 LAB) to {output_path}")

if __name__ == "__main__":
    pass
