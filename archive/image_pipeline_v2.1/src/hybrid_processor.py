import cv2
import numpy as np
import onnxruntime as ort
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridProcessor:
    """Refactored to be pure InSwapper per AA001 directive."""
    def __init__(self, models_dir="image_pipeline/models"):
        self.models_dir = models_dir
        logger.info("Initializing InSwapper...")
        self.face_swapper = ort.InferenceSession(os.path.join(models_dir, "inswapper_128.onnx"), providers=['CPUExecutionProvider'])
        
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
        if not detection_result.face_landmarks: return None, None
        landmarks = detection_result.face_landmarks[0]
        h, w, _ = image.shape
        indices = [473, 468, 1, 291, 61]
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices], dtype=np.float32)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        oval_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in oval_indices], dtype=np.int32)
        cv2.fillPoly(mask, [oval_pts], 255)
        return pts, mask

    def get_validator_mask(self, image):
        _, mask = self.get_face_data(image)
        return mask

    def align_face(self, img, landmarks, is_inswapper=False):
        dst = self.inswapper_dst if is_inswapper else self.arcface_dst
        tform = cv2.estimateAffinePartial2D(landmarks, dst)[0]
        if tform is None: return None, None
        size = (128, 128) if is_inswapper else (112, 112)
        warped = cv2.warpAffine(img, tform, size)
        return warped, tform

    def swap_face(self, target_img, source_embedding, target_pts):
        target_face_align, tform = self.align_face(target_img, target_pts, is_inswapper=True)
        if tform is None: return target_img
        
        face_input = cv2.cvtColor(target_face_align, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        face_input = face_input.transpose((2, 0, 1))
        face_input = np.expand_dims(face_input, axis=0)
        source_emb = np.expand_dims(source_embedding, axis=0)
        
        swapped_face = self.face_swapper.run(None, {
            self.face_swapper.get_inputs()[0].name: face_input,
            self.face_swapper.get_inputs()[1].name: source_emb
        })[0][0]
        
        swapped_face = swapped_face.transpose((1, 2, 0)) * 255.0
        swapped_face = np.clip(swapped_face, 0, 255).astype(np.uint8)
        swapped_face = cv2.cvtColor(swapped_face, cv2.COLOR_RGB2BGR)
        
        inv_tform = cv2.invertAffineTransform(tform)
        h, w, _ = target_img.shape
        swapped_img = cv2.warpAffine(swapped_face, inv_tform, (w, h))
        return swapped_img

    def process_person(self, target_path, reference_paths, output_path):
        logger.info(f"Hybrid Pipeline (Pure InSwapper): Processing {os.path.basename(target_path)}...")
        
        target_img = cv2.imread(target_path)
        if target_img is None: return
        pts, face_mask = self.get_face_data(target_img)
        if pts is None: return
        
        # Load the precomputed embedding from the cache
        cache_path = os.path.join(self.models_dir, "reference_identity.pt")
        if not os.path.exists(cache_path):
            logger.error("No reference_identity.pt found. Please run precompute_identity.py first.")
            return
            
        cached_data = torch.load(cache_path, weights_only=False)
        elite_mean = cached_data["embedding"].detach().cpu().numpy().flatten()
        elite_mean = elite_mean / np.linalg.norm(elite_mean)
        
        swapped_img = self.swap_face(target_img, elite_mean, pts)
        
        # Seamless clone
        kernel_5 = np.ones((5, 5), np.uint8)
        eroded_mask = cv2.erode(face_mask, kernel_5, iterations=1)
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            M = cv2.moments(contours[0])
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (target_img.shape[1] // 2, target_img.shape[0] // 2)
            try:
                output = cv2.seamlessClone(swapped_img, target_img, eroded_mask, center, cv2.NORMAL_CLONE)
            except Exception as e:
                logger.error(f"Clone failed: {e}")
                output = swapped_img
        else:
            output = swapped_img

        # Bit-perfect background
        bg_mask = cv2.bitwise_not(face_mask)
        output = cv2.bitwise_and(target_img, target_img, mask=bg_mask) + cv2.bitwise_and(output, output, mask=face_mask)

        cv2.imwrite(output_path, output)
        logger.info(f"Saved photorealistic result to {output_path}")

if __name__ == "__main__":
    pass
