import abc
import cv2
import numpy as np
import insightface

class FaceSwapEngine(abc.ABC):
    @abc.abstractmethod
    def swap(self, source_img, target_img, profile_embedding):
        pass

class InsightFaceEngine(FaceSwapEngine):
    def __init__(self):
        self.app = insightface.app.FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

    def swap(self, img, target_img, profile_embedding):
        target_faces = self.app.get(target_img)
        if not target_faces: return img
        target_face = sorted(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        
        target_face_kps = np.asarray(target_face.kps, dtype=np.float32)
        aimg, M = insightface.utils.face_align.norm_crop2(img, target_face_kps, self.swapper.input_size[0])
        
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.swapper.input_std, self.swapper.input_size,
                                      (self.swapper.input_mean, self.swapper.input_mean, self.swapper.input_mean), swapRB=True)
        blob = np.asarray(blob, dtype=np.float32)
        
        latent = np.asarray(profile_embedding, dtype=np.float32).reshape((1, -1))
        emap = np.asarray(self.swapper.emap, dtype=np.float32)
        latent = np.dot(latent, emap).astype(np.float32)
        latent /= (np.linalg.norm(latent).astype(np.float32) + 1e-6)
        
        pred = self.swapper.session.run(self.swapper.output_names, {
            self.swapper.input_names[0]: blob, 
            self.swapper.input_names[1]: latent
        })[0]
        
        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        
        IM = cv2.invertAffineTransform(M)
        img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
        
        bgr_fake_warped = cv2.warpAffine(bgr_fake, IM, (img.shape[1], img.shape[0]), borderValue=0.0)
        img_white_warped = cv2.warpAffine(img_white, IM, (img.shape[1], img.shape[0]), borderValue=0.0)
        
        img_mask = np.clip(img_white_warped, 0, 255).astype(np.uint8)
        img_mask = cv2.GaussianBlur(img_mask, (15, 15), 5)
        mask_3ch = cv2.merge([img_mask, img_mask, img_mask]) / 255.0
        
        swapped_base = (bgr_fake_warped.astype(np.float32) * mask_3ch + img.astype(np.float32) * (1.0 - mask_3ch)).astype(np.uint8)
        return swapped_base

class FaceFusionEngine(FaceSwapEngine):
    def swap(self, source_img, target_img, profile_embedding):
        # Placeholder for FaceFusion implementation
        pass

class DreamIDEngine(FaceSwapEngine):
    def swap(self, source_img, target_img, profile_embedding):
        # Placeholder for DreamID implementation
        pass
