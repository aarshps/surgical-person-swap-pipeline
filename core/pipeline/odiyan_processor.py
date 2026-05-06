import os
import cv2
import numpy as np
import insightface
from skimage.exposure import match_histograms
import glob
import requests
import base64
import time
import subprocess
import shutil

import json

def log(msg):
    """Unbuffered logging for real-time status tracking."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def update_task_status(phase, progress=0, details=""):
    """Updates a JSON file for the dashboard to track real-time progress."""
    status = {
        "phase": phase,
        "progress": progress,
        "details": details,
        "timestamp": time.time()
    }
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/task_status.json", "w") as f:
            json.dump(status, f)
    except: pass

def match_colors(source, target):
    """Matches the color distribution of source to target."""
    return match_histograms(source, target, channel_axis=-1)

def inject_noise(image, intensity=0.002):
    """Adds slight gaussian noise to match film grain."""
    noise = np.random.normal(0, intensity * 255, image.shape).astype(np.int16)
    noisy_image = cv2.add(image.astype(np.int16), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def sharpen_image(image, amount=1.2, threshold=0):
    """Applies unsharp masking to enhance high-frequency details."""
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast_mask = cv2.absdiff(image, blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def get_laplacian_pyramid(img, levels=5):
    """Builds a Laplacian pyramid for multi-band blending."""
    gaussian_pyramid = [img.astype(np.float32)]
    for i in range(levels):
        img = cv2.pyrDown(img)
        gaussian_pyramid.append(img.astype(np.float32))
    
    laplacian_pyramid = [gaussian_pyramid[levels]]
    for i in range(levels, 0, -1):
        size = (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
    return laplacian_pyramid

def reconstruct_from_pyramid(laplacian_pyramid):
    """Reconstructs an image from its Laplacian pyramid."""
    levels = len(laplacian_pyramid) - 1
    img = laplacian_pyramid[0]
    for i in range(1, levels + 1):
        size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size)
        img = cv2.add(img, laplacian_pyramid[i])
    return img

def laplacian_blend(img1, img2, mask, levels=5):
    """Blends two images using Laplacian pyramids for seamless, sharp transitions."""
    lp1 = get_laplacian_pyramid(img1, levels)
    lp2 = get_laplacian_pyramid(img2, levels)
    
    # Gaussian pyramid for the mask
    gp_mask = [mask.astype(np.float32) / 255.0]
    for i in range(levels):
        mask = cv2.pyrDown(mask)
        gp_mask.append(mask.astype(np.float32) / 255.0)
    gp_mask.reverse()
    
    blended_pyramid = []
    for l1, l2, m in zip(lp1, lp2, gp_mask):
        m_3ch = cv2.merge([m, m, m])
        blended = l1 * m_3ch + l2 * (1.0 - m_3ch)
        blended_pyramid.append(blended)
    
    return np.clip(reconstruct_from_pyramid(blended_pyramid), 0, 255).astype(np.uint8)

def match_grain(source, target):
    """Extracts high-frequency grain from target and applies it to source."""
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # High-pass filter to get grain
    blurred = cv2.GaussianBlur(target_gray, (3, 3), 0)
    grain = cv2.subtract(target_gray, blurred)
    
    # Analyze target grain intensity
    grain_std = np.std(grain)
    
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    source_blurred = cv2.GaussianBlur(source_gray, (3, 3), 0)
    source_grain = cv2.subtract(source_gray, source_blurred)
    source_std = np.std(source_grain)
    
    # Scale source grain to match target grain intensity if target is sharper
    if grain_std > source_std:
        scale = grain_std / (source_std + 1e-6)
        source = sharpen_image(source, amount=scale * 0.5)
        
    return source

def create_face_mask(img, landmarks):
    """Creates a smooth convex hull mask from facial landmarks."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    points = landmarks.astype(np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.GaussianBlur(mask, (31, 31), 15)
    return mask

def match_texture_and_blend(swapped_img, target_img, mask):
    """Surgically blends texture and colors without losing identity."""
    mask_blur = cv2.GaussianBlur(mask, (31, 31), 15)
    mask_3ch = cv2.merge([mask_blur, mask_blur, mask_blur]) / 255.0
    
    # Preserve sharp details from the swapped image
    blended = (swapped_img.astype(np.float32) * mask_3ch + target_img.astype(np.float32) * (1.0 - mask_3ch)).astype(np.uint8)
    
    # Add high-frequency details back
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    target_high_freq = cv2.Laplacian(target_gray, cv2.CV_64F)
    grain = np.clip(target_high_freq, -10, 10)
    
    final_img = blended.astype(np.float32)
    for i in range(3):
        final_img[:,:,i] += grain * (mask_blur / 255.0) * 0.5
    
    return np.clip(final_img, 0, 255).astype(np.uint8)

class OdiyanSwapPipeline:
    def __init__(self, profile_path="data/profiles/odiyan.npy"):
        log("Initializing Hyper-Realistic Odiyan Pipeline...")
        self.app = insightface.app.FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
        self.odiyan_face = None
        self.profile_path = profile_path
        self.sd_url = "http://127.0.0.1:1234/sdapi/v1/img2img"

    def ensure_sd_server(self):
        try:
            requests.get("http://127.0.0.1:1234", timeout=2)
            return True
        except:
            if os.path.exists("start_server.sh"):
                log("Attempting to start Stable Diffusion Server...")
                subprocess.Popen(["bash", "start_server.sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for _ in range(15):
                    time.sleep(2)
                    try:
                        requests.get("http://127.0.0.1:1234", timeout=2)
                        log("SD Server Ready.")
                        return True
                    except: continue
            return False

    def load_profile(self):
        if os.path.exists(self.profile_path):
            log(f"Loading persistent profile from {self.profile_path}...")
            emb = np.load(self.profile_path).astype(np.float32)
            class MockFace:
                def __init__(self, embedding):
                    self.embedding = embedding.astype(np.float32)
                    self.normed_embedding = embedding.astype(np.float32)
            self.odiyan_face = MockFace(emb)
            return True
        return False

    def save_profile(self):
        if self.odiyan_face is not None:
            os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
            np.save(self.profile_path, self.odiyan_face.embedding.astype(np.float32))

    def learn_odiyan(self, ref_dir="odiyan_refs"):
        if self.load_profile(): return True
        log(f"Learning Odiyan's features from {ref_dir} (Advanced Identity Extraction)...")
        ref_paths = glob.glob(os.path.join(ref_dir, "*"))
        
        data = []
        for path in ref_paths:
            img = cv2.imread(path)
            if img is None: continue
            faces = self.app.get(img)
            if len(faces) > 0:
                # Pick the largest face (presumed to be Aya)
                face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                
                # Quality metrics: Face size and pose (using landmark symmetry)
                w = face.bbox[2] - face.bbox[0]
                h = face.bbox[3] - face.bbox[1]
                size = w * h
                
                # Simple frontality check (eye-to-nose distance symmetry)
                lmk = face.landmark_2d_106
                left_eye = lmk[35]
                right_eye = lmk[93]
                nose = lmk[54]
                dist_l = np.linalg.norm(left_eye - nose)
                dist_r = np.linalg.norm(right_eye - nose)
                symmetry = 1.0 - abs(dist_l - dist_r) / (max(dist_l, dist_r) + 1e-6)
                
                data.append({
                    'emb': face.normed_embedding.astype(np.float32),
                    'score': float(size * (symmetry ** 2))
                })
        
        if not data: return False
        
        # 1. Initial weighted average
        embs = np.array([d['emb'] for d in data], dtype=np.float32)
        scores = np.array([d['score'] for d in data], dtype=np.float32)
        scores = scores / np.sum(scores)
        avg_emb = np.average(embs, axis=0, weights=scores).astype(np.float32)
        
        # 2. Outlier removal (cosine similarity to weighted average)
        similarities = np.dot(embs, avg_emb)
        threshold = np.percentile(similarities, 30)
        
        final_embs = []
        final_scores = []
        for i, sim in enumerate(similarities):
            if sim >= threshold:
                final_embs.append(embs[i])
                final_scores.append(scores[i])
        
        # 3. Final refined average
        final_avg = np.average(final_embs, axis=0, weights=final_scores).astype(np.float32)
        normed_emb = (final_avg / (np.linalg.norm(final_avg) + 1e-6)).astype(np.float32)
        
        class MockFace:
            def __init__(self, embedding):
                self.embedding = embedding.astype(np.float32)
                self.normed_embedding = embedding.astype(np.float32)
        self.odiyan_face = MockFace(normed_emb)
        self.save_profile()
        log(f"Identity learned from {len(final_embs)} optimized frames.")
        return True

    def create_full_mask(self, img, landmarks):
        """Creates a surgically precise gradient mask for face and neck blending."""
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        points = landmarks.astype(np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Use a smaller dilation to keep the mask tight to the face/neck structure
        kernel = np.ones((20, 20), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Multi-stage blurring for a smoother transition at the edges (mitigates neck lines)
        mask = cv2.GaussianBlur(mask, (51, 51), 25)
        mask = cv2.GaussianBlur(mask, (101, 101), 50)
        return mask

    def refine_with_sd(self, image, prompt, denoising=0.35):
        if not self.ensure_sd_server(): return image
        log(f"Baking with Flux.1-schnell (High-Resolution 1024x1024, denoising={denoising})...")
        # Maximum quality as requested by the user
        image_res = cv2.resize(image, (1024, 1024))
        _, img_encoded = cv2.imencode('.png', image_res)
        img_b64 = base64.b64encode(img_encoded).decode('utf-8')
        
        enhanced_prompt = (
            "A professional high-resolution close-up portrait masterpiece. "
            f"The subject is {prompt}. "
            "Natural skin texture with soft pores, realistic human eyes with natural reflections. "
            "The image must strictly maintain the subject's unique facial structure and bone architecture. "
            "Cinematic soft lighting, balanced contrast, raw 8k photography aesthetic, natural focus."
        )
        
        payload = {
            "prompt": enhanced_prompt,
            "init_images": [img_b64],
            "denoising_strength": denoising,
            "steps": 4, 
            "cfg_scale": 1.0, 
            "sampler_name": "Euler",
            "width": 1024,
            "height": 1024
        }
        
        try:
            log("Sending high-res request to Flux Engine (ETA: ~15-20 mins on CPU)...")
            update_task_status("Phase 2: Flux Refinement", 30, "Baking high-res textures (1024x1024)...")
            response = requests.post(self.sd_url, json=payload, timeout=2400)
            if response.status_code == 200:
                log("Flux refinement complete. Decoding results...")
                data = response.json()
                img_res = base64.b64decode(data['images'][0])
                nparr = np.frombuffer(img_res, np.uint8)
                refined = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return sharpen_image(refined, amount=0.35)
            else:
                log(f"Flux Error: Server returned status {response.status_code}")
        except Exception as e:
            log(f"Flux Engine Error: {e}")
        return image

    def process_target(self, target_path, output_path, odiyan_desc="Aya, maintaining her unique facial features and bone structure"):
        if self.odiyan_face is None: return False
        target_img = cv2.imread(target_path)
        if target_img is None: return False
        
        target_faces = self.app.get(target_img)
        if not target_faces: return False
        target_face = sorted(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        
        # Phase 1: Foundational Anchoring
        update_task_status("Phase 1: Anchoring Identity", 10, os.path.basename(target_path))
        log("Phase 1: Anchoring Identity (InsightFace)...")
        
        img = target_img
        target_face_kps = np.asarray(target_face.kps, dtype=np.float32)
        aimg, M = insightface.utils.face_align.norm_crop2(img, target_face_kps, self.swapper.input_size[0])
        
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.swapper.input_std, self.swapper.input_size,
                                      (self.swapper.input_mean, self.swapper.input_mean, self.swapper.input_mean), swapRB=True)
        # Final safety cast for blob
        blob = np.asarray(blob, dtype=np.float32)
        
        # Force float32 for latent and its transformation
        latent = np.asarray(self.odiyan_face.normed_embedding, dtype=np.float32).reshape((1, -1))
        emap = np.asarray(self.swapper.emap, dtype=np.float32)
        latent = np.dot(latent, emap).astype(np.float32)
        latent /= (np.linalg.norm(latent).astype(np.float32) + 1e-6)
        
        # Run inference manually with strictly float32 buffers
        pred = self.swapper.session.run(self.swapper.output_names, {
            self.swapper.input_names[0]: blob, 
            self.swapper.input_names[1]: latent
        })[0]
        
        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        
        # Manual paste-back logic
        IM = cv2.invertAffineTransform(M)
        img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
        
        bgr_fake_warped = cv2.warpAffine(bgr_fake, IM, (img.shape[1], img.shape[0]), borderValue=0.0)
        img_white_warped = cv2.warpAffine(img_white, IM, (img.shape[1], img.shape[0]), borderValue=0.0)
        
        img_mask = np.clip(img_white_warped, 0, 255).astype(np.uint8)
        img_mask = cv2.GaussianBlur(img_mask, (15, 15), 5)
        mask_3ch = cv2.merge([img_mask, img_mask, img_mask]) / 255.0
        
        swapped_base = (bgr_fake_warped.astype(np.float32) * mask_3ch + img.astype(np.float32) * (1.0 - mask_3ch)).astype(np.uint8)
        swapped_base = match_colors(swapped_base, target_img)
        
        # Phase 2: Flux.1 Detail Baking
        update_task_status("Phase 2: Flux Refinement", 25, "Baking textures...")
        log("Phase 2: Surgical Crop & Flux Refinement (Identity-Preserving)...")
        bbox = target_face.bbox.astype(int)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        y1, y2 = max(0, bbox[1]-int(h*0.7)), min(target_img.shape[0], bbox[3]+int(h*1.4))
        x1, x2 = max(0, bbox[0]-int(w*0.9)), min(target_img.shape[1], bbox[2]+int(w*0.9))
        
        head_crop = swapped_base[y1:y2, x1:x2]
        target_crop = target_img[y1:y2, x1:x2]
        
        # Refine with Flux
        # We can't easily track inner SD steps, so we just set phase
        refined_head = self.refine_with_sd(head_crop, odiyan_desc, denoising=0.35)
        refined_head = cv2.resize(refined_head, (x2-x1, y2-y1))
        
        # Phase 3: Frequency Harmonization
        update_task_status("Phase 3: Harmonization", 85, "Matching grain...")
        log("Phase 3: Harmonizing Sharpness & Grain...")
        refined_head = match_colors(refined_head, target_crop)
        refined_head = match_grain(refined_head, target_crop)
        
        # Phase 4: Integration
        update_task_status("Phase 4: Integration", 95, "Laplacian blending...")
        log("Phase 4: Laplacian Integration...")
        mask = self.create_full_mask(target_img, target_face.landmark_2d_106)
        mask_crop = mask[y1:y2, x1:x2]
        
        final_crop = laplacian_blend(refined_head, target_crop, mask_crop, levels=4)
        
        final = target_img.copy()
        final[y1:y2, x1:x2] = final_crop
        
        final = sharpen_image(final, amount=0.03)
        
        cv2.imwrite(output_path, inject_noise(final, 0.0001))
        update_task_status("COMPLETED", 100, os.path.basename(output_path))
        log(f"Final Success (High-Likeness): {output_path}")
        return True

if __name__ == "__main__":
    pipeline = OdiyanSwapPipeline()
    if pipeline.learn_odiyan(ref_dir="odiyan_refs"):
        desc = "Aya, natural realism, maintaining distinct facial features, sharp photography"
        os.makedirs("samples/odiyan_swaps", exist_ok=True)
        os.makedirs("data/failed", exist_ok=True)
        log("Odiyan Daemon is now running. Watching 'target_pics/' for new images...")
        
        while True:
            targets = glob.glob("target_pics/*")
            for t in targets:
                if not os.path.isfile(t): continue
                
                output_name = "surgical_" + os.path.basename(t)
                output_path = os.path.join("samples/odiyan_swaps", output_name)
                
                if not os.path.exists(output_path):
                    log(f"Found new target: {t}")
                    success = pipeline.process_target(t, output_path, odiyan_desc=desc)
                    if not success:
                        failed_path = os.path.join("data/failed", os.path.basename(t))
                        log(f"Moving failed image to {failed_path}")
                        shutil.move(t, failed_path)
            
            time.sleep(5)
