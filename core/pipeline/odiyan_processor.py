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

def log(msg):
    """Unbuffered logging for real-time status tracking."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

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
            emb = np.load(self.profile_path)
            class MockFace:
                def __init__(self, embedding):
                    self.embedding = embedding
                    self.normed_embedding = embedding
            self.odiyan_face = MockFace(emb)
            return True
        return False

    def save_profile(self):
        if self.odiyan_face is not None:
            os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
            np.save(self.profile_path, self.odiyan_face.embedding)

    def learn_odiyan(self, ref_dir="odiyan_refs"):
        if self.load_profile(): return True
        log(f"Learning Odiyan's features from {ref_dir}...")
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
        self.odiyan_face = MockFace(normed_emb)
        self.save_profile()
        return True

    def create_full_mask(self, img, landmarks):
        """Creates a smooth, broad mask covering the face, neck, and upper chest."""
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        points = landmarks.astype(np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Dilate mask heavily so the core remains 255 after extreme blurring
        kernel = np.ones((100, 100), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Broader, aggressive blur to eliminate lines
        mask = cv2.GaussianBlur(mask, (151, 151), 75)
        return mask

    def refine_with_sd(self, image, prompt, denoising=0.48):
        if not self.ensure_sd_server(): return image
        log(f"Baking with Flux.1-schnell (Bleeding-Edge, denoising={denoising})...")
        image_res = cv2.resize(image, (1024, 1024))
        _, img_encoded = cv2.imencode('.png', image_res)
        img_b64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # Flux-optimized prompt for realism and identity retention (Aya likeness)
        enhanced_prompt = f"{prompt}, hyper-realistic photography, perfectly matching original facial structure, detailed pores, natural lighting, sharp focus, 8k"
        
        payload = {
            "prompt": enhanced_prompt,
            "negative_prompt": "cartoon, painting, blurry, deformed, bad anatomy, soft, low quality, watermark, artificial, plastic",
            "init_images": [img_b64],
            "denoising_strength": denoising,
            "steps": 4, 
            "cfg_scale": 1.0, 
            "sampler_name": "Euler",
            "width": 1024,
            "height": 1024
        }
        
        try:
            response = requests.post(self.sd_url, json=payload, timeout=600)
            if response.status_code == 200:
                data = response.json()
                img_res = base64.b64decode(data['images'][0])
                nparr = np.frombuffer(img_res, np.uint8)
                refined = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                # Apply post-SD sharpening
                return sharpen_image(refined, amount=0.5)
        except: pass
        return image

    def process_target(self, target_path, output_path, odiyan_desc="Aya, beautiful woman with long dark hair, natural realism, maintaining distinct facial features"):
        if self.odiyan_face is None: return False
        target_img = cv2.imread(target_path)
        if target_img is None: return False
        
        target_faces = self.app.get(target_img)
        if not target_faces: return False
        target_face = sorted(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        
        # Phase 1: Foundational Anchoring
        log("Phase 1: Anchoring Identity...")
        swapped_base = self.swapper.get(target_img, target_face, self.odiyan_face, paste_back=True)
        swapped_base = match_colors(swapped_base, target_img)
        
        # Phase 2: Ultra-Res Detail Baking
        log("Phase 2: Surgical Crop & SD Bake (1024x1024+)...")
        bbox = target_face.bbox.astype(int)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        # Broader crop to include hair and neck for seamless blending
        y1, y2 = max(0, bbox[1]-int(h*0.6)), min(target_img.shape[0], bbox[3]+int(h*1.3))
        x1, x2 = max(0, bbox[0]-int(w*0.8)), min(target_img.shape[1], bbox[2]+int(w*0.8))
        
        head_crop = swapped_base[y1:y2, x1:x2]
        target_crop = target_img[y1:y2, x1:x2]
        
        refined_head = self.refine_with_sd(head_crop, f"{odiyan_desc}, realistic skin texture, sharp", denoising=0.48)
        refined_head = cv2.resize(refined_head, (x2-x1, y2-y1))
        
        # Phase 3: Frequency Harmonization (Grain & Sharpness Matching)
        log("Phase 3: Frequency Harmonization & Grain Matching...")
        refined_head = match_colors(refined_head, target_crop)
        refined_head = match_grain(refined_head, target_crop)
        
        # Phase 4: Surgical Integration (Laplacian Blending)
        log("Phase 4: Laplacian Integration...")
        mask = self.create_full_mask(target_img, target_face.landmark_2d_106)
        mask_crop = mask[y1:y2, x1:x2]
        
        # Composite refined head back into target image at crop level
        final_crop = laplacian_blend(refined_head, target_crop, mask_crop, levels=4)
        
        # Paste the refined crop back onto the full target image
        final = target_img.copy()
        final[y1:y2, x1:x2] = final_crop
        
        # Final subtle sharpening of the whole image to unify texture
        final = sharpen_image(final, amount=0.1)
        
        cv2.imwrite(output_path, inject_noise(final, 0.0002))
        log(f"Final Success: {output_path}")
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
