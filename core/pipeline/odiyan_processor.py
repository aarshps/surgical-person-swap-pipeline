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

FLUX_WIDTH = 1024
FLUX_HEIGHT = 1024
FLUX_STEPS = 4
FLUX_CFG_SCALE = 1.0
FLUX_DENOISING_STRENGTH = 0.32
FLUX_TIMEOUT_SECONDS = 2400
IDENTITY_ANCHOR_STRENGTH = 0.34

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

def normalize_lighting_and_saturation(source, target, strength=0.75):
    """Bounds contrast/saturation to the target crop without flattening detail."""
    if source.size == 0 or target.size == 0:
        return source

    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_l = src_lab[:, :, 0]
    tgt_l = tgt_lab[:, :, 0]
    src_mean, src_std = cv2.meanStdDev(src_l)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt_l)
    src_mean, src_std = float(src_mean[0][0]), float(src_std[0][0])
    tgt_mean, tgt_std = float(tgt_mean[0][0]), float(tgt_std[0][0])

    contrast_scale = np.clip(tgt_std / (src_std + 1e-6), 0.72, 1.18)
    normalized_l = (src_l - src_mean) * contrast_scale + tgt_mean
    src_lab[:, :, 0] = src_l * (1.0 - strength) + normalized_l * strength

    src_chroma = np.sqrt((src_lab[:, :, 1] - 128.0) ** 2 + (src_lab[:, :, 2] - 128.0) ** 2)
    tgt_chroma = np.sqrt((tgt_lab[:, :, 1] - 128.0) ** 2 + (tgt_lab[:, :, 2] - 128.0) ** 2)
    src_sat = float(np.mean(src_chroma))
    tgt_sat = float(np.mean(tgt_chroma))

    sat_scale = np.clip(tgt_sat / (src_sat + 1e-6), 0.78, 1.08)
    src_lab[:, :, 1] = 128.0 + (src_lab[:, :, 1] - 128.0) * sat_scale
    src_lab[:, :, 2] = 128.0 + (src_lab[:, :, 2] - 128.0) * sat_scale

    return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def create_inner_identity_mask(shape, landmarks, crop_bounds=None):
    """Builds an inner-face mask for eyes, nose, mouth, and likeness-critical planes."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    points = landmarks.astype(np.int32)
    if points.shape[0] > 33:
        points = points[33:]

    if crop_bounds is not None:
        _, _, x1, _ = crop_bounds
        y1 = crop_bounds[0]
        points = points - np.array([x1, y1], dtype=np.int32)

    h, w = mask.shape[:2]
    points[:, 0] = np.clip(points[:, 0], 0, max(0, w - 1))
    points[:, 1] = np.clip(points[:, 1], 0, max(0, h - 1))
    if len(points) < 3:
        return mask

    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    kernel_size = max(15, (min(h, w) // 18) | 1)
    mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), max(1, kernel_size // 2))
    return mask

def identity_anchor_blend(refined, anchor, identity_mask, strength=IDENTITY_ANCHOR_STRENGTH):
    """Reintroduces the InsightFace anchor inside identity-critical regions after Flux."""
    if refined.shape != anchor.shape:
        anchor = cv2.resize(anchor, (refined.shape[1], refined.shape[0]))
    alpha = (identity_mask.astype(np.float32) / 255.0)[:, :, None] * strength
    blended = refined.astype(np.float32) * (1.0 - alpha) + anchor.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)

def compute_square_head_crop(img_shape, bbox):
    """Returns a square crop that preserves Flux geometry and includes hair/neck context."""
    height, width = img_shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)

    crop_size = int(max(face_w * 2.6, face_h * 2.9))
    crop_size = max(crop_size, max(face_w, face_h))
    crop_size = min(crop_size, width, height)

    cx = x1 + face_w // 2
    cy = y1 + int(face_h * 0.58)
    half = crop_size // 2

    crop_x1 = max(0, min(width - 1, cx - half))
    crop_y1 = max(0, min(height - 1, cy - half))
    crop_x2 = min(width, crop_x1 + crop_size)
    crop_y2 = min(height, crop_y1 + crop_size)

    crop_x1 = max(0, crop_x2 - crop_size)
    crop_y1 = max(0, crop_y2 - crop_size)
    return int(crop_y1), int(crop_y2), int(crop_x1), int(crop_x2)

def taper_mask_edges(mask, margin_ratio=0.09):
    """Keeps crop borders anchored to the target image to avoid rectangular seams."""
    h, w = mask.shape[:2]
    margin = max(8, int(min(h, w) * margin_ratio))
    edge = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(edge, (margin, margin), (max(margin, w - margin - 1), max(margin, h - margin - 1)), 255, -1)
    blur = max(9, (margin * 2 + 1) | 1)
    edge = cv2.GaussianBlur(edge, (blur, blur), max(1, margin))
    tapered = (mask.astype(np.float32) * (edge.astype(np.float32) / 255.0))
    return np.clip(tapered, 0, 255).astype(np.uint8)

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
        """Creates a broad face/neck mask for edge-tapered Laplacian integration."""
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        points = landmarks.astype(np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

        x, y, w, h = cv2.boundingRect(hull)
        neck_center = (x + w // 2, y + int(h * 1.12))
        neck_axes = (max(8, int(w * 0.48)), max(8, int(h * 0.45)))
        cv2.ellipse(mask, neck_center, neck_axes, 0, 0, 360, 255, -1)

        kernel_size = max(31, (int(max(w, h) * 0.28) | 1))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        feather = max(31, (int(max(w, h) * 0.42) | 1))
        mask = cv2.GaussianBlur(mask, (feather, feather), max(1, feather // 3))
        return mask

    def refine_with_sd(self, image, prompt, denoising=FLUX_DENOISING_STRENGTH):
        if not self.ensure_sd_server(): return image
        log(f"Baking with Flux.1-schnell (High-Resolution 1024x1024, denoising={denoising})...")
        image_res = cv2.resize(image, (FLUX_WIDTH, FLUX_HEIGHT))
        _, img_encoded = cv2.imencode('.png', image_res)
        img_b64 = base64.b64encode(img_encoded).decode('utf-8')

        # Flux prompt is intentionally restrained; identity comes from the anchor, not text overdrive.
        enhanced_prompt = (
            f"{prompt}, exact same facial geometry and expression as source, "
            "raw documentary portrait, natural skin texture, balanced exposure, "
            "soft real-world lighting, visible pores, same camera perspective"
        )

        payload = {
            "prompt": enhanced_prompt,
            "negative_prompt": (
                "cartoon, painting, illustration, blurry, deformed, bad anatomy, "
                "overprocessed, high contrast, oversaturated, blown highlights, "
                "crushed shadows, plastic skin, waxy, airbrushed, halo, seam, watermark"
            ),
            "init_images": [img_b64],
            "denoising_strength": denoising,
            "steps": FLUX_STEPS,
            "cfg_scale": FLUX_CFG_SCALE,
            "sampler_name": "Euler",
            "width": FLUX_WIDTH,
            "height": FLUX_HEIGHT
        }

        try:
            log("Sending high-res request to Flux Engine (ETA: ~15-20 mins on CPU)...")
            update_task_status("Phase 2: Flux Refinement", 30, "Baking high-res textures (1024x1024)...")
            response = requests.post(self.sd_url, json=payload, timeout=FLUX_TIMEOUT_SECONDS)
            if response.status_code == 200:
                log("Flux refinement complete. Decoding results...")
                data = response.json()
                img_res = base64.b64decode(data['images'][0])
                nparr = np.frombuffer(img_res, np.uint8)
                refined = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return refined
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
        y1, y2, x1, x2 = compute_square_head_crop(target_img.shape, bbox)
        
        head_crop = swapped_base[y1:y2, x1:x2]
        target_crop = target_img[y1:y2, x1:x2]
        
        refined_head = self.refine_with_sd(
            head_crop,
            f"{odiyan_desc}, realistic skin texture, identity-preserving",
            denoising=FLUX_DENOISING_STRENGTH
        )
        refined_head = cv2.resize(refined_head, (x2-x1, y2-y1))
        
        # Phase 3: Frequency Harmonization
        update_task_status("Phase 3: Harmonization", 85, "Matching grain...")
        log("Phase 3: Harmonizing Sharpness & Grain...")
        refined_head = match_colors(refined_head, target_crop)
        refined_head = normalize_lighting_and_saturation(refined_head, target_crop)
        identity_mask = create_inner_identity_mask(refined_head.shape, target_face.landmark_2d_106, crop_bounds=(y1, y2, x1, x2))
        refined_head = identity_anchor_blend(refined_head, head_crop, identity_mask)
        refined_head = match_grain(refined_head, target_crop)
        
        # Phase 4: Integration
        update_task_status("Phase 4: Integration", 95, "Laplacian blending...")
        log("Phase 4: Laplacian Integration...")
        mask = self.create_full_mask(target_img, target_face.landmark_2d_106)
        mask_crop = taper_mask_edges(mask[y1:y2, x1:x2])
        
        pyramid_levels = 5 if min(refined_head.shape[:2]) >= 768 else 4
        final_crop = laplacian_blend(refined_head, target_crop, mask_crop, levels=pyramid_levels)
        
        final = target_img.copy()
        final[y1:y2, x1:x2] = final_crop
        
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
