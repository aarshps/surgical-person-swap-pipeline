import os
import cv2
import numpy as np
import insightface
from gfpgan import GFPGANer
from skimage.exposure import match_histograms
import glob
import requests
import base64
import time
import subprocess

def match_colors(source, target):
    """Matches the color distribution of source to target."""
    return match_histograms(source, target, channel_axis=-1)

def inject_noise(image, intensity=0.002):
    """Adds slight gaussian noise to match film grain."""
    noise = np.random.normal(0, intensity * 255, image.shape).astype(np.int16)
    noisy_image = cv2.add(image.astype(np.int16), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

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
    blended = (swapped_img * mask_3ch + target_img * (1.0 - mask_3ch)).astype(np.uint8)
    
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    target_high_freq = cv2.Laplacian(target_gray, cv2.CV_64F)
    grain = np.clip(target_high_freq, -10, 10)
    
    final_img = blended.astype(np.float32)
    for i in range(3):
        final_img[:,:,i] += grain * (mask_blur / 255.0) * 0.8
    
    return np.clip(final_img, 0, 255).astype(np.uint8)

class OdiyanSwapPipeline:
    def __init__(self, profile_path="data/profiles/odiyan.npy"):
        print("Initializing Hyper-Realistic Odiyan Pipeline...")
        self.app = insightface.app.FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
        self.restorer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None
        )
        self.odiyan_face = None
        self.profile_path = profile_path
        self.sd_url = "http://127.0.0.1:1234/sdapi/v1/img2img"

    def ensure_sd_server(self):
        try:
            requests.get("http://127.0.0.1:1234", timeout=2)
            return True
        except:
            print("Starting Stable Diffusion Server...")
            subprocess.Popen(["bash", "start_server.sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(30):
                time.sleep(2)
                try:
                    requests.get("http://127.0.0.1:1234", timeout=2)
                    print("SD Server Ready.")
                    return True
                except: continue
            return False

    def load_profile(self):
        if os.path.exists(self.profile_path):
            print(f"Loading persistent profile from {self.profile_path}...")
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

    def learn_odiyan(self, ref_dir="data/references/odiyan"):
        if self.load_profile(): return True
        print(f"Learning Odiyan's features from {ref_dir}...")
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

    def refine_with_sd(self, image, prompt, denoising=0.4):
        print(f"Baking details with SD (denoising={denoising})...")
        # Ensure image is exactly 512x512 for memory efficiency and server expectations
        image_res = cv2.resize(image, (512, 512))
        _, img_encoded = cv2.imencode('.png', image_res)
        img_b64 = base64.b64encode(img_encoded).decode('utf-8')
        
        payload = {
            "prompt": prompt,
            "negative_prompt": "cartoon, painting, illustration, blurry, deformed, bad anatomy, (low quality:2)",
            "init_images": [img_b64],
            "denoising_strength": denoising,
            "steps": 15,
            "cfg_scale": 7.0,
            "width": 512,
            "height": 512
        }
        
        try:
            print(f"Sending request to SD server at {self.sd_url}...")
            start_time = time.time()
            response = requests.post(self.sd_url, json=payload, timeout=1200) # 20 minute timeout
            if response.status_code == 200:
                print(f"SD Request successful in {int(time.time() - start_time)}s")
                data = response.json()
                img_res = base64.b64decode(data['images'][0])
                nparr = np.frombuffer(img_res, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                print(f"SD Server returned error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"SD Error: {e}")
        return image

    def process_target(self, target_path, output_path, odiyan_desc="woman with long dark hair, realistic skin"):
        if self.odiyan_face is None: return
        target_img = cv2.imread(target_path)
        if target_img is None: return
        target_faces = self.app.get(target_img)
        if not target_faces: return
        target_face = sorted(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        
        # 1. Anchor
        print("Phase 1: Anchoring identity...")
        swapped_base = self.swapper.get(target_img, target_face, self.odiyan_face, paste_back=True)
        swapped_base = match_colors(swapped_base, target_img)
        
        # 2. Surgical Crop
        print("Phase 2: Surgical Crop & SD Bake...")
        bbox = target_face.bbox.astype(int)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        y1, y2 = max(0, bbox[1]-int(h*1.0)), min(target_img.shape[0], bbox[3]+int(h*0.4))
        x1, x2 = max(0, bbox[0]-int(w*0.6)), min(target_img.shape[1], bbox[2]+int(w*0.6))
        
        head_crop = swapped_base[y1:y2, x1:x2]
        self.ensure_sd_server()
        refined_head = self.refine_with_sd(head_crop, f"{odiyan_desc}, photorealistic", denoising=0.25)
        refined_head = cv2.resize(refined_head, (x2-x1, y2-y1))
        
        # 3. Paste Back
        print("Phase 3: Re-integration...")
        crop_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        cv2.rectangle(crop_mask, (10, 10), (x2-x1-10, y2-y1-10), 255, -1)
        crop_mask = cv2.GaussianBlur(crop_mask, (51, 51), 25)
        crop_mask_3ch = cv2.merge([crop_mask, crop_mask, crop_mask]) / 255.0
        
        composite = target_img.copy()
        composite[y1:y2, x1:x2] = (refined_head * crop_mask_3ch + target_img[y1:y2, x1:x2] * (1.0 - crop_mask_3ch)).astype(np.uint8)

        # 4. Final Identity Lock
        print("Phase 4: Final Identity Lock...")
        final_faces = self.app.get(composite)
        if final_faces:
            f_face = sorted(final_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
            swapped_final = self.swapper.get(composite, f_face, self.odiyan_face, paste_back=True)
            _, _, restored = self.restorer.enhance(swapped_final, has_aligned=False, only_center_face=False, paste_back=True)
            mask = create_face_mask(composite, f_face.landmark_2d_106)
            final = match_texture_and_blend(restored, composite, mask)
        else: final = composite

        cv2.imwrite(output_path, inject_noise(final, 0.001))
        print(f"Success: {output_path}")

if __name__ == "__main__":
    pipeline = OdiyanSwapPipeline()
    if pipeline.learn_odiyan():
        desc = "Odiyan, beautiful woman, long dark wavy hair, elegant, detailed skin"
        targets = glob.glob("data/targets/mixed/*")
        os.makedirs("output/samples/odiyan_swaps", exist_ok=True)
        for t in targets:
            pipeline.process_target(t, os.path.join("output/samples/odiyan_swaps", "surgical_" + os.path.basename(t)), odiyan_desc=desc)
