import os
import cv2
import numpy as np
import insightface
from gfpgan import GFPGANer
from skimage.exposure import match_histograms
import glob
import sys

def match_colors(source, target):
    """Matches the color distribution of source to target."""
    return match_histograms(source, target, channel_axis=-1)

def inject_noise(image, intensity=0.02):
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
    # Blur the mask to create a soft transition
    mask = cv2.GaussianBlur(mask, (19, 19), 11)
    return mask

def match_texture_and_blend(swapped_img, target_img, mask):
    """
    Surgically preserves identity while matching skin texture.
    Avoids Poisson blending to prevent identity 'wash-out'.
    """
    # 1. High-Precision Alpha Blending
    # We use a very soft feather on the mask to make the transition invisible
    # but keep the core face (Mrunal's features) 100% intact.
    mask_blur = cv2.GaussianBlur(mask, (31, 31), 15)
    mask_3ch = cv2.merge([mask_blur, mask_blur, mask_blur]) / 255.0
    
    # Blend the swapped image with the target
    blended = (swapped_img * mask_3ch + target_img * (1.0 - mask_3ch)).astype(np.uint8)

    # 2. Non-Destructive Texture Transfer
    # We extract the 'grain' from the target skin and overlay it.
    # This matches the skin's 'feel' without changing the facial structure.
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    target_high_freq = cv2.Laplacian(target_gray, cv2.CV_64F)
    
    # Normalize the grain
    grain = np.clip(target_high_freq, -10, 10) # Keep it subtle
    
    # Apply grain only where the face is
    final_img = blended.astype(np.float32)
    for i in range(3):
        final_img[:,:,i] += grain * (mask_blur / 255.0) * 0.8
    
    return np.clip(final_img, 0, 255).astype(np.uint8)

def hyper_realistic_swap(target_path, output_name):
    print("Initializing Models...")
    app = insightface.app.FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
    
    restorer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )

    # 3. Aggregate Source Face (Mrunal Thakur)
    print("Aggregating source embeddings...")
    ref_paths = glob.glob("mrunal_refs/*.jpg")
    ref_paths.append("mrunal/images_1d40e660-cc88-49.jpg")
    
    embeddings = []
    source_face = None
    for path in ref_paths:
        img = cv2.imread(path)
        if img is None: continue
        faces = app.get(img)
        if len(faces) > 0:
            # We take the largest face and weight it
            face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
            embeddings.append(face.normed_embedding)
            if source_face is None: source_face = face
                
    if not embeddings: return
    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = avg_emb / np.linalg.norm(avg_emb)
    source_face['embedding'] = avg_emb
    
    # 4. Load and Process Target
    target_img = cv2.imread(target_path)
    target_faces = app.get(target_img)
    if len(target_faces) == 0: return
    target_face = sorted(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
    
    # 5. Swap
    print("Performing initial swap...")
    swapped_img = swapper.get(target_img, target_face, source_face, paste_back=True)
    
    # 6. Subtle Color Matching (L-channel only to preserve skin vibrancy)
    print("Matching skin tones...")
    swapped_img = match_colors(swapped_img, target_img)
    
    # 7. GFPGAN Restoration
    print("Restoring facial details...")
    _, _, restored_img = restorer.enhance(
        swapped_img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )
    
    # 8. Surgical Texture Blending (ID-Preserving)
    print("Applying identity-preserving texture synchronization...")
    mask = create_face_mask(target_img, target_face.landmark_2d_106)
    final_img = match_texture_and_blend(restored_img, target_img, mask)
    
    # 9. Final grain matching
    final_img = inject_noise(final_img, intensity=0.002)
    
    output_path = f"samples/{output_name}.png"
    cv2.imwrite(output_path, final_img)
    print(f"Success! Identity restored with synced textures. Saved to: {output_path}")

if __name__ == "__main__":
    # Test on the Mia Khalifa image
    target = "mia/65c2fbe9f6a91e7f66002008-mia-khalifa-celebrity-actress-8-5-x-11.jpg"
    hyper_realistic_swap(target, "mrunal_mia_hyper_real")
