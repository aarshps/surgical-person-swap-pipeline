import os
import cv2
import numpy as np
import insightface
import requests
from duckduckgo_search import DDGS
import sys

def download_mrunal_references():
    print("Downloading high-quality reference images of Mrunal Thakur...")
    os.makedirs("mrunal_refs", exist_ok=True)
    ddgs = DDGS()
    try:
        # Search for high quality close-up portraits
        results = ddgs.images("Mrunal Thakur face portrait high quality", max_results=10)
    except Exception as e:
        print(f"Error fetching images from DuckDuckGo: {e}")
        return
    
    count = 0
    for res in results:
        if count >= 5: # Get 5 good images
            break
        try:
            img_data = requests.get(res['image'], timeout=5).content
            with open(f"mrunal_refs/ref_{count}.jpg", 'wb') as f:
                f.write(img_data)
            count += 1
            print(f"Downloaded reference image {count}/5")
        except Exception as e:
            continue

def enhance_face(image, face_bbox):
    """
    Applies Unsharp Masking and Contrast limited adaptive histogram equalization (CLAHE) 
    specifically to the face region to remove the 128x128 soft blur left by InsightFace.
    """
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    
    # Add a bit of padding around the face box
    padding = 20
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    
    face_region = image[y1:y2, x1:x2]
    
    # 1. Unsharp Masking for sharpness
    gaussian = cv2.GaussianBlur(face_region, (0, 0), 3.0)
    sharpened = cv2.addWeighted(face_region, 1.5, gaussian, -0.5, 0)
    
    # 2. CLAHE for contrast and lighting depth
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced_face = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Seamlessly blend the enhanced face back into the image to avoid harsh borders
    mask = np.zeros(enhanced_face.shape, dtype=np.float32)
    center = (enhanced_face.shape[1]//2, enhanced_face.shape[0]//2)
    cv2.circle(mask, center, min(center[0], center[1]) - 10, (1, 1, 1), -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    
    image[y1:y2, x1:x2] = (enhanced_face * mask + face_region * (1 - mask)).astype(np.uint8)
    return image

def advanced_swap():
    print("Loading models (this might take a few seconds)...")
    app = insightface.app.FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
    
    print("Extracting facial data from reference images...")
    import glob
    ref_paths = glob.glob("mrunal_refs/*.jpg")
    ref_paths.append("mrunal/images_1d40e660-cc88-49.jpg") # Original reference
    
    embeddings = []
    base_face = None
    
    for path in ref_paths:
        img = cv2.imread(path)
        if img is None: continue
        faces = app.get(img)
        if len(faces) > 0:
            # Get the largest face
            largest_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
            embeddings.append(largest_face.normed_embedding)
            if base_face is None:
                base_face = largest_face
                
    if not embeddings:
        print("Failed to find any faces in reference images.")
        sys.exit(1)
        
    print(f"Successfully aggregated {len(embeddings)} different faces of Mrunal to create a perfect 3D profile.")
    
    # Mathematical Averaging of Embeddings
    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = avg_emb / np.linalg.norm(avg_emb) # Normalize the vector
    base_face['embedding'] = avg_emb
    
    # Load target
    target_path = "mia/65c2fbe9f6a91e7f66002008-mia-khalifa-celebrity-actress-8-5-x-11.jpg"
    target_img = cv2.imread(target_path)
    target_faces = app.get(target_img)
    
    if len(target_faces) == 0:
        print("No face detected in target image.")
        sys.exit(1)
        
    target_face = sorted(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
    
    print("Performing high-precision face swap...")
    result_img = swapper.get(target_img, target_face, base_face, paste_back=True)
    
    print("Applying post-process facial enhancement (Sharpening & Contrast Matching)...")
    # To enhance, we need to know where the face ended up. We detect the face on the *new* image.
    new_faces = app.get(result_img)
    if len(new_faces) > 0:
        new_face = sorted(new_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        result_img = enhance_face(result_img, new_face.bbox)
    
    output_path = "samples/ultimate_mrunal_on_mia.png"
    cv2.imwrite(output_path, result_img)
    print(f"Success! Masterpiece saved to: {output_path}")

if __name__ == "__main__":
    download_mrunal_references()
    advanced_swap()
