import cv2
import insightface
import sys

def swap_face(source_path, target_path, output_path):
    print("Loading models...")
    # Initialize the face analysis model
    app = insightface.app.FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Initialize the face swapper model
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
    
    print(f"Reading source (face): {source_path}")
    print(f"Reading target (body): {target_path}")
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    
    if source_img is None:
        print(f"Error reading source image: {source_path}")
        return
    if target_img is None:
        print(f"Error reading target image: {target_path}")
        return
        
    print("Detecting faces...")
    source_faces = app.get(source_img)
    if len(source_faces) == 0:
        print("No face detected in source image.")
        return
    
    target_faces = app.get(target_img)
    if len(target_faces) == 0:
        print("No face detected in target image.")
        return
        
    # Get the primary face
    source_face = source_faces[0]
    target_face = target_faces[0]
    
    print("Swapping face...")
    result_img = swapper.get(target_img, target_face, source_face, paste_back=True)
    
    cv2.imwrite(output_path, result_img)
    print(f"Success! Face swapped image saved to {output_path}")

if __name__ == "__main__":
    source_img_path = "mrunal/images_1d40e660-cc88-49.jpg"
    target_img_path = "mia/65c2fbe9f6a91e7f66002008-mia-khalifa-celebrity-actress-8-5-x-11.jpg"
    output_img_path = "samples/mrunal_on_mia.png"
    
    swap_face(source_img_path, target_img_path, output_img_path)
