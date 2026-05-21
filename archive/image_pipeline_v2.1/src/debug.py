import os
import cv2
import numpy as np
from processor import ImageProcessor

def debug():
    processor = ImageProcessor()
    
    reference_dir = "image_pipeline/reference"
    input_dir = "image_pipeline/input"
    
    references = [f for f in os.listdir(reference_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    inputs = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    ref_path = os.path.join(reference_dir, references[0])
    in_path = os.path.join(input_dir, inputs[0])
    
    target_img = cv2.imread(in_path)
    ref_img = cv2.imread(ref_path)
    
    ref_data = processor.get_face_data(ref_img)
    target_data = processor.get_face_data(target_img)
    
    ref_pts, _ = ref_data
    ref_align, _ = processor.align_face(ref_img, ref_pts)
    ref_emb = processor.get_face_embedding(ref_align)
    
    target_pts, _ = target_data
    target_face_align, tform = processor.align_face(target_img, target_pts, is_inswapper=True)
    
    cv2.imwrite("debug_target_align.jpg", target_face_align)
    
    face_input = target_face_align.astype(np.float32) / 255.0
    face_input = face_input.transpose((2, 0, 1))
    face_input = np.expand_dims(face_input, axis=0)
    source_emb = np.expand_dims(ref_emb, axis=0)
    
    swapped_face_raw = processor.face_swapper.run(None, {
        processor.face_swapper.get_inputs()[0].name: face_input,
        processor.face_swapper.get_inputs()[1].name: source_emb
    })[0][0]
    
    print("Raw swapped face min/max/mean:", swapped_face_raw.min(), swapped_face_raw.max(), swapped_face_raw.mean())
    
    # If the max is > 1 (e.g. 255), then multiplying by 255 will blow it out
    swapped_face_test1 = swapped_face_raw.transpose((1, 2, 0)) * 255.0
    swapped_face_test1 = np.clip(swapped_face_test1, 0, 255).astype(np.uint8)
    cv2.imwrite("debug_swapped_x255.jpg", swapped_face_test1)
    
    swapped_face_test2 = swapped_face_raw.transpose((1, 2, 0))
    swapped_face_test2 = np.clip(swapped_face_test2, 0, 255).astype(np.uint8)
    cv2.imwrite("debug_swapped_x1.jpg", swapped_face_test2)

if __name__ == '__main__':
    debug()
