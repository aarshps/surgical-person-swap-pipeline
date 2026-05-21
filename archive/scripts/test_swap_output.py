import cv2
from processor import ImageProcessor

processor = ImageProcessor()
ref = cv2.imread("image_pipeline/input/gabbie_1.jpg")
pts, _ = processor.get_face_data(ref)
emb = processor.get_face_embedding(processor.align_face(ref, pts)[0])

swap = processor.swap_face(ref, emb, pts)
cv2.imwrite("test_swap_inspect.jpg", swap)
