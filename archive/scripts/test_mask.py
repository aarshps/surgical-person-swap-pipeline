import cv2
import numpy as np
from PIL import Image

# Dummy image and generated face
h, w = 512, 512
target_img_cv = np.zeros((h, w, 3), dtype=np.uint8)
target_img_cv[:] = 100 # Background is 100

generated_face_full = np.zeros((h, w, 3), dtype=np.uint8)
generated_face_full[:] = 200 # Generated face is 200

# Create a hard face mask (a circle)
face_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(face_mask, (256, 256), 100, 255, -1)

# Method
eroded_mask = cv2.erode(face_mask, np.ones((25, 25), np.uint8), iterations=1)
inward_soft_mask = cv2.GaussianBlur(eroded_mask, (51, 51), 0)
inward_soft_mask = cv2.bitwise_and(inward_soft_mask, face_mask)

soft_blend_mask = np.expand_dims(inward_soft_mask.astype(np.float32) / 255.0, axis=-1)
output = (generated_face_full * soft_blend_mask + target_img_cv * (1.0 - soft_blend_mask)).astype(np.uint8)

bg_mask = cv2.bitwise_not(face_mask)
diff = cv2.absdiff(target_img_cv, output)
bg_diff = cv2.bitwise_and(diff, diff, mask=bg_mask)
print("Changed background pixels:", np.sum(bg_diff > 0))

# Check transition
edge_pixels = cv2.Canny(face_mask, 100, 200)
edge_vals = output[edge_pixels > 0]
print("Values at boundary:", np.unique(edge_vals))
