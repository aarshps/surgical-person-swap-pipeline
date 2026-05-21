import os
import cv2
import numpy as np
from processor import ImageProcessor

def inspect_output():
    out_path = "image_pipeline/output/test_output.jpg"
    in_path = "image_pipeline/input/gabbie_1.jpg"
    
    out_img = cv2.imread(out_path)
    in_img = cv2.imread(in_path)
    
    print("Input shape:", in_img.shape)
    print("Output shape:", out_img.shape)
    
    diff = cv2.absdiff(in_img, out_img)
    print("Max diff:", diff.max())
    print("Mean diff:", diff.mean())
    
    # Save a small patch where difference is highest
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, max_val, _, max_loc = cv2.minMaxLoc(gray_diff)
    print("Max loc:", max_loc)

if __name__ == '__main__':
    inspect_output()
