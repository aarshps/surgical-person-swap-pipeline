import cv2
import os
from gfpgan import GFPGANer

def test_gfpgan():
    # Path to a previously generated image
    input_path = "samples/ultimate_mrunal_on_mia.png"
    output_path = "samples/test_gfpgan_enhanced.png"
    
    if not os.path.exists(input_path):
        print(f"Input image {input_path} not found.")
        return

    img = cv2.imread(input_path)
    
    # Initialize GFPGAN
    # model_path can be 'GFPGANv1.3' or 'GFPGANv1.4'
    # It will attempt to download the model if not present.
    restorer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )

    print("Enhancing image with GFPGAN...")
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )

    cv2.imwrite(output_path, restored_img)
    print(f"Enhanced image saved to {output_path}")

if __name__ == "__main__":
    test_gfpgan()
