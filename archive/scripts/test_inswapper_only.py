import os
import sys
import logging

sys.path.append("image_pipeline/src")
from processor import ImageProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    processor = ImageProcessor()
    input_path = "image_pipeline/input/gabbie_1.jpg"
    ref_dir = "image_pipeline/reference"
    
    if not os.path.exists(ref_dir):
        print(f"Reference directory {ref_dir} not found.")
        return
        
    references = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not references:
        print("No reference images found.")
        return
        
    output_path = "image_pipeline/output/transformed_gabbie_1.png"
    
    print(f"Processing {input_path} with {len(references)} references using InSwapper-only...")
    processor.process_person(input_path, references, output_path)
    print(f"Done. Output saved to {output_path}")

if __name__ == "__main__":
    main()
