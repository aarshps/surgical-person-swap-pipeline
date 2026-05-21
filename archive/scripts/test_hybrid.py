import os
import sys
import logging

sys.path.append("image_pipeline/src")
from hybrid_processor import HybridProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    processor = HybridProcessor()
    input_path = "image_pipeline/input/gabbie_1.jpg"
    ref_dir = "image_pipeline/reference"
    output_path = "image_pipeline/output/transformed_gabbie_1.png"
    
    references = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    processor.process_person(input_path, references, output_path)

if __name__ == "__main__":
    main()
