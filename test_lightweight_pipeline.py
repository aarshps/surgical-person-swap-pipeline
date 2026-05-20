import os
import sys
import shutil
import logging

# Setup project paths
proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_root)

from core.pipeline.lightweight_processor import LightweightProcessor
from core.validators.validator import run_all_validation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestLightweightPipeline")

def setup_sample_media():
    """
    Copies verified local sample media from image_pipeline to hora-odiyan for testing.
    """
    logger.info("Setting up sample targets and references...")
    
    local_input = "/root/image_pipeline/input"
    local_ref = "/root/image_pipeline/reference"
    
    target_dir = os.path.join(proj_root, "target_pics")
    ref_dir = os.path.join(proj_root, "odiyan_refs")
    
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    
    # Copy target
    if os.path.exists(local_input):
        targets = [f for f in os.listdir(local_input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for t in targets[:2]:  # Copy first 2 targets
            src = os.path.join(local_input, t)
            dst = os.path.join(target_dir, t)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                logger.info(f"Copied target {t} -> {target_dir}")
                
    # Copy a subset of references (e.g., 5-10 references to keep it speedy)
    if os.path.exists(local_ref):
        refs = [f for f in os.listdir(local_ref) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for r in refs[:8]:  # Copy 8 references
            src = os.path.join(local_ref, r)
            dst = os.path.join(ref_dir, r)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                logger.info(f"Copied reference {r} -> {ref_dir}")

def main():
    logger.info("Starting lightweight pipeline verification...")
    
    # 1. Setup sample media
    setup_sample_media()
    
    # Directories
    target_dir = os.path.join(proj_root, "target_pics")
    ref_dir = os.path.join(proj_root, "odiyan_refs")
    output_dir = os.path.join(proj_root, "samples/odiyan_swaps")
    models_dir = os.path.join(proj_root, "models")
    
    # Ensure models downloaded
    if not os.path.exists(os.path.join(models_dir, "inswapper_128.onnx")):
        logger.error("ONNX model files missing in models/ directory. Run 'python models/download_models.py' first.")
        return
        
    # 2. Run processor
    processor = LightweightProcessor(models_dir=models_dir)
    
    logger.info("Loading identity extraction profile...")
    if not processor.load_or_extract_identity(ref_dir):
        logger.error("Could not load/extract reference identity profile.")
        return
        
    targets = [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not targets:
        logger.error("No target images found for testing.")
        return
        
    for t in targets:
        input_path = os.path.join(target_dir, t)
        output_path = os.path.join(output_dir, "lightweight_" + t.rsplit('.', 1)[0] + ".png")
        logger.info(f"Swapping {t}...")
        processor.process_image(input_path, output_path)
        
    # 3. Run validation suite
    logger.info("Executing Master Validation Suite...")
    success = run_all_validation(output_dir, target_dir, ref_dir, models_dir=models_dir)
    
    if success:
        logger.info("VERIFICATION SUCCESSFUL: Integrated pipeline passes all quality thresholds.")
    else:
        logger.error("VERIFICATION FAILED: Some quality thresholds were breached.")

if __name__ == "__main__":
    main()
