import time
import os
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from processor import ImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageHandler(FileSystemEventHandler):
    """Handles file system events for new images in the input directory."""
    def __init__(self, processor, reference_dir, output_dir):
        self.processor = processor
        self.reference_dir = reference_dir
        self.output_dir = output_dir

    def on_created(self, event):
        if event.is_directory:
            return
        
        filename = os.path.basename(event.src_path)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            logger.info(f"New image detected: {filename}")
            self.process(event.src_path)

    def process(self, input_path):
        """Triggers the transformation process for a detected image."""
        # Get all reference images
        references = [os.path.join(self.reference_dir, f) for f in os.listdir(self.reference_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not references:
            logger.error("No reference images found in reference folder.")
            return
        
        filename = os.path.basename(input_path)
        output_path = os.path.join(self.output_dir, f"transformed_{filename}")
        
        # Ensure output is png for bit-perfect background preservation
        if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            output_path = output_path.rsplit('.', 1)[0] + '.png'
            
        logger.info(f"Transforming {filename} using {len(references)} reference images...")
        try:
            self.processor.process_person(input_path, references, output_path)
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

def main():
    input_dir = "image_pipeline/input"
    ref_dir = "image_pipeline/reference"
    out_dir = "image_pipeline/output"
    
    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    processor = ImageProcessor()
    event_handler = ImageHandler(processor, ref_dir, out_dir)
    
    # Process existing files
    logger.info("Checking for existing files in input directory...")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            logger.info(f"Found existing file: {filename}")
            event_handler.process(input_path)

    observer = Observer()
    observer.schedule(event_handler, input_dir, recursive=False)
    
    logger.info("Pipeline started.")
    logger.info(f"Watching: {input_dir}")
    logger.info(f"References: {ref_dir}")
    logger.info(f"Output: {out_dir}")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
