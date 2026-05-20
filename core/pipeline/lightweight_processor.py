import os
import cv2
import numpy as np
import torch
import time
import logging
import argparse
import glob
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Try relative/absolute imports depending on how it's executed
try:
    from core.pipeline.swap_engines import LightweightInSwapperEngine
except ImportError:
    try:
        from swap_engines import LightweightInSwapperEngine
    except ImportError:
        # Fallback to local import if needed
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from swap_engines import LightweightInSwapperEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LightweightProcessor")

class LightweightProcessor:
    def __init__(self, models_dir=None, profile_path="data/profiles/lightweight_odiyan.npy"):
        self.profile_path = profile_path
        
        # If models_dir is not provided, look in project root/models
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        
        self.models_dir = models_dir
        self.engine = LightweightInSwapperEngine(models_dir=models_dir)
        self.elite_embedding = None

    def load_or_extract_identity(self, reference_dir):
        """
        Loads the precomputed identity or extracts it from the references folder.
        Uses advanced centroid outlier-filtering (curating top 5 closest to centroid).
        """
        # First, try to load from persistent .npy profile
        if os.path.exists(self.profile_path):
            logger.info(f"Loading persistent profile from {self.profile_path}...")
            self.elite_embedding = np.load(self.profile_path).astype(np.float32)
            return True
            
        # Fallback to .pt cache
        pt_cache = os.path.join(self.models_dir, "reference_identity.pt")
        if os.path.exists(pt_cache):
            logger.info(f"Loading cached identity from {pt_cache}...")
            try:
                cached_data = torch.load(pt_cache, map_location=torch.device('cpu'), weights_only=False)
                emb = cached_data["embedding"].detach().cpu().numpy().flatten()
                self.elite_embedding = emb / np.linalg.norm(emb)
                # Cache as npy
                os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
                np.save(self.profile_path, self.elite_embedding)
                return True
            except Exception as e:
                logger.error(f"Error loading {pt_cache}: {e}")

        # Extract fresh from references
        logger.info(f"Extracting fresh identity from references at {reference_dir}...")
        ref_paths = []
        for ext in ('.png', '.jpg', '.jpeg'):
            ref_paths.extend(glob.glob(os.path.join(reference_dir, f"*{ext}")))
            
        if not ref_paths:
            logger.error(f"No reference images found in {reference_dir}.")
            return False
            
        all_embeds = []
        for ref_path in ref_paths:
            img = cv2.imread(ref_path)
            if img is None: continue
            
            face_data = self.engine.get_face_data(img)
            if face_data is None: continue
            
            pts, _ = face_data
            aligned, _ = self.engine.align_face(img, pts, is_inswapper=False)
            if aligned is None: continue
            
            embedding = self.engine.get_face_embedding(aligned)
            if embedding is not None:
                all_embeds.append(embedding)
                
        if not all_embeds:
            logger.error("Could not extract any face embeddings from the reference images.")
            return False
            
        all_embs_np = np.vstack(all_embeds)
        centroid = np.mean(all_embs_np, axis=0)
        dists = np.linalg.norm(all_embs_np - centroid, axis=1)
        
        # Curate the top 5 highest-quality embeddings (closest to centroid)
        elite_indices = np.argsort(dists)[:5]
        elite_mean = np.mean(all_embs_np[elite_indices], axis=0)
        
        norm = np.linalg.norm(elite_mean)
        if norm > 0:
            elite_mean = elite_mean / norm
            
        self.elite_embedding = elite_mean.astype(np.float32)
        
        # Save to persistent files
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        np.save(self.profile_path, self.elite_embedding)
        
        # Save to torch pt format as well for exact local likeness compatibility
        face_embeds = torch.from_numpy(self.elite_embedding).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
        try:
            torch.save({"embedding": face_embeds, "metadata": {"count": len(ref_paths)}}, pt_cache)
        except Exception as e:
            logger.warning(f"Could not save torch cache to {pt_cache}: {e}")
            
        logger.info(f"Successfully learned elite identity from {len(elite_indices)} optimized references.")
        return True

    def process_image(self, target_path, output_path):
        """
        Runs the full Sprint v2.1 high-fidelity lightweight swap.
        """
        if self.elite_embedding is None:
            logger.error("Identity profile is not loaded. Cannot perform swap.")
            return False
            
        logger.info(f"Processing target image: {os.path.basename(target_path)}")
        target_img = cv2.imread(target_path)
        if target_img is None:
            logger.error(f"Failed to read target image: {target_path}")
            return False
            
        try:
            # Run our modular engine swap
            swapped = self.engine.swap(target_img, target_img, self.elite_embedding)
            
            # Ensure output is png for lossless, high-fidelity background preservation
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                output_path = output_path.rsplit('.', 1)[0] + '.png'
                
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, swapped)
            logger.info(f"Success! Result saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error processing image {target_path}: {e}")
            return False

class LightweightImageHandler(FileSystemEventHandler):
    def __init__(self, processor, reference_dir, output_dir):
        self.processor = processor
        self.reference_dir = reference_dir
        self.output_dir = output_dir

    def on_created(self, event):
        if event.is_directory: return
        filename = os.path.basename(event.src_path)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            logger.info(f"New target image detected: {filename}")
            # Add small delay to let file writing complete
            time.sleep(0.5)
            self.process(event.src_path)

    def process(self, input_path):
        # Refresh identity in case references changed
        self.processor.load_or_extract_identity(self.reference_dir)
        
        filename = os.path.basename(input_path)
        output_path = os.path.join(self.output_dir, "lightweight_" + filename)
        
        success = self.processor.process_image(input_path, output_path)
        if not success:
            failed_dir = "data/failed"
            os.makedirs(failed_dir, exist_ok=True)
            shutil.copy(input_path, os.path.join(failed_dir, filename))
            logger.warning(f"Moved failed image to {os.path.join(failed_dir, filename)}")

def main():
    parser = argparse.ArgumentParser(description="Sprint v2.1 Certified Lightweight Face Swap Pipeline")
    parser.add_argument("--input", default="target_pics", help="Path to input image or watch directory")
    parser.add_argument("--output", default="samples/odiyan_swaps", help="Output directory")
    parser.add_argument("--references", default="odiyan_refs", help="References directory")
    parser.add_argument("--daemon", action="store_true", help="Run as watchdog file watcher daemon")
    args = parser.parse_args()

    # Determine absolute/relative paths
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    input_path = args.input if os.path.isabs(args.input) else os.path.join(proj_root, args.input)
    output_dir = args.output if os.path.isabs(args.output) else os.path.join(proj_root, args.output)
    ref_dir = args.references if os.path.isabs(args.references) else os.path.join(proj_root, args.references)
    
    os.makedirs(input_path if args.daemon else os.path.dirname(input_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    processor = LightweightProcessor()
    
    # Initialize the identity
    logger.info("Initializing identity extraction...")
    success = processor.load_or_extract_identity(ref_dir)
    if not success:
        logger.error("Identity extraction failed. Please ensure reference images are in place.")
        return

    if args.daemon:
        logger.info(f"Starting Lightweight Watchdog Daemon watching: {input_path}")
        event_handler = LightweightImageHandler(processor, ref_dir, output_dir)
        
        # Process existing targets first
        logger.info("Checking for pre-existing targets in watch folder...")
        for ext in ('.png', '.jpg', '.jpeg'):
            for t_path in glob.glob(os.path.join(input_path, f"*{ext}")):
                logger.info(f"Processing existing file: {os.path.basename(t_path)}")
                event_handler.process(t_path)
                
        observer = Observer()
        observer.schedule(event_handler, input_path, recursive=False)
        observer.start()
        logger.info("Lightweight Daemon started.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
        logger.info("Daemon stopped.")
    else:
        # Run in single file/folder batch mode
        if os.path.isdir(input_path):
            logger.info(f"Running in batch mode over directory: {input_path}")
            for ext in ('.png', '.jpg', '.jpeg'):
                for t_path in glob.glob(os.path.join(input_path, f"*{ext}")):
                    filename = os.path.basename(t_path)
                    output_file = os.path.join(output_dir, "lightweight_" + filename)
                    processor.process_image(t_path, output_file)
        else:
            if not os.path.exists(input_path):
                logger.error(f"Input path does not exist: {input_path}")
                return
            filename = os.path.basename(input_path)
            output_file = os.path.join(output_dir, "lightweight_" + filename)
            processor.process_image(input_path, output_file)

if __name__ == "__main__":
    main()
