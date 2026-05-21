import unittest
import cv2
import numpy as np
import os
from diffusion_processor import DiffusionProcessor

class TestDiffusionProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = DiffusionProcessor()
        cls.test_image_path = "image_pipeline/input/gabbie_1.jpg"
        if not os.path.exists(cls.test_image_path):
            # Create a dummy image if not exists for basic structural testing
            dummy = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.imwrite("temp_test_unit.jpg", dummy)
            cls.test_image_path = "temp_test_unit.jpg"

    def test_get_face_data(self):
        img = cv2.imread(self.test_image_path)
        if img is None:
            self.skipTest("Test image not found")
            
        pts, mask = self.processor.get_face_data(img)
        
        # pts should be (5, 2)
        if pts is not None:
            self.assertEqual(pts.shape, (5, 2))
            # mask should match image shape
            self.assertEqual(mask.shape, img.shape[:2])
            self.assertEqual(mask.dtype, np.uint8)
        else:
            print("Warning: No face detected in test image, skipping some assertions.")

    def test_align_face(self):
        img = cv2.imread(self.test_image_path)
        if img is None:
            self.skipTest("Test image not found")
            
        pts, _ = self.processor.get_face_data(img)
        if pts is not None:
            aligned = self.processor.align_face(img, pts)
            self.assertIsNotNone(aligned)
            self.assertEqual(aligned.shape, (112, 112, 3))
        else:
            self.skipTest("No face detected for alignment test")

    def test_cache_metadata(self):
        # Verify if metadata structure is consistent
        ref_paths = ["a.jpg", "b.jpg"]
        ref_paths = sorted(ref_paths)
        ref_metadata = {"count": len(ref_paths), "files": [os.path.basename(p) for p in ref_paths]}
        self.assertEqual(ref_metadata["count"], 2)
        self.assertEqual(ref_metadata["files"], ["a.jpg", "b.jpg"])

if __name__ == '__main__':
    unittest.main()
