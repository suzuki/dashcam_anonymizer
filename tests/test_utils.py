import numpy as np
import cv2
import yaml
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import yolo_to_voc, blur_regions


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


class TestYoloToVoc:
    def test_center_bbox(self):
        result = yolo_to_voc((0.5, 0.5, 0.5, 0.5), 100, 100)
        assert result == (25.0, 25.0, 75.0, 75.0)

    def test_full_image(self):
        result = yolo_to_voc((0.5, 0.5, 1.0, 1.0), 200, 100)
        assert result == (0.0, 0.0, 200.0, 100.0)

    def test_small_bbox(self):
        result = yolo_to_voc((0.1, 0.1, 0.1, 0.1), 1000, 500)
        expected = (50.0, 25.0, 150.0, 75.0)
        assert all(abs(a - b) < 1e-6 for a, b in zip(result, expected))

    def test_rectangular_image(self):
        result = yolo_to_voc((0.5, 0.5, 0.2, 0.4), 1920, 1080)
        x_min, y_min, x_max, y_max = result
        assert abs(x_max - x_min - 0.2 * 1920) < 1e-6
        assert abs(y_max - y_min - 0.4 * 1080) < 1e-6

    def test_edge_zero(self):
        result = yolo_to_voc((0.0, 0.0, 0.0, 0.0), 640, 480)
        assert result == (0.0, 0.0, 0.0, 0.0)


class TestBlurRegions:
    def test_blur_does_not_raise(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        regions = [(10, 10, 50, 50)]
        result = blur_regions(image, regions)
        assert result.shape == (100, 100, 3)

    def test_blur_modifies_region(self):
        # Use a deterministic gradient pattern to ensure blur always changes the ROI
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            image[i, :, :] = i * 2  # horizontal gradient
        original = image.copy()
        regions = [(10, 10, 50, 50)]
        result = blur_regions(image, regions)
        roi_original = original[10:50, 10:50]
        roi_blurred = result[10:50, 10:50]
        assert not np.array_equal(roi_original, roi_blurred)

    def test_blur_empty_regions(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        original = image.copy()
        result = blur_regions(image, [])
        assert np.array_equal(result, original)

    def test_blur_out_of_bounds_clamps(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        regions = [(-10, -10, 100, 100)]
        result = blur_regions(image, regions)
        assert result.shape == (50, 50, 3)


class TestConfigLoading:
    def test_img_config_loads(self):
        config_path = os.path.join(PROJECT_ROOT, "configs", "img_blur.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        assert "model_path" in config
        assert "detection_conf_thresh" in config
        assert "blur_radius" in config
        assert isinstance(config["blur_radius"], int)

    def test_vid_config_loads(self):
        config_path = os.path.join(PROJECT_ROOT, "configs", "vid_blur.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        assert "model_path" in config
        assert "detection_conf_thresh" in config
        assert "blur_radius" in config
