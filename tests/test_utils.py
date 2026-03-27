import numpy as np
import cv2
import yaml
import os


def yolo_to_voc(bbox, img_w, img_h):
    """Copy of the function from blur_images.py for isolated testing."""
    cx, cy, w, h = bbox
    x_min = (cx - w / 2) * img_w
    y_min = (cy - h / 2) * img_h
    x_max = (cx + w / 2) * img_w
    y_max = (cy + h / 2) * img_h
    return (x_min, y_min, x_max, y_max)


def blur_regions(image, regions, blur_radius=31):
    """Copy of the function from blur_images.py/blur_videos.py for isolated testing."""
    for region in regions:
        x1, y1, x2, y2 = region
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        y1, y2 = max(0, y1), min(image.shape[0], y2)
        x1, x2 = max(0, x1), min(image.shape[1], x2)
        if x1 < x2 and y1 < y2:
            roi = image[y1:y2, x1:x2]
            blur_k = blur_radius if blur_radius % 2 != 0 else blur_radius + 1
            blurred_roi = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
            image[y1:y2, x1:x2] = blurred_roi
    return image


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
        assert result == (50.0, 25.0, 150.0, 75.0)

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
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
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
