import cv2
import torch


def get_device(gpu_avail=True):
    """GPU が有効な場合、利用可能な最適デバイスを自動検出して返す。
    優先順位: mps (Apple Silicon) > cuda (NVIDIA) > cpu
    """
    if not gpu_avail:
        return 'cpu'
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'


def yolo_to_voc(bbox, img_w, img_h):
    """Convert normalized YOLO bbox (cx, cy, w, h) to absolute VOC coordinates,
    clamping to the valid image bounds to avoid negative or out-of-range indices."""
    cx, cy, w, h = bbox
    x_min = max(0.0, (cx - w / 2) * img_w)
    y_min = max(0.0, (cy - h / 2) * img_h)
    x_max = min(float(img_w), (cx + w / 2) * img_w)
    y_max = min(float(img_h), (cy + h / 2) * img_h)
    return (x_min, y_min, x_max, y_max)


def blur_regions(image, regions, blur_radius=31):
    """Blurs the specified regions using Gaussian Blur."""
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
