# Dashcam Anonymizer

### This repository blurs human faces and license plates in images and videos using [YOLO by Ultralytics](https://github.com/ultralytics/ultralytics), fine-tuned on images from the [OpenImagesDatasetV7](https://storage.googleapis.com/openimages/web/index.html).

The bundled model is a YOLOv8 fine-tuned checkpoint. The latest ultralytics library also supports YOLO11 and YOLO26 architectures -- see [Retraining](#retraining) for details.

<p align="center">
<img src="media/demo.gif"/>
<img src="media/face_blur.jpg" width="800"/>
</p>

# Setup

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Quick Start

Clone this repo:
```
git clone https://github.com/varungupta31/dashcam_anonymizer.git
cd dashcam_anonymizer
```

### Using uv (recommended)
```
chmod +x setup.sh
./setup.sh
```

The setup script installs all dependencies and downloads the pre-trained YOLO model.

### Using pip
```
pip install -e ".[dev]"
python -m gdown 1uV8IMuGDbmDabdjyeSy4SUKV9OS-ULbe -O model/best.pt
```

## Blurring Images in a Directory

Update `configs/img_blur.yaml` as required, then run:

```
uv run python blur_images.py --config configs/img_blur.yaml
```
The resulting blurred images will be stored in the directory specified in the YAML.

## Blurring Videos in a Directory

```
uv run python blur_videos.py --config configs/vid_blur.yaml
```

Notes:
1. The configuration files are slightly different for videos and images. Make sure to choose and edit the correct ones depending upon the modality.
2. This is designed to process all the contents in a given directory at once. If the blurring is to be re-run, make sure to delete the `runs` directory, as it may lead to new file names within the runs, which will cause errors.

## Running Tests

```
uv run pytest tests/
```

## Retraining

The bundled model (`model/best.pt`) uses the YOLOv8 architecture. You can retrain on the original OpenImagesDatasetV7, or use your own dataset in YOLO format.

### 1. Install training dependencies

```
uv sync --extra train
```

### 2. Prepare the dataset

Download face and license plate images from OpenImagesDatasetV7:

```
uv run python scripts/prepare_dataset.py
```

For a quick test with a small subset:
```
uv run python scripts/prepare_dataset.py --max-samples 100
```

This creates the dataset in `datasets/openimages-face-plate/` with the standard YOLO directory layout.

### 3. Train the model

**YOLOv8** (same architecture as the bundled model):
```
uv run python train.py --model yolov8n.pt --data datasets/openimages-face-plate/dataset.yaml --epochs 100
```

**YOLO26** (NMS-free inference, up to 43% faster CPU inference):
```
uv run python train.py --model yolo26n.pt --data datasets/openimages-face-plate/dataset.yaml --epochs 100
```

Available base models:
- YOLOv8: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`
- YOLO26: `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt`

You can also use a YAML config file:
```
uv run python train.py --config configs/train.yaml
```

### 4. Deploy the trained model

Auto-deploy after training:
```
uv run python train.py --model yolov8n.pt --data datasets/openimages-face-plate/dataset.yaml --deploy
```

Or manually copy:
```
cp runs/train/exp/weights/best.pt model/best.pt
```

No code changes are needed -- the blur scripts will automatically use the new model.

---

## Please Note
* Will this do a 100% perfect job? --> Maybe not.
  - Some specific angles pose a challenge in perfect detections - especially with faces!
* Is there scope for improvement in terms of optimization? --> There always is.
  - _Good_ to question it and call out, even _better_ to help me improve it :)

## If this repository helped you in a research project, please consider to cite and star this Repository!

```
@software{dashcam_anonymizer,
  author = {Varun Gupta},
  month = {8},
  title = {{Dashcam Anonymizer}},
  url = {https://github.com/varungupta31/dashcam_anonymizer},
  version = {1.0.0},
  year = {2023}
}
```
