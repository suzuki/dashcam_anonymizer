# Dashcam Anonymizer

### This repository blurs human faces and license plates in images and videos using [YOLO by Ultralytics](https://github.com/ultralytics/ultralytics), fine-tuned on images from the [OpenImagesDatasetV7](https://storage.googleapis.com/openimages/web/index.html).

The bundled model is a YOLOv8 fine-tuned checkpoint. The latest ultralytics library also supports YOLO11 and YOLO26 architectures -- see [Retraining with YOLO26](#retraining-with-yolo26) for details.

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
mkdir -p model
gdown 1uV8IMuGDbmDabdjyeSy4SUKV9OS-ULbe -O model/best.pt
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

## Retraining with YOLO26

The bundled model (`model/best.pt`) uses the YOLOv8 architecture. To take advantage of newer architectures like YOLO26 (featuring end-to-end NMS-free inference and up to 43% faster CPU inference), you can retrain on your dataset:

1. Prepare your dataset in YOLO format (see `dataset.yaml` for the expected structure).

2. Run the training script:
```
uv run python train.py --model yolo26n.pt --data dataset.yaml --epochs 100
```

Available base models: `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt`

3. Copy the trained model:
```
cp runs/detect/train/weights/best.pt model/best.pt
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
