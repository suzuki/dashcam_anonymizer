"""
Training script for fine-tuning a YOLO model on a custom dataset.

Usage:
    uv run python train.py --model yolo26n.pt --data dataset.yaml --epochs 100

After training, copy the best model to the model directory:
    cp runs/detect/train/weights/best.pt model/best.pt
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO model")
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Base model to fine-tune (e.g., yolo26n.pt, yolo26s.pt, yolo26m.pt)",
    )
    parser.add_argument(
        "--data",
        default="dataset.yaml",
        help="Path to dataset YAML configuration",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
