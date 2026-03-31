"""Training script for fine-tuning a YOLO model on a custom dataset.

Usage:
    uv run python train.py --model yolov8n.pt --data dataset.yaml --epochs 100
    uv run python train.py --model yolo26n.pt --data dataset.yaml --epochs 100
    uv run python train.py --config configs/train.yaml
    uv run python train.py --resume runs/train/exp/weights/last.pt

After training, copy the best model to the model directory:
    cp runs/train/exp/weights/best.pt model/best.pt

Or use --deploy to auto-copy after training:
    uv run python train.py --model yolov8n.pt --data dataset.yaml --deploy

Name the deployed model for easy comparison:
    uv run python train.py --model yolov8n.pt --data dataset.yaml --deploy --deploy-name yolov8n_100ep.pt
    uv run python train.py --model yolo26n.pt --data dataset.yaml --deploy --deploy-name yolo26n_100ep.pt
"""

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

from utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO model")
    parser.add_argument(
        "--config",
        help="Path to training config YAML (CLI args override config values)",
    )
    parser.add_argument(
        "--model",
        help="Base model to fine-tune (e.g., yolov8n.pt, yolo26n.pt)",
    )
    parser.add_argument(
        "--data",
        help="Path to dataset YAML configuration",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        help="Input image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Batch size (-1 = auto)",
    )
    parser.add_argument(
        "--device",
        help="Device: auto, cpu, cuda:0, mps, etc.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Early stopping patience (0 = disabled)",
    )
    parser.add_argument(
        "--optimizer",
        help="Optimizer: auto, AdamW, SGD, etc.",
    )
    parser.add_argument(
        "--resume",
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--project",
        help="Save directory for training results",
    )
    parser.add_argument(
        "--name",
        help="Experiment name",
    )
    deploy_group = parser.add_mutually_exclusive_group()
    deploy_group.add_argument(
        "--deploy",
        action="store_true",
        default=None,
        help="Auto-copy best.pt to model/best.pt after training",
    )
    deploy_group.add_argument(
        "--no-deploy",
        action="store_false",
        dest="deploy",
        help="Do not auto-copy best.pt (overrides config file)",
    )
    parser.add_argument(
        "--deploy-name",
        help="Filename for deployed model (default: best.pt). Example: yolo26n_100ep.pt",
    )
    return parser.parse_args()


def load_config(config_path):
    """Load training config from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_config(args):
    """Merge YAML config with CLI args. CLI args take precedence."""
    defaults = {
        "model": "yolov8n.pt",
        "data": "dataset.yaml",
        "epochs": 100,
        "imgsz": 640,
        "batch": -1,
        "device": "auto",
        "patience": 50,
        "optimizer": "auto",
        "resume": None,
        "workers": 8,
        "project": "runs/train",
        "name": "exp",
        "deploy": False,
        "deploy_name": "best.pt",
    }

    config = dict(defaults)
    if args.config:
        file_config = load_config(args.config)
        for key in defaults:
            if key in file_config:
                config[key] = file_config[key]

    # CLI args override config file values
    for key in defaults:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            config[key] = cli_value

    return config


def resolve_device(device_str):
    """Resolve device string, using auto-detection when 'auto'."""
    if device_str == "auto":
        return get_device(gpu_avail=True)
    return device_str


def deploy_model(save_dir, deploy_name="best.pt"):
    """Copy the best model weights to model/<deploy_name>.

    Args:
        save_dir: The actual training output directory (e.g. from model.trainer.save_dir).
        deploy_name: Filename for the deployed model (default: best.pt).

    Raises FileNotFoundError if the weights file does not exist.
    """
    src = Path(save_dir) / "weights" / "best.pt"
    dst = Path("model") / deploy_name
    if not src.exists():
        raise FileNotFoundError(f"Trained weights not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Deployed: {src} -> {dst}")


def main():
    args = parse_args()
    config = merge_config(args)

    device = resolve_device(config["device"])

    if config["resume"]:
        model = YOLO(config["resume"])
        # ultralytics restores all hyperparameters from the checkpoint
        model.train(resume=True)
    else:
        model = YOLO(config["model"])
        model.train(
            data=config["data"],
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            device=device,
            patience=config["patience"],
            optimizer=config["optimizer"],
            workers=config["workers"],
            project=config["project"],
            name=config["name"],
        )

    if config["deploy"]:
        save_dir = model.trainer.save_dir
        deploy_model(save_dir, config["deploy_name"])


if __name__ == "__main__":
    main()
