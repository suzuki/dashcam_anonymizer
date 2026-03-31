"""Create a subset dataset from an existing YOLO-format dataset.

Usage:
    uv run python scripts/subset_dataset.py --source datasets/openimages-full --output datasets/openimages-small --samples 100
    uv run python scripts/subset_dataset.py --source datasets/openimages-full --output datasets/openimages-small --samples 500 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml

SPLITS = ["train", "val", "test"]


def collect_samples(source_dir, split):
    """Collect image-label pairs for a split.

    Returns list of (image_path, label_path_or_none) tuples.
    """
    images_dir = source_dir / "images" / split
    labels_dir = source_dir / "labels" / split

    if not images_dir.exists():
        return []

    pairs = []
    for img in sorted(images_dir.iterdir()):
        if not img.is_file():
            continue
        label = labels_dir / f"{img.stem}.txt" if labels_dir.exists() else None
        if label and not label.exists():
            label = None
        pairs.append((img, label))
    return pairs


def create_subset(source_dir, output_dir, samples_per_split, seed):
    """Create a subset dataset by sampling from each split."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    rng = random.Random(seed)

    for split in SPLITS:
        pairs = collect_samples(source_dir, split)
        if not pairs:
            print(f"  {split}: skipped (no data)")
            continue

        n = min(samples_per_split, len(pairs))
        selected = rng.sample(pairs, n)

        out_images = output_dir / "images" / split
        out_labels = output_dir / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        for img, label in selected:
            shutil.copy2(img, out_images / img.name)
            if label:
                shutil.copy2(label, out_labels / label.name)

        print(f"  {split}: {n}/{len(pairs)} samples")

    # Generate dataset.yaml
    config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "face", 1: "license_plate"},
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nDataset YAML: {yaml_path}")
    print(f"To train: uv run python train.py --data {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a subset from an existing YOLO dataset"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to source dataset (e.g. datasets/openimages-full)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for subset output (e.g. datasets/openimages-small)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples per split (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory already exists and is not empty: {output_dir}")

    print(f"Creating subset: {args.samples} samples/split from {args.source}")
    create_subset(args.source, args.output, args.samples, args.seed)


if __name__ == "__main__":
    main()
