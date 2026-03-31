"""Download and prepare OpenImagesDatasetV7 for face and license plate detection.

Usage:
    uv run python scripts/prepare_dataset.py
    uv run python scripts/prepare_dataset.py --max-samples 100  # small subset for testing
    uv run python scripts/prepare_dataset.py --output-dir /path/to/dataset

Requires the 'train' extras:
    uv sync --extra train
"""

import argparse
import logging
import re
import shutil
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# OpenImages V7 class names -> project class mapping
CLASS_MAP = {
    "Human face": 0,
    "Vehicle registration plate": 1,
}

# Display names for dataset.yaml output
NAME_MAP = {
    "Human face": "face",
    "Vehicle registration plate": "license_plate",
}

SPLITS = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def download_split(split, classes, max_samples=None):
    """Download a single split from OpenImages V7."""
    import fiftyone.zoo as foz

    kwargs = {
        "split": split,
        "dataset_name": f"open-images-v7-{split}",
        "label_types": ["detections"],
        "classes": classes,
    }
    if max_samples is not None:
        kwargs["max_samples"] = max_samples

    dataset = foz.load_zoo_dataset("open-images-v7", **kwargs)
    return dataset


def export_split(dataset, export_dir, classes):
    """Export a fiftyone dataset to YOLOv5 format."""
    import fiftyone as fo

    dataset.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=classes,
    )


def remap_labels(labels_dir, valid_class_ids=None):
    """Remap class IDs in YOLO label files and remove non-target classes.

    fiftyone exports classes in the order provided, but may include
    co-occurring objects. This function ensures only our target classes
    remain with correct IDs (0: face, 1: license_plate).

    Args:
        labels_dir: Path to directory containing YOLO label .txt files.
        valid_class_ids: Set of class IDs to keep. Defaults to CLASS_MAP values.
    """
    if valid_class_ids is None:
        valid_class_ids = set(CLASS_MAP.values())

    label_pattern = re.compile(r"^(\d+)\s+(\S+\s+\S+\s+\S+\s+\S+)$")
    labels_path = Path(labels_dir)

    if not labels_path.exists():
        return

    for label_file in labels_path.glob("*.txt"):
        lines = label_file.read_text().strip().split("\n")
        remapped = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = label_pattern.match(line)
            if not match:
                logger.warning("Skipping malformed line in %s: %s", label_file, line)
                continue
            class_id = int(match.group(1))
            coords = match.group(2)
            if class_id in valid_class_ids:
                remapped.append(f"{class_id} {coords}")
        if remapped:
            label_file.write_text("\n".join(remapped) + "\n")
        else:
            if lines and lines != [""]:
                logger.warning(
                    "All annotations filtered out in %s (file kept as empty)",
                    label_file,
                )
            label_file.write_text("")


def generate_dataset_yaml(output_dir, class_map):
    """Generate a dataset.yaml for training."""
    config = {
        "path": str(Path(output_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {v: NAME_MAP[k] for k, v in class_map.items()},
    }
    yaml_path = Path(output_dir) / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def reorganize_to_standard_layout(output_dir):
    """Reorganize fiftyone's export layout to the standard YOLO layout.

    fiftyone exports each split as:
        split_dir/images/*.jpg + split_dir/labels/*.txt

    We reorganize to:
        output_dir/images/{train,val,test}/*.jpg
        output_dir/labels/{train,val,test}/*.txt
    """
    output_path = Path(output_dir)
    images_root = output_path / "images"
    labels_root = output_path / "labels"
    images_root.mkdir(exist_ok=True)
    labels_root.mkdir(exist_ok=True)

    for _oid_split, yolo_split in SPLITS.items():
        split_dir = output_path / f"_export_{yolo_split}"
        if not split_dir.exists():
            continue

        src_images = split_dir / "images"
        src_labels = split_dir / "labels"
        dst_images = images_root / yolo_split
        dst_labels = labels_root / yolo_split

        if src_images.exists():
            src_images.rename(dst_images)
        if src_labels.exists():
            src_labels.rename(dst_labels)

        # Clean up the temporary export directory
        shutil.rmtree(split_dir, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Download OpenImages V7 face/license plate data for YOLO training"
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/openimages-face-plate",
        help="Output directory for the dataset (default: datasets/openimages-face-plate)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per split (for testing with a small subset)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = list(CLASS_MAP.keys())

    for oid_split, yolo_split in SPLITS.items():
        print(f"\n{'='*60}")
        print(f"Downloading {oid_split} split...")
        print(f"{'='*60}")

        dataset = download_split(oid_split, classes, args.max_samples)

        export_dir = output_dir / f"_export_{yolo_split}"
        print(f"Exporting to YOLO format: {export_dir}")
        export_split(dataset, export_dir, classes)

        # Remap labels
        remap_labels(export_dir / "labels")

        # Clean up fiftyone dataset to free memory
        import fiftyone as fo
        fo.delete_dataset(dataset.name)

    # Reorganize to standard layout
    print("\nReorganizing to standard YOLO layout...")
    reorganize_to_standard_layout(output_dir)

    # Generate dataset.yaml
    yaml_path = generate_dataset_yaml(output_dir, CLASS_MAP)
    print(f"\nDataset YAML: {yaml_path}")
    print(f"Dataset ready at: {output_dir.resolve()}")
    print(f"\nTo train: uv run python train.py --data {yaml_path}")


if __name__ == "__main__":
    main()
