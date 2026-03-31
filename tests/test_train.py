import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


class TestTrainConfig:
    def test_train_config_loads(self):
        config_path = os.path.join(PROJECT_ROOT, "configs", "train.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "model" in config
        assert "data" in config
        assert "epochs" in config
        assert isinstance(config["epochs"], int)
        assert config["batch"] == -1
        assert config["patience"] == 50

    def test_dataset_yaml_loads(self):
        config_path = os.path.join(PROJECT_ROOT, "dataset.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "path" in config
        assert "train" in config
        assert "val" in config
        assert "test" in config
        assert config["names"] == {0: "face", 1: "license_plate"}


def _fake_args(**overrides):
    """Create a FakeArgs with all None defaults, applying overrides."""
    defaults = dict(
        config=None, model=None, data=None, epochs=None, imgsz=None,
        batch=None, device=None, patience=None, optimizer=None,
        resume=None, workers=None, project=None, name=None, deploy=None,
        deploy_name=None,
    )
    defaults.update(overrides)

    class FakeArgs:
        pass

    for k, v in defaults.items():
        setattr(FakeArgs, k, v)
    return FakeArgs()


class TestMergeConfig:
    def test_defaults(self):
        from train import merge_config

        config = merge_config(_fake_args())
        assert config["model"] == "yolov8n.pt"
        assert config["data"] == "dataset.yaml"
        assert config["epochs"] == 100
        assert config["batch"] == -1
        assert config["device"] == "auto"
        assert config["deploy"] is False
        assert config["deploy_name"] == "best.pt"

    def test_cli_overrides(self):
        from train import merge_config

        config = merge_config(_fake_args(
            model="yolo26s.pt", data="my_data.yaml", epochs=50,
            imgsz=320, batch=16, device="cpu", patience=10,
            optimizer="AdamW", workers=4, project="my_runs",
            name="test", deploy=True, deploy_name="yolo26s_50ep.pt",
        ))
        assert config["model"] == "yolo26s.pt"
        assert config["data"] == "my_data.yaml"
        assert config["epochs"] == 50
        assert config["batch"] == 16
        assert config["device"] == "cpu"
        assert config["deploy"] is True
        assert config["deploy_name"] == "yolo26s_50ep.pt"

    def test_config_file_override(self):
        from train import merge_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"model": "yolo26m.pt", "epochs": 200, "batch": 32}, f)
            tmp_path = f.name

        try:
            config = merge_config(_fake_args(config=tmp_path))
            assert config["model"] == "yolo26m.pt"
            assert config["epochs"] == 200
            assert config["batch"] == 32
            assert config["device"] == "auto"
        finally:
            os.unlink(tmp_path)

    def test_cli_overrides_config_file(self):
        from train import merge_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"model": "yolo26m.pt", "epochs": 200}, f)
            tmp_path = f.name

        try:
            config = merge_config(_fake_args(config=tmp_path, epochs=50))
            assert config["model"] == "yolo26m.pt"  # From config file
            assert config["epochs"] == 50  # CLI override wins
        finally:
            os.unlink(tmp_path)

    def test_no_deploy_overrides_config(self):
        from train import merge_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"deploy": True}, f)
            tmp_path = f.name

        try:
            config = merge_config(_fake_args(config=tmp_path, deploy=False))
            assert config["deploy"] is False
        finally:
            os.unlink(tmp_path)

    def test_unknown_keys_in_config_ignored(self):
        from train import merge_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"model": "yolo26n.pt", "unknown_key": 42}, f)
            tmp_path = f.name

        try:
            config = merge_config(_fake_args(config=tmp_path))
            assert config["model"] == "yolo26n.pt"
            assert "unknown_key" not in config
        finally:
            os.unlink(tmp_path)


class TestResolveDevice:
    def test_explicit_cpu(self):
        from train import resolve_device

        assert resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        from train import resolve_device

        assert resolve_device("cuda:0") == "cuda:0"

    def test_explicit_mps(self):
        from train import resolve_device

        assert resolve_device("mps") == "mps"

    @patch("train.get_device", return_value="cpu")
    def test_auto_calls_get_device(self, mock_get_device):
        from train import resolve_device

        result = resolve_device("auto")
        assert result == "cpu"
        mock_get_device.assert_called_once_with(gpu_avail=True)


class TestDeployModel:
    def test_deploy_copies_weights(self):
        from train import deploy_model

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "save_dir"
            weights_dir = save_dir / "weights"
            weights_dir.mkdir(parents=True)
            src = weights_dir / "best.pt"
            src.write_bytes(b"fake model data")

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                deploy_model(save_dir)
                dst = Path("model") / "best.pt"
                assert dst.exists()
                assert dst.read_bytes() == b"fake model data"
            finally:
                os.chdir(old_cwd)

    def test_deploy_custom_name(self):
        from train import deploy_model

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "save_dir"
            weights_dir = save_dir / "weights"
            weights_dir.mkdir(parents=True)
            (weights_dir / "best.pt").write_bytes(b"custom model")

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                deploy_model(save_dir, "yolo26n_100ep.pt")
                dst = Path("model") / "yolo26n_100ep.pt"
                assert dst.exists()
                assert dst.read_bytes() == b"custom model"
                # best.pt should NOT exist
                assert not (Path("model") / "best.pt").exists()
            finally:
                os.chdir(old_cwd)

    def test_deploy_raises_on_missing_weights(self):
        from train import deploy_model

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                deploy_model(Path(tmpdir) / "nonexistent")


class TestRemapLabels:
    def test_remap_keeps_valid_classes(self):
        from scripts.prepare_dataset import remap_labels

        with tempfile.TemporaryDirectory() as tmpdir:
            label_file = Path(tmpdir) / "test.txt"
            label_file.write_text("0 0.5 0.5 0.1 0.1\n1 0.3 0.3 0.2 0.2\n")
            remap_labels(tmpdir)
            lines = label_file.read_text().strip().split("\n")
            assert len(lines) == 2
            assert lines[0].startswith("0 ")
            assert lines[1].startswith("1 ")

    def test_remap_removes_extra_classes(self):
        from scripts.prepare_dataset import remap_labels

        with tempfile.TemporaryDirectory() as tmpdir:
            label_file = Path(tmpdir) / "test.txt"
            label_file.write_text(
                "0 0.5 0.5 0.1 0.1\n2 0.3 0.3 0.2 0.2\n1 0.7 0.7 0.05 0.05\n"
            )
            remap_labels(tmpdir)
            lines = label_file.read_text().strip().split("\n")
            assert len(lines) == 2
            assert lines[0].startswith("0 ")
            assert lines[1].startswith("1 ")

    def test_remap_custom_valid_ids(self):
        from scripts.prepare_dataset import remap_labels

        with tempfile.TemporaryDirectory() as tmpdir:
            label_file = Path(tmpdir) / "test.txt"
            label_file.write_text("0 0.5 0.5 0.1 0.1\n1 0.3 0.3 0.2 0.2\n2 0.1 0.1 0.3 0.3\n")
            remap_labels(tmpdir, valid_class_ids={0, 2})
            lines = label_file.read_text().strip().split("\n")
            assert len(lines) == 2
            assert lines[0].startswith("0 ")
            assert lines[1].startswith("2 ")

    def test_remap_empty_file(self):
        from scripts.prepare_dataset import remap_labels

        with tempfile.TemporaryDirectory() as tmpdir:
            label_file = Path(tmpdir) / "test.txt"
            label_file.write_text("")
            remap_labels(tmpdir)
            assert label_file.read_text() == ""

    def test_remap_all_filtered_warns(self):
        from scripts.prepare_dataset import remap_labels

        with tempfile.TemporaryDirectory() as tmpdir:
            label_file = Path(tmpdir) / "test.txt"
            label_file.write_text("5 0.5 0.5 0.1 0.1\n3 0.3 0.3 0.2 0.2\n")
            remap_labels(tmpdir)
            assert label_file.read_text() == ""

    def test_remap_malformed_line_skipped(self):
        from scripts.prepare_dataset import remap_labels

        with tempfile.TemporaryDirectory() as tmpdir:
            label_file = Path(tmpdir) / "test.txt"
            label_file.write_text("0 0.5 0.5 0.1 0.1\nbad line\n1 0.3 0.3 0.2 0.2\n")
            remap_labels(tmpdir)
            lines = label_file.read_text().strip().split("\n")
            assert len(lines) == 2

    def test_remap_nonexistent_dir(self):
        from scripts.prepare_dataset import remap_labels

        # Should not raise
        remap_labels("/nonexistent/path")


class TestGenerateDatasetYaml:
    def test_generates_correct_yaml(self):
        from scripts.prepare_dataset import generate_dataset_yaml, CLASS_MAP

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = generate_dataset_yaml(tmpdir, CLASS_MAP)
            assert yaml_path.exists()
            with open(yaml_path) as f:
                config = yaml.safe_load(f)
            assert config["train"] == "images/train"
            assert config["val"] == "images/val"
            assert config["test"] == "images/test"
            assert config["names"] == {0: "face", 1: "license_plate"}
            assert config["path"] == str(Path(tmpdir).resolve())


class TestReorganizeLayout:
    def test_reorganize_standard_layout(self):
        from scripts.prepare_dataset import reorganize_to_standard_layout

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fiftyone-style export structure
            for split in ["train", "val", "test"]:
                export_dir = Path(tmpdir) / f"_export_{split}"
                (export_dir / "images").mkdir(parents=True)
                (export_dir / "labels").mkdir(parents=True)
                (export_dir / "images" / f"{split}_img.jpg").write_bytes(b"img")
                (export_dir / "labels" / f"{split}_lbl.txt").write_text("0 0.5 0.5 0.1 0.1")

            reorganize_to_standard_layout(tmpdir)

            for split in ["train", "val", "test"]:
                assert (Path(tmpdir) / "images" / split / f"{split}_img.jpg").exists()
                assert (Path(tmpdir) / "labels" / split / f"{split}_lbl.txt").exists()
                assert not (Path(tmpdir) / f"_export_{split}").exists()

    def test_reorganize_partial_splits(self):
        from scripts.prepare_dataset import reorganize_to_standard_layout

        with tempfile.TemporaryDirectory() as tmpdir:
            # Only create train split
            export_dir = Path(tmpdir) / "_export_train"
            (export_dir / "images").mkdir(parents=True)
            (export_dir / "images" / "img.jpg").write_bytes(b"img")

            reorganize_to_standard_layout(tmpdir)

            assert (Path(tmpdir) / "images" / "train" / "img.jpg").exists()
            assert not (Path(tmpdir) / "_export_train").exists()


def _create_fake_dataset(root, splits=("train", "val", "test"), n_samples=10):
    """Helper to create a fake YOLO dataset for testing."""
    root = Path(root)
    for split in splits:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(n_samples):
            (img_dir / f"img_{i:04d}.jpg").write_bytes(b"fake image")
            (lbl_dir / f"img_{i:04d}.txt").write_text(f"0 0.5 0.5 0.{i} 0.{i}\n")


class TestSubsetDataset:
    def test_creates_subset(self):
        from scripts.subset_dataset import create_subset

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "full"
            output = Path(tmpdir) / "small"
            _create_fake_dataset(source, n_samples=20)

            create_subset(source, output, samples_per_split=5, seed=42)

            for split in ["train", "val", "test"]:
                images = list((output / "images" / split).iterdir())
                labels = list((output / "labels" / split).iterdir())
                assert len(images) == 5
                assert len(labels) == 5

            # dataset.yaml should exist
            yaml_path = output / "dataset.yaml"
            assert yaml_path.exists()
            with open(yaml_path) as f:
                config = yaml.safe_load(f)
            assert config["names"] == {0: "face", 1: "license_plate"}

    def test_caps_at_available_samples(self):
        from scripts.subset_dataset import create_subset

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "full"
            output = Path(tmpdir) / "small"
            _create_fake_dataset(source, n_samples=3)

            create_subset(source, output, samples_per_split=100, seed=42)

            images = list((output / "images" / "train").iterdir())
            assert len(images) == 3  # capped at available

    def test_reproducible_with_same_seed(self):
        from scripts.subset_dataset import create_subset

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "full"
            _create_fake_dataset(source, n_samples=50)

            out1 = Path(tmpdir) / "sub1"
            out2 = Path(tmpdir) / "sub2"
            create_subset(source, out1, samples_per_split=10, seed=42)
            create_subset(source, out2, samples_per_split=10, seed=42)

            files1 = sorted(f.name for f in (out1 / "images" / "train").iterdir())
            files2 = sorted(f.name for f in (out2 / "images" / "train").iterdir())
            assert files1 == files2

    def test_different_seed_gives_different_subset(self):
        from scripts.subset_dataset import create_subset

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "full"
            _create_fake_dataset(source, n_samples=50)

            out1 = Path(tmpdir) / "sub1"
            out2 = Path(tmpdir) / "sub2"
            create_subset(source, out1, samples_per_split=10, seed=1)
            create_subset(source, out2, samples_per_split=10, seed=2)

            files1 = sorted(f.name for f in (out1 / "images" / "train").iterdir())
            files2 = sorted(f.name for f in (out2 / "images" / "train").iterdir())
            assert files1 != files2

    def test_skips_missing_splits(self):
        from scripts.subset_dataset import create_subset

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "full"
            output = Path(tmpdir) / "small"
            _create_fake_dataset(source, splits=("train",), n_samples=10)

            create_subset(source, output, samples_per_split=5, seed=42)

            assert (output / "images" / "train").exists()
            assert not (output / "images" / "val").exists()
            assert not (output / "images" / "test").exists()
