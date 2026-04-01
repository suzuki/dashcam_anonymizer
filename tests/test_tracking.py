import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import get_tracking_config

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


class TestTrackingConfig:
    def test_vid_config_has_tracking_settings(self):
        config_path = os.path.join(PROJECT_ROOT, "configs", "vid_blur.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        tc = get_tracking_config(config)
        assert isinstance(tc["use_tracking"], bool)
        assert tc["tracker"] in ("botsort.yaml", "bytetrack.yaml")
        assert isinstance(tc["interpolate_frames"], int)
        assert tc["interpolate_frames"] >= 0

    def test_tracking_defaults_when_missing(self):
        """Config without tracking keys should fall back to defaults."""
        config = {
            "model_path": "model/best.pt",
            "videos_path": "videos/",
        }
        tc = get_tracking_config(config)
        assert tc["use_tracking"] is False
        assert tc["tracker"] == "botsort.yaml"
        assert tc["interpolate_frames"] == 0

    def test_tracking_config_overrides(self):
        config = {
            "use_tracking": True,
            "tracker": "bytetrack.yaml",
            "interpolate_frames": 10,
        }
        tc = get_tracking_config(config)
        assert tc["use_tracking"] is True
        assert tc["tracker"] == "bytetrack.yaml"
        assert tc["interpolate_frames"] == 10


class TestInterpolationLogic:
    """Test the interpolation logic extracted from process_video_with_tracking."""

    def _compute_interpolated_boxes(self, last_seen, active_ids, frame_num, interpolate_frames):
        """Reproduce the interpolation logic from blur_videos.py."""
        extra_boxes = []
        for tid, (last_frame, last_box) in list(last_seen.items()):
            if tid not in active_ids:
                gap = frame_num - last_frame
                if 0 < gap <= interpolate_frames:
                    extra_boxes.append(last_box)
                elif gap > interpolate_frames:
                    del last_seen[tid]
        return extra_boxes

    def test_no_interpolation_when_disabled(self):
        last_seen = {1: (5, [10, 10, 50, 50])}
        active_ids = set()
        extra = self._compute_interpolated_boxes(last_seen, active_ids, frame_num=6, interpolate_frames=0)
        assert extra == []

    def test_interpolates_within_window(self):
        last_seen = {1: (10, [10, 10, 50, 50])}
        active_ids = set()
        extra = self._compute_interpolated_boxes(last_seen, active_ids, frame_num=13, interpolate_frames=5)
        assert len(extra) == 1
        assert extra[0] == [10, 10, 50, 50]

    def test_stops_interpolation_after_window(self):
        last_seen = {1: (10, [10, 10, 50, 50])}
        active_ids = set()
        extra = self._compute_interpolated_boxes(last_seen, active_ids, frame_num=16, interpolate_frames=5)
        assert extra == []
        assert 1 not in last_seen  # cleaned up

    def test_no_interpolation_for_active_tracks(self):
        last_seen = {1: (10, [10, 10, 50, 50])}
        active_ids = {1}
        extra = self._compute_interpolated_boxes(last_seen, active_ids, frame_num=11, interpolate_frames=5)
        assert extra == []

    def test_multiple_tracks_interpolation(self):
        last_seen = {
            1: (10, [10, 10, 50, 50]),
            2: (8, [100, 100, 200, 200]),
            3: (12, [300, 300, 400, 400]),
        }
        active_ids = {3}  # only track 3 is active
        extra = self._compute_interpolated_boxes(
            last_seen, active_ids, frame_num=13, interpolate_frames=5
        )
        # Track 1: gap=3 (<= 5), should interpolate
        # Track 2: gap=5 (<= 5), should interpolate
        # Track 3: active, skip
        assert len(extra) == 2

    def test_exact_boundary_interpolates(self):
        last_seen = {1: (10, [10, 10, 50, 50])}
        active_ids = set()
        # gap == interpolate_frames exactly should still interpolate
        extra = self._compute_interpolated_boxes(last_seen, active_ids, frame_num=15, interpolate_frames=5)
        assert len(extra) == 1
