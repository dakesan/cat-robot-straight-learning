import json

import cv2
import numpy as np
import pytest
from synthetic import overhead_camera, project, render

from catrobot.config import RobotConfig
from catrobot.grid import calibrate_from_grid
from catrobot.vision import distance_check, track_video


def draw_grid(camera, camera_from_world, spacing=0.3, count=4):
    width, height = camera["image_size"]
    frame = np.full((height, width, 3), (40, 60, 90), np.uint8)  # brown carpet (BGR)
    span = spacing * (count - 1)
    for k in range(count):
        for a, b in (((k * spacing, 0, 0), (k * spacing, span, 0)), ((0, k * spacing, 0), (span, k * spacing, 0))):
            ends = project([a, b], camera, camera_from_world)
            cv2.line(frame, tuple(np.rint(ends[0]).astype(int)), tuple(np.rint(ends[1]).astype(int)), (0, 140, 255), 7)
    return frame


def test_grid_and_height_recover_the_camera_and_a_taped_distance(tmp_path):
    camera, camera_from_world = overhead_camera(focal=950, size=(1920, 1080), height=1.06, center=(0.45, 0.45))
    cv2.imwrite(str(tmp_path / "grid.png"), draw_grid(camera, camera_from_world))
    result = calibrate_from_grid(tmp_path / "grid.png", tmp_path / "camera.json", tmp_path / "arena.json", 1.06)
    assert result["focal_px"] == pytest.approx(950, rel=0.02)
    assert result["camera_center_m"][2] == pytest.approx(1.06, abs=1e-6)
    assert result["reprojection_rms_px"] < 1.0
    # A hand-drawn 6 cm marker on the battery 13 cm above the floor, placed 0.9 m apart.
    config = RobotConfig(marker_size_m=0.06, marker_height_m=0.13)
    states = [(0.0, 0.45, 0.0, False)] * 45 + [None] * 10 + [(0.9, 0.45, 0.0, False)] * 45
    render(tmp_path / "check.mp4", states, camera, camera_from_world, config)
    summary = track_video(tmp_path / "check.mp4", tmp_path / "camera.json", tmp_path / "arena.json", config, tmp_path / "track.jsonl")
    assert summary["detected"] >= 85
    rows = [json.loads(x) for x in (tmp_path / "track.jsonl").read_text().splitlines()]
    check = distance_check(rows, expected_m=0.9)
    assert abs(check["error_m"]) < 0.005
