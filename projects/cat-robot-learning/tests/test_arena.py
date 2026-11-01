import json
import math

import cv2
import numpy as np
import pytest
from synthetic import draw_marker, overhead_camera, render, write_calibration

from catrobot.config import RobotConfig
from catrobot.vision import calibrate_arena, distance_check, rectangle_layout, track_video


def taped(length, width, diagonal_error=0.0):
    diagonal = math.hypot(length, width)
    return {
        "d12": length,
        "d23": width,
        "d34": length,
        "d41": width,
        "d13": diagonal + diagonal_error,
        "d24": diagonal,
    }


def test_rectangle_layout_places_markers_in_the_driving_frame():
    layout = rectangle_layout(taped(1.6, 0.8))
    assert layout["accepted"] and layout["max_distance_residual_m"] < 1e-6
    corners = {m["id"]: np.array(m["corners_m"]) for m in layout["markers"]}
    centers = {k: v.mean(axis=0) for k, v in corners.items()}
    assert centers[1][:2] == pytest.approx([0, 0.4], abs=1e-6)
    assert centers[2][:2] == pytest.approx([1.6, 0.4], abs=1e-6)
    assert centers[3][:2] == pytest.approx([1.6, -0.4], abs=1e-6)
    # Same corner order as examples/arena-layout.json: top-left faces +Y, right edge +X.
    assert corners[1][0][:2] == pytest.approx([-0.025, 0.425], abs=1e-6)
    assert corners[1][1][:2] == pytest.approx([0.025, 0.425], abs=1e-6)


def test_inconsistent_taped_diagonal_is_not_accepted():
    layout = rectangle_layout(taped(1.6, 0.8, diagonal_error=0.03))
    assert not layout["accepted"] and layout["max_distance_residual_m"] > 0.005


def test_arena_calibration_reproduces_a_taped_one_metre_distance(tmp_path):
    config = RobotConfig()
    layout = rectangle_layout(taped(1.1, 0.6))
    camera, camera_from_world = overhead_camera(size=(1280, 960), center=(0.55, 0))
    camera_path, _ = write_calibration(tmp_path, camera, camera_from_world)
    frame = np.full((960, 1280, 3), 255, np.uint8)
    for marker in layout["markers"]:
        draw_marker(frame, marker["id"], marker["corners_m"], camera, camera_from_world)
    cv2.imwrite(str(tmp_path / "arena.png"), frame)
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout))
    calibrate_arena(tmp_path / "arena.png", camera_path, layout_path, tmp_path / "arena.json")
    states = [(0.05, 0.02, 0.0, False)] * 60 + [None] * 15 + [(1.05, 0.02, 0.0, False)] * 60
    video = tmp_path / "distance.mp4"
    render(video, states, camera, camera_from_world, config, floor_markers=layout["markers"])
    track = tmp_path / "track.jsonl"
    track_video(video, camera_path, tmp_path / "arena.json", config, track)
    rows = [json.loads(x) for x in track.read_text().splitlines()]
    result = distance_check(rows)
    assert result["accepted"]
    assert abs(result["error_m"]) < 0.001
