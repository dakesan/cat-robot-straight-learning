import json

import av
import cv2
import numpy as np
import pytest

from catrobot.config import RobotConfig
from catrobot.sync import analyze_run, fit_clock, pulse_groups
from catrobot.vision import marker_points, marker_pose, track_video, video_frames


def synthetic_camera():
    camera = {
        "camera_matrix": [[1000, 0, 480], [0, 1000, 360], [0, 0, 1]],
        "distortion": [0, 0, 0, 0, 0],
        "image_size": [960, 720],
    }
    rotation = cv2.Rodrigues(np.array([np.pi - 0.08, 0, 0]))[0]
    camera_from_world = np.eye(4)
    camera_from_world[:3, :3] = rotation
    camera_from_world[:3, 3] = -rotation @ np.array([0, 0, 1.2])
    return camera, camera_from_world


def test_raised_marker_pose_is_not_projected_onto_the_floor():
    camera, camera_from_world = synthetic_camera()
    config = RobotConfig(marker_height_m=0.08, marker_offset_x_m=0.03)
    expected_robot = np.array([0.2, 0.1, 0])
    marker = expected_robot + [0.03, 0, 0.08]
    rotation = camera_from_world[:3, :3]
    translation = rotation @ marker + camera_from_world[:3, 3]
    points = cv2.projectPoints(
        marker_points(0.05),
        cv2.Rodrigues(rotation)[0],
        translation,
        np.array(camera["camera_matrix"], float),
        np.zeros(5),
    )[0].reshape(4, 2)
    pose = marker_pose(points, camera, np.linalg.inv(camera_from_world), config)
    assert pose["x_m"] == pytest.approx(0.2, abs=1e-5)
    assert pose["y_m"] == pytest.approx(0.1, abs=1e-5)


def test_actual_video_decode_and_aruco_tracking(tmp_path):
    camera, camera_from_world = synthetic_camera()
    config = RobotConfig(marker_height_m=0.07)
    video = tmp_path / "generated-fixture.mp4"
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 300)
    expected = []
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 960, 720, "yuv420p"
        for i in range(45):
            position = np.array([-0.15 + i * 0.006, 0.02, 0.07])
            rotation = camera_from_world[:3, :3]
            translation = rotation @ position + camera_from_world[:3, 3]
            corners = cv2.projectPoints(
                marker_points(0.05),
                cv2.Rodrigues(rotation)[0],
                translation,
                np.array(camera["camera_matrix"], float),
                np.zeros(5),
            )[0].reshape(4, 2)
            transform = cv2.getPerspectiveTransform(
                np.array([[0, 0], [299, 0], [299, 299], [0, 299]], np.float32),
                corners.astype(np.float32),
            )
            image = cv2.warpPerspective(marker, transform, (960, 720), borderValue=255)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
            expected.append(position)
        for packet in stream.encode():
            container.mux(packet)
    (tmp_path / "camera.json").write_text(json.dumps(camera))
    (tmp_path / "arena.json").write_text(
        json.dumps({"world_from_camera": np.linalg.inv(camera_from_world).tolist()})
    )
    result = track_video(
        video, tmp_path / "camera.json", tmp_path / "arena.json", config, tmp_path / "track.jsonl"
    )
    assert result["coverage"] > 0.95
    rows = [json.loads(s) for s in (tmp_path / "track.jsonl").read_text().splitlines()]
    errors = [
        np.linalg.norm(np.array([row["x_m"], row["y_m"]]) - expected[i][:2])
        for i, row in enumerate(rows)
        if row["valid"]
    ]
    assert max(errors) < 0.006
    assert list(video_frames(video))[-1][0] == pytest.approx(44 / 30)


def test_optical_clock_mapping_recovers_offset_and_drift():
    video = np.array([0, 0.5, 1.5, 15, 15.5, 16.5])
    device = video * 1.0003 + 123.4
    fit = fit_clock(np.column_stack([video, device]).tolist(), 1 / 30)
    assert fit["scale"] == pytest.approx(1.0003)
    assert fit["offset_s"] == pytest.approx(123.4)
    assert pulse_groups(video.tolist()) == video.tolist()
    broken = device.copy()
    broken[3] += 0.3
    with pytest.raises(ValueError, match="residual"):
        fit_clock(np.column_stack([video, broken]).tolist(), 1 / 30)


def test_filmed_led_to_telemetry_pairing_end_to_end(tmp_path):
    camera, camera_from_world = synthetic_camera()
    camera["camera_matrix"][0][0] = camera["camera_matrix"][1][1] = 900
    config = RobotConfig()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 300)
    video = tmp_path / "synthetic-optical-fixture.mp4"

    def light(t):
        elapsed = t if t < 4.3 else t - 4.3
        return elapsed < 0.15 or 0.5 <= elapsed < 0.8 or 1.5 <= elapsed < 1.95

    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 960, 720, "yuv420p"
        for i in range(200):
            t = i / 30
            position = np.array([-0.52 + 1.04 * np.clip((t - 2) / 2, 0, 1), 0, 0.07])
            rotation = camera_from_world[:3, :3]
            translation = rotation @ position + camera_from_world[:3, 3]
            corners = cv2.projectPoints(
                marker_points(0.05),
                cv2.Rodrigues(rotation)[0],
                translation,
                np.array(camera["camera_matrix"], float),
                np.zeros(5),
            )[0].reshape(4, 2)
            warp = cv2.getPerspectiveTransform(
                np.array([[0, 0], [299, 0], [299, 299], [0, 299]], np.float32),
                corners.astype(np.float32),
            )
            frame = cv2.cvtColor(
                cv2.warpPerspective(marker, warp, (960, 720), borderValue=255), cv2.COLOR_GRAY2BGR
            )
            led = cv2.perspectiveTransform(
                np.array([[[config.led_offset[0] * 299, config.led_offset[1] * 299]]], np.float32),
                warp,
            )[0, 0]
            cv2.circle(
                frame,
                tuple(np.rint(led).astype(int)),
                3,
                (0, 0, 255) if light(t) else (0, 0, 0),
                -1,
            )
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    camera_path, arena_path = tmp_path / "camera.json", tmp_path / "arena.json"
    camera_path.write_text(json.dumps(camera))
    arena_path.write_text(
        json.dumps({"world_from_camera": np.linalg.inv(camera_from_world).tolist()})
    )
    track = tmp_path / "track.jsonl"
    result = track_video(video, camera_path, arena_path, config, track)
    assert result["coverage"] > 0.95
    run = tmp_path / "fixture-run"
    run.mkdir()
    # This source label exercises the measured-run parser in a temporary test fixture;
    # it is never exported as physical evidence or used to train a deployable model.
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "source": "measured",
                "fixture": True,
                "trial_id": "fixture-run",
                "split": "test",
                "mode": "pwm",
                "speed_mps": 0.52,
                "correction": 0,
                "config_hash": config.fingerprint(),
                "dropped_samples": 0,
            }
        )
    )
    telemetry = [
        {
            "time_ms": 123400 + round(t * 1000 * 1.0003),
            "sync_led": bool(light(t)),
            "epoch": 1,
            "armed": bool(2 <= t <= 4.1),
        }
        for t in np.arange(0, 6.6, 0.02)
    ]
    (run / "telemetry.jsonl").write_text("".join(json.dumps(s) + "\n" for s in telemetry))
    metrics = analyze_run(run, track)
    assert metrics["quality_ok"] and metrics["reached"]
    assert metrics["max_lateral_m"] < 0.02
    assert json.loads((run / "sync.json").read_text())["residual_s"] < 1 / 30
