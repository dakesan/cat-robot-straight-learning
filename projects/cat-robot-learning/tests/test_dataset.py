import csv
import json

import numpy as np
import pytest

from catrobot.config import RobotConfig, write_json
from catrobot.dataset import build_dataset, kinematic_check
from catrobot.simulation import run_episode


def read_rows(dataset):
    with (dataset / "frames.csv").open() as handle:
        return list(csv.DictReader(handle))


def steady_median(rows, trial, column, after=1.5):
    values = [
        float(r[column])
        for r in rows
        if r["trial_id"] == trial and r["pose_valid"] == "True" and float(r["t_s"]) > after
    ]
    return float(np.median(values))


def linear_plant_run(directory, correction, *, speed=0.12, bias=0.012, track_eff=0.16):
    """Yaw rate = 2 v (c + bias) / track_eff, with track_eff unknown to the config."""
    config = RobotConfig(provenance="measured")
    directory.mkdir(parents=True)
    write_json(directory / "config.json", config.model_dump())
    write_json(directory / "metadata.json", {"trial_id": directory.name, "split": "train",
                                             "source": "measured", "speed_mps": speed,
                                             "correction": correction})
    write_json(directory / "metrics.json", {"quality_ok": True})
    rate = 2 * speed * (correction + bias) / track_eff
    samples = []
    for i in range(400):
        t = i * 0.02
        counts = round(speed * t / config.left.meters_per_count)
        samples.append({"time_ms": 1000 + i * 20, "armed": True, "speed_mps": speed,
                        "correction": correction, "measured_left_mps": speed,
                        "measured_right_mps": speed, "left": 25, "right": 25,
                        "encoder_left": counts, "encoder_right": counts})
    (directory / "telemetry.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    poses = []
    for j in range(int(7.9 * 30)):
        t = j / 30
        yaw = rate * t
        poses.append({"time_s": t, "video_time_s": 50 + t, "valid": True, "yaw_rad": yaw,
                      "x_m": speed * t, "y_m": 0.5 * speed * rate * t * t})
    (directory / "paired.jsonl").write_text("".join(json.dumps(p) + "\n" for p in poses))


def test_measured_gain_label_recovers_a_hidden_bias_without_trusting_the_ruler(tmp_path):
    runs = []
    for i, correction in enumerate((-0.02, 0.0, 0.02)):
        linear_plant_run(tmp_path / f"run-{i}", correction)
        runs.append(tmp_path / f"run-{i}")
    meta = build_dataset(runs, tmp_path / "dataset")
    gain = meta["measured_gain_from_train_split"]["0.12"]
    assert gain["effective_track_m"] == pytest.approx(0.16, rel=0.01)
    assert gain["zero_yaw_correction"] == pytest.approx(-0.012, abs=1e-4)
    rows = read_rows(tmp_path / "dataset")
    for trial in ("run-0", "run-1", "run-2"):
        assert steady_median(rows, trial, "label_correction") == pytest.approx(-0.012, abs=3e-4)
    # The ruler track (0.14 m) would give corrections that depend on the commanded one.
    kinematic = [steady_median(rows, t, "label_kinematic") for t in ("run-0", "run-2")]
    assert abs(kinematic[0] - kinematic[1]) > 0.004


def test_simulated_runs_flow_through_alignment_and_labels(tmp_path):
    config = RobotConfig()
    runs = []
    for i, correction in enumerate((0.0, 0.02)):
        directory = tmp_path / f"sim-{i}"
        run_episode(config, directory, mode="speed", speed=0.12, correction=correction)
        poses = [json.loads(x) for x in (directory / "poses.jsonl").read_text().splitlines()]
        samples = [json.loads(x) for x in (directory / "telemetry.jsonl").read_text().splitlines()]
        paired = [dict(p, video_time_s=p["time_s"], telemetry=s) for p, s in zip(poses, samples)]
        (directory / "paired.jsonl").write_text("".join(json.dumps(p) + "\n" for p in paired))
        metadata = json.loads((directory / "metadata.json").read_text())
        write_json(directory / "metadata.json", dict(metadata, split="train"))
        runs.append(directory)
    meta = build_dataset(runs, tmp_path / "dataset")
    assert meta["included"] == ["sim-0", "sim-1"] and meta["labeled_frames"] > 100
    rows = read_rows(tmp_path / "dataset")
    first = next(r for r in rows if r["trial_id"] == "sim-0")
    assert float(first["encoder_heading_rad"]) == pytest.approx(0, abs=1e-9)
    # A symmetric simulated robot needs no correction once the sweep gain is used. The gain
    # fit includes the 1-2 s start transient, so allow 0.002 (about 1.6 degrees over 1 m).
    for trial in ("sim-0", "sim-1"):
        assert steady_median(rows, trial, "label_correction", after=2) == pytest.approx(0, abs=2e-3)


def test_kinematic_check_reports_effective_track():
    records = [
        {"reached": True, "quality_ok": True, "correction": c, "heading_at_goal_rad": 2 * c / 0.15}
        for c in (-0.02, 0.0, 0.02)
    ]
    result = kinematic_check(records, track_m=0.14)
    assert result["effective_track_m"] == pytest.approx(0.15)
    assert result["ratio"] == pytest.approx(0.14 / 0.15)
