import json

import numpy as np
import pytest
from catrobot._native import Controller, Mode

from catrobot.config import RobotConfig, load_config, write_json
from catrobot.metrics import trajectory_metrics
from catrobot.model import export_header, train
from catrobot.report import summarize
from catrobot.simulation import calibrate_simulated_motors, run_episode, schedule


def training_records():
    records = []
    for job in schedule([0.08, 0.14], "train"):
        v, d = job["speed_mps"], job["correction"]
        records.append(
            dict(
                job,
                source="simulated",
                config_hash="fixture",
                reached=True,
                quality_ok=True,
                lateral_at_goal_m=0.03 + 0.1 * v - 1.2 * d,
                heading_at_goal_rad=0.01 - 0.4 * d,
            )
        )
    return records


def test_saved_calibration_roundtrips_without_changing_model_identity(tmp_path):
    config = RobotConfig()
    config.left.counts_per_revolution = 1319.8
    destination = tmp_path / "config.json"
    write_json(destination, config.model_dump())
    reloaded = load_config(destination)
    assert reloaded.left.counts_per_revolution == 1319.8
    assert reloaded.fingerprint() == config.fingerprint()


def test_schedule_has_distinct_train_and_test_trials():
    training, testing = schedule([0.08, 0.14], "train"), schedule([0.08, 0.14], "test")
    assert len(training) == len(testing) == 30
    assert len({(r["speed_mps"], r["correction"]) for r in training}) == 10
    assert not {r["trial_id"] for r in training} & {r["trial_id"] for r in testing}


def test_frozen_regression_matches_native_inference_and_bounds(tmp_path):
    model = train(training_records())
    native = Controller()
    model.attach(native)
    for speed in np.linspace(0.081, 0.139, 9):
        assert native.choose(speed) == pytest.approx(model.choose(speed), abs=1e-6)
        for delta in np.linspace(-0.04, 0.04, 9):
            assert np.allclose(native.predict(speed, delta), model.predict(speed, delta), atol=1e-7)
    assert model.choose(0.1) > 0
    assert -0.04 <= native.choose(0.1) <= 0.04
    with pytest.raises(ValueError):
        model.choose(0.3)
    with pytest.raises(ValueError, match="measured"):
        export_header(model, tmp_path / "unsafe_model.h")
    bad = training_records()
    bad[0]["split"] = "test"
    with pytest.raises(ValueError, match="train split"):
        train(bad)


def test_exported_header_reproduces_python_predictions(tmp_path):
    import subprocess
    from pathlib import Path

    fixture = train(training_records()).model_copy(update={"source": "measured"})
    export_header(fixture, tmp_path / "fixture_model.h")
    probe = tmp_path / "probe.cpp"
    probe.write_text(
        '#include <iostream>\n#include "fixture_model.h"\n'
        "int main(){auto m=deployedModel();float y,h;"
        'm.predict(0.1f,0.02f,y,h);std::cout.precision(9);std::cout<<y<<" "<<h;}\n'
    )
    headers = Path(__file__).resolve().parents[2] / "cat-robot-wiring-visual/sketches/cat_robot"
    subprocess.run(
        ["c++", "-std=c++17", f"-I{headers}", str(probe), "-o", str(tmp_path / "probe")], check=True
    )
    values = subprocess.check_output([str(tmp_path / "probe")], text=True)
    assert np.allclose(np.fromstring(values, sep=" "), fixture.predict(0.1, 0.02), atol=1e-7)


def test_internal_speed_control_does_not_keep_expired_drive_alive():
    native = Controller()
    native.calibration = RobotConfig().native_calibration()
    native.arm(0, 0)
    assert native.drive(0, 1, 1, Mode.speed, 0.1, 0, 20)
    for now in range(10, 301, 10):
        native.tick(now, now, now, 0)
    status = native.state()
    assert not status["armed"] and status["left"] == status["right"] == 0
    assert status["reason"] == "command_timeout" and status["sequence"] == 1


def test_metrics_include_curved_and_incomplete_runs():
    poses = [
        {"time_s": i / 50, "x_m": i / 500, "y_m": i / 10_000, "yaw_rad": 0.02} for i in range(551)
    ]
    metrics = trajectory_metrics(poses)
    assert metrics["reached"] and metrics["lateral_at_goal_m"] == pytest.approx(0.03, abs=0.002)
    partial = trajectory_metrics(poses[:200])
    assert not partial["reached"] and not partial["success"]


def test_metrics_stop_at_interpolated_one_meter_crossing():
    poses = [
        {"time_s": 0, "x_m": 0, "y_m": 0, "yaw_rad": 0},
        {"time_s": 0.02, "x_m": 0.8, "y_m": 0, "yaw_rad": 0},
        {"time_s": 0.04, "x_m": 1.2, "y_m": 1, "yaw_rad": 0},
    ]
    assert trajectory_metrics(poses)["max_lateral_m"] == pytest.approx(0.5)


def test_mujoco_symmetry_and_motor_asymmetry(tmp_path):
    symmetric = RobotConfig()
    symmetric.max_speed_mps = 0.05
    symmetric = calibrate_simulated_motors(symmetric, tmp_path / "calibration-symmetric")
    straight = run_episode(
        symmetric, tmp_path / "symmetric", mode="pwm", speed=symmetric.max_speed_mps * 0.8
    )
    assert straight["reached"] and straight["max_lateral_m"] < 0.005
    asymmetric = RobotConfig()
    asymmetric.right.motor_gain = 0.75
    asymmetric = calibrate_simulated_motors(asymmetric, tmp_path / "calibration-asymmetric")
    speed = asymmetric.max_speed_mps * 0.8
    curved = run_episode(asymmetric, tmp_path / "asymmetric", mode="pwm", speed=speed)
    assert not curved["reached"] or curved["max_lateral_m"] > 0.05
    pi = run_episode(asymmetric, tmp_path / "pi", mode="speed", speed=speed)
    assert pi["reached"] and pi["max_lateral_m"] < curved["max_lateral_m"]
    assert pi["stop_reason"] == "distance_limit"
    assert json.loads((tmp_path / "pi" / "metrics.json").read_text())["source"] == "simulated"


def test_evaluation_retains_failures_and_rejects_training_leak():
    model = train(training_records())
    record = {
        "trial_id": "test-001",
        "split": "test",
        "source": "simulated",
        "config_hash": "fixture",
        "mode": "learned",
        "speed_mps": 0.1,
        "model_id": model.model_id(),
        "reached": False,
        "quality_ok": True,
        "success": False,
        "max_lateral_m": 0.4,
        "heading_at_goal_rad": 0.8,
        "actual_speed_mps": None,
    }
    summary = summarize([record], model)
    assert summary[0]["n"] == 1 and summary[0]["successes"] == 0
    assert summary[0]["median_actual_speed_mps"] is None
    record["trial_id"] = model.training_runs[0]
    with pytest.raises(ValueError, match="leakage"):
        summarize([record], model)
