import json

import numpy as np
import pytest
from fakes import FakeRobot

from catrobot.acquisition import collect, fit_motors, run_session
from catrobot.config import RobotConfig, write_json
from catrobot.simulation import schedule


def test_probe_and_centered_sweep_schedules():
    probe = schedule([0.1, 0.2], "probe")
    assert len(probe) == 4 and {j["correction"] for j in probe} == {0}
    sweep = schedule([0.1, 0.2], "train", center=-0.03, step=0.01)
    assert len(sweep) == 30
    assert sorted({j["correction"] for j in sweep}) == [-0.05, -0.04, -0.03, -0.02, -0.01]
    with pytest.raises(ValueError, match="firmware limit"):
        schedule([0.1, 0.2], "train", center=0.18, step=0.02)


class Robots:
    """Second trial's first attempt loses telemetry after arming, then succeeds."""

    def __init__(self):
        self.made = 0

    def __call__(self, url):
        self.made += 1
        return FakeRobot(fail_armed_reads=10_000 if self.made == 2 else 0)


def test_failed_attempt_is_archived_and_retried_without_overwriting(tmp_path, fast_clock):
    jobs = [
        {"trial_id": f"train-00{i}", "mode": "speed", "speed_mps": 0.1, "correction": 0,
         "split": "train", "repeat": 0}
        for i in (1, 2, 3)
    ]
    asked = []
    session = run_session(
        "http://example.invalid",
        RobotConfig(provenance="measured"),
        jobs,
        tmp_path,
        session_id="s1",
        confirm=lambda job: True,
        retry=lambda job, metadata: asked.append(metadata["stop_reason"]) or True,
        echo=lambda line: None,
        client_factory=Robots(),
    )
    assert asked == ["acquisition_error"]
    failed = json.loads((tmp_path / "failed/train-002-a1/metadata.json").read_text())
    assert failed["stop_reason"] == "acquisition_error" and failed["end_sync_after_error"]
    retried = json.loads((tmp_path / "train-002/metadata.json").read_text())
    assert retried["stop_reason"] == "distance_limit" and retried["attempt"] == 2
    assert retried["session_id"] == "s1" and "code_commit" in retried
    attempts = json.loads((tmp_path / "session-s1.json").read_text())["attempts"]
    assert [a["directory"] for a in attempts] == [
        "train-001", "failed/train-002-a1", "train-002", "train-003"
    ]
    assert len(session["attempts"]) == 4


def test_completed_trials_are_skipped_and_declining_stops_the_session(tmp_path, fast_clock):
    jobs = [
        {"trial_id": f"test-00{i}", "mode": "pwm", "speed_mps": 0.1, "correction": 0,
         "split": "test", "repeat": 0}
        for i in (1, 2)
    ]
    config = RobotConfig(provenance="measured")
    run_session("u", config, jobs[:1], tmp_path, session_id="a", confirm=lambda j: True,
                retry=lambda j, m: False, echo=lambda x: None, client_factory=lambda u: FakeRobot())
    lines = []
    session = run_session("u", config, jobs, tmp_path, session_id="a", confirm=lambda j: False,
                          retry=lambda j, m: False, echo=lines.append,
                          client_factory=lambda u: FakeRobot())
    assert any("記録済み" in line for line in lines)
    assert len(session["attempts"]) == 1 and not (tmp_path / "test-002").exists()


def pwm_run(directory, percent, *, response=True):
    rows, position, velocity = [], 0.0, 0.0
    wheel = RobotConfig().left
    for i in range(200):
        counts = round(position / wheel.meters_per_count)
        rows.append({"time_ms": i * 20, "armed": True, "left": percent, "right": percent,
                     "encoder_left": counts, "encoder_right": counts})
        target = (((percent * 255 + 50) // 100) * 100 / 255 - 8) * 0.005 if response else 0
        updated = target + (velocity - target) * np.exp(-0.02 / 0.12)
        position += (velocity + updated) * 0.02 / 2
        velocity = updated
    directory.mkdir(parents=True)
    write_json(directory / "metadata.json", {"source": "measured", "mode": "pwm",
                                             "pwm": percent, "split": "train"})
    (directory / "telemetry.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_zero_output_dry_runs_do_not_enter_the_motor_fit(tmp_path):
    for percent in (20, 30, 40):
        pwm_run(tmp_path / f"pwm-{percent}", percent)
    pwm_run(tmp_path / "dry-run", 0, response=False)
    fitted = fit_motors(tmp_path, RobotConfig(provenance="measured"))
    assert fitted.left.deadband_percent == pytest.approx(8, abs=0.5)


def test_device_schedule_compares_flashed_models_at_the_higher_speed():
    jobs = schedule([0.1, 0.2], "device")
    assert len(jobs) == 15 and {j["speed_mps"] for j in jobs} == {0.2}
    assert sorted({j["mode"] for j in jobs}) == ["learned", "speed", "student"]
    assert all(j["correction"] == 0 and j["split"] == "device" for j in jobs)


def test_student_trials_require_the_flashed_student(tmp_path, fast_clock):
    config = RobotConfig(provenance="measured")
    with pytest.raises(ValueError, match="expected student id"):
        collect("http://x", config, tmp_path / "a", mode="student", speed=0.1, split="device")
    with pytest.raises(ValueError, match="requested measured student"):
        collect("http://x", config, tmp_path / "b", mode="student", speed=0.1, split="device",
                student_id="abc", client_factory=lambda u: FakeRobot(student_id="other"))
    assert not (tmp_path / "b").exists()
    robot = FakeRobot(student_id="abc")
    metadata = collect("http://x", config, tmp_path / "c", mode="student", speed=0.1, split="device",
                       student_id="abc", client_factory=lambda u: robot)
    assert metadata["model_id"] == "abc" and metadata["stop_reason"] == "distance_limit"
    assert robot.drives and all(d["mode"] == 3 for d in robot.drives)
