import json
import math
import random

import pytest
from synthetic import overhead_camera, render, write_calibration

from catrobot.config import RobotConfig, write_json
from catrobot.sync import analyze_session_runs
from catrobot.vision import track_video

SCALE, OFFSET, FPS = 1.0002, 5000.0, 30
TICK_PHASE = 0.0037


def sync_light(elapsed_ms):
    # Same pattern as robot::syncLight() in telemetry.h.
    return elapsed_ms < 150 or 500 <= elapsed_ms < 800 or 1500 <= elapsed_ms < 1950


def tick_floor(t):
    return TICK_PHASE + math.floor((t - TICK_PHASE) / 0.01) * 0.01


def led(t, starts):
    """Physical LED state: syncLight evaluated on the 10 ms control tick grid."""
    tick = tick_floor(t)
    return any(0 <= tick - start < 2.0 and sync_light(round((tick - start) * 1000)) for start in starts)


def make_attempts():
    rng = random.Random(4)
    attempts, t = [], 5001.0
    plan = [("train-001", True), ("train-002", True), ("train-003", False), ("train-003", True),
            ("train-004", True), ("train-005", True)]
    for trial, complete in plan:
        start = t
        arm = start + 2.2
        stop = arm + (1.5 if complete else 0.5)
        end_sync = stop + 0.1 if complete else None
        finish = (end_sync + 2.2) if complete else stop + 0.1
        attempts.append(
            {"trial": trial, "complete": complete, "start": start, "arm": arm, "stop": stop,
             "end_sync": end_sync, "finish": finish}
        )
        # Operators reset the robot at irregular intervals, so phases versus frames vary.
        t = finish + 1.0 + rng.uniform(0, 0.5)
    return attempts


def x_at(t, a):
    if t < a["arm"]:
        return -0.52
    if t >= a["stop"]:
        return -0.52 + 1.04 * (a["stop"] - a["arm"]) / 1.5
    return -0.52 + 1.04 * (t - a["arm"]) / 1.5


def test_session_pairs_by_time_despite_a_missing_pattern_and_a_failed_attempt(tmp_path):
    config = RobotConfig()
    camera, camera_from_world = overhead_camera()
    camera_path, arena_path = write_calibration(tmp_path, camera, camera_from_world)
    attempts = make_attempts()
    starts = [a["start"] for a in attempts] + [a["end_sync"] for a in attempts if a["end_sync"]]
    hidden = attempts[4]  # train-004: its start pattern is out of view
    runs = tmp_path / "runs"
    session = {"session_id": "s", "attempts": []}
    counter = 0
    for number, a in enumerate(attempts):
        directory = a["trial"] if a["complete"] else f"failed/{a['trial']}-a1"
        run = runs / directory
        run.mkdir(parents=True)
        rows, t = [], a["start"] - 0.1
        while t < a["finish"]:
            sample_time = tick_floor(t)
            counter += 1
            armed = a["arm"] <= sample_time < a["stop"]
            rows.append({"sample": counter, "time_ms": round(sample_time * 1000), "epoch": number + 1,
                         "armed": armed, "sync_led": led(sample_time, starts), "reason": "operator"})
            t += 0.02
        (run / "telemetry.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        write_json(run / "metadata.json", {
            "source": "measured", "trial_id": a["trial"], "split": "train", "mode": "speed",
            "speed_mps": 0.69, "correction": 0, "config_hash": config.fingerprint(),
            "dropped_samples": 0, "boot_id": 1})
        session["attempts"].append({"trial_id": a["trial"], "directory": directory,
                                    "number": 1, "boot_id": 1})
    write_json(runs / "session-s.json", session)

    frames = []
    last = attempts[-1]["finish"] + 0.5
    for j in range(int((last - OFFSET) / SCALE * FPS)):
        device = SCALE * (j / FPS) + OFFSET
        current = next((a for a in attempts if a["start"] - 0.3 <= device <= a["finish"] + 0.3), None)
        if current is None or (current is hidden and device < hidden["arm"]):
            frames.append(None)
        else:
            frames.append((x_at(device, current), 0.0, 0.0, led(device, starts)))
    video = tmp_path / "session.mp4"
    render(video, frames, camera, camera_from_world, config, fps=FPS)
    track = tmp_path / "track.jsonl"
    track_video(video, camera_path, arena_path, config, track)

    report = analyze_session_runs(runs, runs / "session-s.json", track)
    assert report["clocks"]["1"]["matched"] == 10
    verdicts = {r["directory"]: r for r in report["attempts"]}
    for name in ("train-001", "train-002", "train-003", "train-005"):
        assert verdicts[name]["accepted"], verdicts[name]
        assert verdicts[name]["residual_s"] < 1 / FPS
        fitted = json.loads((runs / name / "sync.json").read_text())
        truth = next(a for a in attempts if a["trial"] == name and a["complete"])
        video_arm = (truth["arm"] - OFFSET) / SCALE
        mapped = fitted["scale"] * video_arm + fitted["offset_s"]
        assert mapped == pytest.approx(truth["arm"], abs=1 / FPS)
        metrics = json.loads((runs / name / "metrics.json").read_text())
        # Corner-based heading on the 44 px synthetic marker carries about a degree of
        # rendering bias, so allow 3 cm; a wrong pairing would not reach the goal at all.
        assert metrics["reached"] and metrics["max_lateral_m"] < 0.03
    assert "start sync" in verdicts["train-004"]["reason"]
    assert "end sync" in verdicts["failed/train-003-a1"]["reason"]
    assert not json.loads((runs / "train-004/metrics.json").read_text())["quality_ok"]
