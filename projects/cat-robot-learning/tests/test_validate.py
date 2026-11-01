import json

from fakes import sample

from catrobot.config import write_json
from catrobot.validate import validate_runs

EXTRA = {
    "sequence": 1, "left": 20, "right": 20, "target_left": 20, "target_right": 20,
    "speed_mps": 0.1, "correction": 0.0, "measured_left_mps": 0.1, "measured_right_mps": 0.1,
    "mode": 1, "battery_mv": None,
}  # fmt: skip


def trial(runs, name, *, stop="distance_limit", rows=None, metrics=True, analysis_error=None):
    run = runs / name
    run.mkdir(parents=True)
    rows = rows or [
        dict(sample(i + 1, 1000 + 20 * i, armed=10 <= i < 60, epoch=1), **EXTRA) for i in range(80)
    ]
    (run / "telemetry.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(run / "metadata.json", {
        "trial_id": name, "split": "train", "source": "measured", "mode": "speed",
        "speed_mps": 0.1, "correction": 0.0, "config_hash": "h", "dropped_samples": 0,
        "stop_reason": stop})
    if metrics:
        write_json(run / "metrics.json", {"quality_ok": analysis_error is None,
                                          "analysis_error": analysis_error})
        write_json(run / "sync.json", {"residual_s": 0.01, "frame_period_s": 1 / 60})
    return rows


def test_validator_accepts_clean_trials_and_names_every_failure(tmp_path):
    rows = trial(tmp_path, "train-001")
    trial(tmp_path, "train-002", stop="command_timeout")
    trial(tmp_path, "train-003", rows=[dict(r, measured_left_mps=float("nan")) for r in rows])
    trial(tmp_path, "train-004", rows=rows[:20] + rows[25:])
    trial(tmp_path, "train-005", metrics=False)
    trial(tmp_path, "train-006", analysis_error="start sync pattern not found in the video")
    (tmp_path / "failed" / "train-002-a1").mkdir(parents=True)
    report = validate_runs(tmp_path)
    results = {r["trial"]: r for r in report["results"]}
    assert report["trials"] == 6 and report["accepted"] == 1
    assert results["train-001"]["accepted"]
    assert "command_timeout" in " ".join(results["train-002"]["problems"])
    assert "non-finite ['measured_left_mps']" in " ".join(results["train-003"]["problems"])
    assert "missing from the sequence" in " ".join(results["train-004"]["problems"])
    assert "not analyzed" in " ".join(results["train-005"]["problems"])
    assert "start sync" in " ".join(results["train-006"]["problems"])
