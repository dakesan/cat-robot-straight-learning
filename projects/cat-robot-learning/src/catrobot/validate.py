"""Dataset validator: every trial passes or fails with stated reasons before training."""

import json
import math
from pathlib import Path

from .quality import telemetry_quality

TELEMETRY_COLUMNS = (
    "sample", "time_ms", "epoch", "sequence", "encoder_left", "encoder_right",
    "encoder_left_invalid", "encoder_right_invalid", "left", "right", "target_left",
    "target_right", "speed_mps", "correction", "measured_left_mps", "measured_right_mps",
    "mode", "armed", "reason", "sync_led", "battery_mv",
)  # fmt: skip
METADATA_KEYS = ("trial_id", "split", "source", "mode", "speed_mps", "correction", "config_hash")
NULLABLE = {"battery_mv"}


def finite(value) -> bool:
    return not isinstance(value, float) or math.isfinite(value)


def validate_trial(run: Path) -> dict:
    problems = []
    try:
        metadata = json.loads((run / "metadata.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return {"trial": run.name, "accepted": False, "problems": [f"metadata: {error}"]}
    problems += [f"metadata lacks {k}" for k in METADATA_KEYS if k not in metadata]
    try:
        lines = (run / "telemetry.jsonl").read_text().splitlines()
        samples = [json.loads(line) for line in lines if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError) as error:
        samples = []
        problems.append(f"telemetry: {error}")
    for index, row in enumerate(samples):
        missing = [c for c in TELEMETRY_COLUMNS if c not in row]
        bad = [c for c, v in row.items() if (v is None and c not in NULLABLE) or not finite(v)]
        if missing or bad:
            problems.append(f"sample row {index}: missing {missing} non-finite {bad}")
            break
    if samples:
        quality = telemetry_quality(samples)
        problems += quality["problems"]
        if len(quality["armed_epochs"]) != 1:
            problems.append(f"expected one armed run, found epochs {quality['armed_epochs']}")
    if metadata.get("dropped_samples"):
        problems.append(f"{metadata['dropped_samples']} samples dropped on the device buffer")
    if metadata.get("stop_reason") != "distance_limit":
        problems.append(f"stop reason {metadata.get('stop_reason')}")
    metrics_path = run / "metrics.json"
    if not metrics_path.exists():
        problems.append("not analyzed against the video")
    else:
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("analysis_error"):
            problems.append(f"video analysis: {metrics['analysis_error']}")
        elif not metrics.get("quality_ok"):
            problems.append("video tracking quality below threshold")
        sync = run / "sync.json"
        if sync.exists():
            clock = json.loads(sync.read_text())
            if clock["residual_s"] > clock["frame_period_s"]:
                problems.append("sync residual exceeds one frame")
    return {
        "trial": run.name,
        "split": metadata.get("split"),
        "mode": metadata.get("mode"),
        "accepted": not problems,
        "problems": problems,
    }


def validate_runs(runs: Path) -> dict:
    """Validate trial directories directly under runs; failed/ attempts are not training data."""
    trials = sorted(p for p in runs.iterdir() if (p / "metadata.json").exists())
    results = [validate_trial(p) for p in trials]
    ids = [json.loads((p / "metadata.json").read_text()).get("trial_id") for p in trials]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    hashes = {json.loads((p / "metadata.json").read_text()).get("config_hash") for p in trials}
    return {
        "trials": len(results),
        "accepted": sum(r["accepted"] for r in results),
        "duplicate_trial_ids": duplicates,
        "config_hashes": len(hashes),
        "results": results,
    }
