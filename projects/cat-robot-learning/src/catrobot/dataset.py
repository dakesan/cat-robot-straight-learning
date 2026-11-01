"""Aligned learning frames and hindsight straightening labels from paired runs.

Raw telemetry and video stay untouched; this module only derives a table from them.
Continuous signals are interpolated, discrete commands are held, video poses are
smoothed by a local linear fit before resampling, and unobserved spans stay invalid.
"""

import csv
import json
from pathlib import Path

import numpy as np

from .config import RobotConfig, write_json

# Inputs the ESP32 student can compute at run time without any camera.
STUDENT_FEATURES = (
    "speed_mps",
    "elapsed_s",
    "measured_left_mps",
    "measured_right_mps",
    "encoder_heading_rad",
)
ELAPSED_CAP_S = 3.0
COLUMNS = (
    "trial_id", "split", "t_s", "speed_mps", "correction_cmd", "desired_left_mps",
    "desired_right_mps", "measured_left_mps", "measured_right_mps", "left_output_pct",
    "right_output_pct", "distance_left_m", "distance_right_m", "encoder_heading_rad",
    "elapsed_s", "x_m", "y_m", "yaw_rad", "yaw_rate_rps", "pose_valid", "label_kinematic",
    "label_correction", "label_valid", "video_time_s",
)  # fmt: skip
GAIN_FROM_S = 1.0


def local_fit(times: np.ndarray, values: np.ndarray, query: float, window: float):
    """Value and slope at query from a line fitted to valid observations in the window."""
    near = np.abs(times - query) <= window / 2
    if near.sum() < 5 or np.ptp(times[near]) < 0.6 * window:
        return None
    slope, intercept = np.polyfit(times[near] - query, values[near], 1)
    return float(intercept), float(slope)


def run_frames(
    run: Path, rate_hz: float = 10, window_s: float = 0.3, track_m: float | None = None
) -> list[dict]:
    metadata = json.loads((run / "metadata.json").read_text())
    config = RobotConfig.model_validate_json((run / "config.json").read_text())
    track = track_m or config.track_m
    samples = [json.loads(x) for x in (run / "telemetry.jsonl").read_text().splitlines() if x]
    paired = [json.loads(x) for x in (run / "paired.jsonl").read_text().splitlines() if x]
    armed = [s for s in samples if s["armed"]]
    if not armed or not paired:
        return []
    arm_ms = armed[0]["time_ms"]
    t_dev = np.array([(s["time_ms"] - arm_ms) / 1000 for s in samples])
    duration = (armed[-1]["time_ms"] - arm_ms) / 1000

    def interpolate(key, query):
        return float(np.interp(query, t_dev, [float(s[key]) for s in samples]))

    def hold(key, query):
        index = max(0, int(np.searchsorted(t_dev, query, side="right")) - 1)
        return samples[index][key]

    sides = {
        side: (getattr(config, side).meters_per_count, getattr(config, side).encoder_sign)
        for side in ("left", "right")
    }
    start_counts = {side: interpolate(f"encoder_{side}", 0.0) for side in sides}
    valid = [p for p in paired if p.get("valid", True)]
    t_pose = np.array([p["time_s"] for p in valid])
    if len(valid) >= 3:
        x0, y0, yaw0 = valid[0]["x_m"], valid[0]["y_m"], valid[0]["yaw_rad"]
        rotation = np.array([[np.cos(yaw0), np.sin(yaw0)], [-np.sin(yaw0), np.cos(yaw0)]])
        local = (np.array([[p["x_m"] - x0, p["y_m"] - y0] for p in valid])) @ rotation.T
        yaw = np.unwrap([p["yaw_rad"] for p in valid]) - yaw0
    rows = []
    for query in np.arange(0, duration + 1e-9, 1 / rate_hz):
        query = float(query)
        speed = float(hold("speed_mps", query))
        correction = float(hold("correction", query))
        distance = {
            side: (interpolate(f"encoder_{side}", query) - start_counts[side]) * mpc * sign
            for side, (mpc, sign) in sides.items()
        }
        row = {
            "trial_id": metadata["trial_id"],
            "split": metadata["split"],
            "t_s": query,
            "speed_mps": speed,
            "correction_cmd": correction,
            "desired_left_mps": speed * (1 - correction),
            "desired_right_mps": speed * (1 + correction),
            "measured_left_mps": interpolate("measured_left_mps", query),
            "measured_right_mps": interpolate("measured_right_mps", query),
            "left_output_pct": hold("left", query),
            "right_output_pct": hold("right", query),
            "distance_left_m": distance["left"],
            "distance_right_m": distance["right"],
            "encoder_heading_rad": (distance["right"] - distance["left"]) / track,
            "elapsed_s": min(query, ELAPSED_CAP_S),
            "x_m": None, "y_m": None, "yaw_rad": None, "yaw_rate_rps": None,
            "pose_valid": False, "label_kinematic": None, "label_correction": None,
            "label_valid": False, "video_time_s": None,
        }  # fmt: skip
        if len(valid) >= 3:
            fits = [
                local_fit(t_pose, values, query, window_s)
                for values in (local[:, 0], local[:, 1], yaw)
            ]
            if all(f is not None for f in fits):
                (x, _), (y, _), (heading, rate) = fits
                nearest = valid[int(np.argmin(np.abs(t_pose - query)))]
                row.update(
                    x_m=x,
                    y_m=y,
                    yaw_rad=heading,
                    yaw_rate_rps=rate,
                    pose_valid=True,
                    video_time_s=nearest.get("video_time_s"),
                )
                if speed > 0:
                    # Kinematic hindsight label, assuming yaw rate = 2 v (c + bias) / track.
                    # build_dataset replaces the gain with the one measured in the sweep.
                    row["label_kinematic"] = correction - track * rate / (2 * speed)
        rows.append(row)
    return rows


def measured_gains(rows: list[dict], from_s: float = GAIN_FROM_S) -> dict:
    """Yaw rate per unit correction at each speed, fitted across the correction sweep."""
    gains = {}
    for speed in sorted({r["speed_mps"] for r in rows if r["speed_mps"] > 0}):
        chosen = [
            r for r in rows if r["speed_mps"] == speed and r["pose_valid"] and r["t_s"] >= from_s
        ]
        corrections = np.array([r["correction_cmd"] for r in chosen])
        if len(set(corrections)) < 2:
            gains[str(speed)] = {"gain": None, "reason": "fewer than two corrections"}
            continue
        rates = np.array([r["yaw_rate_rps"] for r in chosen])
        slope, intercept = np.polyfit(corrections, rates, 1)
        residual = rates - (slope * corrections + intercept)
        gains[str(speed)] = {
            "gain": float(slope),
            "zero_yaw_correction": float(-intercept / slope) if slope else None,
            "effective_track_m": float(2 * speed / abs(slope)) if slope else None,
            "residual_rms_rps": float(np.sqrt(np.mean(residual**2))),
            "frames": len(chosen),
            "corrections": sorted(set(corrections.tolist())),
        }
    return gains


def apply_labels(rows: list[dict], gains: dict) -> None:
    """Hindsight label with the measured gain: the correction that zeroes the yaw rate."""
    for row in rows:
        gain = (gains.get(str(row["speed_mps"])) or {}).get("gain")
        # The sign depends on which wheel the firmware calls "left"; only magnitude matters.
        if row["pose_valid"] and gain and abs(gain) > 1e-3:
            row["label_correction"] = row["correction_cmd"] - row["yaw_rate_rps"] / gain
            row["label_valid"] = True


def kinematic_check(records: list[dict], goal_m: float = 1.0, track_m: float = 0.14) -> dict:
    """Compare the measured heading-per-correction slope with 2 * goal / track."""
    usable = [
        r
        for r in records
        if r.get("reached") and r.get("quality_ok") and r.get("heading_at_goal_rad") is not None
    ]
    if len({r["correction"] for r in usable}) < 2:
        return {"usable_runs": len(usable), "slope": None}
    correction = np.array([r["correction"] for r in usable])
    heading = np.array([r["heading_at_goal_rad"] for r in usable])
    slope, intercept = np.polyfit(correction, heading, 1)
    expected = 2 * goal_m / track_m
    return {
        "usable_runs": len(usable),
        "slope_rad_per_correction": float(slope),
        "expected_slope": expected,
        "ratio": float(slope / expected),
        "effective_track_m": float(2 * goal_m / slope) if slope > 0 else None,
        "intercept_rad": float(intercept),
    }


def build_dataset(
    runs: list[Path],
    output: Path,
    rate_hz: float = 10,
    window_s: float = 0.3,
    track_m: float | None = None,
) -> dict:
    if output.exists():
        raise ValueError("Dataset directory exists; derived data is rebuilt into a new one")
    included, excluded, rows = [], {}, []
    for run in runs:
        metrics_path = run / "metrics.json"
        if not (run / "paired.jsonl").exists() or not metrics_path.exists():
            excluded[run.name] = "not paired with the video"
            continue
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("analysis_error") or not metrics.get("quality_ok"):
            excluded[run.name] = metrics.get("analysis_error") or "video tracking quality"
            continue
        frames = run_frames(run, rate_hz, window_s, track_m)
        if not frames:
            excluded[run.name] = "no armed samples"
            continue
        rows += frames
        included.append(run.name)
    training = [r for r in rows if r["split"] == "train"]
    gains = measured_gains(training)
    apply_labels(rows, gains)
    output.mkdir(parents=True)
    with (output / "frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "rate_hz": rate_hz,
        "pose_window_s": window_s,
        "track_m_override": track_m,
        "student_features": list(STUDENT_FEATURES),
        "included": included,
        "excluded": excluded,
        "frames": len(rows),
        "labeled_frames": sum(r["label_valid"] for r in rows),
        "measured_gain_from_train_split": gains,
        "torque_and_current": "not measured",
    }
    write_json(output / "meta.json", meta)
    return meta
