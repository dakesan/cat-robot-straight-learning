"""Clock mapping from the robot's own motion when no sync LED is visible.

Encoders give the distance travelled on the ESP32 clock; the camera gives the marker's
distance from its resting start on the video clock. Both start at zero, ramp up when the
robot moves and stop together, so the video-to-device offset is the shift that makes the
two curves agree. The Mac wall clock only narrows the search window.
"""

import json
from pathlib import Path

import numpy as np

from .config import RobotConfig, write_json
from .sync import analyze_run, read_jsonl


def encoder_distance(samples: list[dict], config: RobotConfig) -> tuple[np.ndarray, np.ndarray]:
    times = np.array([s["time_ms"] / 1000 for s in samples])
    left = np.array([s["encoder_left"] for s in samples], float)
    right = np.array([s["encoder_right"] for s in samples], float)
    forward = (
        (left - left[0]) * config.left.meters_per_count * config.left.encoder_sign
        + (right - right[0]) * config.right.meters_per_count * config.right.encoder_sign
    ) / 2
    return times, forward


def motion_clock(
    samples: list[dict],
    poses: list[dict],
    config: RobotConfig,
    coarse_offset: float,
    search_s: float = 5.0,
    step_s: float = 0.002,
) -> dict:
    """Offset o with device_time = video_time + o, from distance-versus-time curves."""
    armed = [s for s in samples if s["armed"]]
    if not armed:
        raise ValueError("Telemetry has no armed interval")
    t_dev, forward = encoder_distance(samples, config)
    arm = armed[0]["time_ms"] / 1000
    end = armed[-1]["time_ms"] / 1000
    low, high = arm - 1.5 - search_s, end + 1.5 + search_s
    window = [p for p in poses if p.get("valid") and low <= p["time_s"] + coarse_offset <= high]
    if len(window) < 20:
        raise ValueError("Too few tracked frames around the trial")
    t_vid = np.array([p["time_s"] for p in window])
    xy = np.array([[p["x_m"], p["y_m"]] for p in window])
    best = None
    for offset in np.arange(coarse_offset - search_s, coarse_offset + search_s, step_s):
        device = t_vid + offset
        inside = (device >= t_dev[0]) & (device <= t_dev[-1])
        if inside.sum() < 20:
            continue
        rest = inside & (device < arm)
        if rest.sum() < 3:
            continue
        origin = np.median(xy[rest], axis=0)
        observed = np.linalg.norm(xy[inside] - origin, axis=1)
        predicted = np.abs(np.interp(device[inside], t_dev, forward))
        scale = float(np.dot(observed, predicted) / max(np.dot(predicted, predicted), 1e-9))
        if not 0.85 <= scale <= 1.15:
            continue
        rms = float(np.sqrt(np.mean((observed - scale * predicted) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, float(offset), scale, int(inside.sum()))
    if best is None:
        raise ValueError("No offset aligns encoder and video distance")
    rms, offset, scale, frames = best
    speed = float(np.max(np.abs(np.diff(forward)) / np.maximum(np.diff(t_dev), 1e-3)))
    return {
        "scale": 1.0,
        "offset_s": offset,
        "residual_m": rms,
        "residual_s": rms / max(speed, 0.05),
        "distance_scale": scale,
        "frames": frames,
        "anchors": [[arm - 1.0 - offset, arm - 1.0], [end + 1.0 - offset, end + 1.0]],
        "frame_period_s": float(np.median(np.diff(t_vid))),
        "method": "encoder distance vs tracked marker distance from rest",
    }


def analyze_motion_session(
    runs: Path, session: Path, track: Path, video_start_wall: float, config: RobotConfig
) -> dict:
    """Pair each attempt with the session video by motion, using wall time as a hint."""
    record = json.loads(session.read_text())
    poses = read_jsonl(track)
    video_end_wall = video_start_wall + (poses[-1]["time_s"] + 5.0 if poses else 0.0)
    results = []
    for attempt in record["attempts"]:
        run = runs / attempt["directory"]
        result = {k: attempt.get(k) for k in ("trial_id", "directory", "number")}
        started = attempt.get("started_wall_time")
        if started is not None and not video_start_wall <= started <= video_end_wall:
            continue  # filmed in another video of this session; leave its metrics alone
        try:
            samples = read_jsonl(run / "telemetry.jsonl")
            arm_rows = [r for r in read_jsonl(run / "transport.jsonl") if r["kind"] == "arm"]
            armed = [s for s in samples if s["armed"]]
            if not arm_rows or not armed:
                raise ValueError("trial never armed")
            # Wall clock of the arm request versus the device time of the first armed sample.
            coarse = armed[0]["time_ms"] / 1000 - (arm_rows[0]["wall_time"] - video_start_wall)
            clock = motion_clock(samples, poses, config, coarse)
            metrics = analyze_run(run, track, clock=clock)
            result.update(
                accepted=True,
                offset_s=clock["offset_s"],
                residual_m=clock["residual_m"],
                distance_scale=clock["distance_scale"],
                quality_ok=metrics["quality_ok"],
                success=metrics["success"],
            )
        except (ValueError, FileNotFoundError, KeyError) as error:
            metadata = json.loads((run / "metadata.json").read_text())
            write_json(
                run / "metrics.json",
                dict(
                    metadata,
                    reached=False,
                    quality_ok=False,
                    max_lateral_m=None,
                    lateral_at_goal_m=None,
                    heading_at_goal_rad=None,
                    actual_speed_mps=None,
                    success=False,
                    analysis_error=str(error),
                ),
            )
            result.update(accepted=False, reason=str(error))
        results.append(result)
    report = {"session_id": record["session_id"], "track": str(track), "attempts": results}
    write_json(runs / f"motion-sync-{record['session_id']}.json", report)
    return report
