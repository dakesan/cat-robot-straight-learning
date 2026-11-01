"""Optical clock mapping and run-level pairing without using video during control."""

import itertools
import json
from pathlib import Path

import numpy as np

from .config import write_json
from .metrics import trajectory_metrics


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rising_edges(rows: list[dict], key: str, time_key: str, threshold: float) -> list[float]:
    result, previous = [], False
    for row in rows:
        current = float(row.get(key, 0)) > threshold
        if current and not previous:
            result.append(float(row[time_key]))
        previous = current
    return result


def pulse_groups(edges: list[float]) -> list[float]:
    result = []
    i = 0
    while i + 2 < len(edges):
        if (
            abs(edges[i + 1] - edges[i] - 0.5) <= 0.12
            and abs(edges[i + 2] - edges[i + 1] - 1.0) <= 0.12
        ):
            result.extend(edges[i : i + 3])
            i += 3
        else:
            i += 1
    return result


def fit_clock(anchors: list[list[float]], frame_period: float) -> dict:
    values = np.array(anchors, float)
    if len(values) < 3 or np.ptp(values[:, 0]) < 1:
        raise ValueError("Need at least three optical anchors spanning the run")
    matrix = np.column_stack([values[:, 0], np.ones(len(values))])
    scale, offset = np.linalg.lstsq(matrix, values[:, 1], rcond=None)[0]
    residual = float(np.max(np.abs(matrix @ [scale, offset] - values[:, 1])))
    if not 0.95 < scale < 1.05 or residual > frame_period:
        raise ValueError("Optical synchronization residual exceeds one video frame")
    return {
        "scale": float(scale),
        "offset_s": float(offset),
        "residual_s": residual,
        "anchors": anchors,
        "frame_period_s": frame_period,
    }


def analyze_run(
    run: Path,
    track: Path,
    anchors_path: Path | None = None,
    led_threshold: float = 30,
    pulse_group: int = 0,
    clock: dict | None = None,
) -> dict:
    metadata = json.loads((run / "metadata.json").read_text())
    if metadata["source"] != "measured":
        raise ValueError("Use this command for measured runs, not simulator truth")
    samples, poses = read_jsonl(run / "telemetry.jsonl"), read_jsonl(track)
    if len(poses) < 3:
        raise ValueError("Video contains too few frames")
    frame_period = float(np.median(np.diff([p["time_s"] for p in poses])))
    if clock is not None:
        anchors = clock["anchors"]
    elif anchors_path is not None:
        anchors = json.loads(anchors_path.read_text())
    else:
        device = pulse_groups([x / 1000 for x in rising_edges(samples, "sync_led", "time_ms", 0.5)])
        optical = pulse_groups(rising_edges(poses, "led_signal", "time_s", led_threshold))
        optical = optical[pulse_group * 3 :]
        if len(device) < 6 or len(optical) < len(device):
            raise ValueError(
                "Need complete start/end sync patterns; provide explicit anchors if detection fails"
            )
        anchors = [[a, b] for a, b in zip(optical[: len(device)], device)]
    if clock is None:
        clock = fit_clock(anchors, frame_period)
    armed = [s for s in samples if s["armed"]]
    if not armed or len({s["epoch"] for s in armed}) != 1:
        raise ValueError("Telemetry must contain exactly one armed run")
    start, end = armed[0]["time_ms"] / 1000, armed[-1]["time_ms"] / 1000 + 0.02
    if min(p[1] for p in anchors) > start or max(p[1] for p in anchors) < end:
        raise ValueError("Optical anchors must bracket the complete armed interval")
    times = np.array([s["time_ms"] / 1000 for s in samples])
    paired = []
    for pose in poses:
        device_time = clock["scale"] * pose["time_s"] + clock["offset_s"]
        if start <= device_time <= end:
            index = max(0, int(np.searchsorted(times, device_time, side="right")) - 1)
            paired.append(
                dict(
                    pose,
                    time_s=device_time - start,
                    video_time_s=pose["time_s"],
                    telemetry=samples[index],
                )
            )
    metrics = dict(metadata, **trajectory_metrics(paired, goal_m=metadata.get("goal_m", 1.0)))
    if metadata.get("dropped_samples", 0):
        metrics["quality_ok"] = metrics["success"] = False
    write_json(run / "sync.json", clock)
    write_json(run / "metrics.json", metrics)
    (run / "paired.jsonl").write_text("".join(json.dumps(p) + "\n" for p in paired))
    return metrics


# Sync LED pattern from telemetry.h syncLight(): (milliseconds after start, LED state).
PATTERN = ((0.0, True), (0.15, False), (0.5, True), (0.8, False), (1.5, True), (1.95, False))


def transitions(rows: list[dict], key, time_of, max_gap: float) -> list[tuple[float, bool]]:
    """LED state changes between consecutive observations, timed at their midpoint.

    A logged change only says the edge happened between two observations, so the
    midpoint halves the quantization error of either the 20 ms log or the video frames.
    """
    result = []
    for previous, row in itertools.pairwise(rows):
        before, after = key(previous), key(row)
        if before is None or after is None or before == after:
            continue
        if time_of(row) - time_of(previous) > max_gap:
            continue
        result.append(((time_of(previous) + time_of(row)) / 2, after))
    return result


def find_patterns(changes: list[tuple[float, bool]], tolerance: float = 0.06) -> list[list[float]]:
    """Six consecutive edges matching the start/end pattern, as absolute edge times."""
    found, i = [], 0
    while i + len(PATTERN) <= len(changes):
        window = changes[i : i + len(PATTERN)]
        origin = window[0][0]
        if all(
            state == expected and abs(t - origin - nominal) <= tolerance
            for (t, state), (nominal, expected) in zip(window, PATTERN, strict=True)
        ):
            found.append([t for t, _ in window])
            i += len(PATTERN)
        else:
            i += 1
    return found


def device_patterns(samples: list[dict]) -> list[list[float]]:
    changes = transitions(
        samples, lambda s: bool(s.get("sync_led")), lambda s: s["time_ms"] / 1000, 0.06
    )
    return find_patterns(changes)


def optical_patterns(poses: list[dict], threshold: float, frame_period: float) -> list:
    # Only frames where the robot marker was found carry an LED reading.
    seen = [p for p in poses if "x_m" in p]
    changes = transitions(
        seen,
        lambda p: float(p["led_signal"]) > threshold,
        lambda p: p["time_s"],
        1.5 * frame_period,
    )
    return find_patterns(changes)


def match_groups(device: list, optical: list, scale: float, offset: float, tolerance: float):
    """Greedy one-to-one matching of pattern start times under device = scale*video+offset."""
    pairs, used = [], set()
    for d_index, d in enumerate(device):
        best = None
        for o_index, o in enumerate(optical):
            if o_index in used:
                continue
            error = abs(d[0] - (scale * o[0] + offset))
            if error <= tolerance and (best is None or error < best[0]):
                best = (error, o_index)
        if best is not None:
            used.add(best[1])
            pairs.append((d_index, best[1]))
    return pairs


def session_clock(device: list, optical: list) -> tuple[float, float, list]:
    """Find the video-to-device mapping shared by all patterns of one boot."""
    if not device or not optical:
        return 1.0, 0.0, []
    best = None
    for d in device:
        for o in optical:
            pairs = match_groups(device, optical, 1.0, d[0] - o[0], 0.3)
            if best is None or len(pairs) > len(best[2]):
                best = (1.0, d[0] - o[0], pairs)
    scale, offset, pairs = best
    for tolerance in (0.1, 0.05):
        video = np.concatenate([optical[o] for _, o in pairs])
        clock = np.concatenate([device[d] for d, _ in pairs])
        if len(pairs) >= 2 and np.ptp(video) > 1:
            scale, offset = np.polyfit(video, clock, 1)
        else:
            offset = float(np.mean(clock - video))
        pairs = match_groups(device, optical, scale, offset, tolerance)
    return float(scale), float(offset), pairs


def analyze_session_runs(
    runs: Path, session: Path, track: Path, led_threshold: float = 30
) -> dict:
    """Pair every attempt of a filmed session by time, not by its position in the schedule."""
    record = json.loads(session.read_text())
    poses = read_jsonl(track)
    if len(poses) < 3:
        raise ValueError("Video contains too few frames")
    frame_period = float(np.median(np.diff([p["time_s"] for p in poses])))
    optical = optical_patterns(poses, led_threshold, frame_period)
    attempts = []
    for attempt in record["attempts"]:
        run = runs / attempt["directory"]
        if not (run / "telemetry.jsonl").exists():
            attempts.append({**attempt, "accepted": False, "reason": "no telemetry"})
            continue
        samples = read_jsonl(run / "telemetry.jsonl")
        armed = [s for s in samples if s["armed"]]
        attempts.append(
            {
                **attempt,
                "run": run,
                "patterns": device_patterns(samples),
                "armed": (armed[0]["time_ms"] / 1000, armed[-1]["time_ms"] / 1000)
                if armed
                else None,
            }
        )
    clocks = {}
    for boot in {a.get("boot_id") for a in attempts if "patterns" in a}:
        members = [a for a in attempts if "patterns" in a and a.get("boot_id") == boot]
        device = [p for a in members for p in a["patterns"]]
        scale, offset, pairs = session_clock(device, optical)
        owner = [(a, i) for a in members for i in range(len(a["patterns"]))]
        for a in members:
            a["matched"] = []
        for d_index, o_index in pairs:
            attempt, local = owner[d_index]
            attempt["matched"].append((attempt["patterns"][local], optical[o_index]))
        clocks[str(boot)] = {"scale": scale, "offset_s": offset, "matched": len(pairs)}
        for a in members:
            a["scale"] = scale
    summary = []
    for attempt in attempts:
        result = {k: attempt.get(k) for k in ("trial_id", "directory", "number")}
        if "patterns" not in attempt:
            summary.append({**result, "accepted": False, "reason": attempt["reason"]})
            continue
        matched, window = attempt["matched"], attempt["armed"]
        reason = None
        if window is None:
            reason = "robot was never armed"
        elif not any(d[-1] < window[0] for d, _ in matched):
            reason = "start sync pattern not found in the video"
        elif not any(d[0] > window[1] for d, _ in matched):
            reason = "end sync pattern not found in the video"
        clock = None
        if reason is None:
            video = np.concatenate([o for _, o in matched])
            device = np.concatenate([d for d, _ in matched])
            scale = attempt["scale"]
            offset = float(np.mean(device - scale * video))
            residual = float(np.max(np.abs(device - (scale * video + offset))))
            clock = {
                "scale": scale,
                "offset_s": offset,
                "residual_s": residual,
                "anchors": np.column_stack([video, device]).tolist(),
                "frame_period_s": frame_period,
                "method": "session time matching of LED pattern edges (midpoint timed)",
            }
            if residual > frame_period:
                reason = f"sync residual {residual * 1000:.1f} ms exceeds one frame"
        run = attempt["run"]
        if reason is None:
            metrics = analyze_run(run, track, clock=clock)
        else:
            metadata = json.loads((run / "metadata.json").read_text())
            metrics = dict(
                metadata,
                reached=False,
                quality_ok=False,
                max_lateral_m=None,
                lateral_at_goal_m=None,
                heading_at_goal_rad=None,
                actual_speed_mps=None,
                success=False,
                analysis_error=reason,
            )
            write_json(run / "metrics.json", metrics)
        summary.append(
            {
                **result,
                "accepted": reason is None,
                "reason": reason,
                "residual_s": clock["residual_s"] if clock else None,
                "quality_ok": metrics["quality_ok"],
                "success": metrics["success"],
            }
        )
    report = {
        "session_id": record["session_id"],
        "track": str(track),
        "frame_period_s": frame_period,
        "optical_patterns": len(optical),
        "clocks": clocks,
        "attempts": summary,
    }
    write_json(runs / f"sync-{record['session_id']}.json", report)
    return report
