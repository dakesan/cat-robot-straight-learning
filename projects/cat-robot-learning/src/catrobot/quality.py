"""Acquisition quality: device sampling and host transport are summarized separately."""

import json
import threading
import time
from pathlib import Path

import numpy as np

# Initial acceptance thresholds; revise them only from rehearsal measurements.
MAX_SAMPLE_GAP_MS = 50
COMMAND_LEASE_MS = 300


class TransportLog:
    """Times every host request; host receive time is never a device sample time."""

    def __init__(self, path: Path | None = None):
        self.rows: list[dict] = []
        self.handle = path.open("w") if path is not None else None
        self.lock = threading.Lock()

    def call(self, kind: str, function, *args, **kwargs):
        row = {"kind": kind, "sent_s": time.monotonic(), "wall_time": time.time()}
        try:
            result = function(*args, **kwargs)
            row["ok"] = True
            if isinstance(result, dict) and "samples" in result:
                row["samples"] = len(result["samples"])
                row["latest"] = result.get("latest")
                # A read without a cursor only asks for the latest number; its "dropped"
                # counts samples overwritten before the trial and is not a loss.
                row["dropped"] = result.get("dropped", 0) if "after" in kwargs else 0
            return result
        except BaseException as error:
            row["ok"] = False
            row["error"] = type(error).__name__
            raise
        finally:
            row["received_s"] = time.monotonic()
            with self.lock:
                self.rows.append(row)
                if self.handle is not None:
                    self.handle.write(json.dumps(row) + "\n")
                    self.handle.flush()

    def attach(self, path: Path):
        """Start writing to a file, including requests made before it existed."""
        self.handle = path.open("w")
        for row in self.rows:
            self.handle.write(json.dumps(row) + "\n")
        self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def stats(values) -> dict | None:
    values = np.asarray(values, float)
    if not len(values):
        return None
    return {
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "n": len(values),
    }


def telemetry_quality(samples: list[dict]) -> dict:
    """Device-side sampling, from the ESP32 clock and sample counter only."""
    if not samples:
        return {"samples": 0, "accepted": False, "problems": ["no telemetry samples"]}
    numbers = np.array([s["sample"] for s in samples])
    times = np.array([s["time_ms"] for s in samples], float)
    steps = np.diff(numbers)
    gaps = np.diff(times)
    problems = []
    missing = int(np.sum(np.clip(steps - 1, 0, None)))
    if missing:
        problems.append(f"{missing} samples missing from the sequence")
    if np.any(steps <= 0) or np.any(gaps <= 0):
        problems.append("sample counter or device time is not strictly increasing")
    if len(gaps) and gaps.max() > MAX_SAMPLE_GAP_MS:
        problems.append(f"device sample gap {gaps.max():.0f} ms exceeds {MAX_SAMPLE_GAP_MS} ms")
    invalid = {
        side: int(samples[-1][f"encoder_{side}_invalid"] - samples[0][f"encoder_{side}_invalid"])
        for side in ("left", "right")
    }
    for side, value in invalid.items():
        if value:
            problems.append(f"{value} invalid {side} encoder transitions")
    armed = [s for s in samples if s.get("armed")]
    return {
        "samples": len(samples),
        "missing": missing,
        "device_interval_ms": stats(gaps),
        "invalid_delta": invalid,
        "armed_samples": len(armed),
        "armed_epochs": sorted({s["epoch"] for s in armed}),
        "accepted": not problems,
        "problems": problems,
    }


def transport_quality(rows: list[dict]) -> dict:
    """Host-side request timing; the robot clock is not involved."""
    result = {"requests": len(rows), "errors": sum(not r["ok"] for r in rows)}
    for kind in sorted({r["kind"] for r in rows}):
        selected = [r for r in rows if r["kind"] == kind]
        result[kind] = {
            "round_trip_ms": stats([(r["received_s"] - r["sent_s"]) * 1000 for r in selected]),
            "send_interval_ms": stats(np.diff([r["sent_s"] for r in selected]) * 1000),
        }
    result["dropped"] = int(sum(r.get("dropped", 0) for r in rows))
    drive = result.get("drive")
    gap = drive["send_interval_ms"]["max"] if drive and drive["send_interval_ms"] else None
    result["max_drive_gap_ms"] = gap
    result["drive_gap_within_lease"] = gap is None or gap < COMMAND_LEASE_MS
    return result


def summary_line(trial: str, telemetry: dict, transport: dict, stop_reason: str | None) -> str:
    device = telemetry.get("device_interval_ms") or {}
    poll = (transport.get("telemetry") or {}).get("round_trip_ms") or {}
    verdict = "OK" if telemetry["accepted"] and not transport["dropped"] else "NG"
    nan = float("nan")
    interval = f"{device.get('median', nan):.0f}/最大{device.get('max', nan):.0f}"
    round_trip = (
        f"{poll.get('median', nan):.0f}/99%{poll.get('p99', nan):.0f}/最大{poll.get('max', nan):.0f}"
    )
    parts = [
        f"[{verdict}] {trial}",
        f"停止理由={stop_reason}",
        f"採取{telemetry['samples']}件 抜け{telemetry.get('missing', '?')}",
        f"ESP32採取間隔 中央{interval} ms",
        f"ログ回収の往復 中央{round_trip} ms",
    ]
    if transport.get("max_drive_gap_ms") is not None:
        parts.append(f"指令間隔 最大{transport['max_drive_gap_ms']:.0f} ms")
    if telemetry.get("problems"):
        parts.append("問題: " + "; ".join(telemetry["problems"]))
    return " | ".join(parts)
