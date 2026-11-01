"""Wheel conversion from manual turns and a hand-pushed straight roll."""

import math
from pathlib import Path

import numpy as np

from .sync import read_jsonl

SIDES = ("left", "right")


def main_motion(rows: list[dict], side: str, rate: float = 50, gap_s: float = 2.0) -> dict:
    """Count change over the longest continuous turning of one wheel.

    Handling the robot before or after the turns also moves the encoders a little, so
    the first-to-last difference of the whole log is not trusted.
    """
    key = f"encoder_{side}"
    times = np.array([r.get("host_monotonic_s", r.get("time_ms", 0) / 1000) for r in rows])
    counts = np.array([r[key] for r in rows], float)
    active = []
    for i in range(1, len(rows)):
        window = (times >= times[i] - 1.0) & (times <= times[i])
        if np.ptp(counts[window]) >= rate:
            active.append(i)
    if not active:
        return {"delta": 0, "start_s": None, "end_s": None, "outside": 0}
    segments, current = [], [active[0]]
    for i in active[1:]:
        if times[i] - times[current[-1]] <= gap_s:
            current.append(i)
        else:
            segments.append(current)
            current = [i]
    segments.append(current)

    def delta(segment):
        start = max(0, segment[0] - 1)
        while start > 0 and times[segment[0]] - times[start] < 1.0:
            start -= 1
        return counts[segment[-1]] - counts[start], start

    best = max(segments, key=lambda seg: abs(delta(seg)[0]))
    change, start = delta(best)
    total = counts[-1] - counts[0]
    return {
        "delta": int(change),
        "start_s": float(times[start] - times[0]),
        "end_s": float(times[best[-1]] - times[0]),
        "outside": int(total - change),
    }


def count_deltas(log: Path) -> dict:
    rows = read_jsonl(log)
    if len(rows) < 2:
        raise ValueError(f"{log}: need at least two status rows")
    if len({r["boot_id"] for r in rows}) != 1:
        raise ValueError(f"{log}: device rebooted during capture")
    result = {}
    for side in SIDES:
        motion = main_motion(rows, side)
        result[side] = {
            **motion,
            "invalid": rows[-1][f"encoder_{side}_invalid"] - rows[0][f"encoder_{side}_invalid"],
        }
    return result


def spread(values: list[float]) -> float:
    """Relative range across repeats; 0 when only one repeat exists."""
    values = np.asarray(values, float)
    return float(np.ptp(values) / np.mean(values)) if len(values) > 1 else 0.0


def wheel_calibration(
    turn_logs: list[Path],
    roll_logs: list[Path],
    turns: float = 10,
    roll_m: float = 1.0,
    tolerance: float = 0.01,
) -> dict:
    """Counts/rev and sign from turns; a common effective radius from the roll."""
    if turns <= 0 or roll_m <= 0 or not turn_logs or not roll_logs:
        raise ValueError("Need positive turns, roll distance and at least one log of each kind")
    turn = [count_deltas(path) for path in turn_logs]
    roll = [count_deltas(path) for path in roll_logs]
    result = {"turn_logs": [str(p) for p in turn_logs], "roll_logs": [str(p) for p in roll_logs]}
    problems, warnings = [], []
    for side in SIDES:
        deltas = [t[side]["delta"] for t in turn]
        if any(log[side]["invalid"] for log in turn + roll):
            problems.append(f"{side}: invalid encoder transitions; inspect wiring")
        if any(d == 0 for d in deltas) or len({np.sign(d) for d in deltas}) != 1:
            problems.append(f"{side}: no response or inconsistent turn direction")
            continue
        per_rev = [abs(d) / turns for d in deltas]
        rolled = [abs(r[side]["delta"]) for r in roll]
        if any(np.sign(r[side]["delta"]) != np.sign(deltas[0]) for r in roll):
            problems.append(f"{side}: roll direction disagrees with forward turns")
        result[side] = {
            "turn_deltas": deltas,
            "turn_windows_s": [[t[side]["start_s"], t[side]["end_s"]] for t in turn],
            "counts_outside_main_motion": [log[side]["outside"] for log in turn + roll],
            "counts_per_revolution": float(np.mean(per_rev)),
            "encoder_sign": int(np.sign(deltas[0])),
            "turn_spread": spread(per_rev),
            "roll_counts": rolled,
            "roll_spread": spread(rolled),
            "meters_per_count_from_roll": float(roll_m / np.mean(rolled)),
        }
        if result[side]["roll_spread"] > tolerance:
            problems.append(f"{side}: roll_spread {result[side]['roll_spread']:.2%} exceeds {tolerance:.0%}")
        if result[side]["turn_spread"] > tolerance:
            # Hand-counted turns only fix the sign and an approximate counts/rev; the
            # distance per count comes from the taped roll, which does not use them.
            warnings.append(f"{side}: turn_spread {result[side]['turn_spread']:.2%} exceeds {tolerance:.0%}")
    result["warnings"] = warnings
    if problems:
        result["problems"] = problems
        result["accepted"] = False
        return result
    # One counts/rev for both wheels (same gearbox) and one radius preserving the mean
    # rolled metres per count. A hand push is rarely perfectly straight, so the
    # left/right difference is reported but not trusted; video training learns it.
    per_rev = float(np.mean([result[side]["counts_per_revolution"] for side in SIDES]))
    meters_per_count = float(np.mean([result[side]["meters_per_count_from_roll"] for side in SIDES]))
    result["common_counts_per_revolution"] = per_rev
    result["meters_per_count"] = meters_per_count
    result["radius_m"] = meters_per_count * per_rev / (2 * math.pi)
    left, right = (result[side]["meters_per_count_from_roll"] for side in SIDES)
    result["left_right_roll_difference"] = float((left - right) / meters_per_count)
    result["accepted"] = True
    return result
