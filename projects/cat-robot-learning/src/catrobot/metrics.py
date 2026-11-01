"""Trajectory metrics in the initial body frame, including incomplete runs."""

import numpy as np


def trajectory_metrics(poses: list[dict], goal_m: float = 1.0) -> dict:
    valid = [p for p in poses if p.get("valid", True)]
    if len(valid) < 3:
        return {
            "reached": False,
            "quality_ok": False,
            "max_lateral_m": None,
            "lateral_at_goal_m": None,
            "heading_at_goal_rad": None,
            "actual_speed_mps": None,
            "forward_m": 0.0,
            "success": False,
        }
    t = np.array([p["time_s"] for p in valid])
    positions = np.array([[p["x_m"], p["y_m"]] for p in valid])
    yaw = np.unwrap([p["yaw_rad"] for p in valid])
    if np.any(np.diff(t) <= 0):
        raise ValueError("Pose timestamps must be strictly increasing")
    # One frame's marker heading is noisy (about a degree), and a degree at the start is
    # 17 mm of apparent lateral error after 1 m, so the start frame uses the median of
    # the first half second, when the robot has barely moved.
    start = t <= t[0] + 0.5
    angle = float(np.median(yaw[start]))
    origin = np.median(positions[start][:1], axis=0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    local = (positions - origin) @ rotation
    heading = yaw - angle
    reached_indices = np.flatnonzero(local[:, 0] >= goal_m)
    reached = bool(len(reached_indices))
    end = int(reached_indices[0]) if reached else len(valid) - 1
    if reached and end > 0:
        fraction = (goal_m - local[end - 1, 0]) / (local[end, 0] - local[end - 1, 0])
        lateral = float(local[end - 1, 1] + fraction * (local[end, 1] - local[end - 1, 1]))
        final_heading = float(heading[end - 1] + fraction * (heading[end] - heading[end - 1]))
        final_time = float(t[end - 1] + fraction * (t[end] - t[end - 1]))
        # One frame's heading is noisy, so use the median within 10 cm of the goal.
        near = np.abs(local[:, 0] - goal_m) <= 0.1
        if near.sum() >= 3:
            final_heading = float(np.median(heading[near]))
    else:
        lateral, final_heading, final_time = (
            float(local[end, 1]),
            float(heading[end]),
            float(t[end]),
        )
    moving = np.flatnonzero(np.linalg.norm(local[: end + 1], axis=1) >= 0.005)
    begin = max(0, int(moving[0]) - 1) if len(moving) else 0
    duration = final_time - t[begin]
    # Measured 2026-11-01: a hand-drawn marker filmed through the Pixel webcam is read in
    # about 60 % of frames while moving (motion blur), with gaps up to 0.27 s. Tracked
    # positions agree with the encoders to about 1 mm, so half coverage with gaps of at
    # most 0.35 s (7 cm at 0.2 m/s on a near-straight path) is accepted.
    quality = len(valid) / len(poses) >= 0.5 and float(np.max(np.diff(t))) <= 0.35
    before_goal = local[:end, 1] if reached else local[:, 1]
    max_lateral = float(max(abs(lateral), np.max(np.abs(before_goal))))
    return {
        "reached": reached,
        "quality_ok": quality,
        "max_lateral_m": max_lateral,
        "lateral_at_goal_m": lateral,
        "heading_at_goal_rad": final_heading,
        "actual_speed_mps": float(goal_m / duration) if reached and duration > 0 else None,
        "forward_m": float(local[-1, 0]),
        "success": bool(
            reached and quality and max_lateral <= 0.05 and abs(final_heading) <= np.deg2rad(5)
        ),
    }
