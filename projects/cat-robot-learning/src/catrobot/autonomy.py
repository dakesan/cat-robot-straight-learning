"""Camera-guided reset of the robot to the start pose between trials.

The overhead camera is used only to bring the robot back to the start line when no
person can do it. Trials themselves are driven without camera input, as before.
Motion uses short open-loop output pulses through the normal command lease; after each
pulse the robot stops and a fresh camera pose decides the next pulse.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np

from .config import RobotConfig
from .vision import detector, marker_pose, refine_corners


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Pose:
    x: float
    y: float
    yaw: float
    taken: float


class LiveCamera:
    """Robot pose from the snapshot file that the recorder keeps overwriting."""

    def __init__(self, snapshot: Path, camera: Path, arena: Path, config: RobotConfig):
        self.snapshot = snapshot
        self.camera = json.loads(camera.read_text())
        self.world = np.array(json.loads(arena.read_text())["world_from_camera"], float)
        self.config = config
        self.finder = detector()

    def steady_pose(self, newer_than: float = 0.0, count: int = 3) -> Pose | None:
        """Median of consecutive fresh snapshots; one hand-drawn-marker frame can be off by
        about 10 degrees in heading."""
        poses = []
        for _ in range(count):
            pose = self.pose(newer_than=newer_than)
            if pose is None:
                return None
            poses.append(pose)
            newer_than = pose.taken
        yaw = math.atan2(
            float(np.median([math.sin(p.yaw) for p in poses])),
            float(np.median([math.cos(p.yaw) for p in poses])),
        )
        return Pose(
            float(np.median([p.x for p in poses])),
            float(np.median([p.y for p in poses])),
            yaw,
            poses[-1].taken,
        )

    def pose(self, newer_than: float = 0.0, timeout: float = 3.0) -> Pose | None:
        until = time.time() + timeout
        while time.time() < until:
            try:
                taken = self.snapshot.stat().st_mtime
            except FileNotFoundError:
                taken = 0
            if taken > newer_than:
                image = cv2.imread(str(self.snapshot))
                if image is not None:
                    corners, ids, _ = self.finder.detectMarkers(image)
                    if ids is not None and self.config.marker_id in ids.ravel():
                        index = int(np.flatnonzero(ids.ravel() == self.config.marker_id)[0])
                        points = refine_corners(image, corners[index].reshape(4, 2))
                        try:
                            found = marker_pose(points, self.camera, self.world, self.config)
                        except ValueError:
                            found = None
                        # A robot tilted against an obstacle still gives a usable,
                        # less square view; accept it for moving the robot back.
                        if found and found["squareness"] <= 0.2:
                            return Pose(found["x_m"], found["y_m"], found["yaw_rad"], taken)
            time.sleep(0.05)
        return None


class Mover:
    """Short output pulses; every pulse arms, streams commands every 100 ms, then stops."""

    def __init__(self, client):
        self.client = client

    def pulse(self, left: int, right: int, seconds: float) -> dict:
        left, right = (int(max(-50, min(50, v))) for v in (left, right))
        epoch = self.client.post("/api/arm")["epoch"]
        seq, until = 0, time.monotonic() + seconds
        try:
            while time.monotonic() < until:
                seq += 1
                try:
                    self.client.post(
                        "/api/command", epoch=epoch, seq=seq, left=left, right=right, servo=0
                    )
                except httpx.HTTPStatusError:
                    break  # the robot stopped itself (lease or encoder); the next pose decides
                time.sleep(0.1)
        finally:
            self.client.post("/api/stop")
        return self.client.get("/api/status")


@dataclass
class Lane:
    start_x: float = 0.06
    start_y: float = 0.45
    heading: float = 0.0
    # Safe floor box in the grid frame (m); outside it the robot is not moved.
    x_range: tuple[float, float] = (-0.15, 1.2)
    y_range: tuple[float, float] = (-0.1, 1.0)


def reset_to_start(
    camera: LiveCamera,
    mover: Mover,
    lane: Lane,
    position_tolerance: float = 0.04,
    heading_tolerance: float = math.radians(4),
    max_steps: int = 60,
    log=print,
) -> dict:
    """Drive back to the start pose with rotate-then-straight pulses checked by camera."""
    turn_rate = math.radians(120)  # rad/s at 40 % in place; adapted from observations
    speed = 0.18  # m/s at 35 %; adapted from observations
    settled, last = 0.0, None
    for step in range(max_steps):
        pose = camera.steady_pose(newer_than=settled)
        if pose is None:
            return {"ok": False, "reason": "robot marker not visible", "steps": step}
        if last is not None:
            kind, before, seconds = last
            if kind == "turn":
                turned = abs(wrap(pose.yaw - before.yaw))
                if turned > math.radians(3):
                    turn_rate = 0.6 * turn_rate + 0.4 * turned / seconds
            else:
                moved = math.hypot(pose.x - before.x, pose.y - before.y)
                if moved > 0.02:
                    speed = 0.6 * speed + 0.4 * moved / seconds
        inside_x = lane.x_range[0] <= pose.x <= lane.x_range[1]
        inside_y = lane.y_range[0] <= pose.y <= lane.y_range[1]
        if not (inside_x and inside_y):
            return {"ok": False, "reason": f"outside safe area at ({pose.x:.2f}, {pose.y:.2f})", "steps": step}
        dx, dy = lane.start_x - pose.x, lane.start_y - pose.y
        distance = math.hypot(dx, dy)
        if distance < position_tolerance:
            error = wrap(lane.heading - pose.yaw)
            if abs(error) < heading_tolerance:
                return {"ok": True, "steps": step, "x": pose.x, "y": pose.y, "yaw_deg": math.degrees(pose.yaw)}
            kind, amount = "turn", error
        else:
            bearing = math.atan2(dy, dx)
            backward = abs(wrap(bearing - pose.yaw)) > math.pi / 2
            facing = wrap(pose.yaw + math.pi) if backward else pose.yaw
            error = wrap(bearing - facing)
            limit = math.radians(10 if distance < 0.15 else 15)
            if abs(error) > limit:
                kind, amount = "turn", error
            else:
                kind, amount = "drive", -distance if backward else distance
        sign = 1 if amount > 0 else -1
        if kind == "turn":
            seconds = min(0.5, max(0.12, abs(amount) / turn_rate))
            # Measured 2026-11-01: left=-40/right=+40 turned the robot clockwise (the
            # firmware's "left" motor drives the wheel on the right of the travel direction).
            mover.pulse(40 * sign, -40 * sign, seconds)
            log(f"step {step}: turn {math.degrees(amount):+.1f} deg for {seconds:.2f} s")
        else:
            seconds = min(0.6, max(0.15, abs(amount) / speed))
            mover.pulse(35 * sign, 35 * sign, seconds)
            log(f"step {step}: drive {amount:+.3f} m for {seconds:.2f} s")
        last = (kind, pose, seconds)
        settled = time.time() + 0.25
    return {"ok": False, "reason": "did not converge", "steps": max_steps}


def back_off_until_visible(
    camera: LiveCamera, mover: Mover, pulses: int = 5, log=print, forward: bool = False
) -> bool:
    """Retrace the last motion in short pulses until the marker is seen again.

    After a trial the robot drove forward out of view, so it reverses; during a reset it
    mostly drives backward toward the start, so it then moves forward instead.
    """
    sign = 1 if forward else -1
    for pulse in range(pulses):
        if camera.pose(newer_than=time.time() + 0.3, timeout=2.0) is not None:
            return True
        try:
            # Some headings make the hand-drawn marker unreadable even in plain view, so
            # first turn in place (no travel); then retrace the last motion.
            if pulse % 2 == 0:
                mover.pulse(40, -40, 0.25)
            else:
                mover.pulse(35 * sign, 35 * sign, 0.6)
        except httpx.HTTPError:
            time.sleep(1.0)
        log(f"back off {pulse}")
    return camera.pose(newer_than=time.time() + 0.3, timeout=2.0) is not None
