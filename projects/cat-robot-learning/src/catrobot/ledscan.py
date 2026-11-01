"""Find where the blinking sync LED sits relative to the robot marker in a video.

The LED position is expressed in marker units (0..1 across the marker, top-left origin),
the same coordinates as RobotConfig.led_offset. Each candidate position is scored by how
many complete sync patterns its red signal contains; the robot is stationary during
every pattern, so the marker frame is stable while the LED blinks.
"""

from pathlib import Path

import cv2
import numpy as np

from .sync import find_patterns
from .vision import detector, refine_corners, video_frames

SPAN = (-1.0, 2.0)  # marker units scanned around the marker
CELLS = 60


def red_excess_patches(video: Path, marker_id: int = 0):
    finder = detector()
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    scale = (SPAN[1] - SPAN[0]) / CELLS
    grid = np.array(
        [[SPAN[0] + (i + 0.5) * scale, SPAN[0] + (j + 0.5) * scale] for j in range(CELLS) for i in range(CELLS)],
        np.float32,
    )
    times, patches = [], []
    for time, frame in video_frames(video):
        corners, ids, _ = finder.detectMarkers(frame)
        if ids is None or marker_id not in ids.ravel():
            continue
        index = int(np.flatnonzero(ids.ravel() == marker_id)[0])
        points = refine_corners(frame, corners[index].reshape(4, 2))
        transform = cv2.getPerspectiveTransform(unit, points.astype(np.float32))
        pixels = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), transform).reshape(-1, 2)
        x = np.clip(np.rint(pixels[:, 0]).astype(int), 0, frame.shape[1] - 1)
        y = np.clip(np.rint(pixels[:, 1]).astype(int), 0, frame.shape[0] - 1)
        sample = frame[y, x].astype(float)
        patches.append(sample[:, 2] - np.maximum(sample[:, 0], sample[:, 1]))
        times.append(time)
    return np.array(times), np.array(patches), grid


def locate_led(video: Path, marker_id: int = 0) -> dict:
    times, signals, grid = red_excess_patches(video, marker_id)
    if len(times) < 30:
        raise ValueError("Too few frames with the marker")
    period = float(np.median(np.diff(times)))
    best = None
    for k in range(signals.shape[1]):
        series = signals[:, k]
        low, high = np.percentile(series, [20, 99.5])
        if high - low < 20:
            continue
        on = series > (low + high) / 2
        changes = [
            ((times[i - 1] + times[i]) / 2, bool(on[i]))
            for i in range(1, len(times))
            if on[i] != on[i - 1] and times[i] - times[i - 1] <= 1.5 * period
        ]
        found = len(find_patterns(changes))
        score = (found, high - low)
        if best is None or score > best[0]:
            best = (score, k, low, high)
    if best is None or best[0][0] == 0:
        raise ValueError("No position shows the sync pattern")
    (patterns, contrast), k, low, high = best
    return {
        "led_offset": [float(grid[k][0]), float(grid[k][1])],
        "patterns": patterns,
        "contrast": float(contrast),
        "threshold": float((low + high) / 2),
        "frames": len(times),
    }
