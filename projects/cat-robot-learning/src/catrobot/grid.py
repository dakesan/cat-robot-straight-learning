"""Camera and floor calibration from a taped square grid and a measured camera height.

Replaces the printed floor markers and checkerboard when printing is not possible. The
grid gives floor points at a known spacing; the measured lens height fixes the focal
length, which a single near-overhead view of a plane cannot determine on its own.
Lens distortion is assumed negligible and the principal point is the image centre.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import brentq

from .config import write_json


def tape_mask(image: np.ndarray, margin: int = 40) -> np.ndarray:
    """Bright tape (orange or pale) inside the bounding box of the orange tape."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, np.array((5, 80, 120)), np.array((30, 255, 255)))
    ys, xs = np.nonzero(orange)
    if len(xs) < 1000:
        raise ValueError("Grid tape was not found in the image")
    x0, x1 = np.percentile(xs, [1, 99]).astype(int)
    y0, y1 = np.percentile(ys, [1, 99]).astype(int)
    bright = cv2.inRange(hsv, np.array((0, 0, 225)), np.array((40, 255, 255)))
    roi = np.zeros_like(orange)
    roi[max(0, y0 - margin) : y1 + margin, max(0, x0 - margin) : x1 + margin] = 255
    return cv2.bitwise_and(cv2.bitwise_or(bright, orange), roi)


def fit_lines(segments: np.ndarray, position) -> list[tuple[np.ndarray, np.ndarray]]:
    values = np.array([position(s) for s in segments])
    order = np.argsort(values)
    groups, current = [], [order[0]]
    for i in order[1:]:
        if values[i] - values[current[-1]] < 40:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    lines = []
    for group in groups:
        points = np.vstack([segments[group, :2], segments[group, 2:]]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([vx, vy]), np.array([x0, y0])))
    return lines


def grid_intersections(image: np.ndarray, count: int = 4) -> np.ndarray:
    """Pixel intersections [column (image x order), row (bottom to top)] of a count x count grid."""
    mask = tape_mask(image)
    found = cv2.HoughLinesP(mask, 1, np.pi / 720, 200, minLineLength=300, maxLineGap=40)
    if found is None:
        raise ValueError("No grid lines detected")
    segments = found.reshape(-1, 4).astype(float)
    angle = np.degrees(np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0]))
    angle %= 180
    horizontal = segments[(angle < 30) | (angle > 150)]
    vertical = segments[(angle > 60) & (angle < 120)]
    rows = fit_lines(horizontal, lambda s: (s[1] + s[3]) / 2)
    columns = fit_lines(vertical, lambda s: (s[0] + s[2]) / 2)
    if len(rows) != count or len(columns) != count:
        raise ValueError(f"Expected {count} x {count} tape lines, found {len(rows)} x {len(columns)}")
    rows = rows[::-1]  # bottom of the image first, so +Y points up the image
    points = np.zeros((count, count, 2))
    for i, (d_col, p_col) in enumerate(columns):
        for j, (d_row, p_row) in enumerate(rows):
            t = np.linalg.solve(np.column_stack([d_row, -d_col]), p_col - p_row)
            points[i, j] = p_row + t[0] * d_row
    return points


def pose_for_focal(focal: float, world: np.ndarray, pixels: np.ndarray, size) -> tuple:
    matrix = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]])
    ok, rotation, translation = cv2.solvePnP(world, pixels, matrix, None)
    if not ok:
        raise ValueError("Grid pose estimation failed")
    rotation_matrix = cv2.Rodrigues(rotation)[0]
    center = -rotation_matrix.T @ translation.ravel()
    return matrix, rotation, translation, center


def calibrate_from_grid(
    image_path: Path,
    camera_out: Path,
    arena_out: Path,
    height_m: float,
    spacing_m: float = 0.3,
    count: int = 4,
) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Cannot read grid image")
    size = image.shape[1::-1]
    pixels = grid_intersections(image, count).reshape(-1, 2)
    world = np.array(
        [[spacing_m * i, spacing_m * j, 0.0] for i in range(count) for j in range(count)]
    )

    def height_error(focal):
        return pose_for_focal(focal, world, pixels, size)[3][2] - height_m

    focal = brentq(height_error, 0.2 * size[0], 5 * size[0])
    matrix, rotation, translation, center = pose_for_focal(focal, world, pixels, size)
    projected = cv2.projectPoints(world, rotation, translation, matrix, None)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean((projected - pixels) ** 2)))
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rotation)[0]
    transform[:3, 3] = translation.ravel()
    tilt = float(np.degrees(np.arccos(abs(transform[2, 2]))))
    camera = {
        "camera_matrix": matrix.tolist(),
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "image_size": list(size),
        "method": "taped grid + measured lens height; principal point at centre, no distortion",
        "height_m": height_m,
    }
    write_json(camera_out, camera)
    write_json(
        arena_out,
        {
            "world_from_camera": np.linalg.inv(transform).tolist(),
            "reprojection_rms_px": rms,
            "layout": {
                "provenance": "taped_grid",
                "spacing_m": spacing_m,
                "count": count,
                "axes": "X along image right, Y along image up, Z up",
                "intersections_px": pixels.tolist(),
            },
        },
    )
    return {
        "focal_px": float(focal),
        "camera_center_m": center.tolist(),
        "tilt_deg": tilt,
        "reprojection_rms_px": rms,
    }


def load_points(path: Path) -> np.ndarray:
    return np.array(json.loads(path.read_text())["layout"]["intersections_px"])
