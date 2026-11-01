"""Calibrated marker tracking using actual video presentation timestamps."""

import itertools
import json
import math
from pathlib import Path

import av
import cv2
import numpy as np

from .config import RobotConfig, write_json


def video_frames(path: Path):
    with av.open(str(path)) as container:
        first = None
        for frame in container.decode(video=0):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Video frame has no presentation timestamp")
            time = float(frame.pts * frame.time_base)
            if first is None:
                first = time
            yield time - first, frame.to_ndarray(format="bgr24")


def detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # Tuned on the 2026-11-01 overhead video of a hand-drawn 60 mm marker with a thin white
    # margin on a dark chassis: smaller and more numerous threshold windows and a low
    # threshold constant raised resting-frame detection from 3 % to 87 %.
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.adaptiveThreshConstant = 3
    parameters.minMarkerDistanceRate = 0.01
    parameters.polygonalApproxAccuracyRate = 0.05
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def marker_points(size: float) -> np.ndarray:
    half = size / 2
    return np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float64
    )


def markers(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for identity in range(5):
        image = cv2.aruco.generateImageMarker(dictionary, identity, 600)
        image = cv2.copyMakeBorder(image, 150, 150, 150, 150, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(str(output / f"marker-{identity}.png"), image)
        ok, png = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Cannot encode marker")
        import base64

        encoded = base64.b64encode(png).decode()
        (output / f"marker-{identity}.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            'width="90mm" height="100mm" viewBox="0 0 90 100">'
            f'<image x="7.5" y="7.5" width="75" height="75" xlink:href="data:image/png;base64,{encoded}"/>'
            f'<text x="5" y="92" font-size="3">ID {identity}; black square = 50 mm; print at 100%</text></svg>'
        )
    squares = []
    for y in range(7):
        for x in range(10):
            if (x + y) % 2 == 0:
                squares.append(
                    f'<rect x="{23.5 + x * 25}" y="{12.5 + y * 25}" width="25" height="25"/>'
                )
    (output / "checkerboard.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="297mm" height="210mm" viewBox="0 0 297 210">'
        '<rect width="297" height="210" fill="white"/>'
        + "".join(squares)
        + '<text x="24" y="202" font-size="3.5">9 x 6 inner corners; squares = 25 mm; print at 100%</text></svg>'
    )


def calibrate_camera(
    video: Path, output: Path, columns: int = 9, rows: int = 6, square_m: float = 0.025
):
    if columns < 3 or rows < 3 or square_m <= 0:
        raise ValueError("Invalid checkerboard dimensions")
    object_points = np.zeros((columns * rows, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_m
    world, pixels, size = [], [], None
    last = -1.0
    for time, frame in video_frames(video):
        if time - last < 0.5:
            continue
        last = time
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, (columns, rows))
        if found:
            size = gray.shape[::-1]
            world.append(object_points.copy())
            pixels.append(corners)
        if len(world) >= 40:
            break
    if len(world) < 8:
        raise ValueError("Need at least eight detected checkerboard views with varied orientations")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(world, pixels, size, None, None)
    write_json(
        output,
        {
            "camera_matrix": matrix.tolist(),
            "distortion": distortion.ravel().tolist(),
            "image_size": list(size),
            "reprojection_rms_px": float(rms),
            "views": len(world),
        },
    )


def calibrate_arena(image: Path, camera: Path, layout: Path, output: Path):
    calibration = json.loads(camera.read_text())
    arena = json.loads(layout.read_text())
    frame = cv2.imread(str(image))
    if frame is None:
        raise ValueError("Cannot read arena image")
    if frame.shape[1::-1] != tuple(calibration["image_size"]):
        raise ValueError("Arena image and camera calibration resolutions differ")
    corners, ids, _ = detector().detectMarkers(frame)
    detected = (
        {} if ids is None else {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}
    )
    objects, pixels = [], []
    for marker in arena["markers"]:
        if marker["id"] in detected:
            objects.extend(marker["corners_m"])
            pixels.extend(detected[marker["id"]])
    if len(objects) < 12:
        raise ValueError("At least three measured floor markers must be visible")
    matrix = np.array(calibration["camera_matrix"], float)
    distortion = np.array(calibration["distortion"], float)
    objects, pixels = np.array(objects, float), np.array(pixels, float)
    ok, rotation, translation = cv2.solvePnP(objects, pixels, matrix, distortion)
    if not ok:
        raise ValueError("Arena pose estimation failed")
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rotation)[0]
    transform[:3, 3] = translation.ravel()
    projected = cv2.projectPoints(objects, rotation, translation, matrix, distortion)[0].reshape(
        -1, 2
    )
    write_json(
        output,
        {
            "world_from_camera": np.linalg.inv(transform).tolist(),
            "reprojection_rms_px": float(np.sqrt(np.mean((projected - pixels) ** 2))),
            "layout": arena,
        },
    )


def refine_corners(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Sub-pixel corners with a window scaled to the marker; the tuned thresholds can leave
    the detected quad a few pixels inside the black border."""
    side = float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))
    window = max(4, round(side / 8))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    refined = cv2.cornerSubPix(
        gray, corners.astype(np.float32).reshape(-1, 1, 2), (window, window), (-1, -1), criteria
    )
    return refined.reshape(4, 2).astype(np.float64)


def plane_points(
    pixels: np.ndarray, camera: dict, world_from_camera: np.ndarray, height: float
) -> np.ndarray:
    """Intersect each pixel ray with the horizontal plane z = height (floor frame)."""
    matrix = np.array(camera["camera_matrix"], float)
    distortion = np.array(camera["distortion"], float)
    normalized = cv2.undistortPoints(
        np.asarray(pixels, np.float64).reshape(-1, 1, 2), matrix, distortion
    ).reshape(-1, 2)
    rays = (world_from_camera[:3, :3] @ np.column_stack([normalized, np.ones(len(normalized))]).T).T
    origin = world_from_camera[:3, 3]
    if np.any(np.abs(rays[:, 2]) < 1e-6):
        raise ValueError("Camera ray is parallel to the marker plane")
    scale = (height - origin[2]) / rays[:, 2]
    return origin + scale[:, None] * rays


def marker_pose(
    corners: np.ndarray, camera: dict, world_from_camera: np.ndarray, config: RobotConfig
) -> dict:
    """Pose of a horizontal marker at the measured height, from its four corner rays.

    The marker is level at a known height, so each corner ray is intersected with that
    plane instead of solving a perspective pose. Small squares make the perspective
    solution ambiguous in tilt, which shows up as several degrees of heading noise.
    """
    world = plane_points(corners, camera, world_from_camera, config.marker_height_m)[:, :2]
    # ArUco corner order: top-left, top-right, bottom-right, bottom-left. The marker's
    # +x points from its left edge to its right edge.
    right = (world[1] + world[2]) / 2 - (world[0] + world[3]) / 2
    up = (world[0] + world[1]) / 2 - (world[3] + world[2]) / 2
    if right[0] * up[1] - right[1] * up[0] <= 0:
        raise ValueError("Marker appears mirrored; it is not facing up")
    yaw = math.atan2(right[1], right[0]) - math.radians(config.marker_yaw_deg)
    size = config.marker_size_m
    sides = np.array([np.linalg.norm(world[i] - world[(i + 1) % 4]) for i in range(4)])
    diagonals = np.array([np.linalg.norm(world[0] - world[2]), np.linalg.norm(world[1] - world[3])])
    mean_side = float(np.mean(sides))
    # Scale-free squareness, so a hand-drawn square of slightly different size still counts.
    squareness = float(
        np.sqrt(np.mean(np.concatenate([sides / mean_side - 1, diagonals / (mean_side * math.sqrt(2)) - 1]) ** 2))
    )
    center = world.mean(axis=0)
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    position = center - rotation @ [config.marker_offset_x_m, config.marker_offset_y_m]
    return {
        "x_m": float(position[0]),
        "y_m": float(position[1]),
        "yaw_rad": yaw,
        "squareness": squareness,
        "size_ratio": mean_side / size,
    }


def led_signal(frame: np.ndarray, corners: np.ndarray, offset: tuple[float, float]) -> float:
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    transform = cv2.getPerspectiveTransform(unit, corners.astype(np.float32))
    point = cv2.perspectiveTransform(np.array([[offset]], np.float32), transform)[0, 0]
    radius = max(2, round(np.linalg.norm(corners[1] - corners[0]) * 0.08))
    x, y = np.rint(point).astype(int)
    area = frame[
        max(0, y - radius) : min(frame.shape[0], y + radius + 1),
        max(0, x - radius) : min(frame.shape[1], x + radius + 1),
    ].astype(float)
    if area.size == 0:
        return 0.0
    excess = area[:, :, 2] - np.maximum(area[:, :, 0], area[:, :, 1])
    return float(np.percentile(excess, 90))


def track_video(video: Path, camera: Path, arena: Path, config: RobotConfig, output: Path) -> dict:
    calibration = json.loads(camera.read_text())
    world = np.array(json.loads(arena.read_text())["world_from_camera"], float)
    finder = detector()
    count = valid = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for time, frame in video_frames(video):
            if frame.shape[1::-1] != tuple(calibration["image_size"]):
                raise ValueError(
                    "Video and camera calibration resolutions differ; use the same lens and mode"
                )
            row = {"time_s": time, "valid": False, "led_signal": 0.0}
            corners, identities, _ = finder.detectMarkers(frame)
            if identities is not None and config.marker_id in identities:
                index = int(np.flatnonzero(identities.ravel() == config.marker_id)[0])
                points = refine_corners(frame, corners[index].reshape(4, 2))
                try:
                    pose = marker_pose(points, calibration, world, config)
                    # Reject quads that are not square on the marker plane or whose size is
                    # far from the measured marker: misdetections or badly seen frames. A
                    # hand-drawn marker measured 4-5 % squareness and 0.91-0.93 size ratio.
                    row.update(
                        pose,
                        valid=pose["squareness"] <= 0.10 and abs(pose["size_ratio"] - 1) <= 0.25,
                        led_signal=led_signal(frame, points, config.led_offset),
                    )
                except ValueError:
                    pass
            handle.write(json.dumps(row) + "\n")
            count += 1
            valid += bool(row["valid"])
    return {"frames": count, "detected": valid, "coverage": valid / count if count else 0}


# Floor marker IDs placed at the corners of a taped rectangle, viewed from above:
# 1 = start-left, 2 = far-left, 3 = far-right, 4 = start-right. Each marker's top edge
# faces +Y (left) and its right edge faces +X (the driving direction).
RECTANGLE_EDGES = {"d12": (0, 1), "d23": (1, 2), "d34": (2, 3), "d41": (3, 0)}
RECTANGLE_DIAGONALS = {"d13": (0, 2), "d24": (1, 3)}


def rectangle_layout(distances: dict[str, float], marker_size: float = 0.05) -> dict:
    """Solve marker centers from taped side and diagonal lengths (metres)."""
    from scipy.optimize import least_squares

    pairs = {**RECTANGLE_EDGES, **RECTANGLE_DIAGONALS}
    if set(distances) != set(pairs) or min(distances.values()) <= 0:
        raise ValueError(f"Provide positive distances {sorted(pairs)}")

    def centers(x):
        # Gauge: marker 4 at the origin and marker 1 on the +Y axis.
        return np.array([[0, x[0]], [x[1], x[2]], [x[3], x[4]], [0, 0]], float)

    def residual(x):
        points = centers(x)
        return [np.linalg.norm(points[a] - points[b]) - distances[k] for k, (a, b) in pairs.items()]

    guess = [distances["d41"], distances["d12"], distances["d41"], distances["d34"], 0]
    fitted = least_squares(residual, guess)
    points = centers(fitted.x)
    origin = (points[0] + points[3]) / 2
    forward = (points[1] + points[2]) / 2 - origin
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    local = (points - origin) @ np.column_stack([forward, left])
    half = marker_size / 2
    offsets = [(-half, half), (half, half), (half, -half), (-half, -half)]
    markers = [
        {
            "id": identity,
            "corners_m": [[float(x + dx), float(y + dy), 0.0] for dx, dy in offsets],
        }
        for identity, (x, y) in zip((1, 2, 3, 4), local)
    ]
    worst = float(np.max(np.abs(fitted.fun)))
    return {
        "provenance": "measured_taped_rectangle",
        "distances_m": distances,
        "max_distance_residual_m": worst,
        "accepted": worst <= 0.005,
        "markers": markers,
    }


def stationary_segments(rows: list[dict], speed_mps: float = 0.02, min_s: float = 1.0) -> list:
    """Group valid poses where the marker rests, e.g. while placed on taped spots."""
    valid = [r for r in rows if r.get("valid")]
    segments, current = [], []
    for previous, row in itertools.pairwise(valid):
        dt = row["time_s"] - previous["time_s"]
        step = math.hypot(row["x_m"] - previous["x_m"], row["y_m"] - previous["y_m"])
        if 0 < dt <= 0.2 and step / dt <= speed_mps:
            current.append(row)
        else:
            if current:
                segments.append(current)
            current = []
    if current:
        segments.append(current)
    return [s for s in segments if s[-1]["time_s"] - s[0]["time_s"] >= min_s]


def distance_check(rows: list[dict], expected_m: float = 1.0, tolerance_m: float = 0.01) -> dict:
    """Distance between the two longest resting positions of the robot marker."""
    segments = sorted(stationary_segments(rows), key=len, reverse=True)[:2]
    if len(segments) < 2:
        raise ValueError("Need two resting periods of at least one second each")
    segments.sort(key=lambda s: s[0]["time_s"])
    centers, spread = [], []
    for segment in segments:
        xy = np.array([[r["x_m"], r["y_m"]] for r in segment])
        centers.append(np.median(xy, axis=0))
        spread.append(float(np.max(np.std(xy, axis=0))))
    measured = float(np.linalg.norm(centers[1] - centers[0]))
    return {
        "expected_m": expected_m,
        "measured_m": measured,
        "error_m": measured - expected_m,
        "resting_times_s": [[s[0]["time_s"], s[-1]["time_s"]] for s in segments],
        "position_std_m": spread,
        "accepted": abs(measured - expected_m) <= tolerance_m,
    }


def track_segment(args) -> list[dict]:
    """Track one time window (seconds from the first frame) of a video in its own process."""
    video, start, end, first, calibration, world, config_json = args
    import av as _av

    from .config import RobotConfig as _Config

    config = _Config.model_validate_json(config_json)
    finder = detector()
    rows = []
    with _av.open(str(video)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(0.0, first + start - 1.0) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            time = float(frame.pts * frame.time_base) - first
            if time < start:
                continue
            if time > end:
                break
            image = frame.to_ndarray(format="bgr24")
            row = {"time_s": time, "valid": False, "led_signal": 0.0}
            corners, identities, _ = finder.detectMarkers(image)
            if identities is not None and config.marker_id in identities:
                index = int(np.flatnonzero(identities.ravel() == config.marker_id)[0])
                points = refine_corners(image, corners[index].reshape(4, 2))
                try:
                    pose = marker_pose(points, calibration, world, config)
                    row.update(
                        pose,
                        valid=pose["squareness"] <= 0.10 and abs(pose["size_ratio"] - 1) <= 0.25,
                        led_signal=led_signal(image, points, config.led_offset),
                    )
                except ValueError:
                    pass
            rows.append(row)
    return rows


def track_windows(
    video: Path,
    camera: Path,
    arena: Path,
    config: RobotConfig,
    output: Path,
    windows: list[tuple[float, float]],
    workers: int = 6,
) -> dict:
    """Track only the given time windows, in parallel; frames outside them are skipped."""
    from concurrent.futures import ProcessPoolExecutor

    calibration = json.loads(camera.read_text())
    world = np.array(json.loads(arena.read_text())["world_from_camera"], float)
    with av.open(str(video)) as container:
        first_frame = next(container.decode(video=0))
        first = float(first_frame.pts * first_frame.time_base)
    merged = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    jobs = [
        (video, s, e, first, calibration, world, config.model_dump_json()) for s, e in merged
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        segments = list(pool.map(track_segment, jobs))
    rows = sorted((r for segment in segments for r in segment), key=lambda r: r["time_s"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    valid = sum(r["valid"] for r in rows)
    return {"windows": len(merged), "frames": len(rows), "detected": valid}
