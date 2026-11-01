"""Synthetic overhead camera fixtures for marker, LED and floor-marker tests."""

import json

import av
import cv2
import numpy as np

from catrobot.vision import marker_points


def overhead_camera(focal=900.0, size=(960, 720), height=1.2, center=(0.0, 0.0)):
    camera = {
        "camera_matrix": [[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]],
        "distortion": [0, 0, 0, 0, 0],
        "image_size": list(size),
    }
    rotation = cv2.Rodrigues(np.array([np.pi - 0.08, 0, 0]))[0]
    camera_from_world = np.eye(4)
    camera_from_world[:3, :3] = rotation
    camera_from_world[:3, 3] = -rotation @ np.array([center[0], center[1], height])
    return camera, camera_from_world


def project(points, camera, camera_from_world):
    rotation = camera_from_world[:3, :3]
    return cv2.projectPoints(
        np.asarray(points, float),
        cv2.Rodrigues(rotation)[0],
        camera_from_world[:3, 3],
        np.array(camera["camera_matrix"], float),
        np.zeros(5),
    )[0].reshape(-1, 2)


def marker_image(identity, pixels=300):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.generateImageMarker(dictionary, identity, pixels)


def draw_marker(frame, identity, world_corners, camera, camera_from_world):
    image = marker_image(identity)
    corners = project(world_corners, camera, camera_from_world)
    size = image.shape[0] - 1
    warp = cv2.getPerspectiveTransform(
        np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float32),
        corners.astype(np.float32),
    )
    mask = cv2.warpPerspective(np.full_like(image, 255), warp, frame.shape[1::-1])
    warped = cv2.warpPerspective(image, warp, frame.shape[1::-1])
    frame[mask > 0] = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)[mask > 0]
    return warp


def robot_corners(x, y, yaw, height, size=0.05):
    local = marker_points(size)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy = local[:, :2] @ rotation.T + [x, y]
    return np.column_stack([xy, np.full(4, height)])


def render(path, frames, camera, camera_from_world, config, fps=30, floor_markers=()):
    """frames: iterable of (x, y, yaw, led_on) or None when the robot is out of view."""
    width, height = camera["image_size"]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for state in frames:
            frame = np.full((height, width, 3), 255, np.uint8)
            for marker in floor_markers:
                draw_marker(frame, marker["id"], marker["corners_m"], camera, camera_from_world)
            if state is not None:
                x, y, yaw, led_on = state
                corners = robot_corners(x, y, yaw, config.marker_height_m, config.marker_size_m)
                warp = draw_marker(frame, config.marker_id, corners, camera, camera_from_world)
                led = cv2.perspectiveTransform(
                    np.array([[[config.led_offset[0] * 299, config.led_offset[1] * 299]]], np.float32),
                    warp,
                )[0, 0]
                color = (0, 0, 255) if led_on else (0, 0, 0)
                cv2.circle(frame, tuple(np.rint(led).astype(int)), 3, color, -1)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_calibration(tmp_path, camera, camera_from_world):
    camera_path, arena_path = tmp_path / "camera.json", tmp_path / "arena.json"
    camera_path.write_text(json.dumps(camera))
    arena_path.write_text(json.dumps({"world_from_camera": np.linalg.inv(camera_from_world).tolist()}))
    return camera_path, arena_path
