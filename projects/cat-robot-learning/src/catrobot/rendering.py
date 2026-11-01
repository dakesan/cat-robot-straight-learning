"""Readable simulation capture and checks on the actual encoded video."""

from pathlib import Path

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def follow_camera(world: mujoco.MjModel) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    radius = float(np.max(world.geom_rbound[world.geom_bodyid > 0]))
    camera.distance = max(0.3, 4 * radius)
    camera.azimuth, camera.elevation = 135, -30
    return camera


def status_frame(rgb: np.ndarray, elapsed_s: float, distance_m: float, speed_mps: float) -> Image.Image:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    draw.rectangle((0, 0, image.width, 73), fill=(22, 28, 35))
    draw.text((18, 8), "SIMULATION  |  assumed physical parameters", font=font, fill="white")
    draw.text(
        (18, 39),
        f"Time {elapsed_s:5.2f} s     Forward {distance_m:5.2f} m     Target {speed_mps:.2f} m/s",
        font=font,
        fill="white",
    )
    return image


def inspect_simulation_video(path: Path, poster: Path | None = None) -> dict:
    """Check that the known blue chassis is visible after MP4 encoding."""
    fractions, motion = [], []
    previous = None
    frames = 0
    last_time = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width, height = stream.codec_context.width, stream.codec_context.height
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            body = rgb[80:].astype(np.int16)
            blue = (body[:, :, 2] - body[:, :, 0] > 35) & (
                body[:, :, 2] - body[:, :, 1] > 15
            )
            fractions.append(float(blue.mean()))
            if frames == 0 and poster is not None:
                frame.to_image().save(poster)
            if frames % 15 == 0:
                sample = body[::4, ::4]
                if previous is not None:
                    motion.append(float((np.abs(sample - previous).max(axis=2) > 15).mean()))
                previous = sample
            last_time = float(frame.time or 0)
            frames += 1
    return {
        "frames": frames,
        "width": width,
        "height": height,
        "duration_s": last_time + (1 / 30 if frames else 0),
        "min_robot_fraction": min(fractions, default=0),
        "median_robot_fraction": float(np.median(fractions)) if fractions else 0,
        "max_motion_fraction": max(motion, default=0),
        "robot_visible_throughout": bool(fractions) and min(fractions) >= 0.01,
    }
