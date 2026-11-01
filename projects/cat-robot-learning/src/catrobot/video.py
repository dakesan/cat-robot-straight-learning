"""Build a 1..3 minute presentation from explicit clips and factual captions."""

import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .vision import video_frames


def make_video(manifest: Path, output: Path, font_path: Path | None = None):
    segments = json.loads(manifest.read_text())
    duration = sum(float(s["duration_s"]) for s in segments)
    if not 60 <= duration <= 180:
        raise ValueError("Submission video must last between 60 and 180 seconds")
    if output.exists():
        raise ValueError("Output already exists")
    if any(s.get("speed", 1) not in (1, 2) for s in segments):
        raise ValueError("Only 1x and explicitly labeled 2x playback are supported")
    if font_path is None:
        import matplotlib.font_manager

        font_path = Path(matplotlib.font_manager.findfont("DejaVu Sans"))
    font = ImageFont.truetype(str(font_path), 25)
    output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 960, 720, "yuv420p"
        frame_number = 0
        for segment in segments:
            count = round(float(segment["duration_s"]) * 30)
            speed = segment.get("speed", 1)
            caption = segment["caption"] + ("  [2x speed]" if speed == 2 else "")
            if segment["source"] == "simulated":
                caption = "SIMULATION ONLY\n" + caption
            elif segment["source"] != "measured" and segment["source"] != "explanation":
                raise ValueError("Every clip must declare measured/simulated/explanation source")
            image = None
            frames = None
            current = next_frame = None
            if "image" in segment:
                image = cv2.imread(str(manifest.parent / segment["image"]))
                if image is None:
                    raise ValueError("Cannot read presentation image")
            elif "video" in segment:
                frames = iter(video_frames(manifest.parent / segment["video"]))
                current = next(frames)
                next_frame = next(frames, None)
            for i in range(count):
                if frames is not None:
                    target = segment.get("start_s", 0) + i * speed / 30
                    while next_frame is not None and next_frame[0] <= target:
                        current, next_frame = next_frame, next(frames, None)
                    if next_frame is None and target - current[0] > 0.1:
                        raise ValueError("Requested clip extends beyond the recorded video")
                    image = current[1]
                canvas = np.full((720, 960, 3), 245, np.uint8)
                if image is not None:
                    content = image
                    if "crop" in segment:
                        x, y, width, height = segment["crop"]
                        content = content[y : y + height, x : x + width]
                    if content.size == 0:
                        raise ValueError("Empty privacy crop")
                    ratio = min(960 / content.shape[1], 560 / content.shape[0])
                    resized = cv2.resize(
                        content, (round(content.shape[1] * ratio), round(content.shape[0] * ratio))
                    )
                    x, y = (960 - resized.shape[1]) // 2, (560 - resized.shape[0]) // 2
                    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
                picture = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 565, 960, 720), fill=(20, 20, 20))
                draw.multiline_text((24, 580), caption, font=font, fill=(255, 255, 255), spacing=9)
                frame = av.VideoFrame.from_ndarray(np.asarray(picture), format="rgb24")
                frame.pts, frame.time_base = frame_number, Fraction(1, 30)
                frame_number += 1
                for packet in stream.encode(frame):
                    container.mux(packet)
            if frames is not None:
                frames.close()
        for packet in stream.encode():
            container.mux(packet)
