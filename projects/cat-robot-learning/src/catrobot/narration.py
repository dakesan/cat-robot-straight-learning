"""Build a narrated presentation: VOICEVOX speech per sentence, burned subtitles, clips and cards.

The manifest lists scenes. Each scene has one visual and one or more sentences; the scene lasts as
long as its speech. A clip that is shorter than its speech freezes on its last frame instead of
being slowed down, so no footage is shown at a speed it was not recorded at.
"""

import hashlib
import json
import subprocess
import wave
from fractions import Fraction
from pathlib import Path

import av
import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .vision import video_frames

WIDTH, HEIGHT, FPS = 1280, 720, 30
SUBTITLE_TOP = 590
GAP_S, SCENE_TAIL_S = 0.25, 0.45


def synthesize(text: str, speaker: int, output: Path, url: str, speed: float) -> float:
    """Write one sentence as WAV through a VOICEVOX engine and return its length in seconds."""
    if not output.exists():
        query = httpx.post(
            f"{url}/audio_query", params={"text": text, "speaker": speaker}, timeout=60
        )
        query.raise_for_status()
        body = query.json()
        body["speedScale"] = speed
        audio = httpx.post(f"{url}/synthesis", params={"speaker": speaker}, json=body, timeout=120)
        audio.raise_for_status()
        output.write_bytes(audio.content)
    with wave.open(str(output)) as handle:
        return handle.getnframes() / handle.getframerate()


def width(text: str) -> float:
    """Subtitle width in full-width characters; ASCII glyphs take about half."""
    return sum(0.5 if ch.isascii() else 1 for ch in text)


def wrap(text: str, limit: int = 40) -> list[str]:
    """Split a long subtitle at the comma nearest its middle (40 chars of 30 px fit 1280 px)."""
    if width(text) <= limit:
        return [text]
    cuts = [i + 1 for i, ch in enumerate(text[:-1]) if ch in "、。"]
    half = width(text) / 2
    cut = min(cuts, key=lambda i: abs(width(text[:i]) - half)) if cuts else len(text) // 2
    return [*wrap(text[:cut], limit), *wrap(text[cut:], limit)]


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    ratio = min(width / image.shape[1], height / image.shape[0])
    return cv2.resize(image, (round(image.shape[1] * ratio), round(image.shape[0] * ratio)))


def paste(canvas: np.ndarray, image: np.ndarray, box: tuple[int, int, int, int]):
    x, y, width, height = box
    resized = fit(image, width, height)
    top, left = y + (height - resized.shape[0]) // 2, x + (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized


class Clip:
    """Frames of one recorded video at its own speed, holding the last frame once it ends."""

    def __init__(self, path: Path, crop: list[int] | None):
        self.frames = iter(video_frames(path))
        self.crop = crop
        self.current = next(self.frames)
        self.upcoming = next(self.frames, None)

    def at(self, t: float) -> np.ndarray:
        while self.upcoming is not None and self.upcoming[0] <= t:
            self.current, self.upcoming = self.upcoming, next(self.frames, None)
        image = self.current[1]
        if self.crop:
            x, y, width, height = self.crop
            image = image[y : y + height, x : x + width]
        return image

    def close(self):
        self.frames.close()


class Painter:
    def __init__(self, font: Path, bold: Path):
        self.font = {size: ImageFont.truetype(str(font), size) for size in (26, 30, 34)}
        self.bold = {size: ImageFont.truetype(str(bold), size) for size in (24, 40, 52)}

    def card(self, title: str, lines: list[str]) -> np.ndarray:
        picture = Image.new("RGB", (WIDTH, HEIGHT), (246, 244, 238))
        draw = ImageDraw.Draw(picture)
        draw.rectangle((0, 0, 14, HEIGHT), fill=(30, 80, 170))
        draw.text((70, 70), title, font=self.bold[52], fill=(20, 20, 20))
        y = 170
        for line in lines:
            draw.text((80, y), line, font=self.font[34], fill=(30, 30, 30))
            y += 56
        return cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR)

    def finish(self, canvas: np.ndarray, label: str | None, subtitle: str) -> np.ndarray:
        picture = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(picture)
        if label:
            origin = (24, SUBTITLE_TOP - 46)
            box = draw.textbbox(origin, label, font=self.bold[24])
            draw.rectangle((box[0] - 10, box[1] - 7, box[2] + 10, box[3] + 7), fill=(20, 20, 20))
            draw.text(origin, label, font=self.bold[24], fill=(255, 255, 255))
        if subtitle:
            lines = wrap(subtitle)
            draw.rectangle((0, SUBTITLE_TOP, WIDTH, HEIGHT), fill=(16, 16, 16))
            y = SUBTITLE_TOP + (HEIGHT - SUBTITLE_TOP - 44 * len(lines)) // 2
            for line in lines:
                width = draw.textlength(line, font=self.font[30])
                draw.text(((WIDTH - width) / 2, y), line, font=self.font[30], fill=(255, 255, 255))
                y += 44
        return np.asarray(picture)


def scene_frame(scene: dict, clips: list[Clip], still: np.ndarray | None, t: float):
    canvas = np.full((HEIGHT, WIDTH, 3), 18, np.uint8)
    area = (0, 0, WIDTH, SUBTITLE_TOP)
    if still is not None:
        if scene["visual"] == "card":
            canvas[:SUBTITLE_TOP] = still[:SUBTITLE_TOP]
        else:
            paste(canvas, still, area)
    elif len(clips) == 1:
        paste(canvas, clips[0].at(t), area)
    elif len(clips) == 2:
        half = WIDTH // 2
        paste(canvas, clips[0].at(t), (0, 0, half - 4, SUBTITLE_TOP))
        paste(canvas, clips[1].at(t), (half + 4, 0, half - 4, SUBTITLE_TOP))
    return canvas


def make_narrated_video(
    manifest: Path, output: Path, work: Path, font: Path, bold: Path,
    url: str = "http://127.0.0.1:50021", speaker: int = 3, speed: float = 1.1,
) -> dict:  # fmt: skip
    spec = json.loads(manifest.read_text())
    root = manifest.parent / spec.get("root", ".")
    work.mkdir(parents=True, exist_ok=True)
    painter = Painter(font, bold)
    timeline, audio_parts, cursor = [], [], 0.0
    for n, scene in enumerate(spec["scenes"]):
        start = cursor
        subtitles = []
        for k, sentence in enumerate(scene["sentences"]):
            spoken = sentence.get("say", sentence["text"])
            key = hashlib.sha256(f"{spoken}|{speaker}|{speed}".encode()).hexdigest()[:12]
            wav = work / f"s{n:02d}-{k:02d}-{key}.wav"
            length = synthesize(spoken, speaker, wav, url, speed)
            subtitles.append((cursor - start, cursor - start + length, sentence["text"]))
            audio_parts.append((cursor, wav))
            cursor += length + GAP_S
        cursor += SCENE_TAIL_S
        timeline.append((scene, start, cursor, subtitles))
    if cursor > 180:
        raise ValueError(f"Narration lasts {cursor:.1f} s; the submission limit is 180 s")
    silent = work / "video.mp4"
    with av.open(str(silent), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"crf": "20"}
        index = 0
        for scene, start, end, subtitles in timeline:
            visual = scene["visual"]
            clips = [Clip(root / v, scene.get("crop")) for v in scene.get("videos", [])]
            still = None
            if visual == "card":
                still = painter.card(scene["title"], scene.get("lines", []))
            elif visual == "image":
                still = cv2.imread(str(root / scene["image"]))
                if still is None:
                    raise ValueError(f"Cannot read {scene['image']}")
                if "crop" in scene:
                    x, y, w, h = scene["crop"]
                    still = still[y : y + h, x : x + w]
            for i in range(round((end - start) * FPS)):
                t = i / FPS
                subtitle = next((s for a, b, s in subtitles if a <= t < b + GAP_S), "")
                canvas = scene_frame(scene, clips, still, t)
                frame = av.VideoFrame.from_ndarray(
                    painter.finish(canvas, scene.get("label"), subtitle), format="rgb24"
                )
                frame.pts, frame.time_base = index, Fraction(1, FPS)
                index += 1
                for packet in stream.encode(frame):
                    container.mux(packet)
            for clip in clips:
                clip.close()
        for packet in stream.encode():
            container.mux(packet)
    inputs, delays = [], []
    for i, (at, wav) in enumerate(audio_parts):
        inputs += ["-i", str(wav)]
        delays.append(f"[{i + 1}:a]adelay={round(at * 1000)}:all=1[a{i}]")
    mix = ";".join(delays) + ";" + "".join(f"[a{i}]" for i in range(len(audio_parts)))
    mix += f"amix=inputs={len(audio_parts)}:normalize=0[voice]"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(silent), *inputs, "-filter_complex", mix,
         "-map", "0:v", "-map", "[voice]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
         str(output)],
        check=True,
    )  # fmt: skip
    return {
        "duration_s": round(cursor, 2),
        "scenes": [
            {
                "title": s.get("title", s.get("label", "")),
                "start_s": round(a, 2),
                "end_s": round(b, 2),
            }
            for s, a, b, _ in timeline
        ],
    }
