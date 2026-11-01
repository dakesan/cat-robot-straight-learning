"""Cut evaluation trials from the session video and draw the tracked path on them.

Presentation only: the path comes from the same camera track used for the metrics, and
the dashed line is the straight course from the start pose, with the +/-5 cm band up to
the goal. Usage:

    uv run python scripts/overlay_clips.py VIDEO TRACK RUNS SESSION CAMERA ARENA CONFIG OUT TRIAL...
"""

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from catrobot.config import load_config

COLORS = {"pwm": (219, 111, 42), "speed": (58, 100, 232), "learned": (122, 169, 31)}  # BGR
LABELS = {"pwm": "Equal PWM", "speed": "Wheel-speed PI", "learned": "PI + learned correction"}


def project(points_xy, height, camera, camera_from_world):
    world = np.column_stack([points_xy, np.full(len(points_xy), height)])
    local = (camera_from_world[:3, :3] @ world.T).T + camera_from_world[:3, 3]
    pixels, _ = cv2.projectPoints(
        local, np.zeros(3), np.zeros(3), np.array(camera["camera_matrix"]), np.array(camera["distortion"])
    )
    return pixels.reshape(-1, 2)


def circular_median(angles):
    return math.atan2(float(np.median(np.sin(angles))), float(np.median(np.cos(angles))))


def main(video, track, runs, session, camera_path, arena_path, config_path, out, *trials):
    camera = json.loads(Path(camera_path).read_text())
    camera_from_world = np.linalg.inv(np.array(json.loads(Path(arena_path).read_text())["world_from_camera"]))
    config = load_config(Path(config_path))
    height = config.marker_height_m
    poses = [json.loads(line) for line in Path(track).read_text().splitlines()]
    poses = [p for p in poses if p["valid"]]
    offsets = {a["trial_id"]: a for a in json.loads(Path(runs, f"motion-sync-{session}.json").read_text())["attempts"]}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for trial in trials:
        run = Path(runs, offsets[trial]["directory"])
        metrics = json.loads((run / "metrics.json").read_text())
        samples = [json.loads(line) for line in (run / "telemetry.jsonl").read_text().splitlines()]
        armed = [s["time_ms"] / 1000 for s in samples if s["armed"]]
        start, end = armed[0] - offsets[trial]["offset_s"], armed[-1] - offsets[trial]["offset_s"]
        rest = np.array([[p["x_m"], p["y_m"]] for p in poses if start - 1.0 <= p["time_s"] <= start])
        early = [p["yaw_rad"] for p in poses if start <= p["time_s"] <= start + 0.5]
        origin, heading = np.median(rest, axis=0), circular_median(np.array(early))
        forward = np.array([math.cos(heading), math.sin(heading)])
        left = np.array([-forward[1], forward[0]])
        goal = metrics["goal_m"]
        course = project(np.array([origin, origin + forward * metrics["stop_m"]]), height, camera, camera_from_world)
        band = [
            project(np.array([origin + left * s * 0.05, origin + forward * goal + left * s * 0.05]), height, camera,
                    camera_from_world)
            for s in (-1, 1)
        ]
        path = [p for p in poses if start - 0.2 <= p["time_s"] <= end + 1.0]
        pixels = project(np.array([[p["x_m"], p["y_m"]] for p in path]), height, camera, camera_from_world)
        color = COLORS[metrics["mode"]]
        clip_start, clip_end = start - 1.0, end + 1.5
        target = out / f"{trial}-{metrics['mode']}.mp4"
        with av.open(str(video)) as source, av.open(str(target), "w") as sink:
            stream = source.streams.video[0]
            stream.thread_type = "AUTO"
            first = float(stream.start_time * stream.time_base)
            source.seek(int((clip_start - 2.0 + first) / stream.time_base), stream=stream)
            writer = sink.add_stream("libx264", rate=30)
            writer.width, writer.height, writer.pix_fmt = stream.width, stream.height, "yuv420p"
            written, next_time = 0, clip_start
            for frame in source.decode(stream):
                time = float(frame.pts * frame.time_base) - first
                if time < next_time:
                    continue
                if time > clip_end:
                    break
                next_time += 1 / 30
                image = frame.to_ndarray(format="bgr24")
                overlay = image.copy()
                for edge in band:
                    cv2.line(overlay, tuple(map(int, edge[0])), tuple(map(int, edge[1])), (255, 255, 255), 2, cv2.LINE_AA)
                a, b = course
                for k in range(0, 30, 2):
                    p0, p1 = a + (b - a) * k / 30, a + (b - a) * (k + 1) / 30
                    cv2.line(overlay, tuple(map(int, p0)), tuple(map(int, p1)), (255, 255, 255), 3, cv2.LINE_AA)
                image = cv2.addWeighted(overlay, 0.8, image, 0.2, 0)
                seen = pixels[[p["time_s"] <= time for p in path]]
                if len(seen) > 1:
                    cv2.polylines(image, [seen.astype(np.int32)], False, color, 6, cv2.LINE_AA)
                label = f"{LABELS[metrics['mode']]}  target {metrics['speed_mps']:.2f} m/s"
                if metrics["mode"] == "pwm":
                    label = f"{LABELS['pwm']}  {metrics['pwm']}%"
                cv2.rectangle(image, (420, 70), (420 + 26 * len(label) // 2 + 40, 130), (20, 20, 20), -1)
                cv2.putText(image, label, (440, 112), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
                result = av.VideoFrame.from_ndarray(image, format="bgr24")
                result.pts, result.time_base = written, Fraction(1, 30)
                written += 1
                for packet in writer.encode(result):
                    sink.mux(packet)
            for packet in writer.encode():
                sink.mux(packet)
        print(f"{target} {written / 30:.1f} s lateral={metrics['lateral_at_goal_m'] * 100:+.1f} cm "
              f"heading={math.degrees(metrics['heading_at_goal_rad']):+.1f} deg")


if __name__ == "__main__":
    main(*sys.argv[1:])
