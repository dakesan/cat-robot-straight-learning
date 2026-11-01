"""Convert aligned cat-robot frames and the session video into a LeRobotDataset.

The teacher sees one real video frame per step (nearest frame, never interpolated) plus
the student state and privileged video-derived pose. Actions are left/right target
speeds rebuilt from hindsight straightening labels. Only train-split trials are written.
"""

import csv
import json
import random
from pathlib import Path

import av
import numpy as np
import typer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

STUDENT = ("speed_mps", "elapsed_s", "measured_left_mps", "measured_right_mps", "encoder_heading_rad")
PRIVILEGED = ("correction_cmd", "x_m", "y_m", "yaw_rad", "yaw_rate_rps")
STATE = STUDENT + PRIVILEGED
TASK = "drive straight for one meter"
IMAGE_SIZE = 256

app = typer.Typer(no_args_is_help=True)


def letterbox(image: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is not None:
        x, y, w, h = crop
        image = image[y : y + h, x : x + w]
    picture = Image.fromarray(image)
    picture.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    canvas.paste(picture, ((IMAGE_SIZE - picture.width) // 2, (IMAGE_SIZE - picture.height) // 2))
    return np.asarray(canvas)


def frames_at(video: Path, times: set[float], crop) -> dict[float, np.ndarray]:
    """Decode once and keep the frames whose presentation time (from the first frame) is needed."""
    wanted = {round(t, 4) for t in times}
    found = {}
    last = max(wanted, default=0.0)
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        first = None
        for frame in container.decode(stream):
            time = float(frame.pts * frame.time_base)
            first = time if first is None else first
            key = round(time - first, 4)
            if key in wanted:
                found[key] = letterbox(frame.to_ndarray(format="rgb24"), crop)
            if key > last:
                break
    return found


def longest_valid_run(rows: list[dict]) -> list[dict]:
    best, current = [], []
    for row in rows:
        if row["label_valid"] == "True" and row["video_time_s"]:
            current.append(row)
            if len(current) > len(best):
                best = current
        else:
            current = []
    return best


@app.command()
def convert(
    frames: Path,
    output: Path,
    video: Path,
    sync: Path,
    crop: str | None = None,
    holdout: float = 0.2,
    seed: int = 42,
):
    """frames: dataset dir from `catrobot build-dataset`; sync: runs/sync-<session>.json."""
    if output.exists():
        raise typer.BadParameter("Output exists; LeRobot datasets are rebuilt into a new directory")
    meta = json.loads((frames / "meta.json").read_text())
    with (frames / "frames.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "train"]
    accepted = {a["trial_id"] for a in json.loads(sync.read_text())["attempts"] if a["accepted"]}
    trials = sorted({r["trial_id"] for r in rows} & accepted)
    random.Random(seed).shuffle(trials)
    validation = sorted(trials[: round(holdout * len(trials))])
    training = sorted(trials[len(validation) :])
    rate = round(meta["rate_hz"])
    box = tuple(int(v) for v in crop.split(",")) if crop else None
    episodes = {t: longest_valid_run([r for r in rows if r["trial_id"] == t]) for t in training}
    images = frames_at(video, {float(r["video_time_s"]) for e in episodes.values() for r in e}, box)
    features = {
        "observation.images.lane": {
            "dtype": "image",
            "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (len(STATE),), "names": list(STATE)},
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["target_left_mps", "target_right_mps"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id="local/catrobot-teacher", fps=rate, features=features, root=output,
        robot_type="cat_robot_two_wheel", use_videos=False,
    )  # fmt: skip
    written = {}
    for trial, episode in episodes.items():
        if len(episode) < 10:
            continue
        for row in episode:
            speed, label = float(row["speed_mps"]), float(row["label_correction"])
            dataset.add_frame(
                {
                    "observation.images.lane": images[round(float(row["video_time_s"]), 4)],
                    "observation.state": np.array([float(row[k]) for k in STATE], np.float32),
                    "action": np.array([speed * (1 - label), speed * (1 + label)], np.float32),
                    "task": TASK,
                }
            )
        dataset.save_episode()
        written[trial] = len(episode)
    dataset.finalize()
    (output / "catrobot-split.json").write_text(
        json.dumps(
            {"train_episodes": written, "validation_trials": validation, "seed": seed,
             "state": list(STATE), "privileged": list(PRIVILEGED), "crop": box}, indent=2
        )
    )
    typer.echo(f"episodes {len(written)}, frames {sum(written.values())}, validation {validation}")


if __name__ == "__main__":
    app()
