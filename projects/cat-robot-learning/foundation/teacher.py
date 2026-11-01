"""Run a fine-tuned SmolVLA teacher on every aligned frame and write its correction.

The teacher sees the lane image and the privileged state; its first predicted action
(left/right target speeds) is turned back into a correction c = (vR - vL) / (vR + vL),
which becomes the distillation target of the encoder-only student.
"""

import csv
from pathlib import Path

import numpy as np
import torch
import typer
from convert import STATE, TASK, frames_at
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

app = typer.Typer(no_args_is_help=True)


def usable_rows(frames: Path) -> tuple[list[dict], list[dict]]:
    with (frames / "frames.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return rows, [r for r in rows if r["pose_valid"] == "True" and r["video_time_s"]]


@app.command()
def extract(frames: Path, video: Path, output: Path, crop: str | None = None):
    """Save only the letterboxed lane crops the teacher needs, so the raw video stays local."""
    if output.exists():
        raise typer.BadParameter("Output exists")
    _, usable = usable_rows(frames)
    box = tuple(int(v) for v in crop.split(",")) if crop else None
    images = frames_at(video, {float(r["video_time_s"]) for r in usable}, box)
    times = sorted(images)
    np.savez_compressed(output, times=np.array(times), images=np.stack([images[t] for t in times]))
    typer.echo(f"{len(times)} lane images")


@app.command()
def infer(
    checkpoint: Path,
    frames: Path,
    source: Path,
    output: Path,
    crop: str | None = None,
    device: str = "mps",
):
    """checkpoint: .../checkpoints/last/pretrained_model; frames: build-dataset directory;
    source: the session video, or an .npz written by `extract`."""
    if output.exists():
        raise typer.BadParameter("Output exists")
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(config, pretrained_path=str(checkpoint))
    rows, usable = usable_rows(frames)
    if source.suffix == ".npz":
        stored = np.load(source)
        images = {round(float(t), 4): image for t, image in zip(stored["times"], stored["images"])}
    else:
        box = tuple(int(v) for v in crop.split(",")) if crop else None
        images = frames_at(source, {float(r["video_time_s"]) for r in usable}, box)
    for row in rows:
        row["teacher_correction"] = ""
    with torch.no_grad():
        for row in usable:
            image = images.get(round(float(row["video_time_s"]), 4))
            if image is None:
                continue
            observation = {
                "observation.images.lane": torch.from_numpy(image).permute(2, 0, 1).float() / 255,
                "observation.state": torch.tensor([float(row[k]) for k in STATE], dtype=torch.float32),
                "task": TASK,
            }
            policy.reset()
            action = postprocess(policy.select_action(preprocess(observation)))
            left, right = (float(v) for v in np.asarray(action.cpu()).reshape(-1)[:2])
            if left + right > 1e-6:
                row["teacher_correction"] = str((right - left) / (right + left))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    typer.echo(f"teacher corrections for {sum(r['teacher_correction'] != '' for r in rows)} frames")


if __name__ == "__main__":
    app()
