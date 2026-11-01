import json
from pathlib import Path

import av
import numpy as np

from catrobot.config import load_config
from catrobot.rendering import inspect_simulation_video
from catrobot.simulation import run_episode


def test_encoded_simulation_shows_robot_and_motion_without_changing_physics(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "examples/cad-visual-config.json")
    for name, record in (("plain", False), ("filmed", True)):
        result = run_episode(
            config, tmp_path / name, mode="speed", record_video=record, max_seconds=1.3
        )
        assert result["forward_m"] > 0.05
    check = json.loads((tmp_path / "filmed/video-check.json").read_text())
    assert check["frames"] == 39
    assert check["robot_visible_throughout"] and check["min_robot_fraction"] > 0.02
    assert check["max_motion_fraction"] > 0.02
    assert (tmp_path / "filmed/poster.png").stat().st_size > 1000
    assert (tmp_path / "plain/poses.jsonl").read_bytes() == (
        tmp_path / "filmed/poses.jsonl"
    ).read_bytes()
    with av.open(str(tmp_path / "filmed/simulation.mp4")) as video:
        assert video.streams.video[0].codec_context.name == "h264"
        assert video.streams.video[0].codec_context.format.name == "yuv420p"


def test_white_encoded_video_is_not_accepted_as_a_visible_robot(tmp_path):
    path = tmp_path / "white.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for _ in range(15):
            frame = av.VideoFrame.from_ndarray(np.full((240, 320, 3), 255, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    check = inspect_simulation_video(path)
    assert check["frames"] == 15
    assert not check["robot_visible_throughout"]
    assert check["min_robot_fraction"] == 0
