import pytest

from catrobot.acquisition import collect
from catrobot.config import RobotConfig


def test_unmeasured_geometry_cannot_send_a_real_motor_command(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="measured configuration"):
        collect(
            "http://example.invalid",
            RobotConfig(),
            tmp_path / "run",
            mode="speed",
            speed=0.1,
            client_factory=lambda url: calls.append(url),
        )
    assert calls == []


def test_new_real_trial_does_not_take_over_an_armed_robot(tmp_path):
    class Fake:
        def __init__(self, url):
            self.posts = []

        def get(self, path):
            return {"armed": True}

        def post(self, path, **fields):
            self.posts.append(path)

        def close(self):
            pass

    config = RobotConfig(provenance="measured")
    with pytest.raises(ValueError, match="Stop the robot"):
        collect(
            "http://example.invalid",
            config,
            tmp_path / "run",
            mode="speed",
            speed=0.1,
            client_factory=Fake,
        )


def test_effective_motor_response_is_identified_from_position_not_torque():
    import numpy as np

    from catrobot.acquisition import fit_response

    wheel = RobotConfig().left
    traces = []
    for percent in (20, 30, 40):
        rows, position, velocity = [], 0.0, 0.0
        for i in range(200):
            rows.append(
                {
                    "time_ms": i * 20,
                    "left": percent,
                    "encoder_left": round(position / wheel.meters_per_count),
                }
            )
            target = (((percent * 255 + 50) // 100) * 100 / 255 - 8) * 0.005
            updated = target + (velocity - target) * np.exp(-0.02 / 0.12)
            position += (velocity + updated) * 0.02 / 2
            velocity = updated
        traces.append(rows)
    gain, deadband, constant, rms = fit_response(traces, wheel, "left", 0.004, 7)
    assert gain == pytest.approx(0.005, rel=0.01)
    assert deadband == pytest.approx(8, abs=0.1)
    assert constant == pytest.approx(0.12, abs=0.01)
    assert rms < 0.0001


def test_telemetry_error_after_arming_always_requests_stop(tmp_path, monkeypatch):
    import json

    import httpx

    from catrobot import acquisition

    class Fake:
        def __init__(self):
            self.armed = False
            self.posts = []

        def get(self, path, **params):
            if path == "/api/status":
                return {
                    "armed": False,
                    "encoder_left": 0,
                    "encoder_right": 0,
                    "firmware": "fixture",
                    "boot_id": 1,
                }
            if self.armed:
                raise httpx.ReadTimeout("fixture telemetry failure")
            return {"latest": 0, "boot_id": 1, "dropped": 0, "samples": []}

        def post(self, path, **fields):
            self.posts.append(path)
            if path == "/api/arm":
                self.armed = True
            if path == "/api/stop":
                self.armed = False
            return {"epoch": 1}

        def calibrate(self, config):
            pass

        def close(self):
            pass

    clock = [0.0]

    def advance():
        clock[0] += 0.1
        return clock[0]

    monkeypatch.setattr(acquisition.time, "monotonic", advance)
    monkeypatch.setattr(acquisition.time, "sleep", lambda seconds: None)
    fake = Fake()
    with pytest.raises(TimeoutError):
        collect(
            "http://example.invalid",
            RobotConfig(provenance="measured"),
            tmp_path / "run",
            mode="speed",
            speed=0.1,
            client_factory=lambda url: fake,
        )
    assert "/api/arm" in fake.posts and "/api/drive" in fake.posts
    assert "/api/stop" in fake.posts and not fake.armed
    assert (
        json.loads((tmp_path / "run/metadata.json").read_text())["stop_reason"]
        == "acquisition_error"
    )
