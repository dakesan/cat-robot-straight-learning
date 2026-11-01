import json
import math

import pytest

from catrobot import serial_capture
from catrobot.calibration import wheel_calibration


def status(left=0, right=0, armed=False, pwm=0, boot=7, invalid=0):
    return {
        "firmware": "fixture",
        "armed": armed,
        "left": pwm,
        "right": pwm,
        "servo_us": 0,
        "pwm_left": pwm,
        "pwm_right": pwm,
        "pwm_servo": 0,
        "encoder_left": left,
        "encoder_right": right,
        "encoder_left_invalid": invalid,
        "encoder_right_invalid": 0,
        "boot_id": boot,
        "time_ms": 0,
    }


class FakePort:
    def __init__(self, replies):
        self.replies = list(replies)
        self.written = []
        self.pending = []
        self.closed = False

    def write(self, data):
        self.written.append(data)
        if self.replies:
            self.pending = [b"boot noise\n", (json.dumps(self.replies.pop(0)) + "\n").encode()]

    def readline(self):
        return self.pending.pop(0) if self.pending else b""

    def reset_input_buffer(self):
        pass

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(serial_capture.time, "sleep", lambda seconds: None)


def test_serial_capture_is_read_only_and_stops_on_request(tmp_path):
    port = FakePort([status(0, 0), status(100, 0), status(1320, -1300), status(1320, -1300)])
    stop = tmp_path / "stop"
    calls = {"n": 0}

    def echo(_):
        calls["n"] += 1
        if calls["n"] >= 2:
            stop.touch()

    result = serial_capture.record_serial(
        tmp_path / "log.jsonl", port="fixture", stop_file=stop, opener=lambda _: port, echo=echo
    )
    assert {w for w in port.written} == {b"status\n"}
    assert port.closed and result["all_motionless"]
    assert result["encoder_before"] == [0, 0]
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == result["samples"]


def test_serial_capture_refuses_an_armed_robot(tmp_path):
    port = FakePort([status(armed=True)])
    with pytest.raises(ValueError, match="capture refused"):
        serial_capture.record_serial(tmp_path / "log.jsonl", port="x", opener=lambda _: port)
    assert port.written == [b"status\n"] and port.closed


def test_serial_capture_stops_when_outputs_appear_or_device_reboots(tmp_path):
    port = FakePort([status(), status(), status(pwm=20)])
    with pytest.raises(ValueError, match="produced output"):
        serial_capture.record_serial(
            tmp_path / "a.jsonl", port="x", opener=lambda _: port, echo=lambda _: None
        )
    port = FakePort([status(), status(boot=8)])
    with pytest.raises(ValueError, match="rebooted"):
        serial_capture.record_serial(
            tmp_path / "b.jsonl", port="x", opener=lambda _: port, echo=lambda _: None
        )


def write_log(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_wheel_calibration_combines_turns_and_roll(tmp_path):
    turns = [
        write_log(tmp_path / f"turn{i}.jsonl", [status(0, 0), status(13200 + i * 20, -13150)])
        for i in range(2)
    ]
    # A 21.5 mm wheel with 1320 counts/rev moves 1 m in 1320 / (2*pi*0.0215) counts.
    rolled = round(1320 / (2 * math.pi * 0.0215))
    rolls = [write_log(tmp_path / "roll.jsonl", [status(5, 5), status(5 + rolled, 5 - rolled)])]
    result = wheel_calibration(turns, rolls)
    assert result["accepted"] and not result["warnings"]
    assert result["left"]["encoder_sign"] == 1 and result["right"]["encoder_sign"] == -1
    assert result["left"]["counts_per_revolution"] == pytest.approx(1321)
    assert result["meters_per_count"] == pytest.approx(1 / rolled)
    assert result["radius_m"] == pytest.approx(0.0215, rel=0.002)


def test_wheel_calibration_rejects_disagreeing_repeats_and_invalid_edges(tmp_path):
    turns = [
        write_log(tmp_path / "t1.jsonl", [status(0, 0), status(13200, -13200)]),
        write_log(tmp_path / "t2.jsonl", [status(0, 0), status(14000, -13200)]),
    ]
    rolls = [
        write_log(tmp_path / "r1.jsonl", [status(0, 0), status(9000, -9000, invalid=2)]),
        write_log(tmp_path / "r2.jsonl", [status(0, 0), status(9300, -9000)]),
    ]
    result = wheel_calibration(turns, rolls)
    assert not result["accepted"]
    assert any("roll_spread" in p for p in result["problems"])
    assert any("invalid" in p for p in result["problems"])
    # Disagreeing hand-counted turns only warn: they do not set the distance per count.
    assert any("turn_spread" in w for w in result["warnings"])


def test_handling_before_the_turns_is_excluded(tmp_path):
    rows, left = [], 0
    for i in range(600):
        t = i * 0.1
        if 5 <= t < 8:
            left -= 20  # robot picked up and wheel nudged backwards: -600 counts
        if 20 <= t < 33:
            left += 100  # forward turns of 13000 counts in 13 s
        rows.append(dict(status(left, 0), host_monotonic_s=t))
    log = write_log(tmp_path / "turns.jsonl", rows)
    rows_r = [dict(status(0, -13200 * min(1, max(0, (i * 0.1 - 40) / 10))), host_monotonic_s=i * 0.1)
              for i in range(600)]
    for r in rows_r:
        r["encoder_right"] = round(r["encoder_right"])
    log_r = write_log(tmp_path / "turns-r.jsonl", rows_r)
    from catrobot.calibration import count_deltas

    left_motion = count_deltas(log)["left"]
    assert left_motion["delta"] == 13000 and left_motion["outside"] == -600
    assert count_deltas(log_r)["right"]["delta"] == -13200
