import json

import httpx
import pytest
from fakes import FakeRobot, sample

from catrobot import acquisition
from catrobot.acquisition import collect, record_telemetry
from catrobot.config import RobotConfig
from catrobot.quality import telemetry_quality, transport_quality


def test_device_quality_reports_missing_samples_gaps_and_invalid_edges():
    clean = [sample(i + 1, 1000 + 20 * i) for i in range(50)]
    assert telemetry_quality(clean)["accepted"]
    assert telemetry_quality(clean)["device_interval_ms"]["max"] == 20
    # Samples 11 and 12 are lost (60 ms device gap) and the invalid counter rises after them.
    broken = clean[:10] + [dict(s, encoder_left_invalid=1) for s in clean[12:]]
    result = telemetry_quality(broken)
    assert not result["accepted"] and result["missing"] == 2
    joined = " ".join(result["problems"])
    assert "missing" in joined and "gap" in joined and "invalid left" in joined


def test_transport_quality_separates_round_trip_from_command_gap():
    rows = [
        {"kind": "drive", "sent_s": t, "received_s": t + 0.01, "ok": True} for t in (0, 0.1, 0.45)
    ] + [{"kind": "telemetry", "sent_s": 0.05, "received_s": 0.09, "ok": True, "dropped": 3}]
    result = transport_quality(rows)
    assert result["drive"]["round_trip_ms"]["max"] == pytest.approx(10)
    assert result["max_drive_gap_ms"] == pytest.approx(350)
    assert not result["drive_gap_within_lease"]
    assert result["telemetry"]["round_trip_ms"]["median"] == pytest.approx(40)
    assert result["dropped"] == 3


def test_collect_records_transport_and_quality_summary(tmp_path, fast_clock):
    robot = FakeRobot()
    metadata = collect(
        "http://example.invalid",
        RobotConfig(provenance="measured"),
        tmp_path / "train-001",
        mode="speed",
        speed=0.1,
        client_factory=lambda url: robot,
        extra_metadata={"session_id": "s1", "attempt": 1},
    )
    assert metadata["stop_reason"] == "distance_limit"
    assert metadata["session_id"] == "s1" and metadata["quality_accepted"]
    assert metadata["summary"].startswith("[OK] train-001")
    rows = [json.loads(x) for x in (tmp_path / "train-001/transport.jsonl").read_text().splitlines()]
    kinds = {r["kind"] for r in rows}
    assert {"status", "telemetry", "drive", "arm", "stop", "sync", "calibration"} <= kinds
    quality = json.loads((tmp_path / "train-001/quality.json").read_text())
    assert quality["telemetry"]["missing"] == 0
    # Commands come from their own real-time thread, so count them rather than time them.
    assert quality["transport"]["drive"]["round_trip_ms"]["n"] >= 1


def test_collect_marks_lost_samples_as_unaccepted(tmp_path, fast_clock):
    robot = FakeRobot(drop_at=30)
    metadata = collect(
        "http://example.invalid",
        RobotConfig(provenance="measured"),
        tmp_path / "run",
        mode="speed",
        speed=0.1,
        client_factory=lambda url: robot,
    )
    assert metadata["dropped_samples"] == 1 and not metadata["quality_accepted"]
    assert metadata["summary"].startswith("[NG]")


def test_rehearsal_recording_keeps_going_and_reports_device_and_host_timing(
    tmp_path, monkeypatch
):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        after = int(request.url.params.get("after", 0))
        latest = state["n"] * 4
        rows = [sample(i, 20 * i) for i in range(max(after, latest - 4) + 1, latest + 1)]
        return httpx.Response(
            200, json={"boot_id": 3, "latest": latest, "dropped": 0, "samples": rows}
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        acquisition.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setattr(acquisition.time, "sleep", lambda seconds: None)
    result = record_telemetry("http://robot.invalid", tmp_path / "r1.jsonl", duration=0.05)
    assert result["telemetry"]["accepted"]
    assert result["telemetry"]["device_interval_ms"]["median"] == 20
    assert result["transport"]["telemetry"]["round_trip_ms"]["n"] >= 2
    assert (tmp_path / "r1.transport.jsonl").exists() and (tmp_path / "r1.quality.json").exists()


def test_shorter_lane_stops_earlier_and_records_the_goal(tmp_path, fast_clock):
    robot = FakeRobot()
    metadata = collect(
        "http://example.invalid",
        RobotConfig(provenance="measured"),
        tmp_path / "short",
        mode="speed",
        speed=0.1,
        client_factory=lambda url: robot,
        stop_m=0.9,
        goal_m=0.7,
    )
    assert metadata["stop_reason"] == "distance_limit"
    assert metadata["stop_m"] == 0.9 and metadata["goal_m"] == 0.7
    with pytest.raises(ValueError, match="goal < stop"):
        collect("u", RobotConfig(provenance="measured"), tmp_path / "bad", mode="speed",
                speed=0.1, client_factory=lambda url: robot, stop_m=0.6, goal_m=0.7)
