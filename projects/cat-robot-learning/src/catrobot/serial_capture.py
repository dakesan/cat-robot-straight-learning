"""Read-only USB serial status capture for wheel calibration without motor power."""

import json
import time
from pathlib import Path

# USB serial converter on the COM side of the robot, recorded on 2026-10-31.
COM_VID, COM_PID = 0x1A86, 0x55D3


def find_port() -> str:
    from serial.tools import list_ports

    ports = [p.device for p in list_ports.comports() if (p.vid, p.pid) == (COM_VID, COM_PID)]
    if len(ports) != 1:
        raise ValueError(f"Expected one robot COM port, found {ports}; pass --port explicitly")
    return ports[0]


def open_port(port: str):
    import serial

    # Keep DTR/RTS low before opening so the ESP32 is not reset by the host.
    handle = serial.Serial(port=None, baudrate=115200, timeout=0.2, write_timeout=1)
    handle.dtr = False
    handle.rts = False
    handle.port = port
    handle.open()
    return handle


def read_status(handle, timeout: float = 2.0) -> dict:
    handle.write(b"status\n")
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        line = handle.readline().decode("utf-8", "replace").strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "firmware" in value:
            return value
    raise TimeoutError("No status response on the serial port")


def motionless(status: dict) -> bool:
    return (
        not status["armed"]
        and status["left"] == status["right"] == status["servo_us"] == 0
        and status["pwm_left"] == status["pwm_right"] == status["pwm_servo"] == 0
    )


def record_serial(
    output: Path,
    port: str | None = None,
    duration: float = 120,
    stop_file: Path | None = None,
    interval: float = 0.1,
    opener=open_port,
    echo=print,
) -> dict:
    """Poll status only; never arm, command, calibrate or switch a GPIO."""
    if duration <= 0 or output.exists():
        raise ValueError("Choose a positive duration and a new output file")
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = opener(port or find_port())
    rows = []
    try:
        time.sleep(1)
        handle.reset_input_buffer()
        first = read_status(handle)
        if not motionless(first):
            raise ValueError("Robot is armed or has non-zero outputs; capture refused")
        echo(
            "READY: 読み取り専用の記録を開始しました（モーターは無効のまま）。"
            f" 左 {first['encoder_left']} / 右 {first['encoder_right']}"
        )
        until = time.monotonic() + duration
        last_echo = 0.0
        with output.open("w") as log:
            first["host_monotonic_s"] = time.monotonic()
            rows.append(first)
            log.write(json.dumps(first) + "\n")
            while time.monotonic() < until:
                status = read_status(handle)
                status["host_monotonic_s"] = time.monotonic()
                rows.append(status)
                log.write(json.dumps(status) + "\n")
                log.flush()
                if status["boot_id"] != first["boot_id"]:
                    raise ValueError("Device rebooted during capture")
                if not motionless(status):
                    raise ValueError("Robot became armed or produced output; capture stopped")
                now = time.monotonic()
                if now - last_echo >= 1:
                    echo(f"左 {status['encoder_left']} / 右 {status['encoder_right']}")
                    last_echo = now
                if stop_file is not None and stop_file.exists():
                    break
                time.sleep(interval)
    finally:
        handle.close()
    if not rows:
        raise ValueError("No status rows were recorded")
    return {
        "samples": len(rows),
        "boot_id": first["boot_id"],
        "encoder_before": [rows[0]["encoder_left"], rows[0]["encoder_right"]],
        "encoder_after": [rows[-1]["encoder_left"], rows[-1]["encoder_right"]],
        "invalid_delta": [
            rows[-1][f"encoder_{side}_invalid"] - rows[0][f"encoder_{side}_invalid"]
            for side in ("left", "right")
        ],
        "all_motionless": True,
    }
