"""Bounded real-robot trials over the existing authenticated HTTP interface."""

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from scipy.optimize import least_squares

from .config import RobotConfig, write_json
from .model import CorrectionModel
from .quality import TransportLog, summary_line, telemetry_quality, transport_quality

MODES = {"pwm": 0, "speed": 1, "learned": 2, "student": 3}
# Rehearsal, probe and demo runs are stored apart; only "train" runs may fit models.
SPLITS = ("train", "test", "rehearsal", "calibration", "probe", "demo", "device")


def code_version(project: Path | None = None) -> dict:
    """Commit and dirty state of the acquisition code, recorded with every trial."""
    import subprocess

    root = project or Path(__file__).resolve().parents[2]

    def git(*args):
        result = subprocess.run(
            ["git", *args], check=False, cwd=root, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {"code_commit": git("rev-parse", "HEAD"), "code_dirty": bool(git("status", "--short"))}


class RobotClient:
    def __init__(self, url: str):
        self.http = httpx.Client(base_url=url.rstrip("/"), timeout=1.0, trust_env=False)
        for attempt in range(4):
            # The 5 kB page occasionally stalls on a busy home network; retry before failing.
            try:
                response = self.http.get("/", timeout=2.0)
                response.raise_for_status()
                break
            except httpx.TransportError:
                if attempt == 3:
                    self.http.close()
                    raise
                time.sleep(1.0)
        token = re.search(r"const token='([0-9a-f]{32})'", response.text)
        if token is None:
            self.http.close()
            raise ValueError("Robot control token was not found in the local control page")
        self.http.headers["X-Robot-Token"] = token[1]

    def get(self, path: str, timeout: float | None = None, **params):
        response = self.http.get(path, params=params, timeout=timeout or self.http.timeout)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, timeout: float | None = None, **fields):
        response = self.http.post(path, data=fields, timeout=timeout or self.http.timeout)
        response.raise_for_status()
        return response.json()

    def calibrate(self, config: RobotConfig):
        config.require_measured()
        calibration = config.native_calibration()
        names = (
            "left_meters_per_count",
            "right_meters_per_count",
            "left_sign",
            "right_sign",
            "left_feedforward",
            "right_feedforward",
            "left_deadband",
            "right_deadband",
            "kp",
            "ki",
            "max_speed",
        )
        self.post(
            "/api/calibration", confirmed=1, **{name: getattr(calibration, name) for name in names}
        )

    def close(self):
        self.http.close()


def record_telemetry(url: str, output: Path, duration: float = 30) -> dict:
    """Read samples only; never arm, command, calibrate or switch a GPIO."""
    if duration <= 0 or output.exists():
        raise ValueError("Choose a positive duration and a new output file")
    output.parent.mkdir(parents=True, exist_ok=True)
    transport = TransportLog(output.with_name(output.stem + ".transport.jsonl"))
    samples, stop_reason = [], "duration_limit"

    def get(client, **params):
        response = client.get("/api/telemetry", params=params)
        response.raise_for_status()
        return response.json()

    try:
        with (
            httpx.Client(base_url=url, timeout=1, trust_env=False) as client,
            output.open("w") as handle,
        ):
            initial = transport.call("telemetry", get, client)
            cursor, boot_id = initial["latest"], initial["boot_id"]
            until = time.monotonic() + duration
            while time.monotonic() < until:
                batch = transport.call("telemetry", get, client, after=cursor)
                if batch["boot_id"] != boot_id:
                    stop_reason = "device_rebooted"
                    break
                for sample in batch["samples"]:
                    handle.write(json.dumps(sample) + "\n")
                    samples.append(sample)
                    cursor = sample["sample"]
                handle.flush()
                time.sleep(0.08)
    finally:
        transport.close()
    result = {
        "stop_reason": stop_reason,
        "telemetry": telemetry_quality(samples),
        "transport": transport_quality(transport.rows),
    }
    result["summary"] = summary_line(
        output.stem, result["telemetry"], result["transport"], stop_reason
    )
    write_json(output.with_name(output.stem + ".quality.json"), result)
    return result


def collect(
    url: str,
    config: RobotConfig,
    directory: Path,
    *,
    mode: str,
    speed: float,
    correction: float = 0,
    split: str = "train",
    pwm: int | None = None,
    model: CorrectionModel | None = None,
    max_seconds: float = 25,
    client_factory=RobotClient,
    extra_metadata: dict | None = None,
    stop_m: float = 1.3,
    goal_m: float = 1.0,
    host_model: bool = False,
    student_id: str | None = None,
) -> dict:
    """One trial; the robot stops after stop_m of wheel travel and is scored at goal_m.

    host_model runs "learned" trials before the model is flashed: the same C++ choose()
    as the firmware picks the correction on the Mac, and the ESP32 runs PI with it.
    """
    config.require_measured()
    if not 0 < goal_m < stop_m <= 1.3:
        raise ValueError("Need 0 < goal < stop <= 1.3 m")
    if mode not in MODES or split not in SPLITS:
        raise ValueError("Invalid mode or split")
    if not 0 < speed <= config.max_speed_mps or not -0.2 <= correction <= 0.2:
        raise ValueError("Speed/correction outside configured bounds")
    if not 1 <= max_seconds <= 60:
        raise ValueError("Real trials must be limited to 1..60 seconds")
    if mode == "learned" and (
        model is None or model.source != "measured" or model.config_hash != config.fingerprint()
    ):
        raise ValueError("Learned driving requires a measured model matching this configuration")
    if mode == "student" and not student_id:
        raise ValueError("Student driving requires the expected student id")
    if directory.exists():
        raise ValueError("Trial directory already exists; never overwrite a raw run")
    if pwm is None:
        pwm = round(
            (config.left.deadband_percent + config.right.deadband_percent) / 2
            + speed * (config.left.feedforward + config.right.feedforward) / 2
        )
    if not 0 <= pwm <= 50:
        raise ValueError("PWM exceeds the existing 50% ceiling")
    client = client_factory(url)
    transport = TransportLog()

    def get(kind: str, path: str, **params):
        return transport.call(kind, client.get, path, **params)

    def post(kind: str, path: str, **fields):
        return transport.call(kind, client.post, path, **fields)

    status = get("status", "/api/status")
    if status["armed"]:
        client.close()
        raise ValueError("Stop the robot before starting a recorded trial")
    if mode == "learned" and not host_model and (
        status.get("model_id") != model.model_id() or not status.get("model_measured")
    ):
        client.close()
        raise ValueError("Firmware does not contain the requested measured model")
    if mode == "student" and (status.get("student_id") != student_id or not status.get("student_measured")):
        client.close()
        raise ValueError("Firmware does not contain the requested measured student")
    directory.mkdir(parents=True)
    transport.attach(directory / "transport.jsonl")
    metadata = {
        "trial_id": directory.name,
        "split": split,
        "source": "measured",
        "mode": mode,
        "speed_mps": speed,
        "correction": correction,
        "pwm": pwm,
        "config_hash": config.fingerprint(),
        "firmware": status["firmware"],
        "model_id": {"learned": status.get("model_id"), "student": status.get("student_id")}.get(mode),
        "boot_id": status["boot_id"],
        "dropped_samples": 0,
        "late_commands": 0,
        "rejected_commands": 0,
        "late_telemetry": 0,
        "stop_reason": "not_started",
        "started_wall_time": time.time(),
        "stop_m": stop_m,
        "goal_m": goal_m,
        **(extra_metadata or {}),
    }
    wire_mode = mode
    if mode == "learned":
        if host_model:
            from ._native import Controller

            native = Controller()
            model.attach(native)
            correction = float(native.choose(speed))
            wire_mode = "speed"
            metadata["model_id"] = model.model_id()
            metadata["model_host"] = True
        else:
            correction = model.choose(speed)
        metadata["correction"] = correction
    write_json(directory / "config.json", config.model_dump())
    write_json(directory / "metadata.json", metadata)
    last_sample = None
    samples = []
    cursor = get("telemetry", "/api/telemetry")["latest"]
    sequence, epoch = 0, None
    origin_left, origin_right = status["encoder_left"], status["encoder_right"]

    with (directory / "telemetry.jsonl").open("w") as handle:

        def drive_commands(stop: threading.Event):
            """Send a drive command every 50 ms without waiting for earlier replies.

            Each request uses its own connection, so one delayed or retransmitted request
            does not hold back the next. The ESP32 drops commands that arrive out of order
            and still stops by itself when no command arrives for 300 ms.
            """
            nonlocal sequence

            def send(number: int):
                try:
                    post(
                        "drive",
                        "/api/drive",
                        timeout=0.3,
                        epoch=epoch,
                        seq=number,
                        mode=MODES[wire_mode],
                        speed_mps=speed,
                        correction=correction,
                        pwm=pwm,
                    )
                except httpx.TimeoutException:
                    metadata["late_commands"] += 1
                except httpx.HTTPStatusError:
                    metadata["rejected_commands"] += 1
                except httpx.TransportError:
                    metadata["late_commands"] += 1

            in_flight: list = []
            with ThreadPoolExecutor(max_workers=4) as pool:
                next_at = time.monotonic()
                while not stop.is_set():
                    in_flight = [f for f in in_flight if not f.done()]
                    if len(in_flight) < 4:
                        sequence += 1
                        in_flight.append(pool.submit(send, sequence))
                    next_at += 0.05
                    stop.wait(max(0.0, next_at - time.monotonic()))

        def pump(seconds: float, driving: bool):
            nonlocal cursor, last_sample
            until = time.monotonic() + seconds
            stop = threading.Event()
            sender = threading.Thread(target=drive_commands, args=(stop,), daemon=True)
            if driving:
                sender.start()
            heard = time.monotonic()
            try:
                while time.monotonic() < until:
                    started = time.monotonic()
                    try:
                        batch = get("telemetry", "/api/telemetry", timeout=0.5, after=cursor)
                    except httpx.TimeoutException:
                        metadata["late_telemetry"] += 1
                        if time.monotonic() - heard > 2.0:
                            raise TimeoutError("No telemetry reply for 2 s") from None
                        continue
                    heard = time.monotonic()
                    if batch["boot_id"] != metadata["boot_id"]:
                        raise ValueError("Device rebooted during the trial")
                    metadata["dropped_samples"] += batch["dropped"]
                    for sample in batch["samples"]:
                        handle.write(json.dumps(sample) + "\n")
                        samples.append(sample)
                        last_sample = sample
                        cursor = sample["sample"]
                    handle.flush()
                    if driving and last_sample and last_sample["epoch"] == epoch:
                        if not last_sample["armed"]:
                            metadata["stop_reason"] = last_sample["reason"]
                            return
                        distance = (
                            abs(last_sample["encoder_left"] - origin_left)
                            * config.left.meters_per_count
                            + abs(last_sample["encoder_right"] - origin_right)
                            * config.right.meters_per_count
                        ) / 2
                        if distance >= stop_m:
                            metadata["stop_reason"] = "distance_limit"
                            return
                    time.sleep(max(0, 0.1 - (time.monotonic() - started)))
            finally:
                stop.set()
                if driving:
                    sender.join(timeout=1)

        try:
            transport.call("calibration", client.calibrate, config)
            post("sync", "/api/sync", enabled=1)
            pump(2.2, False)
            before = get("status", "/api/status")
            origin_left, origin_right = before["encoder_left"], before["encoder_right"]
            epoch = post("arm", "/api/arm")["epoch"]
            pump(max_seconds, True)
            if metadata["stop_reason"] == "not_started":
                metadata["stop_reason"] = "duration_limit"
            post("stop", "/api/stop")
            post("sync", "/api/sync", enabled=1)
            pump(2.2, False)
        except BaseException as error:
            metadata["stop_reason"] = "acquisition_error"
            metadata["error_type"] = type(error).__name__
            if isinstance(error, KeyboardInterrupt):
                metadata["intervention"] = "operator_interrupt"
            # Stop first, then try to film the end pattern so the run can still be paired.
            try:
                post("stop", "/api/stop")
                post("sync", "/api/sync", enabled=1)
                pump(2.2, False)
                metadata["end_sync_after_error"] = True
            except BaseException:  # noqa: BLE001 - keep the original error for the caller
                metadata["end_sync_after_error"] = False
            raise
        finally:
            try:
                post("stop", "/api/stop")
                post("sync", "/api/sync", enabled=0)
            except httpx.HTTPError:
                metadata["stop_request_failed"] = True
            finally:
                client.close()
                transport.close()
                metadata["ended_wall_time"] = time.time()
                quality = {
                    "telemetry": telemetry_quality(samples),
                    "transport": transport_quality(transport.rows),
                }
                metadata["summary"] = summary_line(
                    directory.name, quality["telemetry"], quality["transport"],
                    metadata["stop_reason"],
                )
                metadata["quality_accepted"] = bool(
                    quality["telemetry"]["accepted"] and not metadata["dropped_samples"]
                )
                write_json(directory / "quality.json", quality)
                write_json(directory / "metadata.json", metadata)
    return metadata


def completed(metadata: dict | None) -> bool:
    return bool(
        metadata
        and metadata.get("stop_reason") == "distance_limit"
        and metadata.get("quality_accepted", False)
    )


def read_metadata(directory: Path) -> dict | None:
    path = directory / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else None


def run_session(
    url: str,
    config: RobotConfig,
    jobs: list[dict],
    output: Path,
    *,
    session_id: str,
    confirm,
    retry,
    echo=print,
    model: CorrectionModel | None = None,
    client_factory=RobotClient,
    stop_m: float = 1.3,
    goal_m: float = 1.0,
    host_model: bool = False,
    student_id: str | None = None,
) -> dict:
    """Run scheduled trials one at a time; failed attempts move to failed/ and stay recorded.

    confirm(job) -> bool starts a trial; retry(job, metadata) -> bool re-runs a failed one.
    """
    config.require_measured()
    output.mkdir(parents=True, exist_ok=True)
    session_path = output / f"session-{session_id}.json"
    if session_path.exists():
        session = json.loads(session_path.read_text())
    else:
        session = {
            "session_id": session_id,
            "created_wall_time": time.time(),
            **code_version(),
            "video": None,
            "attempts": [],
        }

    def save():
        write_json(session_path, session)

    def archive(job: dict) -> str:
        failed = output / "failed"
        failed.mkdir(exist_ok=True)
        number = 1 + len(list(failed.glob(f"{job['trial_id']}-a*")))
        target = failed / f"{job['trial_id']}-a{number}"
        (output / job["trial_id"]).rename(target)
        relative = str(target.relative_to(output))
        for attempt in session["attempts"]:
            if attempt["directory"] == job["trial_id"]:
                attempt["directory"] = relative
        save()
        return relative

    for index, job in enumerate(jobs):
        destination = output / job["trial_id"]
        existing = read_metadata(destination)
        if destination.exists():
            if completed(existing):
                echo(f"記録済みのため飛ばします: {job['trial_id']}")
                continue
            if not retry(job, existing or {}):
                echo(f"失敗した記録を残して次へ進みます: {job['trial_id']}")
                continue
            archive(job)
        while True:
            echo(
                f"試行 {job['trial_id']}（{index + 1}/{len(jobs)}）: 方式 {job['mode']}、"
                f"速度 {job['speed_mps']:.3f} m/s、補正率 {job['correction']:+.3f}"
            )
            if not confirm(job):
                save()
                return session
            attempt = {
                "trial_id": job["trial_id"],
                "directory": job["trial_id"],
                "schedule_index": index,
                "number": 1 + sum(a["trial_id"] == job["trial_id"] for a in session["attempts"]),
            }
            interrupted = False
            try:
                collect(
                    url,
                    config,
                    destination,
                    mode=job["mode"],
                    speed=job["speed_mps"],
                    correction=job["correction"],
                    split=job["split"],
                    pwm=job.get("pwm"),
                    model=model,
                    client_factory=client_factory,
                    stop_m=stop_m,
                    goal_m=goal_m,
                    host_model=host_model,
                    student_id=student_id,
                    extra_metadata={
                        "session_id": session_id,
                        "attempt": attempt["number"],
                        "schedule_index": index,
                        **code_version(),
                    },
                )
            except KeyboardInterrupt:
                interrupted = True
                echo("Ctrl-C を受け付けました。停止指令を送り、介入として記録しました。")
            except Exception as error:  # noqa: BLE001 - every failure is recorded, then offered for retry
                echo(f"試行が途中で失敗しました: {type(error).__name__}: {error}")
            metadata = read_metadata(destination) or {"stop_reason": "not_started"}
            attempt.update(
                {
                    key: metadata.get(key)
                    for key in (
                        "boot_id",
                        "started_wall_time",
                        "ended_wall_time",
                        "stop_reason",
                        "quality_accepted",
                        "summary",
                        "intervention",
                    )
                }
            )
            session["attempts"].append(attempt)
            save()
            if metadata.get("summary"):
                echo(metadata["summary"])
            if completed(metadata):
                break
            if not destination.exists():
                break
            if interrupted or not retry(job, metadata):
                break
            archive(job)
    save()
    return session


def fit_motors(runs: Path, config: RobotConfig) -> RobotConfig:
    """Fit feedforward from steady measured PWM runs, with each run weighted once."""
    rows = []
    trajectories = {"left": [], "right": []}
    for path in sorted(runs.glob("*/metadata.json")):
        metadata = json.loads(path.read_text())
        if metadata["source"] != "measured" or metadata["mode"] != "pwm":
            continue
        if metadata.get("pwm") == 0 or metadata.get("split") == "rehearsal":
            # Zero-output dry runs exercise the command loop; they carry no motor response.
            continue
        samples = [
            json.loads(s) for s in (path.parent / "telemetry.jsonl").read_text().splitlines()
        ]
        armed = [s for s in samples if s["armed"]]
        if len(armed) < 50:
            continue
        for side in ("left", "right"):
            trajectories[side].append(armed.copy())
        begin = armed[0]["time_ms"] + 1000
        armed = [s for s in armed if s["time_ms"] >= begin]
        if len(armed) < 10:
            continue
        record = {}
        for side in ("left", "right"):
            wheel = getattr(config, side)
            t = np.array([s["time_ms"] / 1000 for s in armed])
            counts = np.array([s[f"encoder_{side}"] for s in armed])
            velocity = np.polyfit(
                t - t[0], counts * wheel.meters_per_count * wheel.encoder_sign, 1
            )[0]
            record[side] = (float(np.median([s[side] for s in armed])), velocity)
        rows.append(record)
    if len(rows) < 3:
        raise ValueError("Record at least three measured PWM runs at different outputs")
    result = config.model_copy(deep=True)
    maxima = []
    for side in ("left", "right"):
        x, y = np.array([r[side] for r in rows]).T
        if np.ptp(x) < 10 or np.any(y <= 0):
            raise ValueError("Need positive, correctly signed rotation over at least 10 PWM points")
        slope, intercept = np.polyfit(x, y, 1)
        deadband = -intercept / slope
        if slope <= 0 or not 0 <= deadband < 50:
            raise ValueError("PWM/speed fit is invalid; inspect signs and steady segments")
        wheel = getattr(result, side)
        wheel.feedforward, wheel.deadband_percent = float(1 / slope), float(deadband)
        gain, response_deadband, constant, rms = fit_response(
            trajectories[side], getattr(config, side), side, float(slope), float(deadband)
        )
        wheel.motor_velocity_per_percent = gain
        wheel.motor_deadband_percent = response_deadband
        wheel.time_constant_s = constant
        wheel.motor_gain = 1
        wheel.response_fit_rms_m = rms
        maxima.append(float(slope * min(40, max(x)) + intercept))
    result.max_speed_mps = 0.8 * min(maxima)
    result.calibration_notes += "; feedforward and effective first-order response fitted from measured PWM/encoders; torque and contact physics remain unverified"
    return RobotConfig.model_validate(result.model_dump())


def fit_response(
    runs: list[list[dict]], wheel, side: str, initial_gain: float, initial_deadband: float
) -> tuple[float, float, float, float]:
    """Identify an effective wheel response; this does not measure motor torque."""
    arrays = []
    for rows in runs:
        t = np.array([s["time_ms"] / 1000 for s in rows])
        position = (
            np.array([s[f"encoder_{side}"] for s in rows], float)
            * wheel.meters_per_count
            * wheel.encoder_sign
        )
        command = np.array([((s[side] * 255 + 50) // 100) * 100 / 255 for s in rows])
        if np.any(np.diff(t) <= 0):
            raise ValueError("Response fit requires strictly increasing device times")
        arrays.append((t - t[0], position - position[0], command))

    def residual(parameters):
        gain, deadband, tau = parameters
        errors = []
        for t, measured, command in arrays:
            predicted = np.zeros(len(t))
            velocity = 0.0
            for i, dt in enumerate(np.diff(t), 1):
                target = max(0, command[i - 1] - deadband) * gain
                next_velocity = target + (velocity - target) * np.exp(-dt / tau)
                predicted[i] = predicted[i - 1] + (velocity + next_velocity) * dt / 2
                velocity = next_velocity
            errors.extend(predicted - measured)
        return np.array(errors)

    fitted = least_squares(
        residual, [initial_gain, initial_deadband, 0.08], bounds=([1e-6, 0, 0.005], [0.1, 49, 2])
    )
    if not fitted.success or np.linalg.matrix_rank(fitted.jac) < 3:
        raise ValueError("Effective motor response was not identifiable from these trials")
    return (*map(float, fitted.x), float(np.sqrt(np.mean(fitted.fun**2))))
