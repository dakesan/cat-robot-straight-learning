"""Command-line entry points for the experiment."""

import json
from pathlib import Path
from typing import Annotated

import typer

from .config import RobotConfig, load_config, write_json
from .model import export_header, load_model, read_records, save_model, train

app = typer.Typer(no_args_is_help=True)


@app.command()
def init_config(output: Path):
    """Write explicitly unmeasured example geometry and motor assumptions."""
    if output.exists():
        raise typer.BadParameter("Output already exists")
    write_json(output, RobotConfig().model_dump())


@app.command("import-cad")
def import_cad(source: Path, output: Path, deflection_mm: float = 0.15):
    """Convert STEP to metre-scale meshes and a header-redacted source copy."""
    from .cad import convert_step

    typer.echo(json.dumps(convert_step(source, output, deflection_mm), ensure_ascii=False, indent=2))


@app.command("view-cad")
def view_cad(
    output: Path,
    asset: str = "shapr3d-20261029",
    bundle: Path | None = None,
    config: Path | None = None,
):
    """Write a standalone rotatable CAD view without requiring a server or OCCT."""
    from .cad import ASSET_ROOT, inspection_html

    if output.exists():
        raise typer.BadParameter("Output already exists")
    html = inspection_html(bundle or ASSET_ROOT / asset, load_config(config) if config else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    typer.echo(str(output))


@app.command("train")
def train_command(runs: Path, output: Path):
    """Fit only train-split runs; never consume test runs."""
    fitted = train(read_records(runs))
    save_model(fitted, output)
    typer.echo(
        f"Model {fitted.model_id()}: {len(fitted.training_runs)} runs, source={fitted.source}"
    )


@app.command("train-student")
def train_student_command(
    frames: Path,
    output: Path,
    config: Path,
    target: str = "label_correction",
    source: str = "measured",
    alpha: float = 1e-3,
):
    """Fit the encoder-only ESP32 student on hindsight labels or teacher outputs."""
    from .student import read_frames, save_student, train_student

    if target not in ("label_correction", "teacher_correction"):
        raise typer.BadParameter("Target must be label_correction or teacher_correction")
    settings = load_config(config)
    model = train_student(
        read_frames(frames), target, source, settings.fingerprint(), settings.track_m, alpha=alpha
    )
    save_student(model, output)
    typer.echo(
        f"student {model.model_id()} target={target} trials={len(model.training_trials)} "
        f"cv_rmse={model.cv_rmse:.4f} (speed median {model.cv_rmse_speed_median:.4f}) "
        f"range=[{model.min_correction:+.3f}, {model.max_correction:+.3f}]"
    )


@app.command("compare-students")
def compare_students_command(
    frames: Path,
    truth: Path,
    split: Path,
    output: Path,
    config: Path,
    alpha: float = 1e-3,
):
    """Score direct and distilled students on the teacher's held-out trials.

    frames: teacher output CSV; truth: frames.csv whose labels are the reference;
    split: the teacher dataset's catrobot-split.json.
    """
    from .student import holdout_comparison, read_frames

    if output.exists():
        raise typer.BadParameter("Output already exists")
    validation = set(json.loads(split.read_text())["validation_trials"])
    result = holdout_comparison(
        read_frames(frames),
        read_frames(truth),
        validation,
        ["label_correction", "teacher_correction", "truth_correction"],
        load_config(config).track_m,
        alpha=alpha,
    )
    write_json(output, result)
    typer.echo(json.dumps(result, indent=2))


@app.command("export-student")
def export_student_command(model: Path, output: Path):
    """Write the student's weights as a firmware header."""
    from .student import export_student_header, load_student

    export_student_header(load_student(model), output)


@app.command("export-model")
def export_command(model: Path, output: Path):
    """Export measured model coefficients for ESP32 compilation."""
    export_header(load_model(model), output)


@app.command("simulate")
def simulate_command(
    config: Path,
    output: Path,
    mode: str = "pwm",
    speed: float = 0.1,
    correction: float = 0,
    model: Path | None = None,
    video: bool = False,
    split: str = "test",
):
    """Run the shared native controller inside a MuJoCo physical world."""
    from .simulation import run_episode

    typer.echo(
        json.dumps(
            run_episode(
                load_config(config),
                output,
                mode=mode,
                speed=speed,
                correction=correction,
                model=load_model(model) if model else None,
                record_video=video,
                split=split,
            ),
            indent=2,
        )
    )


def student_id_of(student: Path | None) -> str | None:
    if student is None:
        return None
    from .student import load_student

    return load_student(student).model_id()


@app.command("schedule")
def schedule_command(
    output: Path,
    low: float | None = None,
    high: float | None = None,
    split: str = "train",
    config: Path | None = None,
    center: float = 0.0,
    step: float = 0.02,
):
    """Write randomized probe (4), train (30), evaluation (30) or on-device (15) trials."""
    from .simulation import schedule

    if split not in ("probe", "train", "test", "device"):
        raise typer.BadParameter("Split must be probe, train, test or device")
    if config is not None:
        settings = load_config(config)
        settings.require_measured()
        low = low if low is not None else 0.45 * settings.max_speed_mps
        high = high if high is not None else 0.8 * settings.max_speed_mps
        if high > settings.max_speed_mps:
            raise typer.BadParameter("Scheduled speed exceeds measured calibration")
    if output.exists():
        raise typer.BadParameter("Schedule already exists; keep the recorded order")
    write_json(output, schedule([low or 0.08, high or 0.14], split, center=center, step=step))


@app.command("record")
def record_command(output: Path, url: str = "http://192.168.4.1", duration: float = 30):
    """Read telemetry without arming the robot or changing any output."""
    from .acquisition import record_telemetry

    result = record_telemetry(url, output, duration)
    typer.echo(result["summary"])


@app.command("collect")
def collect_command(
    config: Path,
    output: Path,
    mode: str = "speed",
    speed: float = 0.1,
    correction: float = 0,
    split: str = "train",
    pwm: int | None = None,
    model: Path | None = None,
    url: str = "http://192.168.4.1",
    max_seconds: float = 25,
    stop_m: float = 1.3,
    goal_m: float = 1.0,
    student: Path | None = None,
):
    """Run one supervised real trial; the sync LED must be visible to the camera."""
    from .acquisition import code_version, collect

    typer.echo(
        json.dumps(
            collect(
                url,
                load_config(config),
                output,
                mode=mode,
                speed=speed,
                correction=correction,
                split=split,
                pwm=pwm,
                model=load_model(model) if model else None,
                max_seconds=max_seconds,
                extra_metadata=code_version(),
                stop_m=stop_m,
                goal_m=goal_m,
                student_id=student_id_of(student),
            ),
            indent=2,
        )
    )


@app.command("collect-schedule")
def collect_schedule(
    config: Path,
    schedule: Path,
    output: Path,
    model: Path | None = None,
    url: str = "http://192.168.4.1",
    session_id: str | None = None,
    stop_m: float = 1.3,
    goal_m: float = 1.0,
    student: Path | None = None,
):
    """Prompt between scheduled trials; failed attempts are kept under failed/."""
    import time

    from .acquisition import run_session

    def confirm(job):
        return typer.confirm(
            "車体を開始位置に戻し、レーンを空け、撮影中ですか？この試行を走らせます", default=True
        )

    def retry(job, metadata):
        reason = metadata.get("stop_reason")
        return typer.confirm(f"{job['trial_id']} は {reason} で終わりました。撮り直しますか？")

    session = run_session(
        url,
        load_config(config),
        json.loads(schedule.read_text()),
        output,
        session_id=session_id or time.strftime("%Y%m%d-%H%M%S"),
        confirm=confirm,
        retry=retry,
        echo=typer.echo,
        model=load_model(model) if model else None,
        stop_m=stop_m,
        goal_m=goal_m,
        student_id=student_id_of(student),
    )
    typer.echo(f"セッション {session['session_id']}: 試み {len(session['attempts'])} 件を記録")


@app.command("reset-start")
def reset_start_command(
    config: Path,
    snapshot: Path,
    camera: Path,
    arena: Path,
    url: str = "http://192.168.4.1",
    start_x: float = 0.06,
    start_y: float = 0.45,
):
    """Camera-guided return to the start pose between trials (never during a trial)."""
    from .acquisition import RobotClient
    from .autonomy import Lane, LiveCamera, Mover, reset_to_start

    settings = load_config(config)
    client = RobotClient(url)
    try:
        result = reset_to_start(
            LiveCamera(snapshot, camera, arena, settings),
            Mover(client),
            Lane(start_x=start_x, start_y=start_y),
            log=typer.echo,
        )
    finally:
        client.close()
    typer.echo(json.dumps(result))


@app.command("autocollect")
def autocollect_command(
    config: Path,
    schedule: Path,
    output: Path,
    snapshot: Path,
    camera: Path,
    arena: Path,
    session_id: str,
    url: str = "http://192.168.4.1",
    model: Path | None = None,
    stop_m: float = 0.9,
    goal_m: float = 0.7,
    start_x: float = 0.06,
    start_y: float = 0.45,
    max_retries: int = 2,
    host_model: bool = False,
    student: Path | None = None,
):
    """Unattended session: the camera returns the robot to the start before each trial."""
    import time

    import httpx

    from .acquisition import RobotClient, run_session
    from .autonomy import Lane, LiveCamera, Mover, back_off_until_visible, reset_to_start

    settings = load_config(config)
    live = LiveCamera(snapshot, camera, arena, settings)
    lane = Lane(start_x=start_x, start_y=start_y)
    retries: dict[str, int] = {}

    pose_log = output / f"camera-poses-{session_id}.jsonl"
    output.mkdir(parents=True, exist_ok=True)

    def log_pose(job, phase):
        pose = live.steady_pose(newer_than=time.time() + 0.2)
        row = {"trial_id": job["trial_id"], "phase": phase, "wall_time": time.time()}
        if pose is not None:
            row.update(x_m=pose.x, y_m=pose.y, yaw_rad=pose.yaw)
        with pose_log.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def confirm(job):
        log_pose(job, "before_reset")
        for attempt in range(3):
            try:
                client = RobotClient(url)
                try:
                    result = reset_to_start(live, Mover(client), lane, log=lambda line: None)
                finally:
                    client.close()
            except httpx.HTTPError as error:
                result = {"ok": False, "reason": f"network: {type(error).__name__}"}
            if not result["ok"] and "not visible" in result.get("reason", ""):
                try:
                    client = RobotClient(url)
                    try:
                        # Lost before any reset step: it drove out after the trial.
                        lost_during_reset = result.get("steps", 0) > 0
                        back_off_until_visible(
                            live, Mover(client), log=typer.echo, forward=lost_during_reset
                        )
                    finally:
                        client.close()
                except httpx.HTTPError:
                    pass
            typer.echo(f"reset before {job['trial_id']}: {json.dumps(result)}")
            if result["ok"]:
                time.sleep(1.0)
                log_pose(job, "start")
                return True
            time.sleep(5)
        return False

    def retry(job, metadata):
        # Only acquisition problems are re-run; controller outcomes stay as recorded.
        if metadata.get("stop_reason") not in ("acquisition_error", "command_timeout"):
            return False
        retries[job["trial_id"]] = retries.get(job["trial_id"], 0) + 1
        return retries[job["trial_id"]] <= max_retries

    jobs = json.loads(schedule.read_text())

    def echo(line):
        typer.echo(line)
        if line.startswith("[") and "] " in line:
            # A trial just finished; record where it ended before the next reset.
            trial = line.split("] ", 1)[1].split(" ", 1)[0]
            log_pose({"trial_id": trial}, "end")

    session = run_session(
        url,
        settings,
        jobs,
        output,
        session_id=session_id,
        confirm=confirm,
        retry=retry,
        echo=echo,
        model=load_model(model) if model else None,
        stop_m=stop_m,
        goal_m=goal_m,
        host_model=host_model,
        student_id=student_id_of(student),
    )
    typer.echo(f"session {session['session_id']}: {len(session['attempts'])} attempts")


@app.command("fit-motors")
def fit_motors_command(config: Path, runs: Path, output: Path):
    """Fit each wheel's feedforward from measured steady PWM runs."""
    from .acquisition import fit_motors

    write_json(output, fit_motors(runs, load_config(config)).model_dump())


@app.command("record-serial")
def record_serial_command(
    output: Path,
    port: str | None = None,
    duration: float = 120,
    stop_file: Path | None = None,
):
    """Record status over the COM USB port; refuses if the robot is armed or driving."""
    from .serial_capture import record_serial

    typer.echo(json.dumps(record_serial(output, port, duration, stop_file), indent=2))


@app.command("wheel-calibration")
def wheel_calibration_command(
    turn_log: Annotated[list[Path], typer.Option(help="Log of 10 forward turns per wheel")],
    roll_log: Annotated[list[Path], typer.Option(help="Log of a straight hand-pushed roll")],
    turns: float = 10,
    roll_m: float = 1.0,
):
    """Counts/rev, sign and effective radius with repeat agreement checks."""
    from .calibration import wheel_calibration

    typer.echo(json.dumps(wheel_calibration(turn_log, roll_log, turns, roll_m), indent=2))


@app.command("encoder-counts")
def encoder_counts(log: Path, turns: float = 10):
    """Estimate signed counts/revolution after manually turning each wheel forward."""
    from .sync import read_jsonl

    rows = read_jsonl(log)
    if turns <= 0 or len(rows) < 2:
        raise typer.BadParameter("Need positive revolutions and at least two samples")
    result = {}
    for side in ("left", "right"):
        delta = rows[-1][f"encoder_{side}"] - rows[0][f"encoder_{side}"]
        invalid = rows[-1][f"encoder_{side}_invalid"] - rows[0][f"encoder_{side}_invalid"]
        if invalid:
            raise typer.BadParameter(
                f"Invalid {side} encoder transitions; inspect wiring before calibration"
            )
        result[side] = {
            "counts_per_revolution": abs(delta) / turns,
            "encoder_sign": 1 if delta > 0 else -1,
        }
        if not delta:
            raise typer.BadParameter(f"No {side} encoder response")
    typer.echo(json.dumps(result, indent=2))


@app.command("markers")
def marker_command(output: Path):
    """Generate printable ArUco markers (black square: 50 mm)."""
    from .vision import markers

    markers(output)


@app.command("frame")
def frame_command(video: Path, output: Path, time: float = 0):
    """Extract a video frame at its presentation time for arena calibration."""
    import cv2

    from .vision import video_frames

    for timestamp, image in video_frames(video):
        if timestamp >= time:
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), image):
                raise typer.BadParameter("Cannot write frame")
            return
    raise typer.BadParameter("Requested time is beyond the video")


@app.command("calibrate-camera")
def camera_command(video: Path, output: Path, square_m: float = 0.025):
    """Calibrate from a video of a 9 by 6 inner-corner checkerboard."""
    from .vision import calibrate_camera

    calibrate_camera(video, output, square_m=square_m)


@app.command("calibrate-arena")
def arena_command(image: Path, camera: Path, layout: Path, output: Path):
    """Locate the fixed camera from measured floor-marker coordinates."""
    from .vision import calibrate_arena

    calibrate_arena(image, camera, layout, output)


@app.command("arena-layout")
def arena_layout_command(
    output: Path,
    d12: Annotated[float, typer.Option(help="Marker 1 (start-left) to 2 (far-left), m")],
    d23: Annotated[float, typer.Option(help="Marker 2 (far-left) to 3 (far-right), m")],
    d34: Annotated[float, typer.Option(help="Marker 3 (far-right) to 4 (start-right), m")],
    d41: Annotated[float, typer.Option(help="Marker 4 (start-right) to 1 (start-left), m")],
    d13: Annotated[float, typer.Option(help="Diagonal 1 to 3, m")],
    d24: Annotated[float, typer.Option(help="Diagonal 2 to 4, m")],
    marker_size: float = 0.05,
):
    """Floor marker corners from taped distances between the same corner of each marker."""
    from .vision import rectangle_layout

    if output.exists():
        raise typer.BadParameter("Output already exists")
    layout = rectangle_layout(
        {"d12": d12, "d23": d23, "d34": d34, "d41": d41, "d13": d13, "d24": d24}, marker_size
    )
    write_json(output, layout)
    typer.echo(
        f"accepted={layout['accepted']} max_residual_m={layout['max_distance_residual_m']:.4f}"
    )


@app.command("calibrate-grid")
def calibrate_grid_command(
    image: Path,
    camera: Path,
    arena: Path,
    height_m: Annotated[float, typer.Option(help="Measured lens height above the floor, m")],
    spacing_m: float = 0.3,
    count: int = 4,
):
    """Camera and floor frame from a taped square grid when markers cannot be printed."""
    from .grid import calibrate_from_grid

    if camera.exists() or arena.exists():
        raise typer.BadParameter("Calibration output already exists")
    result = calibrate_from_grid(image, camera, arena, height_m, spacing_m, count)
    typer.echo(json.dumps(result, indent=2))


@app.command("check-distance")
def check_distance_command(track: Path, expected_m: float = 1.0):
    """Compare two resting robot positions from a tracked clip with a taped distance."""
    from .sync import read_jsonl
    from .vision import distance_check

    typer.echo(json.dumps(distance_check(read_jsonl(track), expected_m), indent=2))


@app.command("track")
def track_command(video: Path, camera: Path, arena: Path, config: Path, output: Path):
    """Track robot pose and LED intensity using actual video timestamps."""
    from .vision import track_video

    typer.echo(json.dumps(track_video(video, camera, arena, load_config(config), output)))


@app.command("analyze")
def analyze_command(
    run: Path,
    track: Path,
    anchors: Path | None = None,
    led_threshold: float = 30,
    pulse_group: int = 0,
):
    """Synchronize one measured run and calculate independent trajectory metrics."""
    from .sync import analyze_run

    typer.echo(json.dumps(analyze_run(run, track, anchors, led_threshold, pulse_group), indent=2))


@app.command("analyze-session")
def analyze_session(
    runs: Path,
    session_id: str,
    track: Path,
    led_threshold: float = 30,
):
    """Pair every filmed attempt of a session by LED timing; failures stay recorded."""
    from .sync import analyze_session_runs

    report = analyze_session_runs(
        runs, runs / f"session-{session_id}.json", track, led_threshold
    )
    for attempt in report["attempts"]:
        verdict = "OK" if attempt["accepted"] and attempt["quality_ok"] else "NG"
        residual = attempt.get("residual_s")
        detail = f"残差 {residual * 1000:.1f} ms" if residual is not None else attempt["reason"]
        typer.echo(f"[{verdict}] {attempt['directory']}: {detail}")


@app.command("track-session")
def track_session_command(
    video: Path,
    camera: Path,
    arena: Path,
    config: Path,
    runs: Path,
    session_id: str,
    output: Path,
    workers: int = 6,
):
    """Track only the time around each attempt of a session, in parallel processes."""
    import av

    from .sync import read_jsonl
    from .vision import track_windows

    start_wall = float(Path(str(video) + ".start").read_text())
    with av.open(str(video)) as container:
        duration = float(container.duration or 0) / 1_000_000
    record = json.loads((runs / f"session-{session_id}.json").read_text())
    windows = []
    for attempt in record["attempts"]:
        transport = runs / attempt["directory"] / "transport.jsonl"
        if not transport.exists():
            continue
        rows = read_jsonl(transport)
        if not rows:
            continue
        begin = rows[0]["wall_time"] - start_wall
        end = rows[-1]["wall_time"] - start_wall
        if begin < 0 or end > duration:
            continue  # this attempt was filmed in another video of the session
        windows.append((max(0.0, begin - 6.0), end + 6.0))
    result = track_windows(video, camera, arena, load_config(config), output, windows, workers)
    typer.echo(json.dumps(result))


@app.command("analyze-motion")
def analyze_motion_command(runs: Path, session_id: str, track: Path, video_start: Path, config: Path):
    """Pair a filmed session by encoder-versus-camera motion when the sync LED is not seen."""
    from .motion_sync import analyze_motion_session

    report = analyze_motion_session(
        runs,
        runs / f"session-{session_id}.json",
        track,
        float(video_start.read_text()),
        load_config(config),
    )
    for attempt in report["attempts"]:
        if attempt["accepted"]:
            typer.echo(
                f"[{'OK' if attempt['quality_ok'] else 'NG'}] {attempt['directory']}: "
                f"residual {attempt['residual_m'] * 1000:.1f} mm, scale {attempt['distance_scale']:.3f}"
            )
        else:
            typer.echo(f"[NG] {attempt['directory']}: {attempt['reason']}")


@app.command("validate")
def validate_command(runs: Path):
    """Check every trial before training; each failure carries its reasons."""
    from .validate import validate_runs

    report = validate_runs(runs)
    write_json(runs / "validation.json", report)
    for result in report["results"]:
        verdict = "OK" if result["accepted"] else "NG"
        typer.echo(f"[{verdict}] {result['trial']}: " + "; ".join(result["problems"]))
    typer.echo(
        f"合格 {report['accepted']}/{report['trials']}、重複ID {report['duplicate_trial_ids']}、"
        f"設定の種類 {report['config_hashes']}"
    )


@app.command("build-dataset")
def build_dataset_command(
    output: Path,
    runs: Annotated[list[Path], typer.Option(help="Directory of analyzed trials, repeatable")],
    rate: float = 10,
    window: float = 0.3,
):
    """Aligned 10 Hz frames with hindsight labels; gains come from train trials only."""
    from .dataset import build_dataset

    trials = [p for directory in runs for p in sorted(directory.iterdir()) if (p / "metadata.json").exists()]
    meta = build_dataset(trials, output, rate, window)
    typer.echo(json.dumps({k: meta[k] for k in ("frames", "labeled_frames", "excluded")}, indent=2))
    typer.echo(json.dumps(meta["measured_gain_from_train_split"], indent=2))


@app.command("report")
def report_command(runs: Path, output: Path, model: Path | None = None):
    """Create comparison PNG/PDF, per-trial CSV and a source-labeled report PDF."""
    from .report import make_report

    typer.echo(
        json.dumps(make_report(runs, output, load_model(model) if model else None), indent=2)
    )


@app.command("demo")
def demo_command(output: Path, video: bool = False, config: Path | None = None):
    """Verify the complete workflow on explicitly assumed simulation data."""
    from .report import make_report
    from .simulation import calibrate_simulated_motors, run_episode, schedule

    if output.exists():
        raise typer.BadParameter("Choose a new output directory")
    settings = load_config(config) if config else RobotConfig()
    if config is None:
        settings.right.motor_gain = 0.85
        settings.motor_gain_jitter_fraction = 0.002
    settings = calibrate_simulated_motors(settings, output / "motor-calibration")
    speeds = [0.6 * settings.max_speed_mps, 0.9 * settings.max_speed_mps]
    write_json(output / "assumed-config.json", settings.model_dump())
    fitted = None
    for split in ("train", "test"):
        jobs = schedule(speeds, split)
        write_json(output / f"{split}-schedule.json", jobs)
        for i, job in enumerate(jobs):
            run_episode(
                settings,
                output / split / job["trial_id"],
                mode=job["mode"],
                speed=job["speed_mps"],
                correction=job["correction"],
                split=split,
                model=fitted if split == "test" else None,
                seed=i
                if split == "train"
                else 10000 + (0 if job["speed_mps"] == speeds[0] else 5) + job["repeat"],
            )
            typer.echo(f"{split} {i + 1}/{len(jobs)}")
        if split == "train":
            fitted = train(read_records(output / "train"))
            save_model(fitted, output / "model.json")
    make_report(output / "test", output / "report", fitted)
    if video:
        for mode in ("pwm", "speed", "learned"):
            run_episode(
                settings,
                output / "video" / mode,
                mode=mode,
                speed=speeds[1],
                model=fitted if mode == "learned" else None,
                record_video=True,
            )
    write_json(
        output / "evidence.json",
        {
            "source": "simulated",
            "measured_robot_verified": False,
            "model_id": fitted.model_id(),
            "train_runs": 30,
            "test_runs": 30,
            "cad_visual": settings.cad_visual.model_dump() if settings.cad_visual else None,
        },
    )


@app.command("package-submission")
def package_command(
    output: Path,
    model: Path | None = None,
    report: Path | None = None,
    training_runs: Path | None = None,
):
    """Prepare an allowlisted snapshot for review; never publish or upload it."""
    from .submission import package_submission

    project = Path(__file__).resolve().parents[2]
    result = package_submission(
        project, output, load_model(model) if model else None, report, training_runs
    )
    typer.echo(f"Prepared {len(result['code_files'])} code files; publication requires review")


@app.command("make-video")
def video_command(manifest: Path, output: Path, font: Path | None = None):
    """Assemble a captioned 1..3 minute MP4 with explicit 2x and simulation labels."""
    from .video import make_video

    make_video(manifest, output, font)


@app.command("make-narrated-video")
def narrated_video_command(
    manifest: Path,
    output: Path,
    work: Path,
    font: Path = Path("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"),
    bold: Path = Path("/System/Library/Fonts/ヒラギノ角ゴシック W7.ttc"),
    url: str = "http://127.0.0.1:50021",
    speaker: int = 3,
    speed: float = 1.1,
):
    """Assemble a VOICEVOX-narrated MP4 (180 s max) from clips, figures and text cards."""
    from .narration import make_narrated_video

    result = make_narrated_video(manifest, output, work, font, bold, url, speaker, speed)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
