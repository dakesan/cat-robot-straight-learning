"""MuJoCo plant driven by the same C++ safety, PI and regression code as the ESP32."""

import json
import math
import random
from fractions import Fraction
from pathlib import Path

import mujoco
import numpy as np

from ._native import Controller, Mode, source_sha
from .cad import visual_bundle
from .config import RobotConfig, write_json
from .metrics import trajectory_metrics
from .model import CorrectionModel
from .rendering import follow_camera, inspect_simulation_video, status_frame


def robot_xml(config: RobotConfig) -> str:
    assets, visuals = [], []
    if config.cad_visual is not None:
        _, meshes = visual_bundle(config.cad_visual)
        position = " ".join(str(v) for v in config.cad_visual.position_m)
        quaternion = " ".join(str(v) for v in config.cad_visual.quaternion_wxyz)
        for i, file in enumerate(meshes):
            assets.append(f'<mesh name="cad_mesh_{i}" file="{file}"/>')
            color = ("0.85 0.35 0.2 1", "0.9 0.65 0.2 1", "0.18 0.5 0.7 1")[i % 3]
            visuals.append(
                f'<geom name="cad_solid_{i}" type="mesh" mesh="cad_mesh_{i}" '
                f'pos="{position}" quat="{quaternion}" rgba="{color}" '
                'mass="0" contype="0" conaffinity="0" group="2"/>'
            )
    proxy_alpha = 0 if visuals else 1
    caster_xml = []
    for i, caster in enumerate(config.resolved_ball_casters()):
        name = "caster" if config.ball_casters is None else f"caster_{i}"
        position = " ".join(str(v) for v in caster.position_m)
        caster_xml.append(f"""
          <body name="{name}" pos="{position}">
            <joint name="{name}_joint" type="ball" damping="{caster.joint_damping}"/>
            <geom name="{name}_ball" type="sphere" size="{caster.radius_m}"
                  mass="{caster.mass_kg}" rgba="0.6 0.65 0.7 1"/>
          </body>""")
    wheel_xml = []
    for name, side in (("left", 1), ("right", -1)):
        wheel = getattr(config, name)
        toe = math.radians(wheel.toe_deg) / 2
        radius = wheel.radius_m * wheel.contact_radius_scale
        wheel_xml.append(f"""
        <body name="{name}" pos="0 {side * config.track_m / 2} -0.03"
              quat="{math.cos(toe)} 0 0 {math.sin(toe)}">
          <joint name="{name}_joint" type="hinge" axis="0 1 0" damping="0.00001"/>
          <geom name="{name}_wheel" type="cylinder" size="{radius} 0.008"
                quat="0.70710678 0.70710678 0 0" mass="0.03" rgba="0.1 0.1 0.1 1"/>
          <geom name="{name}_rotation_mark" type="box" pos="0 {side * 0.0084} 0"
                size="{radius * 0.7} 0.0003 0.0015" mass="0" contype="0" conaffinity="0"
                rgba="0.92 0.92 0.92 1" group="2"/>
        </body>""")
    return f"""<mujoco model="catrobot_assumed_physics">
      <compiler angle="radian"/>
      <option timestep="0.001" integrator="implicitfast"/>
      <visual><global offwidth="960" offheight="720"/></visual>
      <asset>
        <texture name="floor_grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1="0.24 0.24 0.24" rgb2="0.36 0.36 0.36"/>
        <material name="floor_material" texture="floor_grid" texuniform="true" texrepeat="5 5"/>
        {"".join(assets)}
      </asset>
      <default><geom friction="{config.floor_friction} 0.005 0.0001" condim="6"
                    solref="0.008 1"/></default>
      <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="3 3 0.1" material="floor_material"/>
        <camera name="overhead" pos="0.6 0 2.1"/>
        <body name="chassis" pos="0 0 0.055">
          <freejoint/>
          <geom name="chassis_proxy" type="box" size="{config.length_m / 2} {config.width_m / 2} 0.015"
                mass="{config.mass_kg}" rgba="0.15 0.4 0.7 {proxy_alpha}"/>
          {"".join(visuals)}
          <geom type="box" pos="0.06 0 0.02" size="0.018 0.012 0.002" mass="0.001"
                contype="0" conaffinity="0" rgba="1 1 1 1"/>
          {"".join(wheel_xml)}
          {"".join(caster_xml)}
        </body>
      </worldbody>
    </mujoco>"""


def run_episode(
    config: RobotConfig,
    directory: Path,
    *,
    mode: str = "pwm",
    speed: float = 0.1,
    correction: float = 0,
    split: str = "test",
    model: CorrectionModel | None = None,
    record_video: bool = False,
    seed: int = 0,
    pwm_override: int | None = None,
    max_seconds: float = 25,
) -> dict:
    if directory.exists():
        raise ValueError(f"Trial already exists: {directory}")
    if mode not in ("pwm", "speed", "learned") or split not in ("train", "test"):
        raise ValueError("Unknown mode or split")
    if not 0 < speed <= config.max_speed_mps or not -0.2 <= correction <= 0.2:
        raise ValueError("Requested speed/correction outside calibration bounds")
    if mode == "learned" and (model is None or model.config_hash != config.fingerprint()):
        raise ValueError("A correction model matching this robot configuration is required")
    directory.mkdir(parents=True)
    xml = robot_xml(config)
    (directory / "robot.xml").write_text(xml)
    assets = {}
    if config.cad_visual is not None:
        manifest, assets = visual_bundle(config.cad_visual)
        write_json(directory / "cad/manifest.json", manifest)
        for name, data in assets.items():
            (directory / name).write_bytes(data)
    world = mujoco.MjModel.from_xml_string(xml, assets)
    state = mujoco.MjData(world)
    chassis = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    joint_ids = [
        mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_JOINT, s + "_joint") for s in ("left", "right")
    ]
    qpos = [world.jnt_qposadr[j] for j in joint_ids]
    dofs = [world.jnt_dofadr[j] for j in joint_ids]
    for _ in range(400):
        for i, side in enumerate(("left", "right")):
            wheel = getattr(config, side)
            inertia = (config.mass_kg / 2 + 0.015) * wheel.radius_m**2
            state.qfrc_applied[dofs[i]] = -inertia / wheel.time_constant_s * state.qvel[dofs[i]]
        mujoco.mj_step(world, state)
    origin = state.xpos[chassis].copy()
    control = Controller()
    control.calibration = config.native_calibration()
    if model is not None:
        model.attach(control)
    control.arm(0, 8000)
    gain_draw = np.random.default_rng(seed).normal(1, config.motor_gain_jitter_fraction, 2)
    pwm = round(
        (config.left.deadband_percent + config.right.deadband_percent) / 2
        + (config.left.feedforward + config.right.feedforward) / 2 * speed
    )
    pwm = min(50, pwm) if pwm_override is None else pwm_override
    if not 0 <= pwm <= 50:
        raise ValueError("PWM outside existing safety bounds")
    poses, samples = [], []
    renderer = encoder = stream = camera = None
    if record_video:
        import av

        renderer = mujoco.Renderer(world, height=720, width=960)
        camera = follow_camera(world)
        encoder = av.open(str(directory / "simulation.mp4"), "w", options={"movflags": "+faststart"})
        stream = encoder.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 960, 720, "yuv420p"
    counts = [0, 0]
    initial_counts = [
        round(state.qpos[qpos[i]] / (2 * math.pi) * getattr(config, side).counts_per_revolution)
        * getattr(config, side).encoder_sign
        for i, side in enumerate(("left", "right"))
    ]
    next_video = 0
    termination = "duration_limit"
    try:
        for step in range(round(max_seconds * 1000)):
            now = step
            for i, side in enumerate(("left", "right")):
                wheel = getattr(config, side)
                counts[i] = (
                    round(state.qpos[qpos[i]] / (2 * math.pi) * wheel.counts_per_revolution)
                    * wheel.encoder_sign
                )
            if now % 100 == 0:
                accepted = control.drive(
                    now,
                    control.state()["epoch"],
                    now // 100 + 1,
                    getattr(Mode, mode),
                    speed,
                    correction,
                    pwm,
                )
                if not accepted:
                    termination = control.state()["reason"]
                    break
            if now % 10 == 0:
                control.tick(now, counts[0], counts[1], 8000)
            output = control.state()
            torque = []
            for i, side in enumerate(("left", "right")):
                wheel = getattr(config, side)
                percent = output[f"pwm_{side}"] * 100 / 255
                desired = (
                    max(0, percent - wheel.motor_deadband_percent)
                    * wheel.motor_velocity_per_percent
                )
                desired = desired * wheel.motor_gain * gain_draw[i] / wheel.radius_m
                effective_inertia = (config.mass_kg / 2 + 0.015) * wheel.radius_m**2
                value = effective_inertia / wheel.time_constant_s * (desired - state.qvel[dofs[i]])
                value = float(np.clip(value, -wheel.max_torque_nm, wheel.max_torque_nm))
                state.qfrc_applied[dofs[i]] = value
                torque.append(value)
            mujoco.mj_step(world, state)
            if now % 20 == 0:
                matrix = state.xmat[chassis].reshape(3, 3)
                pose = {
                    "time_s": now / 1000,
                    "x_m": float(state.xpos[chassis, 0] - origin[0]),
                    "y_m": float(state.xpos[chassis, 1] - origin[1]),
                    "yaw_rad": math.atan2(matrix[1, 0], matrix[0, 0]),
                    "valid": True,
                }
                poses.append(pose)
                samples.append(
                    dict(
                        output,
                        time_ms=now,
                        sample=len(samples) + 1,
                        encoder_left=counts[0],
                        encoder_right=counts[1],
                        simulated_torque_left_nm=torque[0],
                        simulated_torque_right_nm=torque[1],
                    )
                )
                odometry = (
                    abs((counts[0] - initial_counts[0]) * config.left.meters_per_count)
                    + abs((counts[1] - initial_counts[1]) * config.right.meters_per_count)
                ) / 2
                if odometry >= 1.3 or not output["armed"]:
                    termination = "distance_limit" if odometry >= 1.3 else output["reason"]
                    break
            if renderer is not None and step * 30 >= next_video * 1000:
                import av

                camera.lookat[:] = state.xpos[chassis]
                renderer.update_scene(state, camera=camera)
                image = status_frame(
                    renderer.render(), now / 1000, float(state.xpos[chassis, 0] - origin[0]), speed
                )
                frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = next_video, Fraction(1, 30)
                for packet in stream.encode(frame):
                    encoder.mux(packet)
                next_video += 1
    finally:
        control.stop()
        if renderer is not None:
            for packet in stream.encode():
                encoder.mux(packet)
            encoder.close()
            renderer.close()
    metadata = {
        "trial_id": directory.name,
        "split": split,
        "source": "simulated",
        "mode": mode,
        "speed_mps": speed,
        "correction": float(samples[-1]["correction"]) if samples else correction,
        "config_hash": config.fingerprint(),
        "physics_calibrated": config.physics_calibrated,
        "model_id": model.model_id() if mode == "learned" else None,
        "seed": seed,
        "native_control_sha": source_sha,
        "motor_gain_draw": gain_draw.tolist(),
        "stop_reason": termination,
        "collision_model": "primitive_proxy",
        "cad_visual": config.cad_visual.model_dump() if config.cad_visual else None,
        "ball_caster_count": len(config.resolved_ball_casters()),
    }
    metrics = dict(metadata, **trajectory_metrics(poses))
    write_json(directory / "metadata.json", metadata)
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "config.json", config.model_dump())
    for filename, rows in (("poses.jsonl", poses), ("telemetry.jsonl", samples)):
        (directory / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
    if record_video:
        check = inspect_simulation_video(directory / "simulation.mp4", directory / "poster.png")
        write_json(directory / "video-check.json", check)
        if not check["robot_visible_throughout"]:
            raise ValueError("Encoded video does not show the robot clearly; inspect video-check.json")
    return metrics


def calibrate_simulated_motors(config: RobotConfig, output: Path) -> RobotConfig:
    """Identify loaded wheel response independently of the controller calibration."""
    velocities = {"left": [], "right": []}
    for percent in (25, 35, 45):
        directory = output / f"pwm-{percent}"
        run_episode(
            config,
            directory,
            mode="pwm",
            speed=min(0.1, config.max_speed_mps),
            pwm_override=percent,
            max_seconds=5,
        )
        samples = [
            json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()
        ]
        samples = [s for s in samples if s["time_ms"] >= 1500 and s["armed"]]
        if len(samples) < 20:
            raise ValueError("Simulation motor calibration did not produce steady data")
        times = np.array([s["time_ms"] / 1000 for s in samples])
        for side in ("left", "right"):
            wheel = getattr(config, side)
            position = (
                np.array([s[f"encoder_{side}"] for s in samples])
                * wheel.meters_per_count
                * wheel.encoder_sign
            )
            slope = np.polyfit(times - times[0], position, 1)[0]
            velocities[side].append((percent, slope))
    result = config.model_copy(deep=True)
    for side, measurements in velocities.items():
        percent, velocity = np.array(measurements).T
        slope, intercept = np.polyfit(percent, velocity, 1)
        if slope <= 0 or not 0 <= -intercept / slope < 50:
            raise ValueError(
                "Loaded motor calibration cannot be represented by positive feedforward"
            )
        wheel = getattr(result, side)
        wheel.feedforward, wheel.deadband_percent = float(1 / slope), float(-intercept / slope)
    result.max_speed_mps = 0.9 * min(
        (45 - getattr(result, side).deadband_percent) / getattr(result, side).feedforward
        for side in ("left", "right")
    )
    result.calibration_notes += (
        "; controller feedforward identified from three simulated loaded PWM trials"
    )
    write_json(output / "fitted-config.json", result.model_dump())
    return result


def schedule(
    speeds: list[float],
    split: str,
    seed: int = 20261029,
    center: float = 0.0,
    step: float = 0.02,
) -> list[dict]:
    """Randomized trials. "probe" runs PI at zero correction to locate the bias first;
    "train" sweeps five corrections center + step * (-2..2) around it; "device" compares
    PI with the flashed Ridge and student at the higher speed only."""
    if len(speeds) != 2 or len(set(speeds)) != 2 or min(speeds) <= 0:
        raise ValueError("Specify two distinct positive speeds")
    if split not in ("probe", "train", "test", "device") or step <= 0:
        raise ValueError("Split must be probe, train, test or device, with a positive step")
    if abs(center) + 2 * step > 0.2:
        raise ValueError("Correction sweep exceeds the firmware limit of +-0.2")
    jobs = []
    modes = {"test": ("pwm", "speed", "learned"), "device": ("speed", "learned", "student")}.get(split, ("speed",))
    if split == "device":
        speeds = [max(speeds)]
    if split == "train":
        deltas = tuple(round(center + step * k, 6) for k in (-2, -1, 0, 1, 2))
    else:
        deltas = (0,)
    repeats = {"probe": 2, "train": 3, "test": 5, "device": 5}[split]
    for mode in modes:
        for speed in speeds:
            for delta in deltas:
                for repeat in range(repeats):
                    jobs.append(
                        {
                            "mode": mode,
                            "speed_mps": speed,
                            "correction": delta,
                            "repeat": repeat,
                            "split": split,
                        }
                    )
    random.Random(seed).shuffle(jobs)
    for i, job in enumerate(jobs):
        job["trial_id"] = f"{split}-{i + 1:03d}"
    return jobs
