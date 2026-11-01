"""Validated physical assumptions and measured calibration."""

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


class Wheel(Settings):
    radius_m: float = Field(default=0.0215, gt=0, le=0.2)
    counts_per_revolution: float = Field(default=1320, gt=0)
    encoder_sign: Literal[-1, 1] = 1
    feedforward: float = Field(default=180, gt=0)
    deadband_percent: float = Field(default=8, ge=0, lt=50)
    motor_gain: float = Field(default=1, gt=0)
    motor_velocity_per_percent: float = Field(default=1 / 180, gt=0)
    motor_deadband_percent: float = Field(default=8, ge=0, lt=50)
    time_constant_s: float = Field(default=0.08, gt=0)
    toe_deg: float = Field(default=0, ge=-15, le=15)
    contact_radius_scale: float = Field(default=1, ge=0.8, le=1.2)
    max_torque_nm: float = Field(default=0.04, gt=0)
    response_fit_rms_m: float | None = Field(default=None, ge=0)

    @property
    def meters_per_count(self) -> float:
        return 2 * math.pi * self.radius_m / self.counts_per_revolution


class CadVisual(Settings):
    asset: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float] = (1, 0, 0, 0)
    alignment: Literal["assumed", "orientation_confirmed", "measured"] = "assumed"

    @field_validator("quaternion_wxyz")
    @classmethod
    def unit_quaternion(cls, value):
        if not math.isclose(sum(x * x for x in value), 1, abs_tol=1e-6):
            raise ValueError("CAD rotation quaternion must have unit length")
        return value


class BallCaster(Settings):
    position_m: tuple[float, float, float]
    radius_m: float = Field(default=0.01, gt=0)
    mass_kg: float = Field(default=0.01, gt=0)
    joint_damping: float = Field(default=0.00001, ge=0)


class RobotConfig(Settings):
    provenance: Literal["assumed", "measured"] = "assumed"
    calibration_notes: str = "Example assumptions; measure the actual robot before real driving."
    physics_calibrated: bool = False
    left: Wheel = Field(default_factory=Wheel)
    right: Wheel = Field(default_factory=Wheel)
    track_m: float = Field(default=0.14, gt=0)
    length_m: float = Field(default=0.18, gt=0)
    width_m: float = Field(default=0.12, gt=0)
    mass_kg: float = Field(default=0.45, gt=0)
    caster_x_m: float = -0.06
    floor_friction: float = Field(default=0.8, gt=0)
    motor_gain_jitter_fraction: float = Field(default=0, ge=0, le=0.1)
    kp: float = Field(default=100, ge=0)
    ki: float = Field(default=100, ge=0)
    max_speed_mps: float = Field(default=0.3, gt=0, le=1)
    marker_size_m: float = Field(default=0.05, gt=0)
    marker_height_m: float = Field(default=0.07, ge=0)
    marker_offset_x_m: float = 0
    marker_offset_y_m: float = 0
    marker_yaw_deg: float = 0
    marker_id: int = Field(default=0, ge=0, lt=50)
    led_offset: tuple[float, float] = (1.2, 0.15)
    cad_visual: CadVisual | None = None
    ball_casters: list[BallCaster] | None = None

    @field_validator("led_offset")
    @classmethod
    def finite_offset(cls, value):
        if not all(math.isfinite(x) for x in value):
            raise ValueError("LED offset must be finite")
        return value

    def native_calibration(self):
        from ._native import Calibration

        result = Calibration()
        result.calibrated = True
        for side in ("left", "right"):
            wheel = getattr(self, side)
            setattr(result, f"{side}_meters_per_count", wheel.meters_per_count)
            setattr(result, f"{side}_sign", wheel.encoder_sign)
            setattr(result, f"{side}_feedforward", wheel.feedforward)
            setattr(result, f"{side}_deadband", wheel.deadband_percent)
        result.kp, result.ki, result.max_speed = self.kp, self.ki, self.max_speed_mps
        return result

    def require_measured(self):
        if self.provenance != "measured":
            raise ValueError(
                "Real driving requires a measured configuration, not example assumptions"
            )

    def resolved_ball_casters(self) -> list[BallCaster]:
        if self.ball_casters is not None:
            return self.ball_casters
        return [BallCaster(position_m=(self.caster_x_m, 0, -0.045))]

    def fingerprint(self) -> str:
        excluded = {"calibration_notes"}
        if self.cad_visual is None:
            excluded.add("cad_visual")
        if self.ball_casters is None:
            excluded.add("ball_casters")
        else:
            excluded.add("caster_x_m")
        return hashlib.sha256(
            self.model_dump_json(exclude=excluded).encode()
        ).hexdigest()


def load_config(path: Path) -> RobotConfig:
    return RobotConfig.model_validate_json(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
