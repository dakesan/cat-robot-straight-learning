"""Small supervised correction models with portable C++ inference."""

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings, write_json


class CorrectionModel(Settings):
    version: int = 1
    source: Literal["simulated", "measured"]
    config_hash: str
    min_speed_mps: float = Field(gt=0)
    max_speed_mps: float = Field(gt=0)
    min_correction: float = Field(ge=-0.2, le=0.2)
    max_correction: float = Field(ge=-0.2, le=0.2)
    mean: list[float] = Field(min_length=4, max_length=4)
    scale: list[float] = Field(min_length=4, max_length=4)
    coefficients: list[list[float]] = Field(min_length=2, max_length=2)
    intercept: list[float] = Field(min_length=2, max_length=2)
    training_runs: list[str]
    alpha: float
    cv_loss: float

    def model_id(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]

    def attach(self, controller):
        controller.set_model(
            self.mean,
            self.scale,
            self.coefficients,
            self.intercept,
            self.min_speed_mps,
            self.max_speed_mps,
            self.source == "measured",
            self.min_correction,
            self.max_correction,
        )

    def predict(self, speed: float, correction: float) -> np.ndarray:
        x = (features(np.array([speed]), np.array([correction])) - self.mean) / self.scale
        return (x @ np.asarray(self.coefficients).T + self.intercept)[0]

    def choose(self, speed: float) -> float:
        if not self.min_speed_mps <= speed <= self.max_speed_mps:
            raise ValueError("Requested speed is outside the training interval")
        deltas = np.arange(-200, 201) / 1000
        deltas = deltas[
            (deltas >= self.min_correction - 1e-6) & (deltas <= self.max_correction + 1e-6)
        ]
        predictions = np.array([self.predict(speed, float(d)) for d in deltas])
        loss = (predictions[:, 0] / 0.05) ** 2 + (predictions[:, 1] / np.deg2rad(5)) ** 2
        loss += deltas**2 * 1e-5
        return float(deltas[np.argmin(loss)])


def features(speed: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return np.column_stack([speed, delta, speed * delta, delta * delta])


def load_model(path: Path) -> CorrectionModel:
    model = CorrectionModel.model_validate_json(path.read_text())
    if any(x <= 0 for x in model.scale) or any(len(x) != 4 for x in model.coefficients):
        raise ValueError("Invalid feature dimensions or normalization")
    return model


def train(records: list[dict]) -> CorrectionModel:
    if any(r["split"] != "train" for r in records):
        raise ValueError("Training may consume only runs assigned to the train split")
    if len({r["trial_id"] for r in records}) != len(records):
        raise ValueError("Duplicate trial IDs would leak repeated runs into cross-validation")
    if len({r["source"] for r in records}) != 1 or len({r["config_hash"] for r in records}) != 1:
        raise ValueError("Do not mix simulated/measured data or different robot calibrations")
    usable = [r for r in records if r["reached"] and r["quality_ok"] and r["mode"] == "speed"]
    if len(usable) < 10 or len({r["speed_mps"] for r in usable}) < 2:
        raise ValueError("Need at least 10 valid speed-mode runs covering two speeds")
    speed = np.array([r["speed_mps"] for r in usable])
    x = features(speed, np.array([r["correction"] for r in usable]))
    y = np.array([[r["lateral_at_goal_m"], r["heading_at_goal_rad"]] for r in usable])
    groups = np.array([r["trial_id"] for r in usable])
    folds = list(GroupKFold(n_splits=min(5, len(usable))).split(x, y, groups))
    scores = []
    for alpha in (0.01, 0.1, 1.0, 10.0):
        losses = []
        for training, validation in folds:
            pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            pipeline.fit(x[training], y[training])
            error = (pipeline.predict(x[validation]) - y[validation]) / [0.05, np.deg2rad(5)]
            losses.append(float(np.mean(error**2)))
        scores.append((float(np.mean(losses)), alpha))
    loss, alpha = min(scores)
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    pipeline.fit(x, y)
    scaler, regressor = pipeline.steps[0][1], pipeline.steps[1][1]
    return CorrectionModel(
        source=usable[0]["source"],
        config_hash=usable[0]["config_hash"],
        min_speed_mps=float(speed.min()),
        max_speed_mps=float(speed.max()),
        min_correction=min(r["correction"] for r in usable),
        max_correction=max(r["correction"] for r in usable),
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coefficients=regressor.coef_.tolist(),
        intercept=regressor.intercept_.tolist(),
        training_runs=groups.tolist(),
        alpha=alpha,
        cv_loss=loss,
    )


def export_header(model: CorrectionModel, path: Path) -> None:
    if model.source != "measured":
        raise ValueError("Only models trained on measured runs may be exported for real firmware")

    def floats(values):
        return ", ".join(f"{v:.9e}f" for v in values)

    lines = [
        "#pragma once",
        '#include "drive.h"',
        "// Generated from measured runs; camera pose is never an inference input.",
        "inline robot::CorrectionModel deployedModel() {",
        "  robot::CorrectionModel m;",
        "  m.fitted = true; m.measured = true;",
        f"  m.minSpeed = {model.min_speed_mps:.9e}f; m.maxSpeed = {model.max_speed_mps:.9e}f;",
        f"  m.minCorrection = {model.min_correction:.9e}f; m.maxCorrection = {model.max_correction:.9e}f;",
    ]
    for name, values in (("mean", model.mean), ("scale", model.scale)):
        for i, value in enumerate(values):
            lines.append(f"  m.{name}[{i}] = {value:.9e}f;")
    for j in range(2):
        for i, value in enumerate(model.coefficients[j]):
            lines.append(f"  m.coefficients[{j}][{i}] = {value:.9e}f;")
        lines.append(f"  m.intercept[{j}] = {model.intercept[j]:.9e}f;")
    lines += [
        "  return m;",
        "}",
        f'inline const char* deployedModelId() {{ return "{model.model_id()}"; }}',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_records(directory: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*/metrics.json"))]


def save_model(model: CorrectionModel, path: Path):
    write_json(path, model.model_dump())
