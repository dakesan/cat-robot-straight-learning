"""ESP32 student: a 5-16-16-1 tanh network over encoder-only inputs, exported to C++.

The same network is trained either directly on hindsight labels or on a teacher's
outputs (distillation), with identical inputs, architecture and trial split, so the two
can be compared. Inputs follow catrobot.dataset.STUDENT_FEATURES.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pydantic import Field
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor

from .config import Settings, write_json
from .dataset import ELAPSED_CAP_S, STUDENT_FEATURES

HIDDEN = 16


class StudentModel(Settings):
    version: int = 1
    source: str
    target: str
    config_hash: str
    min_speed_mps: float = Field(gt=0)
    max_speed_mps: float = Field(gt=0)
    min_correction: float = Field(ge=-0.2, le=0.2)
    max_correction: float = Field(ge=-0.2, le=0.2)
    track_m: float = Field(gt=0)
    elapsed_cap_s: float = ELAPSED_CAP_S
    mean: list[float]
    scale: list[float]
    w1: list[list[float]]
    b1: list[float]
    w2: list[list[float]]
    b2: list[float]
    w3: list[float]
    b3: float
    training_trials: list[str]
    cv_rmse: float | None = None
    cv_rmse_speed_median: float | None = None
    alpha: float = 1e-3

    def model_id(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        x = (np.atleast_2d(inputs) - self.mean) / self.scale
        h1 = np.tanh(x @ np.asarray(self.w1).T + self.b1)
        h2 = np.tanh(h1 @ np.asarray(self.w2).T + self.b2)
        out = h2 @ np.asarray(self.w3) + self.b3
        return np.clip(out, self.min_correction, self.max_correction)

    def native_weights(self) -> dict:
        return {
            "measured": self.source == "measured",
            "min_speed": self.min_speed_mps,
            "max_speed": self.max_speed_mps,
            "min_correction": self.min_correction,
            "max_correction": self.max_correction,
            "track_m": self.track_m,
            "elapsed_cap_s": self.elapsed_cap_s,
            **{k: getattr(self, k) for k in ("mean", "scale", "w1", "b1", "w2", "b2", "w3", "b3")},
        }


def read_frames(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def fit_network(x: np.ndarray, y: np.ndarray, seed: int = 0, alpha: float = 1e-3) -> MLPRegressor:
    network = MLPRegressor(
        hidden_layer_sizes=(HIDDEN, HIDDEN),
        activation="tanh",
        alpha=alpha,
        learning_rate_init=3e-3,
        max_iter=4000,
        random_state=seed,
    )
    return network.fit(x, y)


def train_student(
    frames: list[dict],
    target: str,
    source: str,
    config_hash: str,
    track_m: float,
    trials: set[str] | None = None,
    alpha: float = 1e-3,
) -> StudentModel:
    """Fit on train-split frames with a valid target; inputs never include video data."""
    chosen = [
        r
        for r in frames
        if r["split"] == "train"
        and float(r["speed_mps"]) > 0  # the student only runs while a speed is commanded
        and r.get(target) not in (None, "", "None")
        and (trials is None or r["trial_id"] in trials)
        and (target != "label_correction" or r["label_valid"] == "True")
    ]
    if len({r["trial_id"] for r in chosen}) < 4:
        raise ValueError("Need frames from at least four training trials")
    x = np.array([[float(r[k]) for k in STUDENT_FEATURES] for r in chosen])
    y = np.array([float(r[target]) for r in chosen])
    groups = np.array([r["trial_id"] for r in chosen])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (x - mean) / scale
    # Corrections are about 0.01; unscaled, the loss changes by less than the solver's
    # tolerance and training stops after a few dozen epochs. Fit standardized targets and
    # fold the scale back into the output layer.
    y_mean, y_scale = float(y.mean()), float(max(y.std(), 1e-6))
    scaled = (y - y_mean) / y_scale
    errors, baseline = [], []
    for train_index, test_index in GroupKFold(n_splits=min(5, len(set(groups)))).split(x, y, groups):
        network = fit_network(normalized[train_index], scaled[train_index], alpha=alpha)
        predicted = network.predict(normalized[test_index]) * y_scale + y_mean
        errors.append((predicted - y[test_index]) ** 2)
        # Baseline: the training folds' median label at each speed.
        for index in test_index:
            same = [i for i in train_index if x[i, 0] == x[index, 0]]
            baseline.append((np.median(y[same]) - y[index]) ** 2 if same else np.nan)
    network = fit_network(normalized, scaled, alpha=alpha)
    low, high = np.percentile(y, [1, 99])
    speeds = x[:, 0]
    return StudentModel(
        source=source,
        target=target,
        config_hash=config_hash,
        min_speed_mps=float(speeds.min()),
        max_speed_mps=float(speeds.max()),
        min_correction=float(max(-0.2, low)),
        max_correction=float(min(0.2, high)),
        track_m=track_m,
        mean=mean.tolist(),
        scale=scale.tolist(),
        w1=network.coefs_[0].T.tolist(),
        b1=network.intercepts_[0].tolist(),
        w2=network.coefs_[1].T.tolist(),
        b2=network.intercepts_[1].tolist(),
        w3=(network.coefs_[2][:, 0] * y_scale).tolist(),
        b3=float(network.intercepts_[2][0] * y_scale + y_mean),
        training_trials=sorted(set(groups)),
        cv_rmse=float(np.sqrt(np.mean(np.concatenate(errors)))),
        cv_rmse_speed_median=float(np.sqrt(np.nanmean(baseline))),
        alpha=alpha,
    )


def holdout_comparison(
    frames: list[dict],
    truth: list[dict],
    validation: set[str],
    targets: list[str],
    track_m: float,
    alpha: float = 1e-3,
) -> dict:
    """Fit one student per target on the trials outside `validation` and score each on the
    validation trials against `truth`'s hindsight labels, joined on trial and time.

    `truth_correction` is offered as a target: `truth`'s labels on the training trials.
    """

    def key(row):
        return row["trial_id"], row["t_s"]

    reference = {key(r): float(r["label_correction"]) for r in truth if r["label_valid"] == "True"}
    rows = [dict(r, truth_correction=str(reference.get(key(r), ""))) for r in frames if r["split"] == "train"]
    training = [r for r in rows if r["trial_id"] not in validation]
    held = [r for r in rows if r["trial_id"] in validation and key(r) in reference]
    if not held:
        raise ValueError("No validation frame has a reference label")
    x = np.array([[float(r[k]) for k in STUDENT_FEATURES] for r in held])
    y = np.array([reference[key(r)] for r in held])

    def rmse(predicted) -> float:
        return float(np.sqrt(np.mean((np.asarray(predicted) - y) ** 2)))

    by_speed: dict[float, list[float]] = {}
    for r in training:
        if key(r) in reference:
            by_speed.setdefault(float(r["speed_mps"]), []).append(reference[key(r)])
    result = {
        "validation_trials": sorted(validation),
        "validation_frames": len(held),
        "speed_median_rmse": rmse([np.median(by_speed[float(r["speed_mps"])]) for r in held]),
        "zero_rmse": rmse(np.zeros_like(y)),
        "students": {},
    }
    for target in targets:
        model = train_student(training, target, "measured", "comparison", track_m, alpha=alpha)
        result["students"][target] = {
            "rmse": rmse(model.predict(x)),
            "training_trials": len(model.training_trials),
            "training_cv_rmse": model.cv_rmse,
        }
    taught = [i for i, r in enumerate(held) if r.get("teacher_correction") not in (None, "")]
    if taught:
        teacher = np.array([float(held[i]["teacher_correction"]) for i in taught])
        result["teacher"] = {
            "rmse": float(np.sqrt(np.mean((teacher - y[taught]) ** 2))),
            "frames": len(taught),
        }
    return result


def export_student_header(model: StudentModel, path: Path) -> None:
    if model.source != "measured":
        raise ValueError("Only students trained on measured runs may be exported for firmware")

    def floats(values):
        return ", ".join(f"{v:.9e}f" for v in values)

    lines = [
        "#pragma once",
        '#include "drive.h"',
        f"// Student ({model.target}) from measured runs; inputs are encoder-only.",
        "inline robot::StudentModel deployedStudent() {",
        "  robot::StudentModel m;",
        "  m.fitted = true; m.measured = true;",
        f"  m.minSpeed = {model.min_speed_mps:.9e}f; m.maxSpeed = {model.max_speed_mps:.9e}f;",
        f"  m.minCorrection = {model.min_correction:.9e}f; m.maxCorrection = {model.max_correction:.9e}f;",
        f"  m.trackM = {model.track_m:.9e}f; m.elapsedCap = {model.elapsed_cap_s:.9e}f;",
        f"  const float mean[] = {{{floats(model.mean)}}}, scale[] = {{{floats(model.scale)}}};",
        "  for (int i = 0; i < robot::StudentModel::INPUTS; ++i) { m.mean[i] = mean[i]; m.scale[i] = scale[i]; }",
        f"  const float w1[] = {{{floats(np.ravel(model.w1))}}};",
        f"  const float w2[] = {{{floats(np.ravel(model.w2))}}};",
        f"  const float b1[] = {{{floats(model.b1)}}}, b2[] = {{{floats(model.b2)}}}, w3[] = {{{floats(model.w3)}}};",
        "  constexpr int I = robot::StudentModel::INPUTS, H = robot::StudentModel::HIDDEN;",
        "  for (int j = 0; j < H; ++j) {",
        "    for (int i = 0; i < I; ++i) m.w1[j][i] = w1[j * I + i];",
        "    for (int i = 0; i < H; ++i) m.w2[j][i] = w2[j * H + i];",
        "    m.b1[j] = b1[j]; m.b2[j] = b2[j]; m.w3[j] = w3[j];",
        "  }",
        f"  m.b3 = {model.b3:.9e}f;",
        "  return m;",
        "}",
        f'inline const char* deployedStudentId() {{ return "{model.model_id()}"; }}',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def save_student(model: StudentModel, path: Path) -> None:
    write_json(path, model.model_dump())


def load_student(path: Path) -> StudentModel:
    return StudentModel.model_validate(json.loads(path.read_text()))
