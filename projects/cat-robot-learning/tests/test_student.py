import subprocess
from pathlib import Path

import numpy as np
import pytest
from catrobot._native import Controller

from catrobot.dataset import STUDENT_FEATURES
from catrobot.student import export_student_header, holdout_comparison, train_student

SKETCH = Path(__file__).resolve().parents[2] / "cat-robot-wiring-visual/sketches/cat_robot"


def synthetic_frames(trials=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(trials):
        speed = [0.112, 0.198][k % 2]
        for i in range(60):
            t = i / 10
            values = {
                "speed_mps": speed,
                "elapsed_s": min(t, 3.0),
                "measured_left_mps": speed * min(1, t / 0.5) + rng.normal(0, 0.003),
                "measured_right_mps": speed * min(1, t / 0.5) + rng.normal(0, 0.003),
                "encoder_heading_rad": rng.normal(0, 0.01),
            }
            # A start transient that decays, plus a speed-dependent constant trim.
            label = 0.01 * np.exp(-t / 0.6) - 0.004 + 0.02 * (speed - 0.15)
            rows.append({"trial_id": f"t{k}", "split": "train", "label_valid": "True",
                         "label_correction": str(label), **{f: str(v) for f, v in values.items()}})
    return rows


def test_student_learns_a_transient_and_matches_native_inference(tmp_path):
    model = train_student(synthetic_frames(), "label_correction", "measured", "hash", 0.14)
    assert model.cv_rmse < 0.003
    controller = Controller()
    controller.set_student(model.native_weights())
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = np.array([rng.choice([0.112, 0.198]), rng.uniform(0, 3), rng.uniform(0, 0.2),
                      rng.uniform(0, 0.2), rng.normal(0, 0.02)])
        assert controller.student_predict(x.tolist()) == pytest.approx(float(model.predict(x)[0]), abs=1e-5)
    header = tmp_path / "student_model.h"
    export_student_header(model, header)
    program = tmp_path / "check.cpp"
    program.write_text(f'#include "{header}"\nint main() {{ return deployedStudent().valid() ? 0 : 1; }}\n')
    binary = tmp_path / "check"
    subprocess.run(["c++", "-std=c++17", f"-I{SKETCH}", str(program), "-o", str(binary)], check=True)
    assert subprocess.run([str(binary)], check=False).returncode == 0
    assert STUDENT_FEATURES[0] == "speed_mps"


def test_simulated_students_are_not_exported(tmp_path):
    model = train_student(synthetic_frames(), "label_correction", "simulated", "hash", 0.14)
    with pytest.raises(ValueError, match="measured"):
        export_student_header(model, tmp_path / "x.h")


def test_holdout_comparison_scores_every_student_on_the_same_frames():
    frames = synthetic_frames(trials=10)
    rng = np.random.default_rng(2)
    for index, row in enumerate(frames):
        row["t_s"] = str(index % 60 / 10)
        row["teacher_correction"] = str(float(row["label_correction"]) + rng.normal(0, 0.001))
    truth = [dict(r) for r in frames]
    result = holdout_comparison(
        frames, truth, {"t8", "t9"}, ["label_correction", "teacher_correction", "truth_correction"], 0.14
    )
    assert result["validation_frames"] == 2 * 60
    assert set(result["students"]) == {"label_correction", "teacher_correction", "truth_correction"}
    for scored in result["students"].values():
        assert scored["training_trials"] == 8
        assert scored["rmse"] < result["zero_rmse"]
    assert result["teacher"]["rmse"] < 0.002
