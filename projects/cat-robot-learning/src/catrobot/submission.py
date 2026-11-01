"""Prepare a reviewable public snapshot without uploading or changing visibility."""

import hashlib
import json
import shutil
from pathlib import Path

from .config import load_config, write_json
from .model import CorrectionModel, export_header

CODE_SUFFIXES = {".py", ".cpp", ".h", ".ino", ".sh", ".html"}


def package_submission(
    project: Path,
    destination: Path,
    model: CorrectionModel | None = None,
    report: Path | None = None,
    training_runs: Path | None = None,
) -> dict:
    if destination.exists():
        raise ValueError("Destination already exists; choose a new review snapshot")
    if model is not None and (model.source != "measured" or training_runs is None):
        raise ValueError(
            "A submission model requires measured runs and their anonymized training metrics"
        )
    wiring = project.parent / "cat-robot-wiring-visual"
    files = [
        project / name
        for name in ("pyproject.toml", "uv.lock", "CMakeLists.txt", "README.md", "CAD.md", ".gitignore")
    ]
    for folder in (project / "src", project / "native", project / "tests", project / "scripts"):
        files.extend(
            p
            for p in folder.rglob("*")
            if p.suffix in CODE_SUFFIXES and "__pycache__" not in p.parts
        )
    files.extend(p for p in (project / "examples").glob("*.json"))
    files.extend(
        p for p in (project / "foundation").glob("*")
        if p.suffix == ".py" or p.name in {"pyproject.toml", "uv.lock"}
    )
    files.extend((project / "report").glob("*.md"))
    files.extend(
        p for p in (project / "src/catrobot/assets").rglob("*")
        if p.suffix in {".step", ".obj", ".json"}
    )
    files.extend(
        p for p in (wiring / "sketches" / "cat_robot").glob("*") if p.suffix in CODE_SUFFIXES
    )
    files.extend(p for p in (wiring / "tests").glob("*") if p.suffix in {".cpp", ".sh"})
    copied = []
    for source in sorted(files):
        relative = source.relative_to(project.parent.parent)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(relative)
    if model is not None:
        target = destination / "projects/cat-robot-wiring-visual/sketches/cat_robot/learned_model.h"
        export_header(model, target)
        write_json(destination / "model.json", model.model_dump())
        calibration = load_config(training_runs / model.training_runs[0] / "config.json")
        if calibration.fingerprint() != model.config_hash:
            raise ValueError("Training calibration does not match the exported model")
        calibration.calibration_notes = "Measured controller calibration; physical-model validation is recorded separately."
        write_json(destination / "measured-config.json", calibration.model_dump())
        fields = {
            "trial_id",
            "split",
            "source",
            "mode",
            "speed_mps",
            "correction",
            "config_hash",
            "reached",
            "quality_ok",
            "lateral_at_goal_m",
            "heading_at_goal_rad",
        }
        for identity in model.training_runs:
            row = json.loads((training_runs / identity / "metrics.json").read_text())
            write_json(
                destination / "sample-data/training" / identity / "metrics.json",
                {key: value for key, value in row.items() if key in fields},
            )
    if report is not None:
        summary = json.loads((report / "summary.json").read_text())
        for name in (
            "summary.json",
            "trials.csv",
            "comparison.png",
            "comparison.pdf",
            "report.pdf",
        ):
            shutil.copyfile(report / name, destination / name)
    else:
        summary = None
    (destination / "README.md").write_text(
        "# Cat robot straight-line learning\n\n"
        "Review snapshot. No repository has been created or published automatically.\n\n"
        "```sh\ncd projects/cat-robot-learning\nuv sync --locked\nuv run pytest -q\n"
        "uv run catrobot simulate examples/cad-visual-config.json artifacts/first-drive --mode speed --speed 0.10 --video\n```\n\n"
        "See projects/cat-robot-learning/README.md for calibration and measured experiments.\n"
        "The firmware headers hold the Ridge model and the encoder-only student fitted on measured runs; "
        "they have not yet been driven on the ESP32. Simulation assumptions are not measured robot data.\n"
    )
    (destination / ".gitignore").write_text(
        "**/.venv/\n**/build/\n**/data/\n**/artifacts/\n**/__pycache__/\n*.mp4\n"
    )
    manifest = [
        {
            "path": str(relative),
            "sha256": hashlib.sha256((destination / relative).read_bytes()).hexdigest(),
        }
        for relative in copied
    ]
    result = {
        "published": False,
        "review_required": True,
        "code_files": manifest,
        "measured_model": model.model_id() if model else None,
        "report_source": summary["source"] if summary else None,
        "excluded": [
            "raw phone videos",
            "credentials",
            "starter-kit manuals",
            "unrelated projects",
            "git history",
        ],
    }
    write_json(destination / "manifest.json", result)
    return result
