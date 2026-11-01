"""Comparable run summaries, house-style figures and a factual PDF report."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import paperfig as pf
from .config import write_json
from .model import CorrectionModel, read_records
from .sync import read_jsonl

COLORS = dict(zip(("pwm", "speed", "learned", "student"), pf.CATEGORICAL[:4]))
LABELS = {
    "pwm": "Equal PWM",
    "speed": "Wheel-speed PI",
    "learned": "PI + learned correction",
    "student": "PI + ESP32 student",
}
SHORT = {"pwm": "PWM", "speed": "PI", "learned": "PI + learned", "student": "PI + student"}


def summarize(records: list[dict], model: CorrectionModel | None = None) -> list[dict]:
    if not records or any(r["split"] != "test" for r in records):
        raise ValueError("Evaluation must contain test-split runs only")
    if len({r["trial_id"] for r in records}) != len(records):
        raise ValueError("Duplicate evaluation trial IDs")
    if len({r["source"] for r in records}) != 1 or len({r["config_hash"] for r in records}) != 1:
        raise ValueError("Do not combine robot calibrations or measured/simulated evidence")
    if model is not None:
        if set(model.training_runs) & {r["trial_id"] for r in records}:
            raise ValueError("Training/evaluation run leakage")
        if any(r["config_hash"] != model.config_hash for r in records):
            raise ValueError("Evaluation configuration differs from training")
        if any(r["mode"] == "learned" and r["model_id"] != model.model_id() for r in records):
            raise ValueError("Evaluation did not use the frozen correction model")
    result = []
    for mode in COLORS:
        for speed in sorted({r["speed_mps"] for r in records if r["mode"] == mode}):
            rows = [r for r in records if r["mode"] == mode and r["speed_mps"] == speed]
            usable = [r for r in rows if r["quality_ok"] and r["reached"]]

            def median(key, selected=usable):
                values = [abs(r[key]) for r in selected if r[key] is not None]
                return float(np.median(values)) if values else None

            result.append(
                {
                    "mode": mode,
                    "target_speed_mps": speed,
                    "n": len(rows),
                    "reached": sum(r["reached"] for r in rows),
                    "quality_failed": sum(not r["quality_ok"] for r in rows),
                    "successes": sum(r["success"] for r in rows),
                    "median_max_lateral_m": median("max_lateral_m"),
                    "median_abs_heading_rad": median("heading_at_goal_rad"),
                    "median_actual_speed_mps": median("actual_speed_mps"),
                }
            )
    return result


def make_report(runs: Path, output: Path, model: CorrectionModel | None = None) -> dict:
    records = read_records(runs)
    summary = summarize(records, model)
    goal = float(np.median([r.get("goal_m", 1.0) for r in records]))
    output.mkdir(parents=True, exist_ok=True)
    source = records[0]["source"]
    write_json(
        output / "summary.json",
        {"source": source, "groups": summary, "model_id": model.model_id() if model else None},
    )
    with (output / "trials.csv").open("w", newline="") as handle:
        fields = [
            "trial_id",
            "mode",
            "speed_mps",
            "correction",
            "reached",
            "quality_ok",
            "success",
            "max_lateral_m",
            "heading_at_goal_rad",
            "actual_speed_mps",
            "stop_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    pf.apply_style()
    speeds = sorted({r["speed_mps"] for r in records})
    if len(speeds) not in (1, 2):
        raise ValueError("The comparison covers one or two evaluation speeds")
    present = [mode for mode in COLORS if any(r["mode"] == mode for r in records)]
    fig, axes = plt.subplots(2, len(speeds), figsize=(3.6 * len(speeds), 6.0), squeeze=False)
    for column, speed in enumerate(speeds):
        trajectory, errors = axes[0, column], axes[1, column]
        for index, mode in enumerate(present):
            color = COLORS[mode]
            rows = [
                r
                for r in records
                if r["mode"] == mode
                and r["speed_mps"] == speed
                and r["quality_ok"]
                and r["reached"]
            ]
            if not rows:
                continue
            median = float(np.median([r["max_lateral_m"] for r in rows]))
            representative = min(rows, key=lambda r: abs(r["max_lateral_m"] - median))
            directory = runs / representative["trial_id"]
            path = directory / ("paired.jsonl" if source == "measured" else "poses.jsonl")
            poses = [p for p in read_jsonl(path) if p.get("valid", True)]
            points = np.array([[p["x_m"], p["y_m"]] for p in poses])
            angle = poses[0]["yaw_rad"]
            rotation = np.array(
                [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
            )
            points = (points - points[0]) @ rotation
            trajectory.plot(points[:, 0], points[:, 1], color=color, label=LABELS[mode])
            data = [r["max_lateral_m"] * 100 for r in rows]
            errors.scatter(
                np.full(len(data), index) + np.linspace(-0.08, 0.08, len(data)),
                data,
                color=color,
                s=12,
                alpha=0.75,
            )
            errors.plot([index - 0.2, index + 0.2], [np.median(data)] * 2, color=color, lw=2)
        for threshold in (-0.05, 0.05):
            trajectory.axhline(threshold, color=pf.TEXT2, lw=0.7, ls=":")
        trajectory.set(
            xlim=(0, goal + 0.25), xlabel="Forward distance (m)", ylabel="Lateral deviation (m)"
        )
        trajectory.set_title(f"Target {speed:.3f} m/s: median run", loc="left")
        trajectory.legend(fontsize=6, loc="best")
        errors.axhline(5, color=pf.TEXT2, lw=0.7, ls=":")
        errors.set(
            xticks=range(len(present)),
            xticklabels=[SHORT[mode] for mode in present],
            ylabel="Maximum lateral deviation (cm)",
        )
        errors.set_title(f"Target {speed:.3f} m/s: held-out runs", loc="left")
    for label, ax in zip(("a", "b", "c", "d"), axes.flat):
        pf.panel(ax, label)
    fig.suptitle(
        "SIMULATION ONLY - assumed physical parameters"
        if source == "simulated"
        else "MEASURED ROBOT - camera used for evaluation only",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.5, h_pad=2.0)
    overlaps = pf.text_overlaps(fig)
    if overlaps:
        raise ValueError(f"Figure label overlap: {overlaps}")
    pf.save(fig, output, "comparison")
    pdf_report(output, source, records, summary, model, goal)
    return {"source": source, "groups": summary}


def pdf_report(
    output: Path,
    source: str,
    records: list[dict],
    groups: list[dict],
    model: CorrectionModel | None,
    goal: float = 1.0,
):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
    body = ParagraphStyle("Japanese", fontName="HeiseiKakuGo-W5", fontSize=9, leading=15)
    title = ParagraphStyle("JapaneseTitle", parent=body, fontSize=16, leading=24, spaceAfter=12)
    source_label = (
        "シミュレーション検証。実機の改善を示す結果ではない。"
        if source == "simulated"
        else "実機で取得した独立評価走行。"
    )
    width, height = ImageReader(str(output / "comparison.png")).getSize()
    # Fit within 490 pt wide and 430 pt tall; the one-speed figure is a single tall column.
    scale = min(490 / width, 430 / height)
    figure = Image(str(output / "comparison.png"), width=width * scale, height=height * scale)
    story = [
        Paragraph("走行データから直進の補正を学ぶ猫じゃらしロボ", title),
        Paragraph(source_label, body),
        Spacer(1, 10),
        Paragraph("目的と工夫", body),
        Paragraph(
            "左右差のある自作ロボットについて、出力指定、エンコーダーによるPI速度制御、"
            "学習した補正を加えた制御を比較する。PIは速度差とその累積から出力を調整する。"
            "撮影は学習と評価に使い、走行中のモデル入力には使わない。",
            body,
        ),
        Spacer(1, 8),
        Paragraph("AIの役割", body),
        Paragraph(
            f"教師あり学習のRidge回帰で、速度と左右補正率から{goal:g} m区間の横ずれ・向きの変化を予測する。"
            "ESP32とシミュレーターは同じC++推論を使い、予測されるずれが小さい補正率を選ぶ。",
            body,
        ),
        Spacer(1, 8),
        figure,
        PageBreak(),
        Paragraph("速度別の評価結果", body),
        Spacer(1, 10),
    ]
    table = [["制御", "指令 m/s", "実速 m/s", "到達/全数", "達成", "横ずれ cm", "向き °"]]
    for group in groups:
        value = group["median_max_lateral_m"]
        table.append(
            [
                {"pwm": "出力指定", "speed": "PI", "learned": "PI＋学習", "student": "PI＋学生"}[group["mode"]],
                f"{group['target_speed_mps']:.3f}",
                f"{group['median_actual_speed_mps']:.3f}"
                if group["median_actual_speed_mps"] is not None
                else "未算出",
                f"{group['reached']}/{group['n']}",
                str(group["successes"]),
                f"{value * 100:.2f}" if value is not None else "未算出",
                f"{math.degrees(group['median_abs_heading_rad']):.2f}"
                if group["median_abs_heading_rad"] is not None
                else "未算出",
            ]
        )
    grid = Table(table, colWidths=[70, 65, 65, 85, 40, 85, 80])
    grid.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "HeiseiKakuGo-W5"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.black),
            ]
        )
    )
    story += [
        grid,
        Spacer(1, 10),
        Paragraph("評価と限界", body),
        Paragraph(
            f"開発目標は{goal:g} mで横ずれ5 cm、向きのずれ5度以内。停止・未到達も全試行数に含める。"
            "実速、横ずれ、向きは到達・品質確認済みの走行の中央値である。測定品質はCSVへ記録する。"
            "同じ車体・床・校正範囲の補正を対象とし、未知の横滑りや外力への位置修正は扱わない。",
            body,
        ),
    ]
    if source == "simulated":
        story += [
            Spacer(1, 8),
            Paragraph(
                "実機の寸法・モーター応答・接触条件は未校正である。"
                "本結果を実機の成功率やトルク測定値として扱わない。",
                body,
            ),
        ]
    if model:
        story += [
            Spacer(1, 8),
            Paragraph(
                f"モデル {model.model_id()}、学習用 {len(model.training_runs)} 走行、"
                f"評価用 {len(records)} 走行。学習と評価は走行単位で分離した。",
                body,
            ),
        ]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("HeiseiKakuGo-W5", 7)
        canvas.drawString(40, 18, source_label)
        canvas.drawRightString(A4[0] - 40, 18, str(document.page))
        canvas.restoreState()

    SimpleDocTemplate(
        str(output / "report.pdf"),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=35,
        bottomMargin=35,
        title="Cat robot straight-line learning",
        author="",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
