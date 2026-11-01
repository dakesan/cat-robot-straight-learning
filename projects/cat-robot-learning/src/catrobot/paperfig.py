"""Unmodified style/helper subset of the local paper-figures paperfig v0.2.0."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox

STYLE_VERSION = "0.2.0"

BLUE = "#2a78d6"

ORANGE = "#eb6834"

AQUA = "#1baf7a"

YELLOW = "#eda100"

MAGENTA = "#e87ba4"

CATEGORICAL: tuple[str, ...] = (BLUE, ORANGE, AQUA, YELLOW, MAGENTA)

BLUE_LIGHT = "#86b6ef"

ORANGE_LIGHT = "#f4a07a"

SEQ: tuple[str, ...] = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

TEXT = "#0b0b0b"

TEXT2 = "#52514e"

GRID = "#e4e3df"

SURFACE = "#ffffff"

NODE_GRAY = "#8a8984"

NS_GRAY = "#b9b8b3"

RED = "#e34948"

DIVERGING_MID = "#f0efec"

UP, DOWN = YELLOW, AQUA

RC: dict[str, object] = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.edgecolor": TEXT2,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT2,
    "ytick.color": TEXT2,
    "text.color": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 2,
    "pdf.fonttype": 42,
}


def apply_style() -> None:
    """Install the house rcParams. Call once before creating figures."""
    mpl.rcParams.update(RC)


def save(fig, outdir: str | Path, name: str, formats: Sequence[str] = ("png", "pdf")) -> list[Path]:
    """Save ``fig`` as ``<outdir>/<name>.<ext>`` for every format, then close it."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = outdir / f"{name}.{ext}"
        fig.savefig(p, facecolor=SURFACE)
        paths.append(p)
    plt.close(fig)
    return paths


def panel(ax, letter: str, x: float = -0.14, y: float = 1.06, fontsize: float = 11) -> Text:
    """Bold panel letter at the top-left of ``ax`` (axes coordinates)."""
    return ax.text(
        x, y, letter, transform=ax.transAxes, fontsize=fontsize, fontweight="bold", va="bottom"
    )


def _label_texts(fig) -> list[Text]:
    """Texts that must not collide: titles, axis labels, annotations, legends, figure texts.

    Tick labels are excluded; the visual check in SKILL.md covers them.
    """
    texts: list[Text] = list(fig.texts)
    for ax in fig.get_axes():
        texts.extend([ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label])
        texts.extend(ax.texts)
        if ax.get_legend() is not None:
            texts.extend(ax.get_legend().get_texts())
    for lg in fig.legends:
        texts.extend(lg.get_texts())
    return [t for t in texts if t.get_visible() and t.get_text().strip()]


def text_overlaps(fig, pad_pt: float = 0.0) -> list[tuple[str, str]]:
    """Return pairs of label texts whose rendered bounding boxes overlap.

    Runs the renderer, so call it after the layout is final (just before ``save``).
    ``pad_pt`` shrinks each box by that many points on every side before testing,
    which ignores touching edges. This is a machine pre-check only: it does not see
    tick labels, marks under text, or clipping at the figure edge.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    shrink = pad_pt * fig.dpi / 72
    boxes = []
    for t in _label_texts(fig):
        bb = Text.get_window_extent(
            t, renderer
        )  # text box only; Annotation would include its arrow
        if bb.width <= 0 or bb.height <= 0:
            continue
        boxes.append(
            (
                t.get_text(),
                Bbox.from_extents(bb.x0 + shrink, bb.y0 + shrink, bb.x1 - shrink, bb.y1 - shrink),
            )
        )
    out = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                out.append((boxes[i][0], boxes[j][0]))
    return out
