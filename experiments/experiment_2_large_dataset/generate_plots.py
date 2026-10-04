"""
generate_plots.py
=================
Visualisations for DeepGuard Experiment 2, rendered with Pillow + Python stdlib
(no matplotlib, no network, no package installation required).

Every value drawn here is a recorded measurement from the completed Experiment 2
run of 2026-10-04. This script adds no new empirical result; it only renders
values that were already produced by training and evaluation.

Inputs (read-only):
    - training_summary.json          run metadata: best epoch, best validation
                                     loss/accuracy, wall-clock, fingerprint
    - reports/evaluation_report.json official 900-image test-split metrics,
                                     confusion matrix and per-class values
    - HISTORY below                   the per-epoch training history, embedded
                                     as literals because neither JSON file
                                     records it (see PROVENANCE)

Outputs (written only into reports/ beside this script):
    reports/confusion_matrix.png
    reports/per_class_metrics.png
    reports/training_loss_curve.png
    reports/training_accuracy_curve.png

PROVENANCE of HISTORY
---------------------
train_run_log.txt is header-only because of the known Tee-wiring defect in
train_experiment_2.py (documented in README.md, deliberately not fixed), so the
per-epoch rows could not be read from it. The values below were recovered from
models/experiment_2_training_state.pt, which the trainer rewrites atomically
after every epoch, and cross-checked against the captured stdout of the run.
They are therefore recorded measurements, not estimates. The script cannot read
that 48 MB state file without a torch dependency, so the values are carried here
as literals; best_epoch and best_validation_loss from training_summary.json are
asserted against them at run time so the two can never drift apart silently.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# -- verified per-epoch history (recovered from experiment_2_training_state.pt) --
# columns: epoch, train_loss, train_accuracy, validation_loss, validation_accuracy
HISTORY = [
    (1, 0.4470, 0.7845, 0.1800, 0.9378),
    (2, 0.2742, 0.8948, 0.2611, 0.9000),
    (3, 0.2176, 0.9160, 0.1288, 0.9478),
    (4, 0.1868, 0.9269, 0.1183, 0.9511),
    (5, 0.1551, 0.9421, 0.0831, 0.9678),
    (6, 0.1340, 0.9467, 0.3183, 0.8656),
    (7, 0.1119, 0.9586, 0.1017, 0.9589),
    (8, 0.1128, 0.9624, 0.0819, 0.9667),
    (9, 0.0892, 0.9674, 0.0773, 0.9667),
    (10, 0.0934, 0.9676, 0.0684, 0.9689),
]

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "reports"
SUMMARY_JSON = SCRIPT_DIR / "training_summary.json"
EVAL_JSON = OUT_DIR / "evaluation_report.json"

# ---------------------------------------------------------------------------
# Safety guards: this script must never be able to write outside its own
# reports/ directory, and in particular never into the repository-level
# reports/ directory that holds Experiment 1's frozen results.
# ---------------------------------------------------------------------------
_EXPECTED_DIR_NAME = "experiment_2_large_dataset"


def _verify_paths() -> None:
    if SCRIPT_DIR.name != _EXPECTED_DIR_NAME:
        raise SystemExit(
            f"Refusing to run: expected to live in a {_EXPECTED_DIR_NAME!r} "
            f"folder, but am in {SCRIPT_DIR.name!r}."
        )
    repo_root = SCRIPT_DIR.parents[1]
    exp1_reports = (repo_root / "reports").resolve()
    if OUT_DIR.resolve() == exp1_reports:
        raise SystemExit(
            f"Refusing to run: output directory {OUT_DIR} resolves to the "
            f"Experiment 1 reports directory {exp1_reports}."
        )
    if not OUT_DIR.is_relative_to(SCRIPT_DIR):
        raise SystemExit(f"Refusing to run: {OUT_DIR} is outside {SCRIPT_DIR}.")


def _save(img: Image.Image, name: str) -> Path:
    """Save into OUT_DIR only, refusing anything that escapes it."""
    target = (OUT_DIR / name).resolve()
    if not target.is_relative_to(OUT_DIR.resolve()):
        raise SystemExit(f"Refusing to write outside {OUT_DIR}: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    img.save(target)
    print("wrote", target)
    return target


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _check_history_against_summary(summary: dict) -> None:
    """Guard against the embedded HISTORY drifting from the recorded summary."""
    best_epoch = int(summary["best_epoch"])
    if not 1 <= best_epoch <= len(HISTORY):
        raise SystemExit(f"best_epoch {best_epoch} is outside the recorded history.")
    recorded_loss = float(summary["best_validation_loss"])
    row = HISTORY[best_epoch - 1]
    if abs(row[3] - recorded_loss) > 5e-4:
        raise SystemExit(
            f"History disagrees with training_summary.json: epoch {best_epoch} "
            f"validation loss {row[3]:.6f} vs recorded {recorded_loss:.6f}."
        )
    lowest = min(HISTORY, key=lambda r: r[3])
    if lowest[0] != best_epoch:
        raise SystemExit(
            f"History disagrees with training_summary.json: lowest validation "
            f"loss is at epoch {lowest[0]}, but best_epoch is {best_epoch}."
        )
    if int(summary["epochs_completed"]) != len(HISTORY):
        raise SystemExit(
            f"History has {len(HISTORY)} epochs but the run recorded "
            f"{summary['epochs_completed']}."
        )


WHITE = (255, 255, 255)
INK = (30, 30, 40)
GRID = (212, 215, 224)
BLUE = (30, 96, 199)
ORANGE = (221, 104, 16)
GREEN = (40, 145, 90)
GOLD = (152, 116, 40)
FONT_DIR = r"C:\Windows\Fonts"

_CLASS_ORDER = {"real": 0, "fake": 1}


def _font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(str(Path(FONT_DIR) / name), size)
    except OSError:  # pragma: no cover - fallback to the built-in font
        return ImageFont.load_default()


def _label(draw, xy, text, font, fill=INK):
    draw.text(xy, text, font=font, fill=fill)


def _px(box, i, n):
    return box[0] + i * (box[2] - box[0]) / max(n - 1, 1)


def _py(box, value, lo, hi):
    return box[3] - (value - lo) * (box[3] - box[1]) / (hi - lo)


def draw_chart(draw, box, series, y_range, y_label, legend, vline_epoch=None):
    """series: list of (color, [(epoch, value), ...]); draws grid, axes, curves."""
    x0, y0, x1, y1 = box
    lo, hi = y_range
    n = len(series[0][1])
    small = _font(15)
    small_bold = _font(15, bold=True)
    for k in range(5):
        tv = lo + (hi - lo) * k / 4
        gy = _py(box, tv, lo, hi)
        draw.line([(x0, gy), (x1, gy)], fill=GRID, width=1)
        _label(draw, (x0 - 66, gy - 9), f"{tv:.3f}", small)
    for color, points in series:
        for i, (_epoch, value) in enumerate(points):
            px = _px(box, i, n)
            py = _py(box, value, lo, hi)
            if i > 0:
                prev_value = points[i - 1][1]
                draw.line(
                    [(_px(box, i - 1, n), _py(box, prev_value, lo, hi)), (px, py)],
                    fill=color,
                    width=3,
                )
        for i, (_epoch, value) in enumerate(points):
            px = _px(box, i, n)
            py = _py(box, value, lo, hi)
            draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=color, outline=WHITE, width=2)
    if vline_epoch is not None:
        index = next(i for i, (e, _v) in enumerate(series[0][1]) if e == vline_epoch)
        lx = _px(box, index, n)
        y = y0
        dash = 8
        while y < y1:
            draw.line([(lx, y), (lx, min(y + dash, y1))], fill=GOLD, width=2)
            y += dash * 2
    draw.rectangle([x0, y0, x1, y1], outline=INK, width=2)
    _label(draw, (x0 + 6, y0 - 26), y_label, small_bold)
    for i in range(n):
        _label(draw, (_px(box, i, n) - 7, y1 + 8), str(series[0][1][i][0]), small)
    lx = x0 + 12
    ly = y0 + 14
    for label, color in legend:
        draw.rectangle([lx, ly - 2, lx + 36, ly + 12], fill=color, outline=INK)
        _label(draw, (lx + 44, ly - 6), label, _font(16))
        lx += 44 + draw.textlength(label, font=_font(16)) + 46


def render_confusion_matrix(report: dict):
    metrics = report["metrics"]
    matrix = metrics["confusion_matrix"]
    names = metrics["class_names"]
    num_samples = metrics["num_samples"]
    accuracy = metrics["accuracy"]
    support = metrics["per_class"]["real"]["support"]

    correct = matrix[0][0] + matrix[1][1]
    fp = matrix[0][1]
    fn = matrix[1][0]

    cell = 170
    margin_x = 130
    base_y = 300
    width = 880
    height = 730
    img = Image.new("RGB", (width, height), WHITE)
    d = ImageDraw.Draw(img)
    title = _font(20, bold=True)
    _label(d, (margin_x - 30, 34), "DeepGuard Experiment 2 - Test Confusion Matrix", title)
    sub = _font(15)
    _label(
        d,
        (margin_x - 30, 72),
        f"rows = actual class, columns = predicted class   |   total = {num_samples}",
        sub,
    )
    accent = _font(15, bold=True)
    _label(
        d,
        (margin_x - 30, 99),
        f"Accuracy = {accuracy:.4f}  ({correct}/{num_samples} correct, "
        f"{num_samples - correct}/{num_samples} incorrect)",
        accent,
    )

    strength = [0.16, 0.40, 0.72, 0.95]
    for row in range(len(matrix)):
        row_total = sum(matrix[row])
        for col in range(len(matrix[row])):
            value = matrix[row][col]
            frac = value / max(row_total, 1)
            tone = strength[int(round(frac * 3))]
            fill = (int(238 * (1 - tone)), int(244 * (1 - tone)), 255)
            x0 = margin_x + col * cell
            y0 = base_y + row * cell
            d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=fill, outline=INK, width=3)
            cx = x0 + cell / 2
            cy = y0 + cell / 2
            _label(d, (cx - 30, cy - 40), str(value), _font(42, bold=True))
            _label(d, (cx - 58, cy + 22), f"({frac * 100:.1f}%)", _font(17))
    for name in names:
        idx = _CLASS_ORDER[name]
        _label(d, (margin_x - 85, base_y + idx * cell + cell / 2 - 16), name, _font(22, bold=True))
        _label(d, (margin_x + idx * cell + cell / 2 - 16, base_y - 58), name, _font(22, bold=True))
    _label(d, (margin_x + 90, base_y - 96), "predicted", _font(16))
    _label(d, (14, base_y + cell - 24), "actual", _font(16))
    _label(
        d,
        (margin_x - 30, base_y + 2 * cell + 26),
        f"{fp} real -> fake (false positives);  {fn} fake -> real (missed);  "
        f"percentages are per actual class (support {support}).",
        _font(16),
    )
    return _save(img, "confusion_matrix.png")


def render_per_class_metrics(report: dict):
    metrics = report["metrics"]
    per_class = metrics["per_class"]
    macro = metrics["macro"]

    width = 1020
    height = 640
    img = Image.new("RGB", (width, height), WHITE)
    d = ImageDraw.Draw(img)
    title = _font(22, bold=True)
    _label(d, (40, 30), "DeepGuard Experiment 2 - Per-Class Precision, Recall, F1", title)
    sub = _font(16)
    _label(
        d,
        (40, 70),
        f"Test set: {per_class['real']['support']} real + "
        f"{per_class['fake']['support']} fake = {metrics['num_samples']} images",
        sub,
    )
    _label(
        d,
        (40, 98),
        f"Macro average - precision {macro['precision']:.4f}, "
        f"recall {macro['recall']:.4f}, f1 {macro['f1']:.4f}   |   "
        f"accuracy {metrics['accuracy']:.4f}",
        sub,
    )

    groups = [("real", per_class["real"]), ("fake", per_class["fake"])]
    metric_keys = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1-score"]
    colours = [BLUE, ORANGE, GREEN]
    bar_w = 34
    gap = 14
    base_y = 350
    max_h = 250
    x_start = 170
    group_w = 150

    small = _font(14)
    for tv in range(0, 101, 10):
        gy = base_y - tv / 100.0 * max_h
        d.line([(x_start - 6, gy), (x_start, gy)], fill=INK, width=2)
        _label(d, (x_start - 56, gy - 8), f"{tv / 100:.1f}", small)

    for gi, (name, values) in enumerate(groups):
        gx = x_start + 70 + gi * (group_w + 90)
        _label(d, (gx + 8, base_y + 22), name, _font(22, bold=True))
        _label(d, (gx - 6, base_y + 52), f"support {values['support']}", _font(15))
        for mi, key in enumerate(metric_keys):
            value = values[key]
            bx = gx + mi * (bar_w + gap)
            bh = value * max_h
            d.rectangle([bx, base_y - bh, bx + bar_w, base_y], fill=colours[mi])
            d.rectangle([bx, base_y - bh, bx + bar_w, base_y], outline=WHITE, width=1)
            _label(d, (bx - 18, base_y - bh - 24), f"{value:.3f}", _font(15, bold=True))
        for mi, label in enumerate(metric_labels):
            lx = gx + mi * (bar_w + gap) - 26
            _label(d, (lx, base_y - max_h - 34), label, _font(14))

    d.line([(x_start, base_y), (width - 60, base_y)], fill=INK, width=3)
    return _save(img, "per_class_metrics.png")


def render_training_loss(summary: dict):
    best_epoch = int(summary["best_epoch"])
    epochs = [r[0] for r in HISTORY]
    train_loss = [(r[0], r[1]) for r in HISTORY]
    val_loss = [(r[0], r[3]) for r in HISTORY]
    best_loss = min(r[3] for r in HISTORY)

    img = Image.new("RGB", (1180, 760), WHITE)
    d = ImageDraw.Draw(img)
    title = _font(24, bold=True)
    _label(d, (44, 26), "DeepGuard Experiment 2 - Training vs Validation Loss", title)
    sub = _font(16)
    _label(
        d,
        (44, 64),
        f"EfficientNet-B0, ImageNet-pretrained, batch size 8, "
        f"{summary['epochs_completed']}/{summary['epochs_requested']} epochs, "
        f"seed {summary['seed']}, {summary['device'].upper()}",
        sub,
    )
    _label(
        d,
        (44, 90),
        f"Gold line marks the best validation epoch {best_epoch} "
        f"(lowest validation loss {best_loss:.4f})",
        sub,
    )
    draw_chart(
        d,
        (120, 210, 1105, 700),
        [(ORANGE, train_loss), (BLUE, val_loss)],
        (0.0, 0.50),
        "Loss (mean)",
        [("Train loss", ORANGE), ("Validation loss", BLUE)],
        vline_epoch=best_epoch,
    )
    _label(d, (560, 190), "epoch", _font(14))
    _label(
        d,
        (120, 720),
        f"Validation loss spikes at epoch 6 (0.3183) then improves monotonically "
        f"to {best_loss:.4f} at epoch {best_epoch}.",
        sub,
    )
    return _save(img, "training_loss_curve.png")


def render_training_accuracy(summary: dict):
    best_epoch = int(summary["best_epoch"])
    train_acc = [(r[0], r[2]) for r in HISTORY]
    val_acc = [(r[0], r[4]) for r in HISTORY]

    img = Image.new("RGB", (1180, 760), WHITE)
    d = ImageDraw.Draw(img)
    title = _font(24, bold=True)
    _label(d, (44, 26), "DeepGuard Experiment 2 - Training vs Validation Accuracy", title)
    sub = _font(16)
    _label(
        d,
        (44, 64),
        f"Same run as the loss curve. Best checkpoint: epoch {best_epoch}, selected by "
        f"validation loss (not accuracy).",
        sub,
    )
    _label(
        d,
        (44, 90),
        f"Validation accuracy is NOT the reported result - the test-split accuracy "
        f"is the Experiment 2 result.",
        sub,
    )
    draw_chart(
        d,
        (120, 210, 1105, 700),
        [(ORANGE, train_acc), (BLUE, val_acc)],
        (0.75, 1.00),
        "Accuracy",
        [("Train accuracy", ORANGE), ("Validation accuracy", BLUE)],
        vline_epoch=best_epoch,
    )
    _label(d, (560, 190), "epoch", _font(14))
    _label(
        d,
        (120, 720),
        f"Validation accuracy dips at epoch 6 (0.8656) then recovers to "
        f"{HISTORY[-1][4]:.4f} at epoch {HISTORY[-1][0]}, which is also the highest "
        f"validation accuracy in the run.",
        sub,
    )
    return _save(img, "training_accuracy_curve.png")


def main() -> None:
    _verify_paths()
    if not SUMMARY_JSON.is_file():
        raise SystemExit(f"Missing input: {SUMMARY_JSON}")
    if not EVAL_JSON.is_file():
        raise SystemExit(
            f"Missing input: {EVAL_JSON}. Run evaluate_model.py with "
            f"--reports-dir pointing at this experiment first."
        )
    summary = _load_json(SUMMARY_JSON)
    report = _load_json(EVAL_JSON)
    _check_history_against_summary(summary)

    print(f"output directory: {OUT_DIR}")
    render_confusion_matrix(report)
    render_per_class_metrics(report)
    render_training_loss(summary)
    render_training_accuracy(summary)
    print("Experiment 2 plots written to", OUT_DIR)


if __name__ == "__main__":
    main()