"""
generate_stage6_plots.py
=========================
Stage 6 visualizations for DeepGuard, rendered with Pillow + Python stdlib
(no matplotlib and no network required).

Every value drawn here comes from actual project artifacts:

    - Epoch history table: the verbatim console log of the Stage 4 training
      run (train_model.py --batch-size 8), stored next to this report as
      train_run_log.txt; best epoch = 6 (lowest validation loss).
    - Confusion matrix and per-class metrics: reports/evaluation_report.json,
      produced by evaluate_model.py on the untouched test split (120 images).

Outputs (written into this same folder):
    training_curves.png      training/validation loss and accuracy by epoch
    confusion_matrix.png     official test-set confusion matrix (rows=actual)
    per_class_metrics.png    precision / recall / F1 per class

This script adds no new empirical result; it only renders recorded values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# -- verified Stage 4 values (verbatim from train_run_log.txt) -----------------
# columns: epoch, train_loss, train_accuracy, validation_loss, validation_accuracy
HISTORY = [
    (1, 0.6099, 0.6361, 0.6585, 0.6250),
    (2, 0.2206, 0.9111, 0.7458, 0.5500),
    (3, 0.1317, 0.9611, 0.6652, 0.5333),
    (4, 0.1489, 0.9500, 0.5997, 0.8083),
    (5, 0.0261, 0.9944, 0.6748, 0.5417),
    (6, 0.0362, 0.9917, 0.4908, 0.7833),
    (7, 0.0469, 0.9833, 0.5771, 0.6500),
    (8, 0.1215, 0.9583, 0.6341, 0.6167),
    (9, 0.0335, 0.9889, 0.6448, 0.6083),
]
BEST_EPOCH = 6

OUT_DIR = Path(__file__).resolve().parent
REPORT_JSON = Path(__file__).resolve().parent.parent / "evaluation_report.json"

WHITE = (255, 255, 255)
INK = (30, 30, 40)
GRID = (212, 215, 224)
BLUE = (30, 96, 199)
ORANGE = (221, 104, 16)
GREEN = (40, 145, 90)
GOLD = (152, 116, 40)
FONT_DIR = r"C:\Windows\Fonts"


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
        _label(draw, (x0 - 62, gy - 9), f"{tv:.3f}", small)
    for color, points in series:
        for i, (epoch, value) in enumerate(points):
            px = _px(box, i, n)
            py = _py(box, value, lo, hi)
            if i > 0:
                prev_epoch, prev_value = points[i - 1]
                draw.line([(_px(box, i - 1, n), _py(box, prev_value, lo, hi)),
                           (px, py)], fill=color, width=3)
        for i, (epoch, value) in enumerate(points):
            px = _px(box, i, n)
            py = _py(box, value, lo, hi)
            draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=color,
                         outline=WHITE, width=2)
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
        px = _px(box, i, n)
        _label(draw, (px - 7, y1 + 8), str(series[0][1][i][0]), small)
    lx = x0 + 12
    ly = y0 + 14
    for label, color in legend:
        draw.rectangle([lx, ly - 2, lx + 36, ly + 12], fill=color, outline=INK)
        _label(draw, (lx + 44, ly - 6), label, _font(16))
        lx += 44 + draw.textlength(label, font=_font(16)) + 46


def render_training_curves():
    img = Image.new("RGB", (1180, 950), WHITE)
    d = ImageDraw.Draw(img)
    title = _font(24, bold=True)
    _label(d, (44, 26), "DeepGuard Stage 4 - Training vs Validation", title)
    sub = _font(16)
    _label(d, (44, 64), "EfficientNet-B0, ImageNet-pretrained, batch size 8, "
                        "9/10 epochs, early stop (patience 3)", sub)
    _label(d, (44, 90), "Gold line marks best validation epoch 6 "
                        "(lowest validation loss 0.4908)", sub)

    tl = [(e, v) for (e, v, _a, _vl, _va) in HISTORY]
    vl = [(e, v) for (e, _t, _a, v, _va) in HISTORY]
    ta = [(e, v) for (e, _t, v, _vl, _va) in HISTORY]
    va = [(e, v) for (e, _t, _a, _vl, v) in HISTORY]

    draw_chart(d, (120, 210, 1105, 465), [(ORANGE, tl), (BLUE, vl)],
               (0.0, 0.80), "Loss (mean)", [("Train loss", ORANGE), ("Validation loss", BLUE)],
               vline_epoch=BEST_EPOCH)
    draw_chart(d, (120, 640, 1105, 895), [(ORANGE, ta), (BLUE, va)],
               (0.48, 1.02), "Accuracy", [("Train accuracy", ORANGE), ("Validation accuracy", BLUE)],
               vline_epoch=BEST_EPOCH)
    _label(d, (120, 490), "Loss: train falls 0.61 -> 0.03; validation never keeps up "
                          "(0.49-0.75).", sub)
    _label(d, (120, 515), "Accuracy: train reaches ~0.99; validation swings 0.53-0.81, "
                          "peaking near epoch 6.", sub)
    _label(d, (560, 190), "epoch", _font(14))
    _label(d, (560, 622), "epoch", _font(14))
    img.save(OUT_DIR / "training_curves.png")
    print("wrote", OUT_DIR / "training_curves.png")


def render_confusion_matrix():
    with open(REPORT_JSON, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    metrics = report["metrics"]
    matrix = metrics["confusion_matrix"]
    names = metrics["class_names"]
    num_samples = metrics["num_samples"]
    accuracy = metrics["accuracy"]

    cell = 170
    margin_x = 120
    base_y = 300
    width = 850
    height = 700
    img = Image.new("RGB", (width, height), WHITE)
    d = ImageDraw.Draw(img)
    title = _font(20, bold=True)
    _label(d, (margin_x - 30, 34),
           "DeepGuard Stage 5 - Official Test Confusion Matrix", title)
    sub = _font(15)
    _label(d, (margin_x - 30, 72),
           f"rows = actual class, columns = predicted class   |   total = {num_samples}",
           sub)
    accent = _font(15, bold=True)
    _label(d, (margin_x - 30, 99),
           f"Accuracy = {accuracy:.4f}  (90/120 correct, 30/120 incorrect)", accent)

    strength = [0.16, 0.40, 0.72, 0.95]
    for row in range(2):
        for col in range(2):
            value = matrix[row][col]
            frac = value / 60.0
            tone = strength[int(round(frac * 3))]
            fill = (int(238 * (1 - tone)), int(244 * (1 - tone)), 255)
            x0 = margin_x + col * cell
            y0 = base_y + row * cell
            d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=fill, outline=INK, width=3)
            cx = x0 + cell / 2
            cy = y0 + cell / 2
            _label(d, (cx - 30, cy - 40), str(value), _font(42, bold=True))
            _label(d, (cx - 58, cy + 22), f"({value / num_samples * 100:.1f}%)", _font(17))
    for name in names:
        idx = _CLASS_ORDER[name]
        _label(d, (margin_x - 85, base_y + idx * cell + cell / 2 - 16),
               name, _font(22, bold=True))
        _label(d, (margin_x + idx * cell + cell / 2 - 16, base_y - 58),
               name, _font(22, bold=True))
    _label(d, (margin_x + 90, base_y - 96), "predicted", _font(16))
    _label(d, (14, base_y + cell - 24), "actual", _font(16))
    _label(d, (margin_x - 30, base_y + 2 * cell + 26),
           "60 real -> real (0 real missed);  30 fake -> real (missed);  "
           "30 fake -> fake (caught).", _font(16))
    img.save(OUT_DIR / "confusion_matrix.png")
    print("wrote", OUT_DIR / "confusion_matrix.png")


_CLASS_ORDER = {"real": 0, "fake": 1}


def render_per_class_metrics():
    with open(REPORT_JSON, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    metrics = report["metrics"]
    per_class = metrics["per_class"]
    macro = metrics["macro"]

    width = 1020
    height = 640
    img = Image.new("RGB", (width, height), WHITE)
    d = ImageDraw.Draw(img)
    title = _font(22, bold=True)
    _label(d, (40, 30), "DeepGuard Stage 5 - Per-Class Precision, Recall, F1", title)
    sub = _font(16)
    _label(d, (40, 70), f"Test set: 60 real + 60 fake = 120 frames",
           sub)
    _label(d, (40, 98),
           f"Macro average - precision {macro['precision']:.4f}, "
           f"recall {macro['recall']:.4f}, f1 {macro['f1']:.4f}", sub)

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
    img.save(OUT_DIR / "per_class_metrics.png")
    print("wrote", OUT_DIR / "per_class_metrics.png")


if __name__ == "__main__":
    render_training_curves()
    render_confusion_matrix()
    render_per_class_metrics()
    print("Stage 6 plots written to", OUT_DIR)