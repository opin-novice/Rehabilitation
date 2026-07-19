#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report_card.py
==============
Renders a compute_report() dict into a clean, judge-friendly report card:

    +--------------------------------------------------+
    |  Trunk Lateral Flexion              2026-07-20   |
    |                                                  |
    |     MOVEMENT QUALITY        84 / 100  ****-       |
    |     [ AI model score  41/50 ]   6 reps  20.0s    |
    |                                                  |
    |   Range of motion   ####################  88%    |
    |   Smoothness        #####################  91%   |
    |   ...                                            |
    |                                                  |
    |   STRENGTHS               WHAT TO IMPROVE        |
    |   + ...                   > ...                  |
    +--------------------------------------------------+

Returns a BGR image (for cv2.imshow, full-window) and saves a PNG to demo/reports/.
Rendered with Pillow; falls back to a plain OpenCV card if Pillow is unavailable.
"""

import os
from datetime import datetime

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False

W, H = 1280, 760
BG = (24, 27, 33)
FG = (235, 238, 242)
MUTED = (150, 158, 170)
PANEL = (34, 38, 46)
GREEN = (46, 204, 113)
AMBER = (243, 156, 18)
RED = (231, 76, 60)
BLUE = (90, 160, 240)


def _band_color(pct):
    return GREEN if pct >= 80 else AMBER if pct >= 55 else RED


def _verdict(pct):
    return "excellent" if pct >= 88 else "good" if pct >= 72 else \
           "fair" if pct >= 55 else "needs work"


def _font(size, bold=False):
    names = (["arialbd.ttf", "seguisb.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _reports_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(d, exist_ok=True)
    return d


def _star(d, cx, cy, r, fill):
    """Draw a 5-point star centred at (cx,cy). Font glyphs for stars aren't reliable, so we
    draw the polygon ourselves."""
    import math
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * 0.42
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    d.polygon(pts, fill=fill)


def _stars(d, x, y, n, size=22, gap=10):
    for i in range(5):
        _star(d, x + i * (2 * size + gap) + size, y + size,
              size, AMBER if i < n else PANEL)


def render_report(report, save=True):
    """report: dict from feedback.compute_report. Returns (bgr_image, png_path_or_None)."""
    if not _HAVE_PIL:
        return _render_opencv(report, save)

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_h1 = _font(44, bold=True)
    f_h2 = _font(28, bold=True)
    f_big = _font(92, bold=True)
    f_lbl = _font(26)
    f_sm = _font(22)

    pad = 48
    # --- header ---
    d.text((pad, 34), report["exercise_name"], font=f_h1, fill=FG)
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M")
    d.text((W - pad - d.textlength(ts, font=f_sm), 50), ts, font=f_sm, fill=MUTED)
    d.line((pad, 96, W - pad, 96), fill=PANEL, width=3)

    # --- headline composite ---
    comp = report["composite"]
    comp_col = _band_color(comp)
    d.text((pad, 116), "MOVEMENT QUALITY", font=f_h2, fill=MUTED)
    d.text((pad, 148), f"{comp:.0f}", font=f_big, fill=comp_col)
    unit_x = pad + d.textlength(f"{comp:.0f}", font=f_big) + 14
    d.text((unit_x, 208), "/ 100", font=f_h2, fill=MUTED)
    _stars(d, unit_x, 150, report["stars"], size=20, gap=8)
    # band word on the far right of the headline row, prominent
    d.text((W - pad - d.textlength(report["band"], font=f_h1), 150),
           report["band"], font=f_h1, fill=comp_col)

    # --- AI badge + reps/duration ---
    bx, by = pad, 258
    if report.get("ai_badge") is not None:
        badge = f"AI model score   {report['ai_badge']:.0f}/50"
        bw = d.textlength(badge, font=f_sm) + 32
        d.rounded_rectangle((bx, by, bx + bw, by + 40), radius=10, fill=PANEL)
        d.text((bx + 16, by + 8), badge, font=f_sm, fill=BLUE)
        bx += bw + 24
    info = f"{report['reps']} reps    {report['duration_s']:.0f}s session"
    d.text((bx, by + 8), info, font=f_sm, fill=MUTED)

    # --- metric rows ---
    y = 312
    row_h = 42
    bar_x, bar_w = pad + 320, 520
    for m in report["metrics"]:
        pct = float(m["pct"])
        col = _band_color(pct)
        d.text((pad, y), m["label"], font=f_lbl, fill=FG)
        d.rounded_rectangle((bar_x, y + 6, bar_x + bar_w, y + 26), radius=8, fill=PANEL)
        fillw = int(bar_w * np.clip(pct / 100.0, 0, 1))
        if fillw > 0:
            d.rounded_rectangle((bar_x, y + 6, bar_x + fillw, y + 26), radius=8, fill=col)
        d.text((bar_x + bar_w + 16, y), f"{pct:.0f}%", font=f_lbl, fill=col)
        d.text((bar_x + bar_w + 90, y + 3), _verdict(pct), font=f_sm, fill=MUTED)
        y += row_h

    # --- strengths / improvements ---
    cy = y + 14
    cbot = H - 46
    colw = (W - 2 * pad - 40) // 2
    d.rounded_rectangle((pad, cy, pad + colw, cbot), radius=12, fill=PANEL)
    d.rounded_rectangle((pad + colw + 40, cy, W - pad, cbot), radius=12, fill=PANEL)
    d.text((pad + 20, cy + 16), "STRENGTHS", font=f_h2, fill=GREEN)
    d.text((pad + colw + 60, cy + 16), "WHAT TO IMPROVE", font=f_h2, fill=AMBER)

    def _wrap(text, font, maxw):
        words, lines, cur = text.split(), [], ""
        for wd in words:
            test = (cur + " " + wd).strip()
            if d.textlength(test, font=font) <= maxw:
                cur = test
            else:
                lines.append(cur); cur = wd
        if cur:
            lines.append(cur)
        return lines

    def _bullets(items, x0, marker, col):
        yy = cy + 56
        for it in items:
            for i, ln in enumerate(_wrap(it, f_sm, colw - 60)):
                if yy > cbot - 24:
                    return
                pre = f"{marker} " if i == 0 else "   "
                d.text((x0 + 20, yy), pre + ln, font=f_sm, fill=col if i == 0 else FG)
                yy += 27
            yy += 6

    _bullets(report["strengths"], pad, "+", FG)
    _bullets(report["improvements"], pad + colw + 40, ">", FG)

    # --- footer ---
    foot = "Powered by an SE(3)-equivariant, viewpoint-invariant motion model"
    d.text((pad, H - 34), foot, font=_font(18), fill=MUTED)

    rgb = np.array(img)
    bgr = rgb[:, :, ::-1].copy()
    png_path = None
    if save:
        png_path = os.path.join(_reports_dir(),
                                f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        img.save(png_path)
    return bgr, png_path


def _render_opencv(report, save):
    """Plain fallback if Pillow is missing (kept minimal but functional)."""
    import cv2
    img = np.full((H, W, 3), BG[::-1], dtype=np.uint8)
    F = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, report["exercise_name"], (40, 60), F, 1.1, FG[::-1], 2)
    cv2.putText(img, f"MOVEMENT QUALITY  {report['composite']:.0f}/100  ({report['band']})",
                (40, 140), F, 1.0, _band_color(report["composite"])[::-1], 2)
    if report.get("ai_badge") is not None:
        cv2.putText(img, f"AI model score {report['ai_badge']:.0f}/50   {report['reps']} reps",
                    (40, 185), F, 0.7, BLUE[::-1], 2)
    y = 240
    for m in report["metrics"]:
        pct = float(m["pct"]); col = _band_color(pct)[::-1]
        cv2.putText(img, f"{m['label']:<20} {pct:3.0f}%", (40, y), F, 0.7, FG[::-1], 1)
        cv2.rectangle(img, (360, y - 18), (360 + int(5.2 * pct), y - 2), col, -1)
        y += 40
    y += 10
    cv2.putText(img, "STRENGTHS", (40, y), F, 0.7, GREEN[::-1], 2); y += 30
    for s in report["strengths"]:
        cv2.putText(img, "+ " + s[:70], (40, y), F, 0.55, FG[::-1], 1); y += 26
    y += 10
    cv2.putText(img, "WHAT TO IMPROVE", (40, y), F, 0.7, AMBER[::-1], 2); y += 30
    for s in report["improvements"]:
        cv2.putText(img, "> " + s[:70], (40, y), F, 0.55, FG[::-1], 1); y += 26
    png_path = None
    if save:
        png_path = os.path.join(_reports_dir(),
                                f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(png_path, img)
    return img, png_path
