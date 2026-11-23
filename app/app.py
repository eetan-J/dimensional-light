"""Dimensional Light - Luminar-style "Light Depth" for Lightroom Classic.

Usage:
  python app.py --input photo.tif --output photo-DimLight.tif [--model large|base|small]
  python app.py --demo            (synthetic image + synthetic depth, no model needed)

Exit code 0 = result written to --output, 1 = cancelled / error.
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import tempfile
import threading
import time
import traceback

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageDraw, ImageFont, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dimlight import engine  # noqa: E402
from dimlight.engine import FogLayer, Params  # noqa: E402
from dimlight.imageio import load_image, save_tiff16  # noqa: E402

LOG = os.path.join(tempfile.gettempdir(), "dimlight.log")

# Luminar Neo-like palette: plum floating panel over a near-black stage
WIN_BG = "#141214"
STAGE_BG = "#0d0c0d"
HEAD = "#5a2c3b"
BODY = "#47252f"
SEP = "#5f3846"
TEXT = "#f4ecef"
MUTED = "#c2a3ad"
TRACK = "#7b5a66"
FILL = "#efe5e8"
THUMB_BG = "#22131a"
WARM_GRAD = ("#4f6fd8", "#9a8f96", "#e6b43c")   # blue -> neutral -> amber

# per-fog-layer controls: key, label, min, max, default, slider options
FOG_SLIDERS = [
    ("density", "Density", 0, 100, 0, {}),
    ("width", "Fog Depth Width", 5, 100, 60, {}),
    ("softness", "Softness", 0, 100, 30, {}),
    ("ground", "Ground Fog", 0, 100, 0, {}),
    ("height", "Fog Height", 0, 100, 50, {}),
    ("cloud", "Cloudiness", 0, 100, 60, {}),
    ("size", "Cloud Size", 0, 100, 80, {}),
    ("seed", "Cloud Shape", 0, 100, 0, {}),
    ("brightness", "Brightness", 0, 100, 80, {}),
    ("warmth", "Warmth", -100, 100, 0, {"gradient": WARM_GRAD}),
]

S = 1.0    # UI scale from screen DPI (set in main)
SS = 3     # supersampling factor for crisp, anti-aliased widgets


def px(n: float) -> int:
    return int(round(n * S))


PREFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefs.json")


def load_prefs() -> dict:
    try:
        import json
        with open(PREFS, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_prefs(p: dict):
    try:
        import json
        cur = load_prefs()
        cur.update(p)
        with open(PREFS, "w", encoding="utf-8") as f:
            json.dump(cur, f)
    except OSError:
        pass


def log(msg: str):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except OSError:
        pass


def _hex(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def _mix(a, b, t):
    a, b = _hex(a), _hex(b)
    t = min(1.0, max(0.0, t))
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


_FONTS: dict = {}


def font(pt: float, bold: bool = False):
    """Segoe UI at `pt` points, sized for the supersampled surface."""
    key = (pt, bold)
    if key not in _FONTS:
        size = max(6, int(round(pt * 4 / 3 * S * SS)))
        fdir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        for name in (("seguisb.ttf", "segoeuib.ttf") if bold else ("segoeui.ttf",)):
            try:
                _FONTS[key] = ImageFont.truetype(os.path.join(fdir, name), size)
                break
            except OSError:
                continue
        else:
            _FONTS[key] = ImageFont.load_default(size)
    return _FONTS[key]


class Hi:
    """Supersampled drawing surface -> anti-aliased PhotoImage. Coordinates in logical px."""

    def __init__(self, w, h, bg):
        self.w, self.h = w, h
        self.im = Image.new("RGB", (w * SS, h * SS), bg)
        self.d = ImageDraw.Draw(self.im)

    @staticmethod
    def k(*v):
        return [x * SS for x in v]

    def photo(self, im=None):
        return ImageTk.PhotoImage((im or self.im).resize((self.w, self.h), Image.LANCZOS))


# --------------------------------------------------------------------------
# Widgets
# --------------------------------------------------------------------------
class LSlider:
    """Luminar-style slider: label + value on one line, thin track, round knob."""

    def __init__(self, parent, label, lo, hi, default, on_change, bipolar=False,
                 gradient=None, fmt=None):
        self.label, self.lo, self.hi, self.default = label, lo, hi, default
        self.value, self.on_change = default, on_change
        self.bipolar, self.gradient = bipolar, gradient
        self.fmt = fmt or (lambda v: f"{int(round(v))}")
        self.W, self.H = px(256), px(39)
        self.c = tk.Canvas(parent, width=self.W, height=self.H, bg=BODY, highlightthickness=0)
        self.c.pack(padx=px(16), pady=(px(2), 0))
        self.c.bind("<Button-1>", self._press)
        self.c.bind("<B1-Motion>", self._press)
        self.c.bind("<Double-Button-1>", lambda e: self.reset())
        self.c.bind("<MouseWheel>", self._wheel)
        self.photo = None
        self.draw()

    def _x(self, v):
        x0, x1 = px(8), self.W - px(8)
        return x0 + (v - self.lo) / (self.hi - self.lo) * (x1 - x0)

    def _v(self, x):
        x0, x1 = px(8), self.W - px(8)
        t = min(1.0, max(0.0, (x - x0) / (x1 - x0)))
        return self.lo + t * (self.hi - self.lo)

    def draw(self):
        h = Hi(self.W, self.H, BODY)
        d, k = h.d, SS
        ty, t = px(28) * k, max(1.0, 1.25 * S) * k
        d.text((px(2) * k, px(10) * k), self.label, font=font(10), fill=TEXT, anchor="lm")
        d.text(((self.W - px(2)) * k, px(10) * k), self.fmt(self.value), font=font(10),
               fill=TEXT, anchor="rm")
        x0, x1, xv = self._x(self.lo) * k, self._x(self.hi) * k, self._x(self.value) * k
        if self.gradient:
            n = int(x1 - x0)
            u = np.linspace(0, 1, n)[:, None]
            a, b, c = (np.array(_hex(g), np.float32) for g in self.gradient)
            row = np.where(u < 0.5, a + (b - a) * (u * 2), b + (c - b) * (u * 2 - 1))
            grad = Image.fromarray(np.repeat(row[None].astype(np.uint8), int(2 * t) + 1, 0))
            m = Image.new("L", grad.size, 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, grad.size[0] - 1, grad.size[1] - 1],
                                                radius=t, fill=255)
            h.im.paste(grad, (int(x0), int(ty - t)), m)
        else:
            d.rounded_rectangle([x0, ty - t, x1, ty + t], radius=t, fill=TRACK)
            org = self._x(0) * k if self.bipolar else x0
            a, b = sorted((org, xv))
            if b - a > 1:
                d.rounded_rectangle([a, ty - t, b, ty + t], radius=t, fill=FILL)
        r = px(7.5) * k
        d.ellipse([xv - r - 3 * k * S, ty - r - 3 * k * S, xv + r + 3 * k * S, ty + r + 3 * k * S],
                  fill=BODY)
        d.ellipse([xv - r, ty - r + k, xv + r, ty + r + k], fill="#2c161d")      # soft shadow
        d.ellipse([xv - r, ty - r, xv + r, ty + r], fill="#ffffff")
        self.photo = h.photo()
        self.c.delete("all")
        self.c.create_image(0, 0, image=self.photo, anchor="nw")

    def _press(self, e):
        self.set(self._v(e.x), notify=True)

    def _wheel(self, e):
        step = (self.hi - self.lo) / 100.0
        self.set(self.value + (step if e.delta > 0 else -step), notify=True)

    def set(self, v, notify=False):
        self.value = float(min(self.hi, max(self.lo, v)))
        self.draw()
        if notify:
            self.on_change()

    def get(self):
        return self.value

    def reset(self):
        self.set(self.default, notify=True)


class Icon:
    """Anti-aliased line icon (eye / undo / info / cube / chevron)."""

    def __init__(self, parent, kind, command=None, bg=HEAD, size=24):
        self.kind, self.on, self.bg = kind, True, bg
        self.up = False
        self.size = px(size)
        self.c = tk.Canvas(parent, width=self.size, height=self.size, bg=bg,
                           highlightthickness=0, cursor="hand2" if command else "")
        if command:
            self.c.bind("<Button-1>", lambda e: command())
            self.c.bind("<Enter>", lambda e: self.draw("#ffffff"))
            self.c.bind("<Leave>", lambda e: self.draw())
        self.photo = None
        self.draw()

    def draw(self, col=None):
        col = col or (MUTED if self.kind == "chevron" else TEXT)
        h = Hi(self.size, self.size, self.bg)
        d = h.d
        q = self.size * SS / 24.0                 # icon grid is 24 x 24
        P = lambda *v: [x * q for x in v]         # noqa: E731
        w = max(1, int(round(1.5 * q)))
        if self.kind == "eye":
            d.ellipse(P(2.5, 7, 21.5, 17), outline=col, width=w)
            d.ellipse(P(9.3, 9.3, 14.7, 14.7), fill=col)
            if not self.on:
                d.line(P(4, 20, 20, 4), fill=col, width=w)
        elif self.kind == "undo":
            d.arc(P(5, 5.5, 19.5, 20), start=200, end=470, fill=col, width=w)
            d.polygon(P(3.2, 13.5, 3.2, 7.2, 9.2, 10.6), fill=col)
        elif self.kind == "info":
            d.ellipse(P(3.5, 3.5, 20.5, 20.5), outline=col, width=w)
            d.rectangle(P(11.2, 10.5, 12.8, 17), fill=col)
            d.ellipse(P(11, 6.8, 13, 8.8), fill=col)
        elif self.kind == "cube":
            d.polygon(P(12, 2.5, 20.5, 7.3, 20.5, 16.7, 12, 21.5, 3.5, 16.7, 3.5, 7.3),
                      outline=col, width=w)
            d.polygon(P(12, 12, 20.5, 7.3, 20.5, 16.7, 12, 21.5), fill=col)
        elif self.kind == "chevron":
            pts = P(7, 14.5, 12, 9.5, 17, 14.5) if self.up else P(7, 9.5, 12, 14.5, 17, 9.5)
            d.line(pts, fill=col, width=w, joint="curve")
        self.photo = h.photo()
        self.c.delete("all")
        self.c.create_image(0, 0, image=self.photo, anchor="nw")


class DepthPad:
    """Depth-map thumbnail with a perspective floor grid, the light plane and its beam.

    Drag the handle up/down to move the light plane, left/right to change beam width."""

    def __init__(self, parent, on_change, on_release, glow=(255, 224, 190, 34), on_rotate=None):
        self.on_change, self.on_release = on_change, on_release
        self.glow = glow
        self.tilt = 0.0                 # -1..1 plane rotation (drawn as a tilted plane)
        self.on_rotate = on_rotate
        self.W, self.H = px(256), px(208)
        self.c = tk.Canvas(parent, width=self.W, height=self.H, bg=THUMB_BG, highlightthickness=0,
                           cursor="fleur")
        self.c.pack(padx=px(16), pady=(px(8), px(4)))
        self.base = None
        self.photo = None
        self.pivot, self.beam = 0.5, 1.0
        self.gy0, self.gy1 = self.H * 0.48, self.H * 0.93
        self.start = None
        self.c.bind("<Button-1>", self._press)
        self.c.bind("<B1-Motion>", self._drag)
        self.c.bind("<ButtonRelease-1>", lambda e: self.on_release())
        self.c.bind("<MouseWheel>", self._wheel)
        self.c.bind("<Shift-MouseWheel>", self._wheel_beam)
        self.c.bind("<Control-MouseWheel>", lambda e: self.on_rotate and self.on_rotate(
            max(-1.0, min(1.0, self.tilt + (0.05 if e.delta > 0 else -0.05)))))
        self.draw()

    def _y(self, v):
        return self.gy0 + (self.gy1 - self.gy0) * v

    def set_depth(self, depth):
        W, H = self.W * SS, self.H * SS
        h, w = depth.shape
        sc = max(W / w, H / h)                               # cover-crop
        d = cv2.resize(depth, (max(W, int(w * sc)), max(H, int(h * sc))), interpolation=cv2.INTER_CUBIC)
        y0, x0 = (d.shape[0] - H) // 2, (d.shape[1] - W) // 2
        d = np.clip(d[y0:y0 + H, x0:x0 + W], 0, 1)
        g = 0.05 + 0.88 * np.power(d, 1.2)
        ramp = np.clip((np.arange(H) - (self.gy0 - px(34)) * SS) / (px(46) * SS), 0, 1)[:, None]
        g = g * (1 - 0.74 * ramp)                            # fade into the 'floor'
        rgb = np.stack([g, g * 0.92, g * 0.95], -1)
        self.base = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        self.draw()

    # interaction: press jumps the plane, drag moves plane (y) and beam width (x)
    def _press(self, e):
        if abs(e.y - self._y(self.pivot)) > px(20):
            self.pivot = min(1.0, max(0.0, (e.y - self.gy0) / (self.gy1 - self.gy0)))
            self.on_change(self.pivot, self.beam)
        self.start = (e.x, e.y, self.pivot, self.beam)

    def _drag(self, e):
        if not self.start:
            return
        x0, y0, p0, b0 = self.start
        p = min(1.0, max(0.0, p0 + (e.y - y0) / (self.gy1 - self.gy0)))
        b = min(1.0, max(0.05, b0 + (e.x - x0) / (self.W * 0.7)))
        self.on_change(p, b)

    def _wheel(self, e):
        self.on_change(min(1.0, max(0.0, self.pivot + (-0.01 if e.delta > 0 else 0.01))), self.beam)

    def _wheel_beam(self, e):
        self.on_change(self.pivot, min(1.0, max(0.05, self.beam + (0.02 if e.delta > 0 else -0.02))))

    def draw(self):
        W, H, k = self.W * SS, self.H * SS, SS
        if self.base is not None:
            im = self.base.copy().convert("RGBA")
        else:
            im = Image.new("RGBA", (W, H), THUMB_BG)
            ImageDraw.Draw(im).text((W / 2, H * 0.26), "Analyzing depth…", font=font(9),
                                    fill=MUTED, anchor="mm")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        gy0, gy1 = self.gy0 * k, self.gy1 * k
        yp = self._y(self.pivot) * k
        end = self.pivot + self.beam
        ye = self._y(end) * k if end < 1.0 else H

        tilt_px = 0.3 * self.tilt * (gy1 - gy0)        # rotated plane: left/right depth offset

        def curve(y, bend):
            return [(W * (0.02 + 0.96 * u),
                     y + bend * (1 - (2 * u - 1) ** 2) - bend * 0.45 + tilt_px * (2 * u - 1))
                    for u in np.linspace(0, 1, 33)]

        bend = px(7) * k
        # warm glow of the beam between the plane and the beam end
        d.polygon(curve(yp, bend) + curve(ye, bend)[::-1], fill=self.glow)
        # perspective floor grid of crosses
        rows, cols = 9, 13
        for i in range(rows):
            t = i / (rows - 1)
            y = gy0 + (gy1 - gy0) * t ** 1.35
            half = W * (0.36 + 0.31 * t)
            arm = (1.4 + 2.1 * t) * S * k
            lit = yp - k <= y <= ye + k
            a = int((150 + 105 * t) if lit else (55 + 60 * t))
            col = (255, 246, 240, a) if lit else (205, 180, 190, a)
            lw = max(1, int(round(0.9 * S * k)))
            for j in range(cols):
                x = W / 2 + (j / (cols - 1) * 2 - 1) * half
                d.line([x - arm, y, x + arm, y], fill=col, width=lw)
                d.line([x, y - arm, x, y + arm], fill=col, width=lw)
        # light plane and beam end
        d.line(curve(yp, bend), fill=(255, 255, 255, 245), width=max(1, int(1.4 * S * k)), joint="curve")
        if end < 1.0:
            d.line(curve(ye, bend), fill=(255, 255, 255, 120), width=max(1, int(1.0 * S * k)),
                   joint="curve")
        cx = W / 2
        d.polygon([cx - 5 * S * k, gy0 - 9 * S * k, cx + 5 * S * k, gy0 - 9 * S * k,
                   cx, gy0 - 16 * S * k], fill=(255, 255, 255, 255))
        d.rounded_rectangle([cx - 8 * S * k, H - 5 * S * k, cx + 8 * S * k, H - 3 * S * k],
                            radius=k * S, fill=(255, 255, 255, 230))
        # handle: ring + white knob with 4-way arrows (up/down = plane, left/right = beam)
        R, r = 16 * S * k, 10.5 * S * k
        d.ellipse([cx - R, yp - R, cx + R, yp + R], outline=(255, 255, 255, 235),
                  width=max(1, int(1.3 * S * k)))
        d.ellipse([cx - r, yp - r + k, cx + r, yp + r + k], fill=(30, 12, 20, 120))
        d.ellipse([cx - r, yp - r, cx + r, yp + r], fill=(255, 255, 255, 255))
        a, o, tip = 2.6 * S * k, 2.0 * S * k, 6.6 * S * k
        dark = (58, 32, 41, 255)
        d.polygon([cx - a, yp - o, cx + a, yp - o, cx, yp - tip], fill=dark)
        d.polygon([cx - a, yp + o, cx + a, yp + o, cx, yp + tip], fill=dark)
        for sgn in (-1, 1):   # small beam arrows outside the ring
            bx = cx + sgn * (R + 7 * S * k)
            d.polygon([bx, yp - 3.2 * S * k, bx, yp + 3.2 * S * k, bx + sgn * 4.2 * S * k, yp],
                      fill=(255, 255, 255, 200))
        im.alpha_composite(ov)
        self.photo = ImageTk.PhotoImage(im.convert("RGB").resize((self.W, self.H), Image.LANCZOS))
        self.c.delete("all")
        self.c.create_image(0, 0, image=self.photo, anchor="nw")


class LightDial:
    """Shaded sphere for aiming the 3D side light: drag the light spot around the ball.
    Distance from the centre sets the height (centre = straight on, rim = grazing),
    the angle sets the direction.  The ball previews how surfaces will be lit."""

    def __init__(self, parent, on_change, az=315.0, el=35.0, title="Light Angle"):
        self.on_change = on_change
        self.az, self.el = az, el
        self.title, self.default = title, (az, el)
        self.W, self.H = px(256), px(124)
        self.R = px(52)
        self.cx, self.cy = px(62), self.H / 2
        self.c = tk.Canvas(parent, width=self.W, height=self.H, bg=BODY, highlightthickness=0,
                           cursor="crosshair")
        self.c.pack(padx=px(16), pady=(px(4), px(2)))
        self.c.bind("<Button-1>", self._drag)
        self.c.bind("<B1-Motion>", self._drag)
        self.c.bind("<Double-Button-1>", lambda e: self.set(*self.default, True))
        self.photo = None
        self.draw()

    def _vec(self):
        az, el = math.radians(self.az), math.radians(self.el)
        return math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el)

    def _drag(self, e):
        dx, dy = (e.x - self.cx) / self.R, (e.y - self.cy) / self.R
        r = min(1.0, math.hypot(dx, dy))
        az = math.degrees(math.atan2(dx, -dy)) % 360.0
        el = max(5.0, min(88.0, 90.0 * (1.0 - r)))
        self.set(az, el, True)

    def set(self, az, el, notify=False):
        self.az, self.el = az, el
        self.draw()
        if notify:
            self.on_change(az, el)

    def draw(self):
        h = Hi(self.W, self.H, BODY)
        k = SS
        R, cx, cy = self.R * k, self.cx * k, self.cy * k
        n = int(2 * R) + 1
        yy, xx = (np.mgrid[0:n, 0:n].astype(np.float32) - R) / R
        rr = xx * xx + yy * yy
        inside = rr <= 1.0
        nz = np.sqrt(np.clip(1.0 - rr, 0, 1))
        lx, ly, lz = self._vec()
        lam = np.clip(xx * lx + yy * ly + nz * lz, 0, 1)
        spec = np.clip(lam, 0, 1) ** 24 * 0.35
        shade = 0.10 + 0.80 * lam + spec
        rgb = np.stack([shade * 0.98, shade * 0.90, shade * 0.93], -1)
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        ball = Image.fromarray(rgb)
        mask = Image.fromarray((inside * 255).astype(np.uint8))
        h.im.paste(ball, (int(cx - R), int(cy - R)), mask)
        d = h.d
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline="#7b5a66", width=max(1, int(S * k)))
        # light spot: projected light direction on the ball
        sx, sy = cx + lx * R, cy + ly * R
        rs = 6 * S * k
        d.ellipse([sx - rs - 2 * k, sy - rs - 2 * k, sx + rs + 2 * k, sy + rs + 2 * k],
                  fill="#2c161d")
        d.ellipse([sx - rs, sy - rs, sx + rs, sy + rs], fill="#ffd27a")
        tx = (self.cx + self.R + px(18)) * k
        d.text((tx, px(30) * k), self.title, font=font(10), fill=TEXT, anchor="lm")
        d.text((tx, px(56) * k), f"Direction  {self.az:.0f}°", font=font(9), fill=MUTED, anchor="lm")
        d.text((tx, px(76) * k), f"Height  {self.el:.0f}°", font=font(9), fill=MUTED, anchor="lm")
        d.text((tx, px(98) * k), "drag the light spot", font=font(8), fill=MUTED, anchor="lm")
        self.photo = h.photo()
        self.c.delete("all")
        self.c.create_image(0, 0, image=self.photo, anchor="nw")


class Pill(tk.Label):
    def __init__(self, parent, text, command, primary=False, bg=BODY):
        self.bg0 = bg
        super().__init__(parent, text=text, padx=px(14), pady=px(6), cursor="hand2",
                         font=("Segoe UI Semibold" if primary else "Segoe UI", 10))
        self.primary = primary
        self.set_active(primary)
        self.bind("<Button-1>", lambda e: command())
        self.bind("<Enter>", lambda e: self.config(bg=_mix(self.base, "#ffffff", 0.15)))
        self.bind("<Leave>", lambda e: self.config(bg=self.base))

    def set_active(self, on):
        self.base = "#ece4e7" if on else _mix(self.bg0, "#ffffff", 0.10)
        self.config(bg=self.base, fg="#2a1a20" if on else TEXT)


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
class Level:
    """One preview resolution: image in sRGB / linear / 8-bit, plus refined depth."""

    def __init__(self, full_u, maxv, long_edge):
        H, W = full_u.shape[:2]
        s = min(1.0, long_edge / max(H, W))
        if s < 1.0:
            img = cv2.resize(full_u, (max(2, round(W * s)), max(2, round(H * s))),
                             interpolation=cv2.INTER_AREA)
        else:
            img = full_u
        self.srgb = (img.astype(np.float32) / maxv).clip(0, 1)
        self.lin = engine.srgb_to_lin(self.srgb)
        self.gray = self.srgb.mean(axis=2).astype(np.float32)
        self.u8 = (self.srgb * 255 + 0.5).astype(np.uint8)
        self.h, self.w = self.u8.shape[:2]
        self.depth = None


class App:
    def __init__(self, root: tk.Tk, args):
        self.root, self.args = root, args
        self.exit_code = 1
        self.q: queue.Queue = queue.Queue()
        self.evt = threading.Event()
        self.lock = threading.Lock()
        self.closing = False
        self.busy_apply = False

        # ---- image: sharp level at screen resolution + fast level for dragging
        if args.demo:
            self.full_u, self.maxv, self.icc = self._demo_image(), 255.0, None
        else:
            self.full_u, self.maxv, self.icc = load_image(args.input)
        sharp = int(min(2600, max(root.winfo_screenwidth(), root.winfo_screenheight())))
        self.hi = Level(self.full_u, self.maxv, sharp)
        self.lo = Level(self.full_u, self.maxv, max(640, sharp // 3))
        self.ph, self.pw = self.hi.h, self.hi.w

        # ---- state
        self.params = Params()
        self.pivot, self.enabled, self.tab = 0.5, True, "adjust"
        self.depth_model = None
        self.depth_rgb = None
        self.result8 = self.hi.u8
        # brush masks per target: "light" or a fog layer key. None = applies everywhere
        self.masks: dict = {"light": None}
        self.mask_target = "light"
        self.fog_layers: list[dict] = []
        self.fog_sel = 0
        self.next_key = 1
        self.brush_mode = "brush"
        self.last_pt = None
        self.depth_info = ""
        self.photo = None
        self.geom = (0, 0, 1.0)
        self.sl: dict[str, LSlider] = {}

        self._build_ui()
        threading.Thread(target=self._render_worker, daemon=True).start()
        threading.Thread(target=self._depth_worker, daemon=True).start()
        self.root.after(20, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

    @staticmethod
    def _demo_image():
        h, w = 1400, 2100
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        sky = np.stack([0.45 + 0.3 * yy / h, 0.6 + 0.2 * yy / h, 0.85 - 0.2 * yy / h], -1)
        ground = np.stack([0.30 + 0.0 * xx, 0.45 + 0.0 * xx, 0.2 + 0.0 * xx], -1)
        ground = ground * (0.6 + 0.4 * ((np.sin(xx / 50) * np.sin(yy / 18)) > 0))[..., None]
        img = np.where((yy > h * 0.3)[..., None], ground, sky)
        r = min(h, w) * 0.22
        m = ((xx - w * 0.5) ** 2 + (yy - h * 0.62) ** 2) < r ** 2
        sph = np.stack([0.85 * np.ones_like(xx), 0.35 * np.ones_like(xx), 0.25 * np.ones_like(xx)], -1)
        img = np.where(m[..., None], sph, img)
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # ---- UI -----------------------------------------------------------
    def _build_ui(self):
        r = self.root
        r.title("Dimensional Light — Light Depth")
        r.configure(bg=WIN_BG)
        r.geometry(f"{px(1440)}x{px(900)}")
        r.minsize(px(1000), px(700))
        try:
            r.state("zoomed")
        except tk.TclError:
            pass
        st = ttk.Style()
        st.theme_use("clam")
        st.configure("L.Horizontal.TProgressbar", troughcolor="#2a2328", background="#ece4e7",
                     bordercolor=WIN_BG, lightcolor="#ece4e7", darkcolor="#ece4e7")

        bar = tk.Frame(r, bg=WIN_BG, height=px(54))
        bar.pack(side="bottom", fill="x")
        self.status = tk.Label(bar, text="", bg=WIN_BG, fg=MUTED, anchor="w", font=("Segoe UI", 9))
        self.status.pack(side="left", padx=px(18))
        self.apply_btn = Pill(bar, "Apply to Lightroom", self.apply, primary=True, bg=WIN_BG)
        self.apply_btn.pack(side="right", padx=(px(6), px(18)), pady=px(10))
        Pill(bar, "Cancel", self.cancel, bg=WIN_BG).pack(side="right", pady=px(10))
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=px(160),
                                        style="L.Horizontal.TProgressbar")

        self.stage = tk.Canvas(r, bg=STAGE_BG, highlightthickness=0)
        self.stage.pack(fill="both", expand=True)
        self.stage.bind("<Configure>", lambda e: self._compose())
        self.stage.bind("<Button-1>", self._stage_press)
        self.stage.bind("<B1-Motion>", self._stage_drag)
        self.stage.bind("<ButtonRelease-1>", lambda e: setattr(self, "last_pt", None))
        self.stage.bind("<Motion>", self._stage_motion)
        self.stage.bind("<Leave>", lambda e: self.stage.delete("brush"))

        P = tk.Frame(r, bg=BODY, highlightthickness=1, highlightbackground="#6b4352")
        self.panel = P
        P.place(in_=self.stage, relx=1.0, x=-px(18), y=px(18), anchor="ne")

        head = tk.Frame(P, bg=HEAD)
        head.pack(fill="x")
        self.icons = [Icon(head, "cube"), Icon(head, "info", self.show_info),
                      Icon(head, "undo", self.reset_all)]     # keep refs (images get GC'd)
        self.icons[0].c.pack(side="left", padx=(px(12), px(6)), pady=px(10))
        tk.Label(head, text="Light Depth", bg=HEAD, fg=TEXT,
                 font=("Segoe UI Semibold", 12)).pack(side="left")
        self.icons[1].c.pack(side="right", padx=(px(4), px(12)))
        self.icons[2].c.pack(side="right", padx=px(4))
        self.eye = Icon(head, "eye", self.toggle_enabled)
        self.eye.c.pack(side="right", padx=px(4))

        tabs = tk.Frame(P, bg=BODY)
        tabs.pack(fill="x", padx=px(14), pady=(px(10), px(4)))
        self.tab_lbl = {}
        for key, text in (("adjust", "Adjustments"), ("fog", "Atmosphere"), ("mask", "Masking")):
            lb = tk.Label(tabs, text=text, bg=BODY, font=("Segoe UI", 11), cursor="hand2", padx=px(4))
            lb.pack(side="left", padx=(0, px(8)))
            lb.bind("<Button-1>", lambda e, k=key: self.set_tab(k))
            self.tab_lbl[key] = lb
        tk.Frame(P, bg=SEP, height=1).pack(fill="x")

        # --- Adjustments
        A = tk.Frame(P, bg=BODY)
        self.adjust_frame = A
        tk.Frame(A, bg=BODY, height=px(8)).pack()
        self.sl["amount"] = LSlider(A, "Amount", 0, 100, 0, self._changed)
        self.pad = DepthPad(A, self._pad_change, lambda: None,
                            on_rotate=lambda t: self.sl["rot"].set(t * 100, notify=True))
        self.sl["beam"] = LSlider(A, "Beam Width", 5, 100, 100, self._beam_changed)
        self.sl["rot"] = LSlider(A, "Plane Rotation", -100, 100, 0, self._rot_changed,
                                 bipolar=True, fmt=lambda v: f"{v * 0.45:+.0f}°")
        self.sl["softness"] = LSlider(A, "Softness", 0, 100, 0, self._changed)
        self.sl["warmth"] = LSlider(A, "Warmth", -100, 100, 0, self._changed, gradient=WARM_GRAD)

        # collapsible sections (accordion: one open at a time keeps the panel on screen)
        self.sections = {}

        def section(key, title):
            tk.Frame(A, bg=SEP, height=1).pack(fill="x", pady=(px(4), 0))
            head = tk.Frame(A, bg=BODY, cursor="hand2")
            head.pack(fill="x")
            lbl = tk.Label(head, text=title, bg=BODY, fg=MUTED, font=("Segoe UI", 10))
            lbl.pack(side="left", padx=px(18), pady=px(6))
            chev = Icon(head, "chevron", lambda: self.toggle_section(key), bg=BODY, size=20)
            chev.c.pack(side="right", padx=px(16))
            for w in (head, lbl):
                w.bind("<Button-1>", lambda e: self.toggle_section(key))
            body = tk.Frame(A, bg=BODY)
            self.sections[key] = {"head": head, "body": body, "chev": chev, "open": False}
            return body

        Sh = section("shadow", "Shadows")
        self.shadow_on = tk.BooleanVar(value=False)
        tk.Checkbutton(Sh, text="Cast shadows from the depth light", variable=self.shadow_on,
                       command=self._changed, bg=BODY, fg=TEXT, selectcolor=HEAD,
                       activebackground=BODY, activeforeground=TEXT,
                       font=("Segoe UI", 9)).pack(anchor="w", padx=px(14), pady=(px(6), 0))
        self.shadow_dial = LightDial(Sh, lambda az, el: self._changed(), az=300.0, el=30.0,
                                     title="Light Comes From")
        self.sl["sh_strength"] = LSlider(Sh, "Shadow Strength", 0, 100, 85, self._changed)
        self.sl["sh_soft"] = LSlider(Sh, "Shadow Softness", 0, 100, 40, self._changed)
        tk.Frame(Sh, bg=BODY, height=px(8)).pack()

        V = section("adv", "Advanced Settings")
        self.sl["near_b"] = LSlider(V, "Near Brightness", -100, 100, 0, self._changed, bipolar=True)
        self.sl["near_w"] = LSlider(V, "Near Warmth", -100, 100, 0, self._changed, gradient=WARM_GRAD)
        self.sl["far_b"] = LSlider(V, "Far Brightness", -100, 100, 0, self._changed, bipolar=True)
        self.sl["far_w"] = LSlider(V, "Far Warmth", -100, 100, 0, self._changed, gradient=WARM_GRAD)
        self.sl["light"] = LSlider(V, "Side Light", -100, 100, 0, self._changed, bipolar=True)
        self.light_dial = LightDial(V, lambda az, el: self._changed())
        tk.Frame(V, bg=BODY, height=px(8)).pack()
        # global depth settings live in the bottom bar
        ai = tk.Frame(bar, bg=WIN_BG)
        ai.pack(side="right", padx=(0, px(24)))
        tk.Label(ai, text="Depth AI", bg=WIN_BG, fg=MUTED, font=("Segoe UI", 9)).pack(side="left")
        self.model_btn = {}
        for key, text in (("large", "Fast"), ("depthpro", "Pro"), ("ultra", "Ultra")):
            b = Pill(ai, text, lambda k=key: self.set_model(k), bg=WIN_BG)
            b.config(padx=px(9), pady=px(3), font=("Segoe UI", 9))
            b.pack(side="left", padx=(px(6) if key == "large" else px(4), 0))
            self.model_btn[key] = b
        for k, b in self.model_btn.items():
            b.set_active(k == self.args.model)

        # --- Atmosphere: fog layers, each with its own depth pad and settings
        F = tk.Frame(P, bg=BODY)
        self.fog_frame = F
        self.layer_bar = tk.Frame(F, bg=BODY)
        self.layer_bar.pack(fill="x", padx=px(16), pady=(px(10), 0))
        self.fog_pad = DepthPad(F, self._fog_pad_change, lambda: None,
                                glow=(190, 215, 255, 46))
        self.fsl: dict[str, LSlider] = {}
        for key, label, lo, hi, df, kw in FOG_SLIDERS:
            self.fsl[key] = LSlider(F, label, lo, hi, df, lambda k=key: self._fog_changed(k), **kw)
        tk.Frame(F, bg=SEP, height=1).pack(fill="x", pady=(px(6), 0))
        self.sl["haze"] = LSlider(F, "Haze (all layers)", 0, 100, 0, self._changed)
        tk.Frame(F, bg=BODY, height=px(6)).pack()
        self.show_depth = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="Show depth map", variable=self.show_depth,
                       command=self._compose, bg=WIN_BG, fg=MUTED, selectcolor=WIN_BG,
                       activebackground=WIN_BG, activeforeground=TEXT,
                       font=("Segoe UI", 9)).pack(side="right", padx=(0, px(16)))

        # --- Masking
        M = tk.Frame(P, bg=BODY)
        self.mask_frame = M
        tk.Label(M, text="Paint on:", bg=BODY, fg=MUTED, font=("Segoe UI", 9)).pack(
            anchor="w", padx=px(18), pady=(px(12), px(2)))
        self.target_bar = tk.Frame(M, bg=BODY)
        self.target_bar.pack(fill="x", padx=px(16), pady=(0, px(6)))
        self.target_hint = tk.Label(M, text="", bg=BODY, fg=MUTED, font=("Segoe UI", 9),
                                    wraplength=px(250), justify="left")
        self.target_hint.pack(anchor="w", padx=px(18), pady=(0, px(2)))
        # controls for the Paint Light brush (shown only when it is the target)
        self.paint_frame = tk.Frame(M, bg=BODY)
        self.psl: dict[str, LSlider] = {}
        for key, label, lo, hi, df, kw in (
                ("p_int", "Light Intensity", -100, 100, 60, {"bipolar": True}),
                ("p_spread", "Light Spread", 0, 100, 40, {}),
                ("p_reach", "Depth Reach", 0, 100, 30, {}),
                ("p_warm", "Light Warmth", -100, 100, 30, {"gradient": WARM_GRAD})):
            self.psl[key] = LSlider(self.paint_frame, label, lo, hi, df, self._changed, **kw)
        self.paint_anchor = tk.Frame(M, bg=SEP, height=1)
        self.paint_anchor.pack(fill="x", pady=(px(4), px(4)))
        tools = tk.Frame(M, bg=BODY)
        tools.pack(fill="x", padx=px(18), pady=px(4))
        self.tool_btn = {}
        for key, text in (("brush", "Brush"), ("erase", "Erase")):
            b = Pill(tools, text, lambda k=key: self.set_brush(k))
            b.pack(side="left", padx=(0, px(6)))
            self.tool_btn[key] = b
        self.sl["b_size"] = LSlider(M, "Size", 1, 100, 30, lambda: None)
        self.sl["b_soft"] = LSlider(M, "Softness", 0, 100, 50, lambda: None)
        row = tk.Frame(M, bg=BODY)
        row.pack(fill="x", padx=px(18), pady=(px(10), px(4)))
        Pill(row, "Invert", self.invert_mask).pack(side="left", padx=(0, px(6)))
        Pill(row, "Clear", self.clear_mask).pack(side="left")
        self.show_mask = tk.BooleanVar(value=True)
        tk.Checkbutton(M, text="Show mask overlay", variable=self.show_mask, command=self._compose,
                       bg=BODY, fg=TEXT, selectcolor=HEAD, activebackground=BODY,
                       activeforeground=TEXT, font=("Segoe UI", 9)).pack(
            anchor="w", padx=px(14), pady=(px(6), px(14)))

        self._add_fog_layer(select=True, notify=False)
        self.set_tab("adjust")
        self.set_brush("brush")

    # ---- fog layers -----------------------------------------------------
    def _new_layer(self):
        layer = {k: df for k, _, _, _, df, _ in FOG_SLIDERS}
        layer.update(key=self.next_key, pos=0.0, width=0.6)
        self.next_key += 1
        return layer

    def _add_fog_layer(self, select=True, notify=True):
        layer = self._new_layer()
        if self.fog_layers:                       # new layers default to a fresh cloud shape
            layer["seed"] = (len(self.fog_layers) * 37) % 100
        self.fog_layers.append(layer)
        if select:
            self.select_fog(len(self.fog_layers) - 1)
        self._rebuild_layer_ui()
        if notify:
            self._changed()

    def remove_fog_layer(self):
        if len(self.fog_layers) <= 1:
            self.fog_layers[0].update(self._new_layer() | {"key": self.fog_layers[0]["key"]})
        else:
            gone = self.fog_layers.pop(self.fog_sel)
            with self.lock:
                self.masks.pop(gone["key"], None)
            if self.mask_target == gone["key"]:
                self.mask_target = "light"
        self.select_fog(min(self.fog_sel, len(self.fog_layers) - 1))
        self._rebuild_layer_ui()
        self._changed()
        self._compose()

    def select_fog(self, i):
        self.fog_sel = i
        L = self.fog_layers[i]
        for k, s in self.fsl.items():
            s.set(L[k] * 100 if k == "width" else L[k])
        self.fog_pad.pivot, self.fog_pad.beam = L["pos"], L["width"]
        self.fog_pad.draw()
        self._rebuild_layer_ui()

    def _rebuild_layer_ui(self):
        for w in self.layer_bar.winfo_children():
            w.destroy()
        for i, L in enumerate(self.fog_layers):
            b = Pill(self.layer_bar, f"Fog {i + 1}", lambda i=i: self.select_fog(i))
            b.config(padx=px(9), pady=px(3), font=("Segoe UI", 9))
            b.set_active(i == self.fog_sel)
            b.pack(side="left", padx=(0, px(4)))
        add = Pill(self.layer_bar, "+", lambda: self._add_fog_layer())
        add.config(padx=px(8), pady=px(3), font=("Segoe UI Semibold", 9))
        add.pack(side="left", padx=(0, px(4)))
        rm = Pill(self.layer_bar, "Remove", self.remove_fog_layer)
        rm.config(padx=px(8), pady=px(3), font=("Segoe UI", 9))
        rm.pack(side="right")
        self._rebuild_targets()

    def _rebuild_targets(self):
        for w in self.target_bar.winfo_children():
            w.destroy()
        keys = ([("light", "Light"), ("paint_light", "Paint Light")]
                + [(L["key"], f"Fog {i + 1}") for i, L in enumerate(self.fog_layers)])
        for key, text in keys:
            b = Pill(self.target_bar, text, lambda k=key: self.set_mask_target(k))
            b.config(padx=px(9), pady=px(3), font=("Segoe UI", 9))
            b.set_active(key == self.mask_target)
            b.pack(side="left", padx=(0, px(4)))

    def set_mask_target(self, key):
        self.mask_target = key
        self._rebuild_targets()
        if key == "paint_light":
            hint = ("Brush adds 3D light where you paint. It lights surfaces at the depth "
                    "you paint on and fades on anything in front or behind.")
            self.paint_frame.pack(fill="x", before=self.paint_anchor)
        else:
            hint = ("Brush limits the depth light to where you paint; Erase removes it."
                    if key == "light" else
                    "Brush paints this fog layer in (only where painted); Erase clears it.")
            self.paint_frame.pack_forget()
        self.target_hint.config(text=hint)
        self._compose()

    def _fog_changed(self, key):
        L = self.fog_layers[self.fog_sel]
        L[key] = self.fsl[key].get()
        if key == "width":
            L["width"] = max(0.05, self.fsl["width"].get() / 100.0)
            self.fog_pad.beam = L["width"]
            self.fog_pad.draw()
        self._changed()

    def _fog_pad_change(self, pos, width):
        L = self.fog_layers[self.fog_sel]
        L["pos"], L["width"] = pos, width
        self.fsl["width"].set(width * 100)
        self.fog_pad.pivot, self.fog_pad.beam = pos, width
        self.fog_pad.draw()
        self._changed()

    @property
    def mask(self):
        return self.masks.get(self.mask_target)

    @mask.setter
    def mask(self, value):
        self.masks[self.mask_target] = value

    # ---- panel actions --------------------------------------------------
    def set_tab(self, key):
        self.tab = key
        for k, lb in self.tab_lbl.items():
            lb.config(fg=TEXT if k == key else MUTED)
        frames = {"adjust": self.adjust_frame, "fog": self.fog_frame, "mask": self.mask_frame}
        for f in frames.values():
            f.pack_forget()
        frames[key].pack(fill="x")
        self.stage.config(cursor="none" if key == "mask" else "crosshair")
        self.stage.delete("brush")
        self._compose()

    def toggle_section(self, key):
        opening = not self.sections[key]["open"]
        for k, s in self.sections.items():
            s["open"] = opening and k == key
            if s["open"]:
                s["body"].pack(fill="x", after=s["head"])
            else:
                s["body"].pack_forget()
            s["chev"].up = s["open"]
            s["chev"].draw()

    def toggle_advanced(self):
        self.toggle_section("adv")

    def set_model(self, key):
        if key == self.args.model or self.args.demo:
            return
        self.args.model = key
        save_prefs({"model": key})
        for k, b in self.model_btn.items():
            b.set_active(k == key)
        self.status.config(text=f"Switching depth AI to {key}…")
        threading.Thread(target=self._depth_worker, daemon=True).start()

    def toggle_enabled(self):
        self.enabled = not self.enabled
        self.eye.on = self.enabled
        self.eye.draw()
        self._compose()

    def show_info(self):
        messagebox.showinfo(
            "Light Depth",
            "Amount: overall strength of the depth light.\n\n"
            "Depth preview: drag the handle up/down to move the light plane through the scene, "
            "and left/right to change the beam width (how deep the lit zone reaches toward the "
            "camera). Mouse wheel moves the plane; Shift+wheel changes the beam. "
            "You can also click the photo to place the plane at that depth.\n\n"
            "Softness: how gradually light fades into shadow.\nWarmth: colour of the light.\n\n"
            "Advanced Settings: near/far brightness and warmth, haze and a 3D side light.\n"
            "Masking: paint where the effect applies.\n"
            "Eye: compare with the original.  Arrow: reset the tool.")

    def reset_all(self):
        for k, s in self.sl.items():
            if not k.startswith("b_"):
                s.set(s.default)
        self.pivot = 0.5
        self.pad.pivot, self.pad.beam, self.pad.tilt = 0.5, 1.0, 0.0
        self.pad.draw()
        self.light_dial.set(315.0, 35.0)
        self.shadow_dial.set(300.0, 30.0)
        self.shadow_on.set(False)
        with self.lock:
            self.masks = {"light": None}
        self.mask_target = "light"
        self.fog_layers = []
        self._add_fog_layer(select=True, notify=False)
        self._changed()
        self._compose()

    # ---- params ---------------------------------------------------------
    def _make_params(self) -> Params:
        g = lambda k: self.sl[k].get() / 100.0           # noqa: E731
        A = g("amount")
        return Params(
            near_ev=A * (1.4 + 2.5 * g("near_b")),
            far_ev=A * (-2.4 + 2.5 * g("far_b")),
            pivot=self.pivot,
            softness=0.02 + 0.28 * g("softness"),
            beam=g("beam"),
            plane_tilt=0.6 * g("rot"),
            shadows=bool(self.shadow_on.get()),
            shadow_az=self.shadow_dial.az,
            shadow_el=self.shadow_dial.el,
            shadow_soft=g("sh_soft"),
            shadow_strength=g("sh_strength"),
            near_warmth=A * float(np.clip(g("warmth") + g("near_w"), -1.5, 1.5)),
            far_warmth=A * g("far_w"),
            fogs=tuple(FogLayer(
                density=L["density"] / 100.0,
                pos=L["pos"],
                width=L["width"],
                softness=0.02 + 0.28 * L["softness"] / 100.0,
                ground=L["ground"] / 100.0,
                height=L["height"] / 100.0,
                cloud=L["cloud"] / 100.0,
                size=L["size"] / 100.0,
                seed=int(round(L["seed"])),
                brightness=0.25 + 0.75 * L["brightness"] / 100.0,
                warmth=L["warmth"] / 100.0,
                key=L["key"]) for L in self.fog_layers),
            haze=g("haze"),
            paint_ev=2.5 * self.psl["p_int"].get() / 100.0,
            paint_spread=self.psl["p_spread"].get() / 100.0,
            paint_falloff=self.psl["p_reach"].get() / 100.0,
            paint_warmth=self.psl["p_warm"].get() / 100.0,
            light_ev=A * 1.8 * g("light"),
            light_azimuth=self.light_dial.az,
            light_elevation=self.light_dial.el,
        )

    def _changed(self):
        with self.lock:
            self.params = self._make_params()
        self.evt.set()

    def _rot_changed(self):
        self.pad.tilt = self.sl["rot"].get() / 100.0
        self.pad.draw()
        self._changed()

    def _beam_changed(self):
        self.pad.beam = self.sl["beam"].get() / 100.0
        self.pad.draw()
        self._changed()

    def _pad_change(self, pivot, beam):
        self.pivot = pivot
        self.pad.pivot, self.pad.beam = pivot, beam
        self.sl["beam"].set(beam * 100)
        self.pad.draw()
        self._changed()
        if self.show_depth.get():
            self._compose()

    # ---- masking --------------------------------------------------------
    def set_brush(self, mode):
        self.brush_mode = mode
        for k, b in self.tool_btn.items():
            b.set_active(k == mode)

    def _brush_radius(self):
        return max(2.0, self.sl["b_size"].get() / 100.0 * 0.25 * max(self.pw, self.ph))

    def clear_mask(self):
        with self.lock:
            self.mask = None
        self.evt.set()
        self._compose()

    def invert_mask(self):
        with self.lock:
            self.mask = (np.zeros((self.ph, self.pw), np.float32) if self.mask is None
                         else 1.0 - self.mask)
        self.evt.set()
        self._compose()

    def _stamp(self, x, y):
        r = self._brush_radius()
        f = max(self.sl["b_soft"].get() / 100.0, 0.02)
        x0, x1 = int(max(0, x - r)), int(min(self.pw, x + r + 1))
        y0, y1 = int(max(0, y - r)), int(min(self.ph, y + r + 1))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        a = np.clip((1.0 - np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / r) / f, 0, 1)
        sub = self.mask[y0:y1, x0:x1]
        if self.brush_mode == "brush":
            np.maximum(sub, a, out=sub)
        else:
            np.minimum(sub, 1.0 - a, out=sub)

    def _paint_to(self, x, y):
        with self.lock:
            if self.mask is None:   # first stroke: brush = only here, erase = everywhere but here
                fill = 0.0 if (self.brush_mode == "brush"
                               or self.mask_target == "paint_light") else 1.0
                self.mask = np.full((self.ph, self.pw), fill, np.float32)
            lx, ly = self.last_pt if self.last_pt else (x, y)
            n = int(math.hypot(x - lx, y - ly) / max(1.0, self._brush_radius() * 0.25)) + 1
            for i in range(1, n + 1):
                t = i / n
                self._stamp(lx + (x - lx) * t, ly + (y - ly) * t)
        self.last_pt = (x, y)
        self.evt.set()

    # ---- stage events ---------------------------------------------------
    def _to_img(self, cx, cy):
        ox, oy, sc = self.geom
        return (cx - ox) / sc, (cy - oy) / sc

    def _stage_press(self, e):
        x, y = self._to_img(e.x, e.y)
        if self.tab == "mask":
            self.last_pt = None
            self._paint_to(x, y)
        elif self.hi.depth is not None and 0 <= x < self.pw and 0 <= y < self.ph:
            xi, yi = int(x), int(y)
            v = float(np.median(self.hi.depth[max(0, yi - 4):yi + 5, max(0, xi - 4):xi + 5]))
            self._pad_change(v, self.pad.beam)

    def _stage_drag(self, e):
        if self.tab == "mask":
            x, y = self._to_img(e.x, e.y)
            self._paint_to(x, y)
            self._stage_motion(e)

    def _stage_motion(self, e):
        self.stage.delete("brush")
        if self.tab == "mask":
            rr = self._brush_radius() * self.geom[2]
            self.stage.create_oval(e.x - rr, e.y - rr, e.x + rr, e.y + rr, outline="#ffffff",
                                   width=1, tags="brush")
            self.stage.create_oval(e.x - 1, e.y - 1, e.x + 1, e.y + 1, fill="#ffffff",
                                   outline="", tags="brush")

    # ---- workers --------------------------------------------------------
    def _depth_worker(self):
        """Estimate depth.  For Ultra, Depth Pro is shown first so you can start editing;
        the Ultra map keeps computing in the background and is swapped in when ready
        (sliders, layers and brush strokes are kept)."""
        self.depth_gen = gen = getattr(self, "depth_gen", 0) + 1
        model = self.args.model
        progress = lambda m: self.q.put(("status", m))       # noqa: E731

        def publish(d, info):
            if gen != self.depth_gen:          # model was switched meanwhile: drop result
                return False
            if model == "large" and not self.args.demo:      # Pro / Ultra are log-metric
                d = engine.spread_depth(d, 0.35)
            levels = [engine.upsample_depth(d, lv.gray) for lv in (self.lo, self.hi)]
            self.depth_model = d
            self.lo.depth, self.hi.depth = levels
            self.depth_info = info
            self.q.put(("depth", None))
            return True

        full_cache = {}

        def image():
            if "img" not in full_cache:
                full = self.full_u.astype(np.float32) / self.maxv
                sc = 2400 / max(full.shape[:2])
                if sc < 1:
                    full = cv2.resize(full, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                full_cache["img"] = full
            return full_cache["img"]

        def run(m, base=None):
            cache = f"{self.args.input}.{m}.v2.depth.npy"
            if os.path.exists(cache):
                return np.load(cache).astype(np.float32), "cached"
            from dimlight.depth import estimate_depth
            d, info = estimate_depth(image(), m, progress=progress, base=base)
            np.save(cache, d.astype(np.float16))
            return d, info

        try:
            t0 = time.time()
            if self.args.demo:
                from dimlight.depth import demo_depth
                publish(demo_depth(self.hi.h, self.hi.w), "synthetic demo depth")
                return
            ultra_cached = os.path.exists(f"{self.args.input}.ultra.v2.depth.npy")
            if model == "ultra" and not ultra_cached:
                base, _ = run("depthpro")
                if not publish(base, "Depth Pro preview · Ultra refining in background…"):
                    return
                self.q.put(("status", "Depth Pro ready — you can edit now. "
                                      "Ultra depth is refining in the background…"))
                d, _ = run("ultra", base=base)
                publish(d, f"Ultra depth swapped in · {time.time() - t0:.0f}s")
            else:
                d, info = run(model)
                publish(d, f"{info} · {time.time() - t0:.1f}s")
        except Exception as ex:                      # noqa: BLE001
            log(traceback.format_exc())
            self.q.put(("error", f"Depth estimation failed:\n{ex}"))

    def _snapshot_masks(self):
        with self.lock:
            return {k: m.copy() for k, m in self.masks.items() if m is not None}

    def _render(self, lv: Level, p: Params, masks: dict):
        if p.is_identity():
            return lv.u8
        fog_masks = {k: m for k, m in masks.items() if k not in ("light", "paint_light")}
        maps = engine.compute_maps(lv.depth, p, fog_masks, masks.get("paint_light"))
        lm = masks.get("light")
        if lm is not None and lm.shape != (lv.h, lv.w):
            lm = cv2.resize(lm, (lv.w, lv.h), interpolation=cv2.INTER_AREA)
        res = engine.apply_maps_lin(lv.lin, maps, p, fast=True, light_mask=lm)
        return (np.clip(res, 0, 1) * 255 + 0.5).astype(np.uint8)

    def _render_worker(self):
        """Fast low-res frame immediately, sharp frame once the sliders settle."""
        while True:
            self.evt.wait()
            self.evt.clear()
            if self.closing:
                return
            if self.hi.depth is None:
                continue
            try:
                p = self.params
                masks = self._snapshot_masks()
                self.q.put(("result", self._render(self.lo, p, masks)))
                if self.evt.wait(0.12):          # still dragging: skip the sharp pass
                    continue
                if self.closing:
                    return
                self.q.put(("result", self._render(self.hi, p, masks)))
            except Exception:                    # noqa: BLE001
                log(traceback.format_exc())

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "status":
                    self.status.config(text=payload)
                elif kind == "depth":
                    self.depth_rgb = engine.depth_colour(self.hi.depth)
                    self.pad.set_depth(self.hi.depth)
                    self.fog_pad.set_depth(self.hi.depth)
                    self.status.config(text=f"Depth ready  ·  {self.depth_info}")
                    self.evt.set()
                    self._compose()
                elif kind == "result":
                    self.result8 = payload
                    self._compose()
                elif kind == "error":
                    messagebox.showerror("Dimensional Light", payload)
                    self.status.config(text=payload.splitlines()[0])
                elif kind == "done":
                    self.exit_code = 0
                    self._quit()
                    return
        except queue.Empty:
            pass
        if not self.closing:
            self.root.after(15, self._poll)

    # ---- drawing --------------------------------------------------------
    def _compose(self):
        cw, ch = self.stage.winfo_width(), self.stage.winfo_height()
        if cw < 20 or ch < 20:
            return
        panel_w = self.panel.winfo_width() + px(36) if cw > px(1100) else 0
        aw, ah = cw - panel_w - px(24), ch - px(24)
        sc = min(aw / self.pw, ah / self.ph)
        nw, nh = max(1, int(self.pw * sc)), max(1, int(self.ph * sc))

        def fit(a, interp=None):
            if a.shape[1] == nw and a.shape[0] == nh:
                return a
            up = a.shape[1] < nw
            return cv2.resize(a, (nw, nh), interpolation=interp or
                              (cv2.INTER_CUBIC if up else cv2.INTER_AREA))

        img = fit(self.result8 if self.enabled else self.hi.u8)
        if self.depth_rgb is not None and self.show_depth.get():
            dep = fit(self.hi.depth, cv2.INTER_AREA)
            img = (img * 0.45 + fit(self.depth_rgb) * 0.55).astype(np.uint8)
            img[engine.focus_line(dep, self.pivot)] = (255, 255, 255)
        if self.tab == "mask" and self.show_mask.get() and self.mask is not None:
            m = fit(self.mask, cv2.INTER_AREA)
            if self.mask_target == "paint_light":     # show painted light strokes in amber
                a = (0.45 * m)[..., None]
                col = np.array([255, 190, 60], np.float32)
            else:                                     # show where the effect is removed
                a = (0.5 * (1.0 - m))[..., None]
                col = np.array([200, 30, 60], np.float32)
            img = (img * (1 - a) + col * a).astype(np.uint8)

        self.photo = ImageTk.PhotoImage(Image.fromarray(np.ascontiguousarray(np.clip(img, 0, 255)).astype(np.uint8)))
        ox, oy = px(12) + (aw - nw) // 2, px(12) + (ah - nh) // 2
        self.geom = (ox, oy, sc)
        self.stage.delete("img")
        self.stage.create_image(ox, oy, image=self.photo, anchor="nw", tags="img")
        self.stage.tag_lower("img")
        if not self.enabled:
            self.stage.create_text(ox + px(12), oy + px(12), text="ORIGINAL", fill="#ffffff",
                                   anchor="nw", font=("Segoe UI Semibold", 9), tags="img")

    # ---- finish ---------------------------------------------------------
    def apply(self):
        if self.busy_apply or self.depth_model is None:
            return
        self.busy_apply = True
        self.apply_btn.config(text="Rendering…")
        self.progress.pack(side="right", padx=px(12))
        self.progress.start(12)
        p = self.params
        masks = self._snapshot_masks()
        threading.Thread(target=self._apply_worker, args=(p, masks), daemon=True).start()

    def _apply_worker(self, p: Params, masks: dict):
        try:
            img = self.full_u
            H, W = img.shape[:2]
            gray = np.empty((H, W), np.float32)
            for y in range(0, H, 512):
                gray[y:y + 512] = img[y:y + 512].astype(np.float32).mean(axis=2) / self.maxv
            depth = engine.upsample_depth(self.depth_model, gray)
            del gray
            fog_masks = {k: m for k, m in masks.items() if k not in ("light", "paint_light")}
            maps = None if p.is_identity() else engine.compute_maps(
                depth, p, fog_masks, masks.get("paint_light"))
            lm = masks.get("light")
            lfull = None if lm is None else cv2.resize(lm, (W, H), interpolation=cv2.INTER_LINEAR)
            out = np.empty((H, W, 3), np.uint16)
            for y0 in range(0, H, 512):
                y1 = min(H, y0 + 512)
                strip = (img[y0:y1].astype(np.float32) / self.maxv).clip(0, 1)
                if maps is not None:
                    strip = engine.apply_maps_srgb_strip(strip, maps, p, y0, y1, light_mask=lfull)
                out[y0:y1] = (np.clip(strip, 0, 1) * 65535 + 0.5).astype(np.uint16)
            save_tiff16(self.args.output, out, self.icc)
            if self.args.save_depth:
                cv2.imwrite(os.path.splitext(self.args.output)[0] + "-depth.png",
                            (depth * 65535).astype(np.uint16))
            self.q.put(("done", None))
        except Exception as ex:                      # noqa: BLE001
            log(traceback.format_exc())
            self.q.put(("error", f"Rendering failed:\n{ex}"))
            self.root.after(0, self._apply_failed)

    def _apply_failed(self):
        self.busy_apply = False
        self.progress.stop()
        self.progress.pack_forget()
        self.apply_btn.config(text="Apply to Lightroom")

    def cancel(self):
        self.exit_code = 1
        self._quit()

    def _quit(self):
        self.closing = True
        self.evt.set()
        self.root.destroy()


# --------------------------------------------------------------------------
def selftest(app: App):
    """Drive the UI programmatically (development testing only)."""
    from PIL import ImageGrab

    def grab(suffix):
        app.root.update()
        x, y = app.root.winfo_rootx(), app.root.winfo_rooty()
        base, ext = os.path.splitext(app.args.screenshot)
        ImageGrab.grab((x, y, x + app.root.winfo_width(), y + app.root.winfo_height())).save(
            base + suffix + ext)

    def wait_depth(then):
        if app.hi.depth is None:
            app.root.after(200, lambda: wait_depth(then))
        else:
            then()

    def step1():
        app.sl["amount"].set(45)
        app.sl["softness"].set(40)
        app.sl["warmth"].set(35)
        app._pad_change(0.5, 1.0)
        app.sl["rot"].set(-20, notify=True)              # plane slightly rotated
        app.root.after(1500, step2)

    def step2():
        grab("")
        app.set_tab("fog")
        app.fsl["density"].set(30, notify=True)           # Fog 1: thin mist far back
        app.fsl["ground"].set(80, notify=True)
        app.fsl["height"].set(55, notify=True)
        app._fog_pad_change(0.0, 0.4)
        app._add_fog_layer()                               # Fog 2: faint low mist mid-depth
        app.fsl["density"].set(22, notify=True)
        app.fsl["ground"].set(90, notify=True)
        app.fsl["height"].set(40, notify=True)
        app.fsl["warmth"].set(25, notify=True)
        app._fog_pad_change(0.4, 0.25)
        app.sl["haze"].set(10, notify=True)
        app.root.after(1800, step3)

    def step3():
        grab("-fog")
        app.set_tab("mask")
        app.set_mask_target(app.fog_layers[1]["key"])      # erase only Fog 2 on the left
        app.set_brush("erase")
        for i in range(30):
            app._paint_to(app.pw * (0.05 + 0.4 * i / 29), app.ph * 0.6)
        app.last_pt = None
        app.set_mask_target("paint_light")                # paint 3D light on the door
        app.set_brush("brush")
        for i in range(20):
            app._paint_to(app.pw * (0.62 + 0.12 * i / 19), app.ph * (0.62 + 0.05 * i / 19))
        app.last_pt = None
        app._compose()
        app.root.after(1800, step4)

    def step4():
        grab("-mask")
        app.set_tab("adjust")
        app.root.after(500, step5)

    def step5():                    # wait for a background Ultra swap, if one is running
        if "refining" in app.depth_info:
            app.root.after(1000, step5)
            return
        app.root.after(1500, lambda: (grab("-final"), app.apply()))

    app.root.after(800, lambda: wait_depth(step1))


def main():
    global S
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--model", default=None, help="large | depthpro (default: last used)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--save-depth", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--screenshot", default="selftest.png")
    args = ap.parse_args()
    args.model = args.model or load_prefs().get("model", "depthpro")
    if args.demo:
        args.input = args.input or "demo.png"
        args.output = args.output or "demo-out.tif"
    if not args.demo and not (args.input and args.output):
        ap.error("--input and --output are required")
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:                            # noqa: BLE001
            pass
    root = tk.Tk()
    S = max(1.0, root.winfo_fpixels("1i") / 96.0)
    try:
        app = App(root, args)
    except Exception as ex:                          # noqa: BLE001
        log(traceback.format_exc())
        messagebox.showerror("Dimensional Light", f"Could not open image:\n{ex}")
        return 1
    if args.selftest:
        selftest(app)
    root.mainloop()
    return app.exit_code


if __name__ == "__main__":
    sys.exit(main())
