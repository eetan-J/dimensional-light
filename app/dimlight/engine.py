"""Depth-aware relighting engine.

Convention: depth is a float32 map in [0, 1] where 1 = nearest to the camera
and 0 = farthest.  Everything here is numpy/OpenCV so it stays fast enough for
live slider preview at ~1400 px and can also run strip-wise on full-res files.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FogLayer:
    """One volume of fog occupying a depth window, like the light's plane + beam."""
    density: float = 0.0
    pos: float = 0.0          # far edge of the fog volume (0 = horizon .. 1 = camera)
    width: float = 0.6        # how far the volume reaches toward the camera
    softness: float = 0.10    # edge softness of the depth window
    ground: float = 0.0       # 0 = fills the frame, 1 = fog only below `height`
    height: float = 0.5       # top of the fog in the frame (0 = bottom edge, 1 = top edge)
    cloud: float = 0.6        # 0 = even volume, 1 = separate cloud banks with gaps
    size: float = 0.5         # cloud scale
    seed: int = 0             # cloud shape variation
    brightness: float = 0.8
    warmth: float = 0.0
    key: int = 0              # identifies the layer's brush mask


@dataclass
class Params:
    # Depth light: exposure applied to everything nearer / farther than `pivot`
    near_ev: float = 0.0
    far_ev: float = 0.0
    pivot: float = 0.5        # focus depth (0 far .. 1 near)
    softness: float = 0.20    # width of the near/far transition
    beam: float = 1.0         # depth range lit in front of the plane (1 = everything nearer)
    plane_tilt: float = 0.0   # rotates the light plane: depth change across the frame width
    # Cast shadows for the depth light: anything with a nearer object between it and the
    # light (seen along the light direction) is in shadow and does not get lit
    shadows: bool = False
    shadow_az: float = 300.0      # direction the light comes FROM, 0 = top, clockwise
    shadow_el: float = 30.0       # light height above the scene (lower = longer shadows)
    shadow_soft: float = 0.4
    shadow_strength: float = 0.85
    near_warmth: float = 0.0  # colour temperature of the near (lit) side, -1..1
    far_warmth: float = 0.0   # colour temperature of the far side, -1..1
    # Atmosphere: stacked volumetric fog layers + aerial haze
    fogs: tuple = field(default_factory=tuple)   # tuple[FogLayer, ...]
    haze: float = 0.0             # aerial perspective: bluish contrast loss with distance
    # Painted 3D light: brushed light that only reaches surfaces at the painted depth
    paint_ev: float = 0.0
    paint_spread: float = 0.4
    paint_falloff: float = 0.3
    paint_warmth: float = 0.0
    # Directional light shaded from the depth-derived surface normals
    light_ev: float = 0.0
    light_azimuth: float = 315.0   # degrees, 0 = from top, clockwise
    light_elevation: float = 35.0  # degrees above the picture plane
    relief: float = 2.0            # how strongly depth is treated as 3D relief

    def is_identity(self) -> bool:
        return (self.light_identity() and self.haze == 0 and not self.active_fogs()
                and self.paint_ev == 0 and self.paint_warmth == 0)

    def light_identity(self) -> bool:
        return (self.near_ev == 0 and self.far_ev == 0 and self.near_warmth == 0
                and self.far_warmth == 0 and self.light_ev == 0)

    def active_fogs(self):
        return [f for f in self.fogs if f.density > 0]


PARAM_NAMES = [f.name for f in fields(Params)]


# --------------------------------------------------------------------------
# Colour helpers
# --------------------------------------------------------------------------
def srgb_to_lin(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92,
                    ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


_LUT_N = 8192
_LUT = None


def lin_to_srgb(x: np.ndarray, fast: bool = False) -> np.ndarray:
    global _LUT
    if fast:
        if _LUT is None:
            g = np.linspace(0, 1, _LUT_N + 1, dtype=np.float32)
            _LUT = lin_to_srgb(g)
        idx = np.clip(x * _LUT_N + 0.5, 0, _LUT_N).astype(np.int32)
        return _LUT[idx]
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _soft_clip(x: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Gentle highlight shoulder (slope 1 at the knee, asymptote 1.0)."""
    span = 1.0 - knee
    t = np.maximum(x - knee, 0.0) / span
    return np.where(x > knee, knee + span * t / (1.0 + t), x)


# --------------------------------------------------------------------------
# Depth refinement (edge-aware upsampling)
# --------------------------------------------------------------------------
def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    k = (2 * radius + 1, 2 * radius + 1)

    def mean(a):
        return cv2.boxFilter(a, -1, k, borderType=cv2.BORDER_REPLICATE)

    mi, mp = mean(guide), mean(src)
    cov = mean(guide * src) - mi * mp
    var = mean(guide * guide) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return mean(a) * guide + mean(b)


def spread_depth(depth: np.ndarray, amount: float = 0.8) -> np.ndarray:
    """Redistribute depth so each slice of the scene gets a fair share of 0..1.

    Monocular models output disparity (1/distance): the foreground eats most of the
    range and the whole background is crushed near 0, so the light plane races to the
    back once it passes the subject.  Histogram equalisation (monotonic, so edges and
    ordering are preserved) blended with the raw values makes the handle move evenly."""
    d = depth.astype(np.float32)
    xs = np.quantile(d, np.linspace(0, 1, 257)).astype(np.float64)
    xs = np.maximum.accumulate(xs) + np.arange(257) * 1e-7     # strictly increasing
    eq = np.interp(d, xs, np.linspace(0, 1, 257)).astype(np.float32)
    return np.clip(amount * eq + (1 - amount) * d, 0, 1)


def upsample_depth(depth: np.ndarray, guide_gray: np.ndarray) -> np.ndarray:
    """Resize `depth` to the guide's size and snap its edges to the photo."""
    h, w = guide_gray.shape
    d = cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    guide = guide_gray.astype(np.float32)
    radius = max(3, int(round(max(w, h) / 350)))
    d = guided_filter(guide, d, radius, 2e-4)
    d = guided_filter(guide, d, max(2, radius // 2), 1e-4)
    return np.clip(d, 0.0, 1.0)


# --------------------------------------------------------------------------
# Relighting
# --------------------------------------------------------------------------
_GRAD: dict = {}


_SHADOW: dict = {}


def shadow_map(depth: np.ndarray, p: Params) -> np.ndarray:
    """Screen-space cast shadows from the depth map (0 = lit, 1 = fully in shadow).

    The depth map is treated as a height field rising toward the camera.  From every
    pixel a ray is marched toward the light (in the image plane, climbing at the light's
    elevation); if a nearer surface blocks it, the pixel is in shadow.  Computed at a
    fixed working resolution so preview and full-size export match, then softened."""
    h, w = depth.shape
    key = (id(depth), h, w, round(p.shadow_az, 1), round(p.shadow_el, 1), round(p.shadow_soft, 3))
    if _SHADOW.get("key") == key:
        return _SHADOW["map"]
    sc = min(1.0, 640.0 / max(h, w))
    sw, sh = max(16, int(w * sc)), max(16, int(h * sc))
    ds = cv2.resize(depth.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    height = ds * (0.35 * max(sw, sh))               # depth range ~ a third of the frame
    az, el = math.radians(p.shadow_az), math.radians(max(3.0, min(85.0, p.shadow_el)))
    ux, uy = math.sin(az), -math.cos(az)              # image-plane step toward the light
    rise = math.tan(el)                               # ray climbs this much per pixel
    occ = np.zeros_like(height)
    steps = 56
    max_len = 0.45 * max(sw, sh)
    for i in range(1, steps + 1):
        t = max_len * (i / steps) ** 1.6              # dense near, sparse far
        M = np.float32([[1, 0, -ux * t], [0, 1, -uy * t]])
        shifted = cv2.warpAffine(height, M, (sw, sh), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        np.maximum(occ, shifted - (height + rise * t), out=occ)
    shadow = np.clip(occ / (0.02 * max(sw, sh)), 0, 1)    # soft onset instead of a hard step
    blur = 0.6 + p.shadow_soft * 0.02 * max(sw, sh)       # penumbra
    shadow = cv2.GaussianBlur(shadow, (0, 0), blur)
    shadow = cv2.resize(shadow, (w, h), interpolation=cv2.INTER_CUBIC)
    shadow = np.clip(shadow, 0, 1).astype(np.float32)
    _SHADOW.clear()
    _SHADOW.update(key=key, map=shadow)
    return shadow


def paint_light_maps(depth: np.ndarray, paint: np.ndarray, p: Params):
    """3D painted light.  The light lives at the depth of the surfaces you painted on:
    it glows outward across the frame but fades on anything nearer or farther than
    that depth, so it wraps the subject instead of washing over the whole picture."""
    h, w = depth.shape
    sc = min(1.0, 1024.0 / max(h, w))                  # blur at low res, resolution-free
    sw, sh = max(8, int(w * sc)), max(8, int(h * sc))
    m = cv2.resize(paint.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    ds = cv2.resize(depth.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    sigma = 2.0 + p.paint_spread * sw / 10.0
    mb = cv2.GaussianBlur(m, (0, 0), sigma)
    dref = cv2.GaussianBlur(m * ds, (0, 0), sigma) / (mb + 1e-4)   # depth of the strokes
    glow = np.clip(np.maximum(mb * 1.8, m), 0, 1) ** 0.8
    glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_CUBIC)
    dref = cv2.resize(dref, (w, h), interpolation=cv2.INTER_CUBIC)
    tol = 0.03 + 0.35 * p.paint_falloff
    reach = glow * np.exp(-((depth - dref) / tol) ** 2)
    ev = (p.paint_ev * reach).astype(np.float32)
    warm = (p.paint_warmth * reach).astype(np.float32) if p.paint_warmth else None
    return ev, warm


def compute_maps(depth: np.ndarray, p: Params, fog_masks: dict | None = None,
                 paint: np.ndarray | None = None) -> dict:
    """Per-pixel maps (exposure EV, warmth, fog transmittance/light, haze, painted light).
    fog_masks: {layer key: float mask (any size, resized here)} painted by the user.
    paint: the 'Paint Light' brush mask."""
    h, w = depth.shape
    d = depth.astype(np.float32)

    # near / far exposure split around the focus depth
    s2 = max(2.0 * p.softness, 1e-3)
    # rotated light plane: its depth changes linearly across the frame (left -> right)
    piv = p.pivot
    if p.plane_tilt:
        piv = p.pivot + p.plane_tilt * np.linspace(-0.5, 0.5, w, dtype=np.float32)[None, :]
    near_w = _smoothstep((d - piv) / s2 + 0.5)
    if p.pivot + p.beam < 1.0 or p.plane_tilt:   # beam ends before the camera: light a band
        end = piv + p.beam
        near_w = near_w * np.where(end < 1.0, _smoothstep((end - d) / s2 + 0.5), 1.0)
    if p.shadows and p.shadow_strength > 0:
        near_w = near_w * (1.0 - p.shadow_strength * shadow_map(depth, p))
    ev = p.far_ev + (p.near_ev - p.far_ev) * near_w

    # directional light from depth-derived normals
    if p.light_ev != 0.0:
        key = id(depth)
        if _GRAD.get("key") != key:            # gradients depend only on the depth map
            sigma = max(1.0, w / 700.0)
            ds = cv2.GaussianBlur(d, (0, 0), sigma)
            _GRAD.clear()
            _GRAD.update(key=key, ref=depth,
                         gx=cv2.Sobel(ds, cv2.CV_32F, 1, 0, ksize=3) / 8.0,
                         gy=cv2.Sobel(ds, cv2.CV_32F, 0, 1, ksize=3) / 8.0)
        gx, gy = _GRAD["gx"], _GRAD["gy"]
        s = p.relief * w * 0.25            # resolution-independent slope scale
        nx, ny = -gx * s, -gy * s
        inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
        az, el = math.radians(p.light_azimuth), math.radians(p.light_elevation)
        lx, ly, lz = math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el)
        ndl = (nx * lx + ny * ly + 1.0 * lz) * inv
        ev = ev + p.light_ev * (ndl - lz) * 3.0

    dist = 1.0 - d                                   # 0 = camera, 1 = far
    fog_t = fog_l = None
    if p.active_fogs():
        fog_t, fog_l = volumetric_fog(dist, p, fog_masks or {})

    haze = None
    if p.haze > 0.0:
        haze = (0.75 * p.haze * np.power(dist, 1.6)).astype(np.float32)

    warm = None
    if p.near_warmth != 0.0 or p.far_warmth != 0.0:
        warm = (p.far_warmth + (p.near_warmth - p.far_warmth) * near_w).astype(np.float32)

    pev = pwarm = None
    if paint is not None and (p.paint_ev or p.paint_warmth):
        if paint.shape != (h, w):
            paint = cv2.resize(paint, (w, h), interpolation=cv2.INTER_LINEAR)
        pev, pwarm = paint_light_maps(d, paint, p)

    return {"ev": ev.astype(np.float32), "haze": haze, "warm": warm,
            "fog_t": fog_t, "fog_l": fog_l, "pev": pev, "pwarm": pwarm}


# --------------------------------------------------------------------------
# Volumetric fog: a stack of cloud slices between camera and horizon.  Each pixel's
# view ray only gathers the slices in FRONT of the surface it hits, so fog banks pass
# behind near objects and in front of far ones, with edges as sharp as the depth map.
# --------------------------------------------------------------------------
FOG_SLICES = 20
_SLICE_SMALL: dict = {}
_SLICE_FULL: dict = {}


def _slice_clouds(aspect: float, f: FogLayer) -> list:
    """Small (384 px wide) cloud density layers in 0..1, one per slice.  Farther slices
    get smaller features (perspective); clouds are stretched horizontally like fog banks."""
    key = (round(aspect, 3), round(f.cloud, 3), round(f.size, 3), int(f.seed))
    if key in _SLICE_SMALL:
        return _SLICE_SMALL[key]
    sw = 384
    sh = max(8, round(sw * aspect))
    rng = np.random.default_rng(9173 + 7919 * int(f.seed))
    layers = []
    for k in range(FOG_SLICES):
        s = (k + 0.5) / FOG_SLICES
        cells = (12.0 - 9.5 * f.size) * (1.0 + 2.2 * s)           # cloud cells across
        acc = np.zeros((sh, sw), np.float32)
        amp, total = 1.0, 0.0
        for octave in range(4):
            cx = max(2, int(round(cells * 2 ** octave)))
            cy = max(2, int(round(cells * 2 ** octave * aspect * 1.8)))
            g = rng.random((cy, cx)).astype(np.float32)
            acc += amp * cv2.resize(g, (sw, sh), interpolation=cv2.INTER_CUBIC)
            total += amp
            amp *= 0.5
        n = acc / total
        n = (n - n.mean()) / (n.std() + 1e-6) * 0.16 + 0.5           # normalise contrast
        clouds = _smoothstep((n - 0.5) / 0.2 + 0.5)                   # banks with gaps
        even = 0.55 + 0.6 * (n - 0.5)                                  # gentle variation
        layers.append(np.clip(even + (clouds - even) * f.cloud, 0, 1).astype(np.float32))
    if len(_SLICE_SMALL) > 16:
        _SLICE_SMALL.clear()
    _SLICE_SMALL[key] = layers
    return layers


def _ev_at(dval: float, p: Params) -> float:
    """Exposure of the depth light at a scalar depth (lights the fog slices)."""
    s2 = max(2.0 * p.softness, 1e-3)
    w = float(_smoothstep(np.float32((dval - p.pivot) / s2 + 0.5)))
    if p.pivot + p.beam < 1.0:
        w *= float(_smoothstep(np.float32(((p.pivot + p.beam) - dval) / s2 + 0.5)))
    return p.far_ev + (p.near_ev - p.far_ev) * w


def fog_window(dval: float, f: FogLayer) -> float:
    """How much of fog layer `f` sits at scalar depth dval (plane + width, soft edges)."""
    s2 = max(2.0 * f.softness, 1e-3)
    w = float(_smoothstep(np.float32((dval - f.pos) / s2 + 0.5)))
    if f.pos + f.width < 1.0:
        w *= float(_smoothstep(np.float32(((f.pos + f.width) - dval) / s2 + 0.5)))
    return w


def _layer_slices(h: int, w: int, f: FogLayer):
    small = _slice_clouds(h / w, f)
    fkey = (h, w, id(small))
    if fkey not in _SLICE_FULL:                 # cache resized layers per preview size
        if len(_SLICE_FULL) >= 8:
            _SLICE_FULL.clear()
        _SLICE_FULL[fkey] = ([cv2.resize(L, (w, h), interpolation=cv2.INTER_CUBIC)
                              for L in small] if h * w <= 4_000_000 else None)
    full = _SLICE_FULL[fkey]
    return lambda k: full[k] if full is not None else cv2.resize(
        small[k], (w, h), interpolation=cv2.INTER_CUBIC)


def volumetric_fog(dist: np.ndarray, p: Params, masks: dict):
    """All fog layers integrated together along each view ray.

    Returns (T, Ls): scene transmittance (h, w) and a list with each layer's
    accumulated fog light (h, w), so every layer keeps its own colour."""
    h, w = dist.shape
    layers = p.active_fogs()
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    per = []
    for f in layers:
        ground = None
        if f.ground > 0:                    # fog sits below a height line, soft top
            top = 1.0 - f.height
            below = _smoothstep((yy - top) / 0.28 + 0.5)
            ground = (1.0 - f.ground) + f.ground * below
        m = masks.get(f.key)
        if m is not None and m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        # density is 'visible opacity': narrow volumes are denser per slice
        scale = 3.0 * f.density / (FOG_SLICES * max(f.width, 0.12))
        per.append((f, _layer_slices(h, w, f), ground, m, scale))

    inv_edge = np.float32(1.0 / max(0.6 / FOG_SLICES, 0.015))
    dist = dist.astype(np.float32, copy=False)
    T = np.ones((h, w), np.float32)
    Ls = [np.zeros((h, w), np.float32) for _ in layers]
    vis = np.empty((h, w), np.float32)
    tmp = np.empty((h, w), np.float32)
    for k in range(FOG_SLICES):
        s = (k + 0.5) / FOG_SLICES
        dval = 1.0 - s
        active = [(i, it, fog_window(dval, it[0])) for i, it in enumerate(per)]
        active = [x for x in active if x[2] > 1e-3]
        if not active:
            continue
        # slice only counts where it is in FRONT of the surface (behind = occluded)
        np.subtract(dist, np.float32(s - 0.5 / inv_edge), out=vis)
        vis *= inv_edge
        np.clip(vis, 0, 1, out=vis)
        lit = np.float32(2.0 ** (0.45 * _ev_at(dval, p)))      # the beam glows through fog
        a_tot = None
        alphas = []
        for i, (f, slices, ground, m, scale), win in active:
            a = slices(k) * vis
            a *= np.float32(scale * win)
            if ground is not None:
                a *= ground
            if m is not None:
                a *= m
            np.minimum(a, 1.5, out=a)
            a *= 1.0 - 0.5 * a                                   # ~ 1 - exp(-t)
            alphas.append((i, a))
            a_tot = a.copy() if a_tot is None else a_tot + a
        np.minimum(a_tot, 0.97, out=a_tot)
        for i, a in alphas:
            np.multiply(T, a, out=tmp)
            tmp *= lit
            Ls[i] += tmp
        np.subtract(np.float32(1.0), a_tot, out=tmp)
        T *= tmp
    return T, Ls


def apply_maps_lin(lin: np.ndarray, maps: dict, p: Params, fast: bool = False,
                   light_mask: np.ndarray | None = None) -> np.ndarray:
    """lin: linear-light float32 (h, w, 3).  Returns display sRGB float32 0..1.
    light_mask (h, w) limits only the depth light; fog has its own per-layer masks."""
    out = lin
    if not p.light_identity():
        out = lin * np.exp2(maps["ev"])[..., None]
        if maps["warm"] is not None:
            wm = maps["warm"][..., None]
            gain = np.concatenate([1.0 + 0.18 * wm, np.ones_like(wm), 1.0 - 0.18 * wm], axis=-1)
            out = out * gain
        if light_mask is not None:
            out = lin + (out - lin) * light_mask[..., None]

    if maps.get("pev") is not None:                   # painted 3D light (not limited by mask)
        out = out * np.exp2(maps["pev"])[..., None]
        if maps.get("pwarm") is not None:
            wm = maps["pwarm"][..., None]
            out = out * np.concatenate([1.0 + 0.25 * wm, np.ones_like(wm), 1.0 - 0.25 * wm], -1)

    if maps.get("fog_t") is not None:
        out = out * maps["fog_t"][..., None]
        for f, L in zip(p.active_fogs(), maps["fog_l"]):
            base = float(srgb_to_lin(np.float32(f.brightness)))
            col = np.array([1 + 0.22 * f.warmth, 1.0, 1 - 0.22 * f.warmth], np.float32) * base
            out = out + col * L[..., None]

    if maps["haze"] is not None:
        hz = maps["haze"][..., None]
        col = np.array([0.62, 0.70, 0.82], np.float32)       # cool sky-scattered light
        lum = (out * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1, keepdims=True)
        out = out * (1.0 - hz) + (0.45 * lum + 0.55 * col * 0.42) * hz

    return lin_to_srgb(_soft_clip(out), fast)


def apply_maps_srgb_strip(strip_srgb: np.ndarray, maps: dict, p: Params,
                          y0: int, y1: int, light_mask: np.ndarray | None = None) -> np.ndarray:
    def cut(v):
        if v is None:
            return None
        return [x[y0:y1] for x in v] if isinstance(v, list) else v[y0:y1]
    sub = {k: cut(v) for k, v in maps.items()}
    lm = None if light_mask is None else light_mask[y0:y1]
    return apply_maps_lin(srgb_to_lin(strip_srgb), sub, p, light_mask=lm)


# --------------------------------------------------------------------------
# Visualisation helpers
# --------------------------------------------------------------------------
def depth_colour(depth: np.ndarray) -> np.ndarray:
    """RGB uint8 turbo visualisation (red = near, blue = far)."""
    u8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def focus_line(depth: np.ndarray, pivot: float) -> np.ndarray:
    """Boolean mask of the contour where depth crosses the focus plane."""
    ds = cv2.GaussianBlur(depth, (0, 0), 1.5)
    m = (ds > pivot).astype(np.uint8)
    edge = m - cv2.erode(m, np.ones((3, 3), np.uint8))
    return cv2.dilate(edge, np.ones((2, 2), np.uint8)).astype(bool)
