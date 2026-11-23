"""High-quality monocular depth estimation (Depth Anything V2 via transformers)."""
from __future__ import annotations

import math

import cv2
import numpy as np

# Large = best quality (CC-BY-NC-4.0 weights).  Small is Apache-2.0.
MODELS = {
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "depthpro": "apple/DepthPro-hf",      # sharper edges, metric depth
    "ultra": "prs-eth/marigold-depth-v1-1",   # diffusion detail fused with Depth Pro
}
LABELS = {"large": "Depth Anything V2", "depthpro": "Apple Depth Pro",
          "ultra": "Ultra (Marigold + Depth Pro)"}


ULTRA_PASSES = 10


def _estimate_ultra(rgb_f32, device, progress, base=None):
    """Ultra quality (~30-60 s): Marigold diffusion depth, a 10-sample ensemble at high
    processing resolution, for very fine detail (foliage, hair, thin branches), fused
    with Depth Pro for correct large-scale structure and distance spacing.
    `base` = an already computed Depth Pro map (saves running it twice)."""
    import torch
    from diffusers import MarigoldDepthPipeline
    from PIL import Image

    if base is None:
        base = _estimate_depthpro(rgb_f32, device, progress)        # near = 1, log-metric

    key = ("marigold", device)
    if key not in _cache:
        progress("Loading Marigold for ultra depth…")
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = MarigoldDepthPipeline.from_pretrained(
            MODELS["ultra"], variant="fp16" if device == "cuda" else None,
            torch_dtype=dtype).to(device)
        pipe.set_progress_bar_config(disable=True)
        _cache[key] = pipe
    pipe = _cache[key]

    img = Image.fromarray((np.clip(rgb_f32, 0, 1) * 255).astype(np.uint8))
    # Own ensemble: N diffusion samples, each affinely aligned to the first, then the
    # per-pixel median.  (diffusers' built-in ensembling runs a slow CPU optimiser.)
    preds = []
    for i in range(ULTRA_PASSES):
        progress(f"Ultra depth: diffusion pass {i + 1}/{ULTRA_PASSES}…")
        gen = torch.Generator(device=device).manual_seed(2024 + i)
        with torch.inference_mode():
            out = pipe(img, num_inference_steps=4, ensemble_size=1, processing_resolution=1280,
                       match_input_resolution=True, generator=gen)
        pr = np.asarray(out.prediction).astype(np.float32)
        preds.append(pr.reshape(pr.shape[-3], pr.shape[-2]) if pr.ndim > 2 else pr)
    ref = preds[0]
    sub = (slice(None, None, 4), slice(None, None, 4))
    r = ref[sub].ravel()
    aligned = [ref]
    for pr in preds[1:]:
        x = pr[sub].ravel()
        A = np.stack([x, np.ones_like(x)], 1)
        (a, b), *_ = np.linalg.lstsq(A, r, rcond=None)
        aligned.append(a * pr + b)
    m = 1.0 - np.median(np.stack(aligned), axis=0)                  # near = 1

    progress("Fusing ultra detail with Depth Pro structure…")
    H, W = m.shape
    base = cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC)
    # 1) give Marigold Depth Pro's distance distribution (monotonic, keeps ordering/edges)
    qs = np.linspace(0, 1, 513)
    xm = np.maximum.accumulate(np.quantile(m, qs)) + np.arange(513) * 1e-7
    mm = np.interp(m, xm, np.quantile(base, qs)).astype(np.float32)
    # 2) Depth Pro decides the layout down to object scale; Marigold only adds the fine
    #    detail (edges of leaves, hair, thin parts), via an edge-aware correction so the
    #    Depth Pro layout doesn't bleed across object boundaries
    gray = cv2.cvtColor((np.clip(rgb_f32, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    from .engine import guided_filter
    rad = max(6, int(max(H, W) / 40))
    fused = mm + guided_filter(gray, base - mm, rad, 1e-3)
    return np.clip(fused, 0, 1).astype(np.float32)


def _estimate_depthpro(rgb_f32, device, progress):
    """Apple Depth Pro, as log inverse depth: near = 1 and distance is perceptually
    even (doubling the distance = an equal step)."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    mid = MODELS["depthpro"]
    key = (mid, device)
    if key not in _cache:
        progress(f"Loading Apple Depth Pro ({device})…")
        proc = AutoImageProcessor.from_pretrained(mid)
        # fp32: Depth Pro's field-of-view head is unstable in fp16
        model = AutoModelForDepthEstimation.from_pretrained(mid).to(device).eval()
        _cache[key] = (proc, model)
    proc, model = _cache[key]

    h, w = rgb_f32.shape[:2]
    progress("Estimating depth with Apple Depth Pro…")
    u8 = (np.clip(rgb_f32, 0, 1) * 255).astype(np.uint8)
    inputs = proc(images=u8, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs)
    long = 1536
    tw, th = (long, max(14, round(long * h / w))) if w >= h else (max(14, round(long * w / h)), long)
    # metric distance in metres (uses the model's own field-of-view estimate)
    metres = proc.post_process_depth_estimation(out, target_sizes=[(th, tw)])[0]["predicted_depth"]
    metres = metres.float().cpu().numpy()
    return metric_to_control(metres)


def metric_to_control(metres: np.ndarray) -> np.ndarray:
    """Metres -> 0..1 control depth (1 = nearest).

    Log scale, so every doubling of distance is an equal step of the handle.  The range
    runs from the nearest subject to the 'far limit' (90th percentile, capped at 25x the
    near distance): sky and extreme background are pinned to 0 so they are only reached
    when the plane is pushed all the way back, instead of grabbing the middle of the range."""
    z = np.log(np.clip(metres, 0.05, 1e5))
    near = np.percentile(z, 1.0)
    far = min(np.percentile(z, 90.0), near + math.log(25.0))
    far = max(far, near + math.log(1.5))
    return np.clip(1.0 - (z - near) / (far - near), 0.0, 1.0).astype(np.float32)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_cache = {}


def _normalise(d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(d, [0.5, 99.5])
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def estimate_depth(rgb_f32: np.ndarray, model_key: str = "large", min_side: int = 1036,
                   tta: bool = True, progress=lambda msg: None, base=None):
    """rgb_f32: (h, w, 3) float32 sRGB 0..1.  Returns (depth, info) with depth in
    [0, 1], 1 = nearest, at the model's working resolution (aspect preserved)."""
    import torch
    from transformers import AutoModelForDepthEstimation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = MODELS.get(model_key, model_key)
    if model_key == "depthpro":
        d = _estimate_depthpro(rgb_f32, device, progress)
        return d, f"Apple Depth Pro · {device}"
    if model_key == "ultra":
        d = _estimate_ultra(rgb_f32, device, progress, base)
        return d, f"Ultra (Marigold + Depth Pro) · {device}"

    key = (model_id, device)
    if key not in _cache:
        progress(f"Loading {model_id} ({device})…")
        m = AutoModelForDepthEstimation.from_pretrained(model_id).to(device, dtype).eval()
        _cache[key] = m
    model = _cache[key]

    h, w = rgb_f32.shape[:2]
    if device == "cpu":
        min_side = min(min_side, 518)       # keep CPU runs tolerable
    scale = min_side / min(h, w)
    tw = max(14, int(round(w * scale / 14)) * 14)
    th = max(14, int(round(h * scale / 14)) * 14)
    small = cv2.resize(rgb_f32, (tw, th), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    x = ((small - _MEAN) / _STD).transpose(2, 0, 1)[None]
    x = torch.from_numpy(np.ascontiguousarray(x)).to(device, dtype)

    def run(t):
        with torch.inference_mode():
            return model(pixel_values=t).predicted_depth[0].float().cpu().numpy()

    progress("Estimating depth…")
    d = _normalise(run(x))
    if tta:
        progress("Refining depth (flip pass)…")
        d2 = _normalise(run(torch.flip(x, dims=[3]))[:, ::-1])
        d = 0.5 * (d + d2)
    d = cv2.resize(d, (tw, th), interpolation=cv2.INTER_CUBIC) if d.shape != (th, tw) else d
    return np.clip(d, 0, 1).astype(np.float32), f"{model_id.split('/')[-1]} · {device}"


def demo_depth(h: int, w: int) -> np.ndarray:
    """Synthetic depth for --demo / self-test: ground plane receding + a sphere."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = 0.1 + 0.8 * (yy / h) ** 1.3
    r = min(h, w) * 0.22
    cx, cy = w * 0.5, h * 0.62
    q = 1 - ((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2
    d = np.where(q > 0, np.maximum(d, 0.55 + 0.4 * np.sqrt(np.clip(q, 0, 1))), d)
    return np.clip(d, 0, 1).astype(np.float32)
