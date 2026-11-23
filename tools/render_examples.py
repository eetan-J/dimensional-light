"""Render the README before/after examples.

Usage:  python tools/render_examples.py <folder with tree.jpg barn.jpg lighthouse.jpg> <out folder>

The example photos are public-domain (CC0) images from Wikimedia Commons; see README.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from dimlight import engine as e                      # noqa: E402
from dimlight.depth import estimate_depth             # noqa: E402

F = e.FogLayer

EXAMPLES = {
    "tree": dict(
        title="Depth light with cast shadows: the tree is lit, the distance falls away",
        params=e.Params(near_ev=0.7, far_ev=-0.9, pivot=0.45, softness=0.12,
                        near_warmth=0.35, far_warmth=-0.15, shadows=True,
                        shadow_az=300, shadow_el=25, shadow_strength=0.8, shadow_soft=0.4)),
    "barn": dict(
        title="A thin layer of mist between the barn and the forest",
        params=e.Params(near_ev=0.3, far_ev=-0.35, pivot=0.5, softness=0.15, haze=0.12,
                        fogs=(F(density=0.3, pos=0.0, width=0.4, cloud=0.7, size=0.9,
                                ground=0.8, height=0.55, brightness=0.9, key=1),))),
    "lighthouse": dict(
        title="Warm light on the foreground pilings, the water and treeline pushed back",
        params=e.Params(near_ev=0.6, far_ev=-0.8, pivot=0.55, softness=0.1,
                        near_warmth=0.4, haze=0.08)),
}


def main(src, out):
    os.makedirs(out, exist_ok=True)
    for name, ex in EXAMPLES.items():
        bgr = cv2.imread(os.path.join(src, name + ".jpg"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        dm, _ = estimate_depth(rgb, "depthpro")
        d = e.upsample_depth(dm, rgb.mean(2))
        p = ex["params"]
        res = e.apply_maps_lin(e.srgb_to_lin(rgb), e.compute_maps(d, p), p)
        after = cv2.cvtColor((res * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        h = 520
        w = int(bgr.shape[1] * h / bgr.shape[0])
        b = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        a = cv2.resize(after, (w, h), interpolation=cv2.INTER_AREA)
        depth = cv2.applyColorMap((cv2.resize(d, (w, h)) * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
        gap = np.full((h, 8, 3), 20, np.uint8)
        cv2.imwrite(os.path.join(out, f"{name}-before-after.jpg"),
                    np.hstack([b, gap, a]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(os.path.join(out, f"{name}-depth.jpg"), depth,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("rendered", name, "-", ex["title"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
