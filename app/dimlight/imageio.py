"""Image load/save keeping 16-bit data and the embedded ICC profile."""
from __future__ import annotations

import os

import cv2
import numpy as np


def load_image(path: str):
    """Returns (rgb array uint8/uint16, max_value, icc_bytes_or_None)."""
    icc = None
    ext = os.path.splitext(path)[1].lower()
    arr = None
    if ext in (".tif", ".tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(path) as tf:
                arr = tf.pages[0].asarray()
                tag = tf.pages[0].tags.get(34675)
                if tag is not None:
                    icc = bytes(tag.value)
            if arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
        except Exception:
            arr = None
    if arr is None:
        bgr = cv2.imread(path, getattr(cv2, "IMREAD_COLOR_BGR", cv2.IMREAD_COLOR) | cv2.IMREAD_ANYDEPTH)
        if bgr is None:
            raise IOError(f"Cannot read image: {path}")
        arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    maxv = 65535.0 if arr.dtype == np.uint16 else 255.0
    return arr, maxv, icc


def save_tiff16(path: str, rgb16: np.ndarray, icc: bytes | None):
    try:
        import tifffile
        extra = [(34675, "B", len(icc), icc, True)] if icc else None
        tifffile.imwrite(path, rgb16, photometric="rgb", extratags=extra)
    except ImportError:
        cv2.imwrite(path, cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR))
