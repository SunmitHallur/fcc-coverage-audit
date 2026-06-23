"""Reconcile downloadable vector coverage against the rendered map tiles.

The green map tiles on broadbandmap.fcc.gov are rendered *from* the vector data,
so they should agree. This module cross-checks them and, where they disagree
(IoU below threshold), falls back to computer-vision segmentation of the green
raster for that tile - covering rendering/simplification gaps or cases where the
published image shows coverage the downloadable polygons omit.

Imaging deps (rasterio/Pillow/scikit-image) are imported lazily so the core
vector pipeline runs without them.
"""
from __future__ import annotations

import io
import logging

import numpy as np

log = logging.getLogger(__name__)

# Green coverage intensity -> approximate signal band (dBm). Darker/saturated
# green = stronger signal on the FCC heat map.
_GREEN_TO_DBM = [
    (0.85, -85.0),   # strong, saturated green
    (0.55, -95.0),   # medium green
    (0.0, -105.0),   # light green
]


def segment_green_coverage(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segment coverage from a rendered RGB tile.

    Returns (coverage_mask, signal_dbm_grid). Non-coverage pixels are NaN in the
    signal grid. Uses HSV thresholding on the green hue band.
    """
    from skimage.color import rgb2hsv

    arr = rgb[..., :3].astype(np.float32) / 255.0
    hsv = rgb2hsv(arr)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Green hue ~ 0.20..0.45 in [0,1]; require some saturation to exclude grey basemap.
    green = (h > 0.20) & (h < 0.45) & (s > 0.20) & (v > 0.20)

    signal = np.full(green.shape, np.nan, dtype=np.float32)
    intensity = np.clip(s * v, 0.0, 1.0)
    for thresh, dbm in _GREEN_TO_DBM:
        band = green & (intensity >= thresh) & np.isnan(signal)
        signal[band] = dbm
    return green, signal


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union of two boolean coverage masks."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def load_png_rgb(data: bytes) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def reconcile_tile(
    vector_mask: np.ndarray, tile_png: bytes, min_iou: float
) -> dict:
    """Compare one tile; if disagreement, return CV-segmented coverage as fallback.

    Returns dict with keys: iou, agree (bool), source ("vector"|"cv"),
    cv_mask, cv_signal.
    """
    rgb = load_png_rgb(tile_png)
    cv_mask, cv_signal = segment_green_coverage(rgb)
    # Resize vector mask to tile resolution if needed (nearest).
    if vector_mask.shape != cv_mask.shape:
        from skimage.transform import resize

        vector_mask = resize(
            vector_mask.astype(float), cv_mask.shape, order=0, preserve_range=True
        ) > 0.5
    score = iou(vector_mask, cv_mask)
    agree = score >= min_iou
    return {
        "iou": score,
        "agree": agree,
        "source": "vector" if agree else "cv",
        "cv_mask": cv_mask,
        "cv_signal": cv_signal,
    }
