"""Optional camera-image preprocessing shared by training and evaluation.

clahe: brightness normalisation + CLAHE on the L channel (CIELAB) of each 160x120 image:
  1. stretch L so its 1st..99th percentiles span 0..255 (removes most of a global light-strength change),
  2. Contrast Limited Adaptive Histogram Equalisation, 4x4 tiles, clip limit 2 (evens out local contrast).
CLAHE alone reduced the image change caused by halving the light by only ~8 % in the simulated cell;
with the stretch first it fell from 62 to 13 grey levels while the change caused by moving the bin 10 mm
grew (0.9 -> 1.4). The policy must be trained and run with the same preprocessing.
"""

import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def clahe_rgb(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    lo, hi = np.percentile(L, (1, 99))
    L = np.clip((L.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0)), 0, 255).astype(np.uint8)
    lab[..., 0] = _CLAHE.apply(L)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply(kind: str | None, img: np.ndarray) -> np.ndarray:
    if kind is None:
        return img
    if kind == "clahe":
        return clahe_rgb(img)
    raise ValueError(f"unknown preprocessing {kind!r}")
