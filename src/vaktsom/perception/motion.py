"""Is the body moving, or only the water above it? Limb motion from the video, per detection.

Under a rippling surface everything shimmers, a motionless body included. For each detected person
the probe compares the frame with one about half a second older: the mean absolute change inside
the body box, minus the same change in a ring of water around it (the shimmer baseline), scaled by
the contrast of the body. A swimmer scores high, a body lying still under waves scores near zero,
whatever the waves do. The tracker smooths the value over time.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from vaktsom.perception.types import Detection


class MotionProbe:
    def __init__(self, lag_s: float = 0.5, scale: float = 0.5, gain: float = 6.0):
        self.lag_s, self.scale, self.gain = lag_s, scale, gain
        self.history: deque[tuple[float, np.ndarray]] = deque(maxlen=40)

    def _gray(self, image: np.ndarray) -> np.ndarray:
        small = cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)

    def annotate(self, ts: float, image: np.ndarray, detections: list[Detection]) -> None:
        cur = self._gray(image)
        past = None
        for t, g in reversed(self.history):
            if ts - t >= self.lag_s:
                past = g
                break
        self.history.append((ts, cur))
        if past is None or past.shape != cur.shape:
            return  # no reference yet: leave motion unknown, the tracker keeps its prior
        h, w = cur.shape
        diff = np.abs(cur - past)
        for d in detections:
            x0, y0, x1, y1 = (d.box * self.scale).astype(int)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            bw, bh = x1 - x0, y1 - y0
            # inner part of the box (the body), and a ring around the box (the water)
            ix0, iy0 = x0 + bw // 6, y0 + bh // 6
            ix1, iy1 = x1 - bw // 6, y1 - bh // 6
            inner = diff[iy0:iy1, ix0:ix1]
            pad = max(bw, bh) // 2 + 2
            ox0, oy0, ox1, oy1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w), min(y1 + pad, h)
            outer = diff[oy0:oy1, ox0:ox1].copy()
            outer[y0 - oy0 : y1 - oy0, x0 - ox0 : x1 - ox0] = np.nan
            ring = float(np.nanmean(outer)) if np.isfinite(outer).any() else 0.0
            contrast = float(cur[iy0:iy1, ix0:ix1].std()) + 6.0
            d.motion = float(np.clip(self.gain * (float(inner.mean()) - ring) / contrast, 0.0, 1.5))
