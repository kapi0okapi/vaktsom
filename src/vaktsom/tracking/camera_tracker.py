"""Per-camera multi-person tracker (SORT/ByteTrack family) in image space.

A constant-velocity Kalman filter per person, two-stage association (confident detections first,
then weak ones for tracks that would otherwise drop, as in ByteTrack) and a cost that mixes IoU
with normalised centre distance, so small, fast swimmers far from the camera still associate.
All timing is in seconds, because camera frame rates on a busy network are not constant.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from vaktsom.perception.types import Detection


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N, 4) and b (M, 4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


class BoxKalman:
    """State [cx, cy, w, h, vx, vy, vw, vh]; measurements [cx, cy, w, h]."""

    def __init__(self, box: np.ndarray, ts: float):
        cx, cy, w, h = self._to_cxcywh(box)
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], float)
        s = max(w, h)
        self.P = np.diag([s, s, s, s, 10 * s, 10 * s, s, s]) ** 2 * 0.01
        self.ts = ts

    @staticmethod
    def _to_cxcywh(box: np.ndarray) -> np.ndarray:
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])

    def predict(self, ts: float) -> None:
        dt = max(ts - self.ts, 0.0)
        if dt == 0:
            return
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * dt
        s = max(self.x[2], self.x[3], 1.0)
        # white-acceleration noise, scaled with the box so near and far swimmers behave alike
        q = np.array([0.5, 0.5, 0.2, 0.2]) * s
        G = np.vstack([np.eye(4) * dt**2 / 2, np.eye(4) * dt])
        self.P = F @ self.P @ F.T + G @ np.diag(q**2) @ G.T
        self.x = F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 2.0)
        self.ts = ts

    def update(self, box: np.ndarray) -> None:
        z = self._to_cxcywh(box)
        H = np.hstack([np.eye(4), np.zeros((4, 4))])
        s = max(z[2], z[3], 1.0)
        R = np.diag([0.05 * s, 0.05 * s, 0.1 * s, 0.1 * s]) ** 2
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(8) - K @ H) @ self.P

    @property
    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class LocalTrack:
    id: int
    kf: BoxKalman
    first_ts: float
    last_ts: float
    hits: int = 1
    confirmed: bool = False
    p_under: float = 0.0
    p_vertical: float = 0.65  # neutral prior until a posture estimate arrives
    motion: float = 1.0
    score: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=400))  # (ts, box, p_under, motion)

    def observe(self, det: Detection, ts: float, tau: float) -> None:
        dt = max(ts - self.last_ts, 1e-3)
        a = 1.0 - np.exp(-dt / tau)  # time-based smoothing, independent of the frame rate
        self.p_under += a * (det.p_under - self.p_under)
        if det.p_vertical is not None:
            self.p_vertical += a * (det.p_vertical - self.p_vertical)
        if det.motion is not None:
            self.motion += a * (det.motion - self.motion)
        self.score = det.score
        self.kf.update(det.box)
        self.last_ts = ts
        self.hits += 1
        self.history.append((ts, self.kf.box.copy(), self.p_under, self.motion))

    @property
    def box(self) -> np.ndarray:
        return self.kf.box


@dataclass
class TrackerConfig:
    high_score: float = 0.5
    low_score: float = 0.15
    confirm_hits: int = 3
    max_lost_s: float = 2.0  # the world tracker keeps a person much longer; this is image-level
    tentative_lost_s: float = 0.5
    gate_iou: float = 0.05
    gate_dist: float = 1.2  # centre distance in units of the box size
    smoothing_tau_s: float = 0.6


class CameraTracker:
    def __init__(self, config: TrackerConfig | None = None):
        self.cfg = config or TrackerConfig()
        self.tracks: list[LocalTrack] = []
        self._ids = itertools.count(1)

    def _cost(self, tracks: list[LocalTrack], dets: list[Detection]) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        tb = np.array([t.box for t in tracks])
        db = np.array([d.box for d in dets])
        iou = iou_matrix(tb, db)
        tc = (tb[:, :2] + tb[:, 2:]) / 2
        dc = (db[:, :2] + db[:, 2:]) / 2
        size = np.sqrt(np.prod(np.maximum(tb[:, 2:] - tb[:, :2], 1.0), axis=1))
        dist = np.linalg.norm(tc[:, None] - dc[None], axis=2) / size[:, None]
        cost = (1.0 - iou) + 0.5 * np.minimum(dist, 2.0)
        gated = (iou < self.cfg.gate_iou) & (dist > self.cfg.gate_dist)
        cost[gated] = 1e6
        return cost

    def _match(self, tracks: list[LocalTrack], dets: list[Detection]):
        cost = self._cost(tracks, dets)
        if cost.size == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        rows, cols = linear_sum_assignment(cost)
        pairs = [(r, c) for r, c in zip(rows, cols, strict=True) if cost[r, c] < 1e5]
        mt = {r for r, _ in pairs}
        md = {c for _, c in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(dets)) if j not in md]

    def update(self, detections: list[Detection], ts: float) -> list[LocalTrack]:
        """Advance to ``ts`` with this frame's detections; returns the confirmed, live tracks."""
        for t in self.tracks:
            t.kf.predict(ts)
        high = [d for d in detections if d.score >= self.cfg.high_score]
        low = [d for d in detections if self.cfg.low_score <= d.score < self.cfg.high_score]

        pairs, un_tracks, un_high = self._match(self.tracks, high)
        for ti, di in pairs:
            self.tracks[ti].observe(high[di], ts, self.cfg.smoothing_tau_s)
        # second stage: weak detections only rescue confirmed tracks (a sinking swimmer fades)
        rest = [self.tracks[i] for i in un_tracks if self.tracks[i].confirmed]
        pairs2, _, _ = self._match(rest, low)
        for ti, di in pairs2:
            rest[ti].observe(low[di], ts, self.cfg.smoothing_tau_s)

        for di in un_high:
            d = high[di]
            t = LocalTrack(
                next(self._ids),
                BoxKalman(d.box, ts),
                ts,
                ts,
                p_under=d.p_under,
                p_vertical=0.65 if d.p_vertical is None else d.p_vertical,
                motion=1.0 if d.motion is None else d.motion,
                score=d.score,
            )
            t.history.append((ts, d.box.copy(), d.p_under, t.motion))
            self.tracks.append(t)

        alive = []
        for t in self.tracks:
            if not t.confirmed and t.hits >= self.cfg.confirm_hits:
                t.confirmed = True
            silent = ts - t.last_ts
            limit = self.cfg.max_lost_s if t.confirmed else self.cfg.tentative_lost_s
            if silent <= limit:
                alive.append(t)
        self.tracks = alive
        return [t for t in self.tracks if t.confirmed]
