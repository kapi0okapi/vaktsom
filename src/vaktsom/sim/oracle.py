"""Detections from simulated ground truth, with the errors a real detector makes.

The oracle projects each body into each camera (refraction-aware) and then degrades the result:
missed detections that grow with depth, glare and occlusion, box jitter, noisy submersion and
posture estimates, false positives on the water and on objects lying on the bottom. This lets the
whole chain after perception (tracking, fusion, rules, alarms, user interface) be tested and
measured before and independently of the trained detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vaktsom.geometry import Camera
from vaktsom.perception.types import Detection
from vaktsom.sim.scene import BodyState
from vaktsom.tracking.camera_tracker import iou_matrix


@dataclass
class GlareSpot:
    """Specular reflection of a window or lamp on the water, as seen by one camera (pool metres)."""

    centre: tuple[float, float]
    radii: tuple[float, float]
    strength: float = 1.0  # 0..1, can vary over the day

    def contains(self, xy: np.ndarray) -> np.ndarray:
        d = (np.atleast_2d(xy) - np.asarray(self.centre)) / np.asarray(self.radii)
        return (d**2).sum(1) <= 1.0


@dataclass
class NoiseModel:
    p_detect_surface: float = 0.97
    p_detect_under: float = 0.93
    depth_penalty: float = 0.04  # per metre of depth
    glare_factor: float = 0.3
    occlusion_factor: float = 0.6
    box_jitter: float = 0.04
    p_flip: float = 0.03  # submersion estimate wrong in this frame
    fp_rate: float = 0.03  # spurious detections per frame and camera
    objects: list[tuple[float, float, float, float]] = field(default_factory=list)  # (x, y, depth, p) on the bottom


def body_points(s: BodyState) -> np.ndarray:
    """A dozen 3D points on the body (pool frame, z up) for projection."""
    L = s.length
    h = np.array([np.cos(s.heading), np.sin(s.heading), 0.0])
    side = np.array([-np.sin(s.heading), np.cos(s.heading), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    axis = (1 - s.vertical) * h + s.vertical * up
    axis /= np.linalg.norm(axis)
    centre = np.array([s.xy[0], s.xy[1], -s.z])
    pts = [centre + axis * a * L / 2 for a in np.linspace(-1, 1, 5)]
    pts += [centre + axis * a * L / 2 + side * w for a in (0.35, 0.1) for w in (-0.22, 0.22)]
    pts = np.array(pts)
    pts[:, 2] = np.minimum(pts[:, 2], 0.35)  # nothing floats far above the surface
    return pts


def project_bodies(cam: Camera, states: list[BodyState], pts: np.ndarray | None = None) -> list[np.ndarray | None]:
    """Image boxes of many bodies at once (refraction-aware); None where a body is out of view.
    ``pts`` (N, P, 3): body points when the caller already has them (they do not depend on the camera)."""
    if not states:
        return []
    if pts is None:
        pts = np.stack([body_points(s) for s in states])  # (N, P, 3)
    n, k, _ = pts.shape
    flat = pts.reshape(-1, 3)
    above = flat[:, 2] >= 0
    px = np.empty((len(flat), 2))
    if above.any():
        px[above] = cam.project(flat[above])
    if (~above).any():
        px[~above] = cam.project_underwater(flat[~above])
    px = px.reshape(n, k, 2)
    w, h = cam.size
    inside = (px[..., 0] >= 0) & (px[..., 0] < w) & (px[..., 1] >= 0) & (px[..., 1] < h)
    in_front = ((pts - cam.center) @ cam.R[2] > 0).all(1)
    boxes: list[np.ndarray | None] = []
    for i in range(n):
        if inside[i].mean() < 0.5 or not in_front[i]:
            boxes.append(None)
            continue
        q = px[i][inside[i]]
        x0, y0 = q.min(0)
        x1, y1 = q.max(0)
        pad = 0.12 * max(x1 - x0, y1 - y0, 8)
        boxes.append(np.array([max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w - 1), min(y1 + pad, h - 1)]))
    return boxes


def project_body(cam: Camera, s: BodyState) -> np.ndarray | None:
    return project_bodies(cam, [s])[0]


class OracleDetector:
    def __init__(
        self,
        cameras: dict[str, Camera],
        noise: NoiseModel | None = None,
        glare: dict[str, list[GlareSpot]] | None = None,
        seed: int = 0,
    ):
        self.cameras = cameras
        self.noise = noise or NoiseModel()
        self.glare = glare or {}
        self.rng = np.random.default_rng(seed)

    def detect(self, camera_id: str, states: list[BodyState], pts: np.ndarray | None = None) -> list[Detection]:
        cam, n, rng = self.cameras[camera_id], self.noise, self.rng
        boxes, kept = [], []
        for s, b in zip(states, project_bodies(cam, states, pts), strict=True):
            if b is not None:
                boxes.append(b)
                kept.append(s)
        if not kept:
            return self._false_positives(cam, camera_id)
        boxes = np.array(boxes)
        dist = np.array([np.linalg.norm(np.append(s.xy, -s.z) - cam.center) for s in kept])
        iou = iou_matrix(boxes, boxes)
        np.fill_diagonal(iou, 0)
        dets = []
        for i, s in enumerate(kept):
            p = n.p_detect_under - n.depth_penalty * s.z if s.head_under else n.p_detect_surface
            if any(g.contains(s.xy)[0] and rng.random() < g.strength for g in self.glare.get(camera_id, [])):
                p *= n.glare_factor
            if ((iou[i] > 0.4) & (dist < dist[i])).any():
                p *= n.occlusion_factor
            if rng.random() > p:
                continue
            b = boxes[i]
            size = np.sqrt((b[2] - b[0]) * (b[3] - b[1]))
            b = b + rng.normal(0, n.box_jitter * size, 4)
            under = s.head_under if rng.random() > n.p_flip else not s.head_under
            p_under = float(np.clip(rng.normal(0.86, 0.07) if under else rng.normal(0.08, 0.05), 0, 1))
            score = rng.normal(0.74 - 0.04 * s.z, 0.07) if s.head_under else rng.normal(0.86, 0.06)
            shimmer = 0.06 if s.head_under else 0.0
            dets.append(
                Detection(
                    box=b,
                    score=float(np.clip(score, 0.2, 0.99)),
                    p_under=p_under,
                    p_vertical=float(np.clip(s.vertical + rng.normal(0, 0.12), 0, 1)),
                    motion=float(np.clip(s.motion + shimmer + rng.normal(0, 0.06), 0, 1.5)),
                    extra={"truth": s.pid},
                )
            )
        return dets + self._false_positives(cam, camera_id)

    def _false_positives(self, cam: Camera, camera_id: str) -> list[Detection]:
        n, rng = self.noise, self.rng
        out = []
        if rng.random() < n.fp_rate:
            w, h = cam.size
            cx, cy, s = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(20, 70)
            out.append(
                Detection(
                    np.array([cx - s, cy - s, cx + s, cy + s]),
                    float(rng.uniform(0.5, 0.7)),
                    p_under=float(rng.uniform(0.2, 0.9)),
                    motion=float(rng.uniform(0, 0.6)),
                    extra={"truth": None},
                )
            )
        for x, y, depth, p in n.objects:  # a toy or a towel on the bottom, sometimes taken for a person
            if rng.random() < p:
                s = BodyState(-1, np.array([x, y]), depth - 0.05, 0.0, 0.0, True, 0.0, "object", 0.5, (0, 0, 0))
                b = project_body(cam, s)
                if b is not None:
                    out.append(
                        Detection(b, float(rng.uniform(0.5, 0.65)), p_under=0.9, motion=0.02, extra={"truth": "object"})
                    )
        return out
