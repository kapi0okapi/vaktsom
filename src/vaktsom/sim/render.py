"""Fast synthetic CCTV views of the simulated pool (real time on a laptop).

Every pixel's ray refracts at the water surface and meets the bottom, so depth and viewing angle
shift the tiles exactly as in a real pool; on top come animated ripples, caustics, Fresnel
reflection of the ceiling at grazing angles, glare spots, lane ropes, and the bathers, tinted and
blurred by the water above them. The views are schematic (for the demo, the user interface and
end-to-end tests with the oracle detector); photorealistic training images come from the
rendering pipeline in ``ml/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vaktsom.geometry import Basin, Camera, refract
from vaktsom.sim.oracle import GlareSpot, body_points
from vaktsom.sim.scene import BodyState

WATER = np.array([196, 160, 70], np.float32)  # BGR tint of pool water
CEILING = np.array([215, 210, 200], np.float32)
DECK = (175, 185, 190)


def pool_texture(basins: dict[str, Basin], ppm: float, margin: float = 6.0, seed: int = 0):
    """Top-down texture of the hall floor and basin bottoms; returns (image, origin, ppm)."""
    rng = np.random.default_rng(seed)
    pts = np.vstack([b.outline for b in basins.values()])
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    w, h = int((hi[0] - lo[0]) * ppm), int((hi[1] - lo[1]) * ppm)
    img = np.full((h, w, 3), DECK, np.uint8)
    deck_tile = max(int(0.3 * ppm), 2)
    img[::deck_tile, :] = (160, 168, 172)
    img[:, ::deck_tile] = (160, 168, 172)

    def to_px(p):
        return np.asarray((np.asarray(p) - lo) * ppm, np.int32)

    for b in basins.values():
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [to_px(b.outline)], 1)
        tiles = np.full((h, w, 3), (235, 215, 160), np.uint8)
        t = max(int(0.25 * ppm), 2)
        tiles[::t, :] = (215, 190, 130)
        tiles[:, ::t] = (215, 190, 130)
        noise = rng.normal(0, 6, (h, w, 1))
        tiles = np.clip(tiles + noise, 0, 255).astype(np.uint8)
        img[mask.astype(bool)] = tiles[mask.astype(bool)]
        cv2.polylines(img, [to_px(b.outline)], True, (140, 120, 80), max(int(0.15 * ppm), 1))
        bl, bh = b.outline.min(0), b.outline.max(0)
        if bh[0] - bl[0] >= 20 and bh[1] - bl[1] >= 8:  # lane lines on the bottom of a lap pool
            n_lanes = int((bh[1] - bl[1]) // 2.5)
            for k in range(n_lanes):
                y = bl[1] + (k + 0.5) * (bh[1] - bl[1]) / n_lanes
                a, c = to_px([bl[0] + 2.0, y]), to_px([bh[0] - 2.0, y])
                cv2.line(img, tuple(a), tuple(c), (90, 50, 20), max(int(0.25 * ppm), 1))
                for x_end in (bl[0] + 2.0, bh[0] - 2.0):
                    p = to_px([x_end, y])
                    cv2.line(
                        img,
                        (p[0], p[1] - int(0.5 * ppm)),
                        (p[0], p[1] + int(0.5 * ppm)),
                        (90, 50, 20),
                        max(int(0.25 * ppm), 1),
                    )
    return img, lo, ppm


def smooth_noise(shape, scale: float, rng: np.random.Generator) -> np.ndarray:
    small = rng.normal(0, 1, (max(int(shape[0] / scale), 2), max(int(shape[1] / scale), 2))).astype(np.float32)
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


@dataclass
class _View:
    map_x: np.ndarray
    map_y: np.ndarray
    water: np.ndarray  # bool, pixel sees water
    absorb: np.ndarray  # 1 - exp(-k * path length in water)
    fresnel: np.ndarray
    surface: np.ndarray  # (H, W, 2) surface point of each pixel, pool metres
    sky: np.ndarray  # rays that never reach the floor (walls, ceiling)


class SceneRenderer:
    def __init__(
        self,
        basins: dict[str, Basin],
        cameras: dict[str, Camera],
        size: tuple[int, int] = (960, 540),
        ppm: float = 40.0,
        glare: dict[str, list[GlareSpot]] | None = None,
        lane_ropes: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
        seed: int = 0,
    ):
        self.basins, self.cameras, self.size = basins, cameras, size
        self.glare = glare or {}
        self.lane_ropes = lane_ropes or []
        self.rng = np.random.default_rng(seed)
        self.tex, self.origin, self.ppm = pool_texture(basins, ppm, seed=seed)
        caustic = np.zeros(self.tex.shape[:2], np.float32)
        for k in range(3):
            n = smooth_noise(self.tex.shape[:2], 0.35 * ppm * (k + 1), self.rng)
            caustic += np.exp(-np.abs(n) * 6)  # bright filaments where the noise crosses zero
        self.caustic = (caustic / caustic.max()).astype(np.float32)
        self.views = {cid: self._prepare(cam) for cid, cam in cameras.items()}
        w, h = size
        self.ripples = [(smooth_noise((h, w), 18, self.rng), smooth_noise((h, w), 18, self.rng)) for _ in range(4)]
        self.noise = [self.rng.normal(0, 2.5, (h, w, 3)).astype(np.float32) for _ in range(6)]
        self._k = 0

    def _scaled(self, cam: Camera) -> Camera:
        sx, sy = self.size[0] / cam.size[0], self.size[1] / cam.size[1]
        K = cam.K.copy()
        K[0] *= sx
        K[1] *= sy
        return Camera(K, cam.R, cam.t, cam.dist, self.size)

    def _prepare(self, cam: Camera) -> _View:
        cam = self._scaled(cam)
        w, h = self.size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5, np.arange(h, dtype=np.float64) + 0.5)
        pix = np.stack([u.ravel(), v.ravel()], 1)
        d = cam.rays(pix)
        c = cam.center
        down = d[:, 2] < -1e-6
        s = np.where(down, -c[2] / np.where(down, d[:, 2], -1), 0)
        surf = c[:2] + s[:, None] * d[:, :2]
        water = np.zeros(len(pix), bool)
        depth = np.zeros(len(pix))
        for b in self.basins.values():
            lo, hi = b.outline.min(0), b.outline.max(0)
            cand = down & (surf[:, 0] >= lo[0]) & (surf[:, 0] <= hi[0]) & (surf[:, 1] >= lo[1]) & (surf[:, 1] <= hi[1])
            idx = np.flatnonzero(cand)
            if len(idx):
                inside = b.contains(surf[idx])
                idx = idx[inside]
                water[idx] = True
                depth[idx] = b.depth_at(surf[idx])
        t = refract(d)
        bottom = surf.copy()
        path = np.zeros(len(pix))
        wi = np.flatnonzero(water)
        if len(wi):
            k = depth[wi] / np.maximum(-t[wi, 2], 1e-6)
            bottom[wi] = surf[wi] + k[:, None] * t[wi, :2]
            path[wi] = k
        mx = ((bottom[:, 0] - self.origin[0]) * self.ppm).astype(np.float32)
        my = ((bottom[:, 1] - self.origin[1]) * self.ppm).astype(np.float32)
        cos_i = np.clip(-d[:, 2], 0, 1)
        fresnel = 0.02 + 0.98 * (1 - cos_i) ** 5
        return _View(
            mx.reshape(h, w),
            my.reshape(h, w),
            water.reshape(h, w),
            (1 - np.exp(-0.18 * path)).reshape(h, w).astype(np.float32),
            np.where(water, fresnel, 0).reshape(h, w).astype(np.float32),
            surf.reshape(h, w, 2).astype(np.float32),
            (~down).reshape(h, w),
        )

    def render(self, camera_id: str, t: float, bodies: list[BodyState], wave: float = 1.0) -> np.ndarray:
        view = self.views[camera_id]
        cam = self._scaled(self.cameras[camera_id])
        w, h = self.size
        # animated ripples: blend of fixed noise fields, stronger where the water is deep
        ph = [np.sin(t * f + p) for f, p in ((1.3, 0), (0.9, 1.7), (1.7, 3.1), (0.6, 4.2))]
        dx = sum(a * r[0] for a, r in zip(ph, self.ripples, strict=True)) * (1.2 * wave)
        dy = sum(a * r[1] for a, r in zip(ph, self.ripples, strict=True)) * (1.2 * wave)
        amp = view.absorb * 2.2 + 0.25
        mx = (view.map_x + dx * amp * view.water).astype(np.float32)
        my = (view.map_y + dy * amp * view.water).astype(np.float32)
        img = cv2.remap(self.tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        caus = cv2.remap(
            self.caustic,
            (mx + 25 * np.sin(t * 0.3)).astype(np.float32),
            (my + 25 * np.cos(t * 0.27)).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        img *= (1 + 0.35 * caus * view.water)[..., None]
        a = view.absorb[..., None]
        img = img * (1 - a) + WATER * a
        f = view.fresnel[..., None]
        img = img * (1 - f) + CEILING * f
        img[view.sky] = (190, 196, 200)

        self._draw_bodies(img, cam, view, bodies, t)
        for p0, p1 in self.lane_ropes:
            xs = np.linspace(0, 1, 120)
            pts = np.array([[p0[0] + (p1[0] - p0[0]) * x, p0[1] + (p1[1] - p0[1]) * x, 0.02] for x in xs])
            px = cam.project(pts)
            for k in range(len(px) - 1):
                col = (40, 40, 220) if (k // 6) % 2 else (230, 230, 230)
                cv2.line(img, tuple(np.int32(px[k])), tuple(np.int32(px[k + 1])), col, 2, cv2.LINE_AA)
        for g in self.glare.get(camera_id, []):
            d = (view.surface - np.asarray(g.centre, np.float32)) / np.asarray(g.radii, np.float32)
            m = np.clip(1.2 - (d**2).sum(-1), 0, 1) * view.water * g.strength
            flicker = 0.85 + 0.15 * self.ripples[0][0]
            img += (m * flicker * 255)[..., None] * 0.9
        self._k += 1
        img += self.noise[self._k % len(self.noise)]
        return np.clip(img, 0, 255).astype(np.uint8)

    def _draw_bodies(self, img: np.ndarray, cam: Camera, view: _View, bodies: list[BodyState], t: float) -> None:
        order = sorted(bodies, key=lambda s: -np.linalg.norm(np.append(s.xy, -s.z) - cam.center))
        for s in order:
            pts = body_points(s)
            under = pts[:, 2] < 0
            px = np.empty((len(pts), 2))
            if (~under).any():
                px[~under] = cam.project(pts[~under])
            if under.any():
                px[under] = cam.project_underwater(pts[under])
            if not np.isfinite(px).all() or ((pts - cam.center) @ cam.R[2] <= 0.5).any():
                continue
            w_img, h_img = self.size
            if (
                px[:, 0].max() < -50
                or px[:, 0].min() > w_img + 50
                or px[:, 1].max() < -50
                or px[:, 1].min() > h_img + 50
                or np.abs(px).max() > 8 * max(w_img, h_img)
            ):
                continue
            width = float(np.clip(np.linalg.norm(px[5] - px[6]), 3.0, 120.0))
            pad = int(width * 2 + 12)
            x0, y0 = int(max(px[:, 0].min() - pad, 0)), int(max(px[:, 1].min() - pad, 0))
            x1, y1 = int(min(px[:, 0].max() + pad, img.shape[1])), int(min(px[:, 1].max() + pad, img.shape[0]))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            q = px - [x0, y0]
            layer = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
            mask = np.zeros((y1 - y0, x1 - x0), np.float32)
            skin = (150, 175, 215)
            axis = q[:5]
            for k in range(4):  # torso and legs as thick segments
                p0, p1 = tuple(np.int32(axis[k])), tuple(np.int32(axis[k + 1]))
                col = s.colour if k in (1, 2) else skin
                th = max(int(width * (0.9 if k in (1, 2) else 0.55)), 2)
                cv2.line(layer, p0, p1, col, th, cv2.LINE_AA)
                cv2.line(mask, p0, p1, 1.0, th, cv2.LINE_AA)
            head = tuple(np.int32(axis[4]))
            r = max(int(width * 0.45), 2)
            cv2.circle(layer, head, r, (60, 70, 90) if s.face_down else skin, -1, cv2.LINE_AA)
            cv2.circle(mask, head, r, 1.0, -1, cv2.LINE_AA)
            swing = np.sin(t * 6 + s.pid) * s.motion
            for side in (5, 7):  # arms, swinging with the limb motion
                sh = q[side]
                tip = sh + (q[side] - q[side + 1]) * (1.2 + swing) + (axis[4] - axis[2]) * 0.4 * swing
                th = max(int(width * 0.25), 1)
                cv2.line(layer, tuple(np.int32(sh)), tuple(np.int32(tip)), skin, th, cv2.LINE_AA)
                cv2.line(mask, tuple(np.int32(sh)), tuple(np.int32(tip)), 1.0, th, cv2.LINE_AA)
            depth = max(s.z, 0.0)
            blur = int(1 + depth * 2.5) * 2 + 1
            layer = cv2.GaussianBlur(layer, (blur, blur), 0)
            mask = cv2.GaussianBlur(mask, (blur, blur), 0)
            atten = 1 - np.exp(-0.35 * depth)
            layer = layer * (1 - atten) + WATER * atten
            alpha = (mask * (0.95 - 0.35 * min(depth, 1.5) / 1.5))[..., None]
            roi = img[y0:y1, x0:x1]
            roi[:] = roi * (1 - alpha) + layer * alpha
            head = (head[0] + x0, head[1] + y0)
            if not s.head_under and s.motion > 0.5:  # splash ring for an active swimmer
                cv2.circle(img, head, int(r * 1.8), (245, 245, 245), 1, cv2.LINE_AA)
