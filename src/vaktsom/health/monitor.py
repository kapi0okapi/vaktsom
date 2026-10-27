"""Is the system able to detect right now? Camera health and live coverage of every basin.

ISO 20380 4.3.3: staff must be told immediately when detection performance deteriorates, and the
interface must show the level and the nature of the deterioration at all times. Each camera is
checked on every analysed frame (no frames, frozen picture, darkness, blur or fog on the lens, sun
glare on the water) and every basin is divided into 0.5 m cells that know which cameras see them.
A cell that loses its last good view is reported as uncovered, with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from vaktsom.geometry import Basin, Camera, points_in_polygon


@dataclass
class CameraStatus:
    camera_id: str
    state: str = "unknown"  # ok | degraded | down | unknown
    reasons: list[str] = field(default_factory=list)
    fps: float = 0.0
    last_frame: float = 0.0
    glare: float = 0.0  # fraction of visible water lost to glare
    sharpness: float = 0.0
    brightness: float = 0.0
    glare_map: np.ndarray | None = None  # low-res boolean map of glare pixels

    def public(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "state": self.state,
            "reasons": self.reasons,
            "fps": round(self.fps, 1),
            "glare": round(self.glare, 3),
            "sharpness": round(self.sharpness, 1),
            "brightness": round(self.brightness, 1),
            "last_frame": self.last_frame,
        }


class CameraHealth:
    """Per-camera frame checks. Thresholds are per site, tuned during the technical study."""

    def __init__(
        self,
        camera_id: str,
        water_mask: np.ndarray | None,
        size: tuple[int, int],
        min_fps: float = 5.0,
        glare_limit: float = 0.25,
        min_sharpness: float = 20.0,
    ):
        self.status = CameraStatus(camera_id)
        self.min_fps, self.glare_limit, self.min_sharpness = min_fps, glare_limit, min_sharpness
        w, h = size
        self.scale = 320 / w
        self.small = (320, max(int(h * self.scale), 1))
        mask = np.zeros(self.small[::-1], np.uint8)
        if water_mask is not None:
            cv2.fillPoly(mask, [np.asarray(np.asarray(water_mask) * self.scale, np.int32)], 1)
        else:
            mask[:] = 1
        self.mask = mask.astype(bool)
        self._prev: np.ndarray | None = None
        self._frozen_since: float | None = None

    def frame(self, image: np.ndarray, ts: float) -> CameraStatus:
        s = self.status
        if s.last_frame:
            dt = max(ts - s.last_frame, 1e-3)
            s.fps = 0.8 * s.fps + 0.2 / dt if s.fps else 1.0 / dt
        s.last_frame = ts
        small = cv2.resize(image, self.small, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = hsv[..., 2]
        water = self.mask
        s.brightness = float(gray[water].mean()) if water.any() else float(gray.mean())
        glare = (hsv[..., 2] > 235) & (hsv[..., 1] < 40) & water
        s.glare_map = glare
        s.glare = float(glare.sum() / max(water.sum(), 1))
        s.sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        if self._prev is not None and np.abs(small.astype(np.int16) - self._prev).mean() < 0.3:
            self._frozen_since = self._frozen_since or ts
        else:
            self._frozen_since = None
        self._prev = small.astype(np.int16)

        reasons = []
        if self._frozen_since is not None and ts - self._frozen_since > 5.0:
            reasons.append("frozen")
        if s.brightness < 25:
            reasons.append("dark")
        if s.sharpness < self.min_sharpness:
            reasons.append("blurred_or_fogged")
        if s.glare > self.glare_limit:
            reasons.append("glare")
        if s.fps and s.fps < self.min_fps:
            reasons.append("low_fps")
        s.reasons = reasons
        s.state = "down" if "frozen" in reasons else ("degraded" if reasons else "ok")
        return s

    def check_alive(self, now: float, timeout: float = 3.0) -> CameraStatus:
        s = self.status
        if not s.last_frame or now - s.last_frame > timeout:
            s.state, s.reasons = "down", ["no_frames"]
        return s


@dataclass
class BasinCoverage:
    basin_id: str
    state: str  # ok | reduced (some cells with one view) | degraded (cells without a view) | down
    uncovered_fraction: float
    single_view_fraction: float
    reasons: list[str]
    uncovered_cells: list[tuple[float, float]]

    def public(self) -> dict:
        return {
            "basin_id": self.basin_id,
            "state": self.state,
            "uncovered_fraction": round(self.uncovered_fraction, 3),
            "single_view_fraction": round(self.single_view_fraction, 3),
            "reasons": self.reasons,
            "uncovered_cells": [[round(x, 2), round(y, 2)] for x, y in self.uncovered_cells[:400]],
        }


class CoverageMap:
    """Which cameras see each 0.5 m cell of each basin (design time), and which still do (live)."""

    def __init__(
        self,
        basins: dict[str, Basin],
        cameras: dict[str, Camera],
        camera_basins: dict[str, list[str]],
        water_masks: dict[str, np.ndarray | None],
        cell: float = 0.5,
        min_px_per_m: float = 30.0,
    ):
        self.cell = cell
        self.cells: dict[str, np.ndarray] = {}
        self.views: dict[str, dict[str, np.ndarray]] = {}  # basin -> camera -> visible cell mask
        self.pixels: dict[str, dict[str, np.ndarray]] = {}
        self.cameras = cameras
        for bid, basin in basins.items():
            lo, hi = basin.outline.min(0), basin.outline.max(0)
            xs = np.arange(lo[0] + cell / 2, hi[0], cell)
            ys = np.arange(lo[1] + cell / 2, hi[1], cell)
            grid = np.array([(x, y) for x in xs for y in ys])
            grid = grid[basin.contains(grid)] if len(grid) else grid.reshape(0, 2)
            self.cells[bid] = grid
            self.views[bid], self.pixels[bid] = {}, {}
            for cid, cam in cameras.items():
                if bid not in camera_basins.get(cid, []) or len(grid) == 0:
                    continue
                visible = cam.sees(grid) & (cam.ground_resolution(grid) >= min_px_per_m)
                px = cam.project(np.hstack([grid, np.zeros((len(grid), 1))]))
                mask = water_masks.get(cid)
                if mask is not None:
                    visible &= points_in_polygon(px, mask)
                self.views[bid][cid] = visible
                self.pixels[bid][cid] = px

    def design_report(self) -> dict[str, dict]:
        """Coverage at design time: share of each basin seen by >= 1 and >= 2 cameras."""
        out = {}
        for bid, grid in self.cells.items():
            n = sum(v.astype(int) for v in self.views[bid].values()) if self.views[bid] else np.zeros(len(grid))
            out[bid] = {
                "cells": int(len(grid)),
                "covered_1": float((n >= 1).mean()) if len(grid) else 0.0,
                "covered_2": float((n >= 2).mean()) if len(grid) else 0.0,
            }
        return out

    def live(self, status: dict[str, CameraStatus]) -> dict[str, BasinCoverage]:
        out = {}
        for bid, grid in self.cells.items():
            good = np.zeros(len(grid), int)
            reasons: set[str] = set()
            for cid, visible in self.views[bid].items():
                st = status.get(cid)
                if st is None or st.state == "down":
                    if st is not None:
                        reasons.update(f"{cid}:{r}" for r in st.reasons)
                    continue
                ok = visible.copy()
                if st.glare_map is not None and st.glare > 0:  # cells under glare in this view
                    px = self.pixels[bid][cid]
                    scale = st.glare_map.shape[1] / self.cameras[cid].size[0]
                    gx = np.clip((px[:, 0] * scale).astype(int), 0, st.glare_map.shape[1] - 1)
                    gy = np.clip((px[:, 1] * scale).astype(int), 0, st.glare_map.shape[0] - 1)
                    blinded = st.glare_map[gy, gx] & visible
                    if blinded.any():
                        reasons.add(f"{cid}:glare")
                    ok &= ~blinded
                good += ok
            n = max(len(grid), 1)
            uncovered = good == 0
            state = "ok"
            if uncovered.all() and len(grid):
                state = "down"
            elif uncovered.any():
                state = "degraded"
            elif (good == 1).any():
                state = "reduced"
            out[bid] = BasinCoverage(
                bid,
                state,
                float(uncovered.sum() / n),
                float((good == 1).sum() / n),
                sorted(reasons),
                [tuple(c) for c in grid[uncovered]],
            )
        return out
