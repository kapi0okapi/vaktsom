"""Pool geometry: basin outlines, depth, and camera rays that bend at the water surface.

Pool frame: metres, x along the basin, y across, z up, water surface at z = 0, bottom at
z = -depth(x, y). A camera sees a submerged body along a ray that refracts at the surface
(Snell's law, water n = 1.333). Ignoring this puts a body on the bottom of a 5 m diving pit several
metres from where it really is, so every submerged position goes through ``Camera.to_depth``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

WATER_N = 1.333


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Boolean mask: which (N, 2) points lie inside the closed polygon (M, 2) (even-odd rule)."""
    pts = np.atleast_2d(np.asarray(points, np.float64))
    poly = np.asarray(polygon, np.float64)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    on_edge = np.zeros(len(pts), bool)
    xj, yj = poly[-1]
    for xi, yi in poly:
        crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= crosses
        # points on the outline count as inside (a body at the wall is still in the basin)
        ex, ey = xj - xi, yj - yi
        t = np.clip(((x - xi) * ex + (y - yi) * ey) / max(ex * ex + ey * ey, 1e-12), 0, 1)
        on_edge |= np.hypot(x - (xi + t * ex), y - (yi + t * ey)) < 1e-6
        xj, yj = xi, yi
    return inside | on_edge


def polygon_area(polygon: np.ndarray) -> float:
    p = np.asarray(polygon, np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


@dataclass
class DepthRegion:
    """Part of a basin with a constant depth, or a linear slope from depth_a at point a to depth_b at b."""

    polygon: np.ndarray
    depth: float
    slope_to: tuple[tuple[float, float], tuple[float, float], float] | None = None  # (a, b, depth_b)
    movable: bool = False  # depth follows the movable floor setting

    def contains(self, xy: np.ndarray) -> np.ndarray:
        lo, hi = self.polygon.min(0), self.polygon.max(0)
        box = (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
        if len(self.polygon) == 4 and len(set(self.polygon[:, 0])) == 2 and len(set(self.polygon[:, 1])) == 2:
            return box
        out = np.zeros(len(xy), bool)
        if box.any():
            out[box] = points_in_polygon(xy[box], self.polygon)
        return out

    def depth_at(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy)
        if self.slope_to is None:
            return np.full(len(xy), self.depth)
        a, b, depth_b = self.slope_to
        a, b = np.asarray(a, float), np.asarray(b, float)
        ab = b - a
        t = np.clip(((xy - a) @ ab) / float(ab @ ab), 0.0, 1.0)
        return self.depth + t * (depth_b - self.depth)


@dataclass
class Basin:
    id: str
    name: str
    outline: np.ndarray  # (M, 2) metres
    regions: list[DepthRegion] = field(default_factory=list)
    default_depth: float = 1.0
    movable_floor_depth: float | None = None  # current setting, if the basin has a movable floor

    def __post_init__(self) -> None:
        o = np.asarray(self.outline, float)
        self._lo, self._hi = o.min(0), o.max(0)
        xs, ys = set(np.round(o[:, 0], 9)), set(np.round(o[:, 1], 9))
        self._rect = len(o) == 4 and len(xs) == 2 and len(ys) == 2

    def contains(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, float))
        inside = (
            (xy[:, 0] >= self._lo[0])
            & (xy[:, 0] <= self._hi[0])
            & (xy[:, 1] >= self._lo[1])
            & (xy[:, 1] <= self._hi[1])
        )
        if self._rect or not inside.any():
            return inside
        out = np.zeros(len(xy), bool)
        out[inside] = points_in_polygon(xy[inside], self.outline)
        return out

    def clamp(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The nearest point of the basin to each point (the point itself when inside), and how far
        away it is."""
        xy = np.atleast_2d(np.asarray(xy, float))
        out, dist = xy.copy(), np.zeros(len(xy))
        outside = ~self.contains(xy)
        if outside.any():
            a = np.asarray(self.outline, float)
            ab = np.roll(a, -1, axis=0) - a  # edges
            p = xy[outside][:, None, :]
            t = np.clip(((p - a) * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
            near = a + t[..., None] * ab  # nearest point on each edge
            d = np.linalg.norm(near - p, axis=-1)
            k = d.argmin(1)
            out[outside] = near[np.arange(len(k)), k]
            dist[outside] = d[np.arange(len(k)), k]
        return out, dist

    def depth_at(self, xy: np.ndarray) -> np.ndarray:
        """Water depth (m, positive) at each point; later regions override earlier ones."""
        xy = np.atleast_2d(np.asarray(xy, float))
        depth = np.full(len(xy), self.default_depth)
        for region in self.regions:
            inside = region.contains(xy)
            if not inside.any():
                continue
            if region.movable and self.movable_floor_depth is not None:
                depth[inside] = self.movable_floor_depth
            else:
                depth[inside] = region.depth_at(xy[inside])
        return depth

    def floor_unknown(self, xy: np.ndarray) -> np.ndarray:
        """True where the depth follows a movable floor whose current setting is not known."""
        xy = np.atleast_2d(np.asarray(xy, float))
        out = np.zeros(len(xy), bool)
        if self.movable_floor_depth is None:
            for region in self.regions:
                if region.movable:
                    out |= region.contains(xy)
        return out

    @property
    def area(self) -> float:
        return polygon_area(self.outline)


def refract(direction: np.ndarray, n_ratio: float = 1.0 / WATER_N) -> np.ndarray:
    """Refract downward unit rays (N, 3) at the horizontal surface z = 0 (normal +z)."""
    d = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    normal = np.array([0.0, 0.0, 1.0])
    cos_i = -(d @ normal)
    k = 1.0 - n_ratio**2 * (1.0 - cos_i**2)  # > 0 always when going from air into water
    t = n_ratio * d + (n_ratio * cos_i - np.sqrt(np.maximum(k, 0.0)))[:, None] * normal
    return t / np.linalg.norm(t, axis=-1, keepdims=True)


@dataclass
class Camera:
    """Pinhole camera in the pool frame. ``R``, ``t`` map pool points to camera coordinates."""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    size: tuple[int, int] = (1920, 1080)  # width, height

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @classmethod
    def from_pnp(
        cls,
        image_points: np.ndarray,
        pool_points: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray | None = None,
        size: tuple[int, int] = (1920, 1080),
    ) -> Camera:
        """Pose from at least 4 surveyed points (pool corners, lane marks) and their pixels."""
        dist = np.zeros(5) if dist is None else np.asarray(dist, float)
        pool_points = np.asarray(pool_points, float)
        if pool_points.shape[1] == 2:
            pool_points = np.hstack([pool_points, np.zeros((len(pool_points), 1))])
        ok, rvec, tvec = cv2.solvePnP(
            pool_points, np.asarray(image_points, float), K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            raise ValueError("solvePnP failed: check the point correspondences")
        R, _ = cv2.Rodrigues(rvec)
        return cls(K=np.asarray(K, float), R=R, t=tvec.ravel(), dist=dist, size=size)

    @classmethod
    def looking_at(
        cls, position: np.ndarray, target: np.ndarray, hfov_deg: float, size: tuple[int, int] = (1920, 1080)
    ) -> Camera:
        """A camera at ``position`` aimed at ``target`` (both pool frame), for design and simulation."""
        w, h = size
        f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        position, target = np.asarray(position, float), np.asarray(target, float)
        forward = target - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        if np.linalg.norm(right) < 1e-6:  # looking straight down
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.stack([right, down, forward])  # rows: camera x (right), y (down), z (forward)
        return cls(K=K, R=R, t=-R @ position, size=size)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Pool points (N, 3) seen through air only -> pixels (N, 2)."""
        rvec, _ = cv2.Rodrigues(self.R)
        px, _ = cv2.projectPoints(np.asarray(points, float).reshape(-1, 1, 3), rvec, self.t, self.K, self.dist)
        return px.reshape(-1, 2)

    def project_underwater(self, points: np.ndarray, iterations: int = 24) -> np.ndarray:
        """Pool points below the surface (N, 3) -> pixels, through the refracting surface.

        For each point the ray crosses the surface where Snell's law holds (Fermat's principle):
        a bisection along the horizontal line from the camera foot to the point, vectorised.
        Points at or above the surface are projected directly.
        """
        points = np.atleast_2d(np.asarray(points, float))
        c = self.center
        out = points.copy()
        under = points[:, 2] < 0
        if under.any():
            p = points[under]
            horizontal = p[:, :2] - c[:2]
            dist = np.linalg.norm(horizontal, axis=1)
            u = horizontal / np.maximum(dist, 1e-12)[:, None]
            h_air, h_water = c[2], -p[:, 2]
            lo, hi = np.zeros(len(p)), dist.copy()
            for _ in range(iterations):  # sin_air - n * sin_water is increasing in s
                s = 0.5 * (lo + hi)
                sin_air = s / np.hypot(s, h_air)
                sin_water = (dist - s) / np.maximum(np.hypot(dist - s, h_water), 1e-12)
                go_right = sin_air < WATER_N * sin_water
                lo = np.where(go_right, s, lo)
                hi = np.where(go_right, hi, s)
            s = 0.5 * (lo + hi)
            out[under, :2] = c[:2] + u * s[:, None]
            out[under, 2] = 0.0
        return self.project(out)

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        """Unit ray directions (N, 3) in the pool frame for pixels (N, 2)."""
        pix = np.asarray(pixels, float).reshape(-1, 1, 2)
        norm = cv2.undistortPoints(pix, self.K, self.dist).reshape(-1, 2)
        d_cam = np.hstack([norm, np.ones((len(norm), 1))])
        d = d_cam @ self.R  # R^T applied to each row
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def to_surface(self, pixels: np.ndarray) -> np.ndarray:
        """Pixels -> (N, 2) points on the water surface z = 0."""
        d = self.rays(pixels)
        c = self.center
        s = -c[2] / np.where(d[:, 2] < -1e-9, d[:, 2], -1e-9)
        return c[:2] + s[:, None] * d[:, :2]

    def underwater_rays(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(surface points (N, 2), horizontal step per metre of depth (N, 2)) of the refracted rays.

        The point at depth d along pixel i's refracted ray is ``surface[i] + d * step[i]``.
        """
        surface = self.to_surface(pixels)
        t = refract(self.rays(pixels))
        return surface, t[:, :2] / np.maximum(-t[:, 2], 1e-9)[:, None]

    def to_depth(self, pixels: np.ndarray, depth: float | np.ndarray) -> np.ndarray:
        """Pixels -> (N, 2) points at ``depth`` metres below the surface, with refraction."""
        surface = self.to_surface(pixels)
        t = refract(self.rays(pixels))
        depth = np.broadcast_to(np.asarray(depth, float), (len(surface),))
        s = depth / np.where(t[:, 2] < -1e-9, -t[:, 2], 1e-9)
        return surface + s[:, None] * t[:, :2]

    def to_bottom(self, pixels: np.ndarray, basin: Basin, height_above_bottom: float = 0.25) -> np.ndarray:
        """Pixels of something lying on the bottom -> (N, 2) pool points (iterates on varying depth)."""
        xy = self.to_surface(pixels)
        for _ in range(3):
            depth = np.maximum(basin.depth_at(xy) - height_above_bottom, 0.0)
            xy = self.to_depth(pixels, depth)
        return xy

    def ground_resolution(self, xy: np.ndarray) -> np.ndarray:
        """Approximate pixels per metre on the surface at pool points (N, 2)."""
        xy = np.atleast_2d(np.asarray(xy, float))
        p0 = np.hstack([xy, np.zeros((len(xy), 1))])
        p1 = p0 + np.array([0.5, 0.0, 0.0])
        p2 = p0 + np.array([0.0, 0.5, 0.0])
        a, b, c = self.project(p0), self.project(p1), self.project(p2)
        return np.linalg.norm(b - a, axis=1) + np.linalg.norm(c - a, axis=1)  # px per 0.5 m * 2

    def sees(self, xy: np.ndarray, margin: int = 8) -> np.ndarray:
        """Which pool surface points (N, 2) fall inside the image and in front of the camera."""
        xy = np.atleast_2d(np.asarray(xy, float))
        p = np.hstack([xy, np.zeros((len(xy), 1))])
        in_front = (p - self.center) @ self.R[2] > 0
        px = self.project(p)
        w, h = self.size
        inside = (px[:, 0] >= margin) & (px[:, 0] < w - margin) & (px[:, 1] >= margin) & (px[:, 1] < h - margin)
        return in_front & inside
