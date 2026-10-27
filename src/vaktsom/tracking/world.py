"""People in pool coordinates, fused from every camera that sees them.

Each camera's tracks are projected into the pool frame (refraction-aware for submerged people) and
linked to one person per camera. Links persist while they stay consistent, so a swimmer keeps one
identity across camera hand-overs. A submerged person seen by two or more cameras is triangulated:
the depth at which the refracted rays from all views meet tells whether the body is on the bottom
(the ISO 20380 definition of a detection) or still sinking. Views that never meet are two bodies (a
camera's tracker swapped two people who crossed): the odd view is let go instead of averaged in.

A person no camera sees any more is kept as *lost* at the last known position, because a body
that sank under glare must not silently disappear; the missing-swimmer rule acts on that.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from vaktsom.geometry import Basin, Camera
from vaktsom.tracking.camera_tracker import LocalTrack


@dataclass
class Observation:
    camera_id: str
    track_id: int
    ts: float
    pixel: np.ndarray
    xy_surface: np.ndarray
    p_under: float
    p_vertical: float
    motion: float
    weight: float
    box: np.ndarray
    step: np.ndarray  # horizontal shift per metre of depth along the refracted ray
    xy: np.ndarray | None = None  # final position estimate, set during fusion
    pixel_prev: np.ndarray | None = None  # the same track's pixel about a second earlier ...
    dt_prev: float = 0.0  # ... and how much earlier

    def at_depth(self, depth: float | np.ndarray) -> np.ndarray:
        return self.xy_surface + np.multiply.outer(np.asarray(depth, float), self.step)

    def ray_distance(self, ref: np.ndarray, top: float, bottom: float) -> float:
        """Distance from ``ref`` to where this body can be between ``top`` and ``bottom`` metres deep:
        one camera cannot tell how deep a submerged body is, only along which refracted ray it lies."""
        a = self.at_depth(max(top, 0.0))
        seg = self.at_depth(max(bottom, top, 0.0)) - a
        n2 = float(seg @ seg)
        t = 0.0 if n2 == 0.0 else float(np.clip((ref - a) @ seg / n2, 0.0, 1.0))
        return float(np.linalg.norm(a + t * seg - ref))


class PointKalman:
    """Constant-velocity filter on (x, y) in metres."""

    def __init__(self, xy: np.ndarray, ts: float):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([0.5, 0.5, 1.0, 1.0])
        self.ts = ts

    def predict(self, ts: float, accel: float = 1.5) -> None:
        dt = max(ts - self.ts, 0.0)
        if dt == 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        self.P = F @ self.P @ F.T + G @ G.T * accel**2
        self.x = F @ self.x
        self.ts = ts

    def update(self, xy: np.ndarray, sigma: float) -> None:
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        S = H @ self.P @ H.T + np.eye(2) * sigma**2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(xy) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    @property
    def xy(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))


@dataclass
class Person:
    id: int
    kf: PointKalman
    first_ts: float
    last_seen: float
    basin_id: str | None = None
    links: dict[str, int] = field(default_factory=dict)  # camera id -> local track id
    observations: dict[str, Observation] = field(default_factory=dict)
    p_under: float = 0.0
    p_vertical: float = 0.0
    motion: float = 1.0
    depth: float | None = None  # metres below the surface, from multi-view triangulation
    # working depth for placing single-view observations: grows at a plausible descent rate while the
    # person is under water instead of jumping to the bottom (a duck dive is not a body on the bottom)
    depth_est: float = 0.0
    # speed of the body in the images, measured at one depth: a working depth that is still growing
    # moves the computed position, but is not motion of the body
    still_speed: float | None = None
    bottom_depth: float | None = None  # water depth at the position
    lost: bool = False
    updates: int = 0
    distress: float = 0.0  # early-distress score, 0..1 (analysis.distress)
    history: deque = field(default_factory=lambda: deque(maxlen=600))

    @property
    def xy(self) -> np.ndarray:
        return self.kf.xy

    @property
    def n_views(self) -> int:
        return len(self.links)

    @property
    def on_bottom(self) -> bool | None:
        """True/False when triangulated, None when only one camera sees the person."""
        if self.depth is None or self.bottom_depth is None:
            return None
        return self.depth >= self.bottom_depth - 0.6


@dataclass
class WorldConfig:
    link_gate_m: float = 1.6  # new link: observation within this distance of the person
    keep_gate_m: float = 2.8  # existing link survives while it stays this close
    relink_gate_m: float = 2.5  # a lost person is re-acquired this close to the last position ...
    underwater_speed: float = 1.0  # ... plus this many metres per second spent unseen under water
    max_relink_m: float = 8.0
    keep_lost_s: float = 90.0
    max_obs_age_s: float = 0.6
    # a camera only speaks for its own basins (technical study), and only where it sees well
    min_px_per_m: float = 20.0
    max_view_deg: float = 75.0
    smoothing_tau_s: float = 0.8
    under_threshold: float = 0.5
    descent_mps: float = 0.8  # how fast the working depth of an unseen-depth submerged person may grow
    # a new view of a submerged body links along its ray only where the person can be: near the
    # triangulated depth, or the start of a dive for someone seen at the surface (a swimmer right above
    # a body on the bottom of a deep basin is not that body)
    depth_margin_m: float = 1.0
    surface_band_m: float = 0.5
    split_window_s: float = 4.0  # a newcomer born this close in time to a submerged person's loss ...
    split_radius_m: float = 1.5  # ... and this close in space is the same person
    duplicate_radius_m: float = 1.2  # two people this close, from disjoint cameras ...
    duplicate_hold_s: float = 2.0  # ... for this long, are one person
    # one person's views this far apart for this long show two bodies (a camera's tracker swapped two
    # people who crossed); views of one body stay within about a metre of each other
    disagree_m: float = 1.5
    disagree_s: float = 1.0
    let_go_s: float = 30.0  # the view let go is not taken back by that person for this long
    edge_margin_m: float = 0.5  # a body by the wall is now and then estimated this far outside it
    still_baseline_s: float = 1.0  # stillness: image motion over this baseline, at a fixed depth


class WorldTracker:
    def __init__(
        self,
        cameras: dict[str, Camera],
        basins: dict[str, Basin],
        config: WorldConfig | None = None,
        camera_basins: dict[str, list[str]] | None = None,
    ):
        self.cameras = cameras
        self.basins = basins
        self.camera_basins = camera_basins or {}
        self.cfg = config or WorldConfig()
        self.persons: dict[int, Person] = {}
        self._ids = itertools.count(1)
        self.pinned: set[int] = set()  # persons in an open alarm: never dropped
        self._close_since: dict[tuple[int, int], float] = {}
        self._apart_since: dict[int, float] = {}
        self._let_go: dict[tuple[int, str, int], float] = {}  # (person, camera, track) views not to take back

    # projections -------------------------------------------------------------------------------
    def basin_at(self, xy: np.ndarray) -> Basin | None:
        for b in self.basins.values():
            if b.contains(xy[None])[0]:
                return b
        return None

    def _in_basin(self, xy: np.ndarray) -> tuple[Basin | None, np.ndarray]:
        """The basin at ``xy``, or the nearest one within ``edge_margin_m`` (a body lying by the wall
        must not drop out of its basin because its estimate strays past the edge), and the point of
        the basin to read the depth at."""
        b = self.basin_at(xy)
        if b is not None:
            return b, xy
        best, at, gap = None, xy, self.cfg.edge_margin_m
        for c in self.basins.values():
            near, d = c.clamp(xy)
            if d[0] <= gap:
                best, at, gap = c, near[0], d[0]
        return best, at

    def observe(self, camera_id: str, tracks: list[LocalTrack], ts: float) -> list[Observation]:
        cam = self.cameras[camera_id]
        live = [t for t in tracks if ts - t.last_ts <= self.cfg.max_obs_age_s]
        if not live:
            return []
        boxes = np.array([t.box for t in live])
        pixels = (boxes[:, :2] + boxes[:, 2:]) / 2
        surface, step = cam.underwater_rays(pixels)
        # view quality: ground resolution, and a steep view beats a grazing one (glare, refraction)
        px_per_m = cam.ground_resolution(surface)
        ray = np.hstack([surface, np.zeros((len(surface), 1))]) - cam.center
        cos_angle = np.abs(ray[:, 2]) / np.maximum(np.linalg.norm(ray, axis=1), 1e-9)
        weight = np.minimum(px_per_m, 200.0) / 200.0 * np.maximum(cos_angle, 0.15)
        usable = (px_per_m >= self.cfg.min_px_per_m) & (cos_angle >= np.cos(np.radians(self.cfg.max_view_deg)))
        own = self.camera_basins.get(camera_id)
        if own:  # far views into other basins are grazing, refracted and mislocated: not this camera's job
            near = np.zeros(len(surface), bool)
            for bid in own:
                b = self.basins.get(bid)
                if b is not None:
                    lo, hi = b.outline.min(0) - 1.5, b.outline.max(0) + 1.5
                    near |= (
                        (surface[:, 0] >= lo[0])
                        & (surface[:, 0] <= hi[0])
                        & (surface[:, 1] >= lo[1])
                        & (surface[:, 1] <= hi[1])
                    )
            usable &= near
        out = []
        for i, t in enumerate(live):
            if not usable[i]:
                continue
            o = Observation(
                camera_id,
                t.id,
                t.last_ts,
                pixels[i],
                surface[i],
                t.p_under,
                t.p_vertical,
                t.motion,
                float(weight[i]),
                boxes[i],
                step[i],
            )
            for ts_h, box_h, *_ in reversed(t.history):  # about one second back on the smoothed track
                if t.last_ts - ts_h >= self.cfg.still_baseline_s:
                    o.pixel_prev = np.array([(box_h[0] + box_h[2]) / 2, (box_h[1] + box_h[3]) / 2])
                    o.dt_prev = t.last_ts - ts_h
                    break
            out.append(o)
        return out

    def _position(self, obs: Observation, depth: float) -> np.ndarray:
        """Where the person seen in ``obs`` is, if submerged at ``depth`` metres."""
        if obs.p_under < self.cfg.under_threshold or depth <= 0:
            return obs.xy_surface
        return obs.at_depth(depth)

    def _triangulate(self, obs: list[Observation]) -> tuple[np.ndarray, float] | None:
        """Depth where the refracted rays of all views agree best, and the point there."""
        mean_surface = np.mean([o.xy_surface for o in obs], axis=0)
        basin, at = self._in_basin(mean_surface)
        if basin is None or len(obs) < 2:
            return None
        max_depth = float(basin.depth_at(at[None])[0])
        depths = np.linspace(0.0, max_depth, 41)
        w = np.array([o.weight for o in obs])
        pts = np.stack([o.at_depth(depths) for o in obs], axis=1)  # (D, V, 2)
        centre = (w[None, :, None] * pts).sum(1) / w.sum()
        spread = (w[None, :] * ((pts - centre[:, None]) ** 2).sum(-1)).sum(1) / w.sum()
        k = int(np.argmin(spread))
        if np.sqrt(spread[k]) > self.cfg.disagree_m / 2:  # the rays never meet: not one body, no depth
            return None
        return centre[k], float(depths[k])

    # association ---------------------------------------------------------------------------------
    def update(self, observations: list[Observation], ts: float) -> list[Person]:
        for p in self.persons.values():
            if not p.lost:
                p.kf.predict(ts)
        by_key = {(o.camera_id, o.track_id): o for o in observations}
        claimed: set[tuple[str, int]] = set()
        assigned: dict[int, list[Observation]] = {pid: [] for pid in self.persons}

        # 1. keep consistent links
        for p in self.persons.values():
            for cam_id, tid in list(p.links.items()):
                o = by_key.get((cam_id, tid))
                if o is None:
                    del p.links[cam_id]
                    continue
                xy = self._position(o, p.depth_est)
                if np.linalg.norm(xy - p.xy) > self.cfg.keep_gate_m + p.kf.speed * 0.5:
                    del p.links[cam_id]
                    continue
                o.xy = xy
                assigned[p.id].append(o)
                claimed.add((cam_id, tid))

        # 1b. a person is one body: a view that stays far from the person's other views (a camera's
        #     tracker swapped two people who crossed) is let go, and not taken back for a while
        for p in self.persons.values():
            obs = assigned[p.id]
            odd = self._odd_view(p, obs) if len(obs) >= 2 else None
            if odd is None:
                self._apart_since.pop(p.id, None)
                continue
            if ts - self._apart_since.setdefault(p.id, ts) < self.cfg.disagree_s:
                continue
            del p.links[odd.camera_id]
            obs.remove(odd)
            claimed.discard((odd.camera_id, odd.track_id))
            self._let_go[(p.id, odd.camera_id, odd.track_id)] = ts
            del self._apart_since[p.id]

        # 2. link the rest to people without a link from that camera (Hungarian, active first)
        free = [o for o in observations if (o.camera_id, o.track_id) not in claimed]
        for o in free:
            o.xy = o.xy_surface  # a newcomer starts at the surface; its depth estimate grows if it stays under
        floors = {id(o): self._floor(o) for o in free if o.p_under >= self.cfg.under_threshold}
        candidates = [p for p in self.persons.values()]
        if free and candidates:
            cost = np.full((len(free), len(candidates)), 1e6)
            for i, o in enumerate(free):
                for j, p in enumerate(candidates):
                    if o.camera_id in p.links or (p.id, o.camera_id, o.track_id) in self._let_go:
                        continue
                    ref = p.xy
                    gate = self.cfg.link_gate_m
                    if p.lost:  # a diver can resurface metres away: the search radius grows while lost
                        swim = self.cfg.underwater_speed if p.p_under >= self.cfg.under_threshold else 0.3
                        gate = min(self.cfg.relink_gate_m + swim * (ts - p.last_seen), self.cfg.max_relink_m)
                    # compare at this person's working depth, or at the surface
                    d = min(
                        float(np.linalg.norm(self._position(o, p.depth_est) - ref)),
                        float(np.linalg.norm(o.xy_surface - ref)),
                    )
                    if id(o) in floors:  # a body under water: along this camera's ray, where p can be
                        d = min(d, o.ray_distance(ref, *self._depths(p, floors[id(o)], ts)))
                    if d <= gate:
                        cost[i, j] = d + (1.0 if p.lost else 0.0)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols, strict=True):
                if cost[i, j] >= 1e5:
                    continue
                o, p = free[i], candidates[j]
                o.xy = self._position(o, p.depth_est)
                p.links[o.camera_id] = o.track_id
                assigned.setdefault(p.id, []).append(o)
                claimed.add((o.camera_id, o.track_id))

        # 3. new people from what is left, merging simultaneous views of the same newcomer
        rest = sorted((o for o in free if (o.camera_id, o.track_id) not in claimed), key=lambda o: -o.weight)
        new: list[Person] = []
        for o in rest:
            match = None
            for p in new:
                if o.camera_id not in p.links and np.linalg.norm(o.xy - p.xy) <= self.cfg.link_gate_m:
                    match = p
                    break
            if match is None:
                match = Person(next(self._ids), PointKalman(o.xy, ts), ts, ts)
                self.persons[match.id] = match
                new.append(match)
                assigned[match.id] = []
            match.links[o.camera_id] = o.track_id
            assigned[match.id].append(o)

        # 4. fuse
        for pid, obs in assigned.items():
            p = self.persons[pid]
            if obs:
                self._fuse(p, obs, ts)
            elif not p.lost:
                p.lost = True
                p.kf.x[2:] = 0.0  # a person we stop seeing stays where we last saw them
        # 5. one human held by two people: co-located for a while, each seen by other cameras (the
        #    cameras' projections of an upright body disagreed when it entered); and a submerged person
        #    who vanishes while a newcomer appears on the same spot; merge both kinds
        self._merge_duplicates(ts)
        self._merge_splits(ts)
        self._merge_ghosts(ts)
        # 6. forget people lost too long (unless an alarm still needs them)
        for pid in [
            pid
            for pid, p in self.persons.items()
            if p.lost and ts - p.last_seen > self.cfg.keep_lost_s and pid not in self.pinned
        ]:
            del self.persons[pid]
        self._apart_since = {k: v for k, v in self._apart_since.items() if k in self.persons}
        self._let_go = {k: v for k, v in self._let_go.items() if k[0] in self.persons and ts - v < self.cfg.let_go_s}
        return list(self.persons.values())

    def _merge_duplicates(self, ts: float) -> None:
        active = [p for p in self.persons.values() if not p.lost and p.links]
        if len(active) < 2:
            self._close_since.clear()
            return
        xy = np.array([p.xy for p in active])
        d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
        seen = set()
        for i, j in zip(*np.nonzero(np.triu(d <= self.cfg.duplicate_radius_m, 1)), strict=True):
            a, b = active[i], active[j]
            if set(a.links) & set(b.links):  # both seen by one camera as two bodies: two people
                continue
            key = (min(a.id, b.id), max(a.id, b.id))
            seen.add(key)
            since = self._close_since.setdefault(key, ts)
            if ts - since < self.cfg.duplicate_hold_s:
                continue
            if a.id in self.pinned or b.id in self.pinned:
                keep, drop = (a, b) if a.id in self.pinned else (b, a)
            else:  # the one that has been on this spot longer goes on (a still body keeps its stillness
                # when a swimmer's identity runs into it), else the older one
                mid = (a.xy + b.xy) / 2
                here_a, here_b = self._here_since(a, mid), self._here_since(b, mid)
                first = here_a < here_b - 1.0 or (abs(here_a - here_b) <= 1.0 and a.first_ts <= b.first_ts)
                keep, drop = (a, b) if first else (b, a)
            if drop.id in self.pinned or drop.id not in self.persons or keep.id not in self.persons:
                continue
            keep.links.update(drop.links)
            keep.first_ts = min(keep.first_ts, drop.first_ts)
            del self.persons[drop.id]
        self._close_since = {k: v for k, v in self._close_since.items() if k in seen}

    def _merge_splits(self, ts: float) -> None:
        active = [q for q in self.persons.values() if not q.lost]
        for p in [p for p in self.persons.values() if p.lost and p.p_under >= self.cfg.under_threshold]:
            if p.id in self.pinned:
                continue
            for q in active:
                born_near_loss = (
                    p.last_seen - self.cfg.split_window_s <= q.first_ts <= p.last_seen + self.cfg.split_window_s
                )
                radius = self.cfg.split_radius_m + self.cfg.underwater_speed * max(0.0, q.first_ts - p.last_seen)
                if born_near_loss and q.first_ts > p.first_ts and self._reach(p, q.xy) <= radius:
                    q.first_ts = p.first_ts  # q continues p's story
                    del self.persons[p.id]
                    break

    def _merge_ghosts(self, ts: float) -> None:
        """A second identity that one camera made for a person the other cameras still see. When it
        is lost under water right beside that person, who was under water too, it was never a body
        of its own: it goes, instead of becoming a missing person. Only a single-camera identity
        younger than the visible person qualifies, so a swimmer seen from two angles before
        vanishing is never folded into someone else."""
        visible = [q for q in self.persons.values() if not q.lost and len(q.links) >= 2]
        for p in [p for p in self.persons.values() if p.lost and p.p_under >= self.cfg.under_threshold]:
            if p.id in self.pinned or len(p.observations) != 1 or ts - p.last_seen > 2.0:
                continue
            o = next(iter(p.observations.values()))  # its one camera saw it somewhere along a ray
            floor = self._floor(o)
            for q in visible:
                if q.first_ts >= p.first_ts or o.ray_distance(q.xy, 0.0, floor) > self.cfg.duplicate_radius_m:
                    continue
                under = [row[4] for row in q.history if row[0] >= p.last_seen - 3.0]
                if under and max(under) >= self.cfg.under_threshold:
                    del self.persons[p.id]
                    break

    def _here_since(self, p: Person, xy: np.ndarray) -> float:
        """Since when ``p`` has been within ``duplicate_radius_m`` of ``xy``, by its track."""
        since = p.last_seen
        for row in reversed(p.history):
            if np.hypot(row[1] - xy[0], row[2] - xy[1]) > self.cfg.duplicate_radius_m:
                break
            since = row[0]
        return since

    def _reach(self, p: Person, xy: np.ndarray) -> float:
        """How far ``xy`` is from where ``p`` can be: someone last seen under water by one camera, at
        an unknown depth, can be anywhere along that camera's ray, not only at the working depth."""
        d = float(np.linalg.norm(xy - p.xy))
        if p.depth is None and len(p.observations) == 1:
            o = next(iter(p.observations.values()))
            d = min(d, o.ray_distance(xy, 0.0, self._floor(o)))
        return d

    def _odd_view(self, p: Person, obs: list[Observation]) -> Observation | None:
        """The view that agrees least with the person's other views, if it is more than ``disagree_m``
        from them: a view of a body under water anywhere along its ray, at the depth where the views
        come closest, a view at the surface where it is. Of two views, the poorer view of the pool
        goes."""
        basin, at = self._in_basin(p.xy)
        depths = np.linspace(0.0, float(basin.depth_at(at[None])[0]) if basin else 0.0, 21)
        under = p.depth_est > 0.0
        pts = np.stack(
            [
                o.at_depth(depths)
                if under or o.p_under >= self.cfg.under_threshold
                else np.broadcast_to(o.xy, (len(depths), 2))
                for o in obs
            ],
            axis=1,
        )  # (D, V, 2)
        w = np.array([o.weight for o in obs])
        gaps = np.empty(len(obs))
        for i in range(len(obs)):
            rest = np.arange(len(obs)) != i
            centre = (w[rest][None, :, None] * pts[:, rest]).sum(1) / w[rest].sum()
            gaps[i] = np.min(np.linalg.norm(pts[:, i] - centre, axis=1))
        i = int(np.argmax(gaps - 1e-6 * w))
        return obs[i] if gaps[i] > self.cfg.disagree_m else None

    def _floor(self, o: Observation) -> float:
        """How deep a body seen in ``o`` can be: the water depth where its ray meets the surface."""
        basin, at = self._in_basin(o.xy_surface)
        return max(float(basin.depth_at(at[None])[0]) - 0.1, 0.0) if basin else 0.0

    def _depths(self, p: Person, floor: float, ts: float) -> tuple[float, float]:
        """How deep ``p`` can be by now: anywhere down to ``floor`` while under water at an unknown
        depth, near the triangulated depth, or only just under for someone seen at the surface. The
        band widens at the descent rate while the person is unseen."""
        grow = self.cfg.descent_mps * max(ts - p.last_seen, 0.0)
        if p.depth is None:
            return 0.0, floor
        if p.depth > 0.0:
            m = self.cfg.depth_margin_m + grow
            return max(p.depth - m, 0.0), min(p.depth + m, floor)
        return 0.0, min(self.cfg.surface_band_m + grow, floor)

    def _fuse(self, p: Person, obs: list[Observation], ts: float) -> None:
        w = np.array([o.weight for o in obs])
        w = w / w.sum()
        p_under = float((w * [o.p_under for o in obs]).sum())
        tri = self._triangulate(obs) if p_under >= self.cfg.under_threshold and len(obs) >= 2 else None
        dt_est = max(ts - p.last_seen, 0.0) if p.updates else 0.0
        if tri is not None:
            xy, p.depth = tri
            p.depth_est = p.depth
        else:
            p.depth = None if p_under >= self.cfg.under_threshold else 0.0
            xy = (w[:, None] * np.array([o.xy for o in obs])).sum(0)
            if p_under >= self.cfg.under_threshold:
                basin, at = self._in_basin(xy)
                floor = max(float(basin.depth_at(at[None])[0]) - 0.25, 0.0) if basin else 0.0
                p.depth_est = min(p.depth_est + self.cfg.descent_mps * dt_est, floor)
            else:
                p.depth_est = 0.0
        if p.lost:  # re-acquired: restart the filter where the person is now
            p.kf = PointKalman(xy, ts)
            p.lost = False
        else:
            p.kf.update(xy, sigma=0.25 if len(obs) > 1 else 0.4)
        dt = max(ts - p.last_seen, 1e-3)
        a = 1.0 if p.updates == 0 else 1.0 - np.exp(-dt / self.cfg.smoothing_tau_s)
        p.p_under += a * (p_under - p.p_under)
        p.p_vertical += a * (float((w * [o.p_vertical for o in obs]).sum()) - p.p_vertical)
        p.motion += a * (float((w * [o.motion for o in obs]).sum()) - p.motion)
        speeds = []
        for o in obs:
            if o.pixel_prev is None or o.dt_prev <= 0:
                continue
            cam = self.cameras[o.camera_id]
            d = p.depth_est if o.p_under >= self.cfg.under_threshold else 0.0
            both = np.array([o.pixel, o.pixel_prev])
            pts = cam.to_depth(both, d) if d > 0 else cam.to_surface(both)
            speeds.append(float(np.linalg.norm(pts[0] - pts[1])) / o.dt_prev)
        if speeds:  # still if one view sees it still: a swimmer passing over a body joins its views
            v = float(min(speeds))
            p.still_speed = v if p.still_speed is None else p.still_speed + a * (v - p.still_speed)
        p.last_seen = ts
        p.updates += 1
        p.observations = {o.camera_id: o for o in obs}
        basin, at = self._in_basin(p.xy)
        p.basin_id = basin.id if basin else None
        p.bottom_depth = float(basin.depth_at(at[None])[0]) if basin else None
        if basin and basin.floor_unknown(at[None])[0]:
            p.bottom_depth = None  # a movable floor at an unknown height: any still body under water counts
        p.history.append((ts, *p.xy, p.kf.speed, p.p_under, p.motion, p.p_vertical, len(obs), p.depth))
