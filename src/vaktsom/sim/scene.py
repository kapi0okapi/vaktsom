"""Simulated bathers with ground truth, for tests, evaluation and the demo.

Each bather follows a behaviour (lap swimming, play with short dives, standing, floating, breath
holds, a full drowning sequence, an ISO 20380 test mannequin) and exposes its true body state at
every time step. The drowning sequence follows the literature: instinctive drowning response at the
surface (upright, no forward progress, head low, 20-60 s), slide under, sinking, then motionless on
the bottom. Ground-truth events give the ISO onset time: the moment the body is fully immersed,
on the bottom and still.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vaktsom.geometry import Basin


@dataclass
class BodyState:
    pid: int
    xy: np.ndarray
    z: float  # depth of the body centre below the surface, metres (>= 0)
    heading: float
    vertical: float  # 0 horizontal (swimming, floating), 1 upright (standing, treading, IDR)
    head_under: bool
    motion: float  # limb motion 0..1
    label: str
    length: float
    colour: tuple[int, int, int]
    face_down: bool = False


@dataclass
class Events:
    distress: float | None = None
    submerged: float | None = None
    bottom: float | None = None  # ISO onset: fully immersed, on the bottom, still


class Behaviour:
    """Base class. ``step`` advances the body by ``dt`` and returns its state."""

    def __init__(self, pid: int, basin: Basin, rng: np.random.Generator, length: float | None = None):
        self.pid, self.basin, self.rng = pid, basin, rng
        self.length = length if length is not None else float(rng.uniform(1.25, 1.9))
        self.colour = tuple(int(c) for c in rng.integers(30, 230, 3))
        self.xy = self._random_point()
        self.heading = float(rng.uniform(-np.pi, np.pi))
        self.events = Events()
        self.done = False

    def _random_point(self, margin: float = 0.8) -> np.ndarray:
        lo, hi = self.basin.outline.min(0) + margin, self.basin.outline.max(0) - margin
        for _ in range(200):
            p = self.rng.uniform(lo, hi)
            if self.basin.contains(p[None])[0]:
                return p
        return (lo + hi) / 2

    def depth_here(self) -> float:
        return float(self.basin.depth_at(self.xy[None])[0])

    def _keep_inside(self, new_xy: np.ndarray) -> np.ndarray:
        if self.basin.contains(new_xy[None])[0]:
            return new_xy
        self.heading += np.pi + self.rng.normal(0, 0.4)  # turn at the wall
        return self.xy

    def state(
        self, z: float, vertical: float, head_under: bool, motion: float, label: str, face_down: bool = False
    ) -> BodyState:
        return BodyState(
            self.pid,
            self.xy.copy(),
            z,
            self.heading,
            vertical,
            head_under,
            motion,
            label,
            self.length,
            self.colour,
            face_down,
        )

    def step(self, t: float, dt: float) -> BodyState:  # pragma: no cover - interface
        raise NotImplementedError


class LapSwimmer(Behaviour):
    def __init__(self, *a, lane_y: float | None = None, **k):
        super().__init__(*a, **k)
        lo, hi = self.basin.outline.min(0), self.basin.outline.max(0)
        self.lane_y = lane_y if lane_y is not None else float(self.rng.uniform(lo[1] + 1, hi[1] - 1))
        self.xy[1] = self.lane_y
        self.heading = 0.0 if self.rng.random() < 0.5 else np.pi
        self.speed = float(self.rng.uniform(0.6, 1.3))

    def step(self, t: float, dt: float) -> BodyState:
        d = np.array([np.cos(self.heading), np.sin(self.heading)])
        new = self.xy + d * self.speed * dt
        new[1] += (self.lane_y - new[1]) * 0.5 * dt
        lo, hi = self.basin.outline.min(0), self.basin.outline.max(0)
        if new[0] < lo[0] + 0.7 or new[0] > hi[0] - 0.7:
            self.heading = np.pi - self.heading  # tumble turn: a short dive at the wall
            self._turn_until = t + 2.0
        self.xy = self._keep_inside(new)
        under = t < getattr(self, "_turn_until", -1)
        return self.state(0.6 if under else 0.15, 0.0, under, 0.9, "swim")


class Player(Behaviour):
    """Leisure bathing: wandering, pauses, short duck dives (the classic false-alarm source)."""

    def __init__(self, *a, dive_rate: float = 1 / 40, **k):
        super().__init__(*a, **k)
        self.dive_rate = dive_rate
        self.speed = float(self.rng.uniform(0.2, 0.6))
        self.dive_until = -1.0
        self.pause_until = -1.0

    def step(self, t: float, dt: float) -> BodyState:
        if t > self.dive_until and self.rng.random() < self.dive_rate * dt:
            self.dive_until = t + float(self.rng.uniform(1.5, 6.0))
        if t > self.pause_until and self.rng.random() < dt / 20:
            self.pause_until = t + float(self.rng.uniform(2, 8))
        diving = t < self.dive_until
        moving = t >= self.pause_until or diving
        self.heading += self.rng.normal(0, 0.6) * np.sqrt(dt)
        if moving:
            d = np.array([np.cos(self.heading), np.sin(self.heading)])
            self.xy = self._keep_inside(self.xy + d * self.speed * dt * (1.3 if diving else 1.0))
        depth = self.depth_here()
        standing = depth <= 1.35 and not diving
        treading = not standing and not diving and not moving
        z = min(0.9, depth - 0.2) if diving else (depth * 0.5 if standing else 0.2)
        vertical = 1.0 if standing else (0.85 if treading else 0.35)
        label = "dive" if diving else ("stand" if standing else ("tread" if treading else "play"))
        return self.state(max(z, 0.1), vertical, diving, 0.8 if moving else 0.55, label)


class Stander(Behaviour):
    """Stands still in shallow water, head above the surface (ISO non-detection test)."""

    def __init__(self, *a, at: tuple[float, float] | None = None, **k):
        super().__init__(*a, **k)
        if at is not None:
            self.xy = np.asarray(at, float)

    def step(self, t: float, dt: float) -> BodyState:
        depth = self.depth_here()
        return self.state(depth * 0.5, 1.0, False, 0.12, "stand")


class Floater(Behaviour):
    """Relaxing on the back at the surface, almost motionless."""

    def step(self, t: float, dt: float) -> BodyState:
        self.xy = self._keep_inside(self.xy + self.rng.normal(0, 0.02, 2))
        return self.state(0.1, 0.0, False, 0.15, "float")


class BreathHolder(Behaviour):
    """Goes down, sits still on the bottom for ``hold_s`` - 1.5 seconds, then swims back up
    (intentional, but ISO alarms). Shallow water is reached in 1.5 s, deeper water at 1.2 m/s, and the
    way up takes 1 m/s, so the diving pool's five metres take seconds each way, not an instant."""

    def __init__(self, *a, start: float = 5.0, hold_s: float = 20.0, **k):
        super().__init__(*a, **k)
        self.start, self.hold_s = start, hold_s

    def step(self, t: float, dt: float) -> BodyState:
        floor = max(self.depth_here() - 0.3, 0.3)
        down = max(1.5, floor / 1.2)
        bottom_s = max(self.hold_s - 1.5, 0.0)
        u = t - self.start
        if 0.0 <= u < down + bottom_s + floor:
            if self.events.submerged is None:
                self.events.submerged = t
            if u >= down and self.events.bottom is None:
                self.events.bottom = t
            if u < down:
                z, motion = u / down * floor, 0.6
            elif u < down + bottom_s:
                z, motion = floor, 0.1
            else:
                z, motion = floor - (u - down - bottom_s), 0.6
            return self.state(max(z, 0.3), 0.6, True, motion, "breath_hold")
        return self.state(0.15, 0.5, False, 0.5, "play")


class Drowner(Behaviour):
    """Normal bathing, then instinctive drowning response, slide under, sink, lie on the bottom."""

    def __init__(self, *a, start: float = 20.0, idr_s: float | None = None, passive: bool = False, **k):
        super().__init__(*a, **k)
        self.start = start
        self.idr_s = idr_s if idr_s is not None else float(self.rng.uniform(15, 40))
        self.passive = passive  # sudden loss of consciousness: no surface struggle
        self.sink_rate = float(self.rng.uniform(0.15, 0.35))
        self.speed = float(self.rng.uniform(0.2, 0.5))
        self.z = 0.2

    def step(self, t: float, dt: float) -> BodyState:
        depth = self.depth_here()
        if t < self.start:  # bathing normally
            self.heading += self.rng.normal(0, 0.5) * np.sqrt(dt)
            d = np.array([np.cos(self.heading), np.sin(self.heading)])
            self.xy = self._keep_inside(self.xy + d * self.speed * dt)
            return self.state(0.2, 0.3, False, 0.8, "play")
        struggle_end = self.start + (0.0 if self.passive else self.idr_s)
        if t < struggle_end:  # instinctive drowning response: upright, bobbing, no progress
            if self.events.distress is None:
                self.events.distress = t
            bob = np.sin(2 * np.pi * 0.7 * t)
            self.xy = self._keep_inside(self.xy + self.rng.normal(0, 0.03, 2))
            return self.state(0.35, 0.95, bob < -0.6, 0.85, "distress")
        if self.passive and t < struggle_end + 4.0:  # face down at the surface
            if self.events.distress is None:
                self.events.distress = t
            return self.state(0.15, 0.0, True, 0.05, "face_down", face_down=True)
        bottom_z = max(depth - 0.25, 0.3)
        if self.z < bottom_z:  # sinking
            if self.events.submerged is None:
                self.events.submerged = t
            self.z = min(bottom_z, self.z + self.sink_rate * dt)
            self.xy = self._keep_inside(self.xy + self.rng.normal(0, 0.01, 2))
            return self.state(self.z, 0.5, True, 0.2, "sinking")
        if self.events.bottom is None:
            self.events.bottom = t
        return self.state(bottom_z, 0.1, True, 0.02, "bottom")


class Mannequin(Behaviour):
    """ISO 20380 detection test: a rescue mannequin placed on the bottom at ``at`` from ``start``.

    At ``end`` it is retrieved the way a lifeguard would: lifted to the surface and towed to the
    nearest wall, where it leaves the water.
    """

    def __init__(self, *a, at: tuple[float, float], start: float, end: float, **k):
        super().__init__(*a, length=1.7, **k)
        self.at = np.asarray(at, float)  # where it lies during the test (xy moves during retrieval)
        self.xy = self.at.copy()
        self.start, self.end = start, end
        self.colour = (40, 90, 200)
        self.z = None
        lo, hi = self.basin.outline.min(0), self.basin.outline.max(0)
        gaps = [
            (self.xy[0] - lo[0], (-1, 0)),
            (hi[0] - self.xy[0], (1, 0)),
            (self.xy[1] - lo[1], (0, -1)),
            (hi[1] - self.xy[1], (0, 1)),
        ]
        self.exit_dir = np.asarray(min(gaps)[1], float)

    def step(self, t: float, dt: float) -> BodyState | None:
        if t < self.start:
            return None
        bottom = max(self.depth_here() - 0.2, 0.2)
        if t < self.end:
            if self.events.bottom is None:
                self.events.submerged = self.events.bottom = t
            self.z = bottom
            return self.state(bottom, 0.0, True, 0.0, "mannequin")
        if self.z > 0.15:  # lifted to the surface
            self.z = max(0.15, self.z - 1.2 * dt)
            return self.state(self.z, 0.3, self.z > 0.3, 0.6, "rescue")
        new = self.xy + self.exit_dir * 1.0 * dt  # towed to the wall
        if not self.basin.contains(new[None])[0]:
            return None
        self.xy = new
        return self.state(0.15, 0.2, False, 0.7, "rescue")


@dataclass
class Scene:
    basins: dict[str, Basin]
    bathers: list[Behaviour] = field(default_factory=list)
    t: float = 0.0

    def step(self, dt: float) -> list[BodyState]:
        self.t += dt
        states = []
        for b in self.bathers:
            s = b.step(self.t, dt)
            if s is not None:
                states.append(s)
        return states
