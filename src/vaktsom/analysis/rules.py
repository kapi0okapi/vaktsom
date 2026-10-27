"""Alarm rules over fused people: when does a situation become a finding for the lifeguards?

- ``BottomRule`` (ISO 20380): fully submerged, stationary, on or near the bottom for the basin's
  fixed set-off time -> alarm. The timer starts at the onset of the condition, not when the rule
  first notices it, so the alarm sounds at the set-off time (the standard allows 2 s).
- ``MissingRule``: a person last seen submerged who no camera sees any more for the set-off time
  -> alarm at the last known position. Glare and occlusion cannot hide a sinking swimmer.
- ``DistressRule``: early signs at the surface (instinctive drowning response) -> pre-alarm.
- ``FaceDownRule``: motionless at the surface, body horizontal, for a long time -> pre-alarm.

Rules never fire inside an active non-detection zone (rescue drills, ISO 4.3.2) or in a basin that
is closed for detection. One finding per person and kind per episode (ISO 4.3.3: one alarm per
detection); a new episode needs the condition to clear first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from vaktsom.config import BasinConfig, RuleConfig
from vaktsom.geometry import points_in_polygon
from vaktsom.tracking.world import Person

Kind = Literal["bottom", "missing", "distress", "face_down"]
Severity = Literal["alarm", "pre_alarm", "system"]


@dataclass
class Finding:
    kind: Kind
    severity: Severity
    person_id: int
    basin_id: str
    xy: np.ndarray
    ts: float
    onset: float
    confidence: float
    detail: dict = field(default_factory=dict)


@dataclass
class NonDetectionZone:
    id: str
    basin_id: str
    polygon: np.ndarray  # pool metres
    until: float  # detection comes back by itself at this time (ISO 4.3.2)
    created_by: str = ""
    reason: str = ""

    def covers(self, xy: np.ndarray, ts: float) -> bool:
        return ts < self.until and bool(points_in_polygon(xy[None], self.polygon)[0])


class _EpisodeRule:
    """Tracks for each person when a condition started and whether it already fired."""

    gap_tolerance_s = 1.0  # short dropouts (a missed frame, a wave) do not restart the timer

    def __init__(self) -> None:
        self.onset: dict[int, float] = {}
        self.last_true: dict[int, float] = {}
        self.fired: set[int] = set()

    def step(self, pid: int, condition: bool, ts: float, backdate: float = 0.0) -> float | None:
        """Returns the episode onset while the condition holds, None otherwise."""
        if condition:
            if pid not in self.onset or ts - self.last_true.get(pid, -1e9) > self.gap_tolerance_s:
                self.onset[pid] = ts - backdate
                self.fired.discard(pid)
            self.last_true[pid] = ts
            return self.onset[pid]
        if pid in self.onset and ts - self.last_true.get(pid, -1e9) > self.gap_tolerance_s:
            del self.onset[pid]
            self.fired.discard(pid)
        return None

    def forget(self, alive: set[int]) -> None:
        for d in (self.onset, self.last_true):
            for pid in [k for k in d if k not in alive]:
                del d[pid]
        self.fired &= alive


def still_since(p: Person, ts: float, radius: float = 0.35, max_back: float = 8.0, blip_s: float = 1.0) -> float:
    """When the person last moved: the earliest time from which every fused position stayed within
    ``radius`` of the current one, while under water. The stillness estimators confirm a still body
    a second or two late; ISO 20380 counts the set-off time from the moment it became still. A
    position that strays for under ``blip_s`` and comes back (one camera briefly following a wrong
    detection) is not the body moving."""
    if not p.history:
        return ts
    _, x, y = p.history[-1][:3]
    since = p.history[-1][0]
    strayed = None  # latest time of the run of strayed positions being stepped over
    for row in reversed(p.history):
        if ts - row[0] > max_back or row[4] < 0.5:
            break
        depth = row[8]
        if depth is not None and p.bottom_depth is not None and depth < p.bottom_depth - 0.6:
            break  # still sinking: a body is on the bottom only once the triangulated depth says so
        if np.hypot(row[1] - x, row[2] - y) > radius:
            strayed = row[0] if strayed is None else strayed
            if strayed - row[0] >= blip_s:
                break
            continue
        strayed = None
        since = row[0]
    return since


class BottomRule(_EpisodeRule):
    # at least this much backdating: stillness is judged on about a second of motion
    onset_lag_s = 0.8
    # a still body is a place, not an identity: a swimmer crossing over it can make the cameras swap
    # its identity with the swimmer's, so whoever lies still on the same spot within a few seconds of
    # another identity's still episode ending there takes over its time
    hand_over_s = 3.0
    hand_over_m = 0.5

    def __init__(self) -> None:
        super().__init__()
        self.spot: dict[int, np.ndarray] = {}  # where each running episode's body lies
        self.ended: list[tuple[int, np.ndarray, float, float]] = []  # (person, where, onset, last still)

    def step(self, pid: int, condition: bool, ts: float, backdate: float = 0.0) -> float | None:
        was, last = self.onset.get(pid), self.last_true.get(pid)
        onset = super().step(pid, condition, ts, backdate)
        if was is not None and self.onset.get(pid) != was and pid in self.spot:  # over, or restarted
            self.ended.append((pid, self.spot.pop(pid), was, last if last is not None else ts))
        return onset

    def forget(self, alive: set[int]) -> None:
        for pid in [k for k in self.onset if k not in alive and k in self.spot]:  # merged into another
            self.ended.append((pid, self.spot.pop(pid), self.onset[pid], self.last_true.get(pid, 0.0)))
        super().forget(alive)
        self.spot = {k: v for k, v in self.spot.items() if k in alive}

    def _handed_over(self, p: Person, ts: float, onset: float) -> float:
        self.ended = [e for e in self.ended if ts - e[3] <= self.hand_over_s]
        for pid, where, o, _ in self.ended:
            if pid != p.id and np.hypot(*(where - p.xy)) <= self.hand_over_m:
                onset = min(onset, o)
        return onset

    def evaluate(self, p: Person, basin: BasinConfig, rules: RuleConfig, ts: float) -> Finding | None:
        speed = p.still_speed if p.still_speed is not None else p.kf.speed
        still = speed <= rules.still_speed_mps and p.motion <= rules.still_motion_bottom
        condition = (not p.lost) and p.p_under >= rules.submerged_prob and still and p.on_bottom is not False
        new = condition and (p.id not in self.onset or ts - self.last_true.get(p.id, -1e9) > self.gap_tolerance_s)
        backdate = max(self.onset_lag_s, ts - still_since(p, ts)) if new else 0.0
        onset = self.step(p.id, condition, ts, backdate)
        if onset is None:
            return None
        onset = self.onset[p.id] = self._handed_over(p, ts, onset)
        self.spot[p.id] = p.xy.copy()
        if p.id in self.fired or ts - onset < basin.set_off_time_s:
            return None
        self.fired.add(p.id)
        return Finding(
            "bottom",
            "alarm",
            p.id,
            basin.id,
            p.xy,
            ts,
            onset,
            float(p.p_under),
            {"views": p.n_views, "depth": p.depth, "on_bottom": p.on_bottom},
        )


class MissingRule(_EpisodeRule):
    min_followed_s = 2.0

    def __init__(self) -> None:
        super().__init__()
        self.anchor: dict[int, float] = {}

    def evaluate(
        self, p: Person, basin: BasinConfig, rules: RuleConfig, ts: float, still_since: float | None = None
    ) -> Finding | None:
        """``still_since``: onset of a bottom episode that was running when the person vanished."""
        followed = p.last_seen - p.first_ts >= self.min_followed_s and p.updates >= 10
        condition = p.lost and p.p_under >= rules.submerged_prob and followed
        onset = self.step(p.id, condition, ts)
        if onset is None:
            self.anchor.pop(p.id, None)
            return None
        if p.id not in self.anchor:
            # the clock runs from the last sighting, or from the start of the stillness if the body
            # was already lying still when it disappeared from view
            self.anchor[p.id] = min(p.last_seen, still_since) if still_since is not None else p.last_seen
        if p.id in self.fired:
            return None
        wait = rules.missing_after_s or basin.set_off_time_s
        if ts - self.anchor[p.id] < wait:
            return None
        self.fired.add(p.id)
        return Finding(
            "missing",
            "alarm",
            p.id,
            basin.id,
            p.xy,
            ts,
            self.anchor[p.id],
            float(p.p_under),
            {"last_seen": p.last_seen},
        )

    def forget(self, alive: set[int]) -> None:
        super().forget(alive)
        for pid in [k for k in self.anchor if k not in alive]:
            del self.anchor[pid]


class DistressRule(_EpisodeRule):
    gap_tolerance_s = 2.0

    def evaluate(self, p: Person, basin: BasinConfig, rules: RuleConfig, ts: float) -> Finding | None:
        condition = (not p.lost) and p.distress >= rules.distress_prob
        onset = self.step(p.id, condition, ts)
        if onset is None or p.id in self.fired or ts - onset < rules.distress_hold_s:
            return None
        self.fired.add(p.id)
        return Finding("distress", "pre_alarm", p.id, basin.id, p.xy, ts, onset, float(p.distress))


class FaceDownRule(_EpisodeRule):
    def evaluate(self, p: Person, basin: BasinConfig, rules: RuleConfig, ts: float) -> Finding | None:
        deep_enough = (p.bottom_depth or 0.0) > 0.6
        condition = (
            not p.lost
            and deep_enough
            and 0.15 <= p.p_under < rules.submerged_prob
            and p.p_vertical < 0.3
            and p.kf.speed <= rules.still_speed_mps
            and p.motion <= rules.still_motion
        )
        onset = self.step(p.id, condition, ts)
        if onset is None or p.id in self.fired or ts - onset < rules.face_down_s:
            return None
        self.fired.add(p.id)
        return Finding("face_down", "pre_alarm", p.id, basin.id, p.xy, ts, onset, float(1 - p.motion))


class RuleEngine:
    def __init__(self, basins: list[BasinConfig], rules: RuleConfig):
        self.basins = {b.id: b for b in basins}
        self.rules = rules
        self.bottom = BottomRule()
        self.missing = MissingRule()
        self.distress = DistressRule()
        self.face_down = FaceDownRule()
        self.zones: dict[str, NonDetectionZone] = {}
        self.closed: set[str] = set()  # basins with detection switched off (closed to the public)
        self.turbulent: set[str] = set()  # basins in a mode that disturbs the water (wave machine on)
        self.face_down_basins: set[str] | None = None  # None = every basin

    def suppressed(self, p: Person, ts: float) -> bool:
        if p.basin_id in self.closed:
            return True
        return any(z.basin_id == p.basin_id and z.covers(p.xy, ts) for z in self.zones.values())

    def evaluate(self, persons: list[Person], ts: float) -> list[Finding]:
        if any(ts >= z.until for z in self.zones.values()):  # replaced, never changed in place (threads)
            self.zones = {k: z for k, z in self.zones.items() if ts < z.until}
        findings: list[Finding] = []
        alive = {p.id for p in persons}
        for p in persons:
            basin = self.basins.get(p.basin_id) if p.basin_id else None
            if basin is None or not basin.covered or self.suppressed(p, ts):
                for rule in (self.bottom, self.missing, self.distress, self.face_down):
                    rule.step(p.id, False, ts)
                continue
            still_since = self.bottom.onset.get(p.id)  # read before the bottom rule resets it
            candidates = [
                self.missing.evaluate(p, basin, self.rules, ts, still_since),
                self.bottom.evaluate(p, basin, self.rules, ts),
            ]
            if basin.id in self.turbulent:  # bathers bob in the waves: surface signs mean nothing
                self.distress.step(p.id, False, ts)
                self.face_down.step(p.id, False, ts)
            else:
                candidates.append(self.distress.evaluate(p, basin, self.rules, ts))
                if self.face_down_basins is None or basin.id in self.face_down_basins:
                    candidates.append(self.face_down.evaluate(p, basin, self.rules, ts))
            findings += [f for f in candidates if f is not None]
        for rule in (self.bottom, self.missing, self.distress, self.face_down):
            rule.forget(alive)
        return findings
