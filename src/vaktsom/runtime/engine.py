"""The detection engine: frames -> people -> rules -> alarms, at a fixed tick rate.

One tick (10 Hz by default): take the newest frame of every camera, detect people (batched over
cameras), track them per camera, fuse them into pool coordinates, score early distress, evaluate the
alarm rules, advance escalation, check camera health and live coverage, and keep the alarm clip
rings filled. ``snapshot()`` is what the user interface shows.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vaktsom.alarms.manager import AlarmManager
from vaktsom.analysis.distress import HeuristicDistress
from vaktsom.analysis.rules import NonDetectionZone, RuleEngine
from vaktsom.config import SiteConfig
from vaktsom.geometry import Basin, Camera
from vaktsom.health.monitor import CameraHealth, CoverageMap
from vaktsom.perception.motion import MotionProbe
from vaktsom.perception.types import Detection, Frame
from vaktsom.storage.clips import ClipStore, FrameRing
from vaktsom.storage.db import EventStore
from vaktsom.tracking.camera_tracker import CameraTracker
from vaktsom.tracking.world import WorldTracker

log = logging.getLogger(__name__)


@dataclass
class TickResult:
    ts: float
    frames: dict[str, Frame]
    detections: dict[str, list[Detection]] = field(default_factory=dict)


def location_label(basin: Basin, name: str, xy: np.ndarray) -> str:
    """Position in words for the lifeguards: lane and metre mark in a lap pool, else a quadrant."""
    lo, hi = basin.outline.min(0), basin.outline.max(0)
    length, width = hi - lo
    x, y = float(xy[0]), float(xy[1])
    if length >= 20 and width >= 8:
        lanes = int(width // 2.5)
        lane = min(int((y - lo[1]) / (width / lanes)) + 1, lanes)
        return f"{name}, bane {lane}, {x - lo[0]:.0f} m"
    ns = "nord" if y > (lo[1] + hi[1]) / 2 else "sør"
    ew = "øst" if x > (lo[0] + hi[0]) / 2 else "vest"
    central = abs(x - (lo[0] + hi[0]) / 2) < length / 6 and abs(y - (lo[1] + hi[1]) / 2) < width / 6
    return f"{name}, midt" if central else f"{name}, {ns}{ew}"


class Engine:
    def __init__(
        self,
        cfg: SiteConfig,
        source,
        detector=None,
        dispatch=None,
        store: EventStore | None = None,
        clips: ClipStore | None = None,
        hub=None,
        min_fps: float = 5.0,
        attributes=None,
    ):
        """``source.tick(ts)`` returns a TickResult; detections in it are used as they are (simulation),
        otherwise ``detector(list of images) -> list of detection lists`` runs on the frames."""
        self.cfg = cfg
        self.source = source
        self.detector = detector
        self.attributes = attributes  # optional posture/submersion model on person crops
        self.hub = hub
        self.basins: dict[str, Basin] = {b.id: b.build() for b in cfg.basins}
        self.cameras: dict[str, Camera] = {c.id: c.build() for c in cfg.cameras}
        self.trackers = {c.id: CameraTracker() for c in cfg.cameras}
        self.motion = {c.id: MotionProbe() for c in cfg.cameras}
        self.world = WorldTracker(self.cameras, self.basins, camera_basins={c.id: c.basins for c in cfg.cameras})
        self.rules = RuleEngine(cfg.basins, cfg.rules)
        self.distress = HeuristicDistress({b.id: b.standing_depth_m for b in cfg.basins})
        masks = {c.id: (np.asarray(c.water_mask) if c.water_mask else None) for c in cfg.cameras}
        self.min_fps = min_fps
        self.health = {c.id: CameraHealth(c.id, masks[c.id], c.size, min_fps=min_fps) for c in cfg.cameras}
        self.coverage = CoverageMap(self.basins, self.cameras, {c.id: c.basins for c in cfg.cameras}, masks)
        self.store = store or EventStore(cfg.data_dir / "events.db")
        self.clips = clips or ClipStore(cfg.data_dir / "clips", cfg.retention.clip_days)
        self.rings = {c.id: FrameRing(seconds=cfg.retention.pre_s + 20) for c in cfg.cameras}
        self.privacy = {c.id: [np.asarray(m) for m in c.privacy_masks] for c in cfg.cameras}
        self.alarms = AlarmManager(
            self.store, cfg.escalation, dispatch or (lambda *a, **k: None), {b.id: b.name for b in cfg.basins}
        )
        self.now = time.time()
        self.alarms.clock = lambda: self.now
        self.latest: dict[str, Frame] = {}
        self.live_coverage: dict = {}
        self.modes: dict[str, set[str]] = {b.id: set() for b in cfg.basins}
        self._health_seen: dict[str, str] = {}
        self._last_housekeeping = float("-inf")
        self._restore_settings()
        self.persons: list = []
        self.stats = {"ticks": 0, "tick_ms": 0.0, "detections": 0}
        self._last_cov = 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.last_tick_wall = time.time()  # wall-clock end of the last completed tick, for the watchdog
        self._down_alarm = None

    # one tick -----------------------------------------------------------------------------------------
    def step(self, ts: float) -> None:
        t0 = time.perf_counter()
        self.now = ts
        res: TickResult = self.source.tick(ts)
        dets = res.detections
        if not dets and self.detector is not None:
            ids = [cid for cid, f in res.frames.items() if f.image is not None]
            out = self.detector([res.frames[c].image for c in ids]) if ids else []
            dets = dict(zip(ids, out, strict=True))
        observations = []
        for cid, frame in res.frames.items():
            if frame.image is not None:
                self.health[cid].frame(frame.image, frame.ts)
                self.rings[cid].push(frame.ts, frame.image, self.privacy[cid])
                self.latest[cid] = frame
            if res.detections:  # the simulator hands over detections for this very tick
                tracks = self.trackers[cid].update(dets.get(cid, []), ts)
            elif frame.image is not None:  # a new picture from this camera was analysed
                if self.attributes is not None:
                    self.attributes.annotate(frame.image, dets.get(cid, []))
                self.motion[cid].annotate(frame.ts, frame.image, dets.get(cid, []))
                tracks = self.trackers[cid].update(dets.get(cid, []), frame.ts)
            else:  # no new picture this tick: keep what the camera last saw
                tracks = [t for t in self.trackers[cid].tracks if t.confirmed]
            observations += self.world.observe(cid, tracks, ts)
        self.world.pinned = self.alarms.pinned_persons()
        persons = self.world.update(observations, ts)
        for p in persons:
            p.distress = self.distress.score(p, ts)
        findings = self.rules.evaluate(persons, ts)
        if findings:
            raised = self.alarms.handle(
                findings,
                cameras_for=lambda pid: sorted(self.world.persons[pid].links) if pid in self.world.persons else [],
                label_for=lambda bid, xy: location_label(self.basins[bid], self.basins[bid].name, xy),
            )
            for a in raised:
                cams = a.cameras or [c.id for c in self.cfg.cameras if a.basin_id in c.basins]
                a.clip_id = self.clips.start(
                    a.id, cams, a.onset, a.raised_at, self.cfg.retention.pre_s, self.cfg.retention.post_s, self.rings
                )
        self.alarms.tick(ts)
        self.clips.poll(ts, self.rings)
        if hasattr(self.source, "interest"):  # simulator: render the cameras that see an alarm often
            for a in self.alarms.open_alarms():
                for cid in a.cameras:
                    self.source.interest(cid, 2.0)
        for cid in self.health:  # a camera is down after three missed frame intervals (at least 3 s)
            self.health[cid].check_alive(ts, timeout=max(3.0, 3.0 / self.min_fps))
        if ts - self._last_cov >= 1.0:
            coverage = self.coverage.live({c: h.status for c, h in self.health.items()})
            for bid, modes in self.modes.items():  # waves and bubbles weaken detection in the whole basin
                c = coverage.get(bid)
                if modes and c is not None:
                    c.state = "down" if c.state == "down" else "degraded"
                    c.reasons = sorted(set(c.reasons) | {f"mode:{m}" for m in modes})
            self.live_coverage = coverage
            self._record_health(ts)
            self._last_cov = ts
        if ts - self._last_housekeeping >= 3600.0:
            self.housekeeping()
            self._last_housekeeping = ts
        with self._lock:
            self.persons = persons
        self.stats["ticks"] += 1
        self.stats["detections"] = sum(len(v) for v in dets.values())
        self.stats["tick_ms"] = 0.9 * self.stats["tick_ms"] + 0.1 * (time.perf_counter() - t0) * 1000
        self.last_tick_wall = time.time()
        if self.hub is not None:
            self.hub.publish("state", self.snapshot())

    def run(self, realtime: bool = True, duration: float | None = None, start: float | None = None) -> None:
        period = 1.0 / self.cfg.tick_hz
        ts = start if start is not None else time.time()
        end = None if duration is None else ts + duration
        self.store.log("system_start", detail={"site": self.cfg.site})
        failures = 0
        while not self._stop.is_set() and (end is None or ts < end):
            try:
                self.step(ts)
                failures = 0
            except Exception:  # one bad frame or bug must not stop detection; the watchdog warns staff
                failures += 1
                if failures == 1 or failures % 600 == 0:
                    log.exception("detection step failed (%d in a row)", failures)
                    self.store.log("engine_error", detail={"in_a_row": failures})
            ts += period
            if realtime:
                delay = ts - time.time()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -2.0 and not getattr(self.source, "simulated", False):
                    ts = time.time()  # live cameras: skip ahead rather than run on stale time
        self.store.log("system_stop")

    def stop(self) -> None:
        self._stop.set()

    def watchdog(self, stale_s: float = 5.0) -> None:
        """Own thread: if the detection loop stops (a crash, a hang), every output is told at once,
        and the warning closes itself when detection runs again (ISO 20380 4.3.3)."""
        while not self._stop.wait(1.0):
            self.check_alive(time.time(), stale_s)

    def check_alive(self, now: float, stale_s: float = 5.0) -> None:
        stale = now - self.last_tick_wall > stale_s
        if stale and self._down_alarm is None:
            first = self.cfg.basins[0].id
            label = "Deteksjonen har stoppet. Overvåk uten Vaktsom."
            self._down_alarm = self.alarms.raise_manual(
                "system", "system", first, self.basins[first].outline.mean(0), "system", {"label": label}
            )
            self.store.log("detection_stopped", detail={"last_tick": self.last_tick_wall})
        elif not stale and self._down_alarm is not None:
            self.alarms.resolve(self._down_alarm.id, "system", "other", "detection running again")
            self.store.log("detection_resumed")
            self._down_alarm = None
        if stale and self.hub is not None:  # the screens keep showing the warning
            self.hub.publish("state", self.snapshot())

    def housekeeping(self, now: float | None = None) -> None:
        """Hourly: clips past ``retention.clip_days`` are crypto-shredded and log rows
        past ``retention.log_days`` (at least 30, ISO 20380 4.4.1) are deleted. Wall-clock time, like
        the clip and log timestamps."""
        now = time.time() if now is None else now
        shredded = self.clips.enforce_retention(now)
        self.store.prune(self.cfg.retention.log_days, now)
        if shredded:
            self.store.log("clips_shredded", detail={"clips": shredded, "days": self.cfg.retention.clip_days})

    def _record_health(self, ts: float) -> None:
        """Every change in a camera's state or a basin's coverage goes to the health log, with the
        reasons (ISO 20380 4.3.3 and 4.4.1); the monthly report adds up the time with reduced coverage."""
        current = {f"camera:{c}": (h.status.state, h.status.reasons) for c, h in self.health.items()}
        current |= {f"basin:{b}": (c.state, c.reasons) for b, c in self.live_coverage.items()}
        for source, (state, reasons) in current.items():
            if state == "unknown" or self._health_seen.get(source, "ok") == state:
                continue
            self.store.health_event(source, state, {"reasons": list(reasons)}, ts=ts)
            self._health_seen[source] = state

    # control from the user interface ---------------------------------------------------------------
    def add_zone(self, basin_id: str, polygon, minutes: float, actor: str, reason: str = "") -> NonDetectionZone:
        now = time.time()
        z = NonDetectionZone(
            f"z{int(now * 1000) % 10**9}", basin_id, np.asarray(polygon, float), now + minutes * 60, actor, reason
        )
        # shared with the detection thread: replaced, never changed in place
        self.rules.zones = {**self.rules.zones, z.id: z}
        self.store.log("zone_created", actor, {"zone": z.id, "basin": basin_id, "minutes": minutes, "reason": reason})
        return z

    def remove_zone(self, zone_id: str, actor: str) -> None:
        self.rules.zones = {k: z for k, z in self.rules.zones.items() if k != zone_id}
        self.store.log("zone_removed", actor, {"zone": zone_id})

    def set_basin_open(self, basin_id: str, open_: bool, actor: str) -> None:
        self.rules.closed = self.rules.closed - {basin_id} if open_ else self.rules.closed | {basin_id}
        self.store.log("basin_opened" if open_ else "basin_closed", actor, {"basin": basin_id})

    def set_mode(self, basin_id: str, mode: str, on: bool, actor: str) -> None:
        """An operating mode that disturbs the water (``turbulent_when``), such as the wave machine."""
        modes = self.modes[basin_id] | {mode} if on else self.modes[basin_id] - {mode}
        self.modes = {**self.modes, basin_id: modes}
        self.rules.turbulent = {b for b, m in self.modes.items() if m}
        self._last_cov = 0.0  # show it at once
        self._save_settings()
        self.store.log("mode_on" if on else "mode_off", actor, {"basin": basin_id, "mode": mode})

    def set_floor(self, basin_id: str, depth: float, actor: str) -> None:
        """The movable floor's current height, as water depth over it in metres."""
        self.basins[basin_id].movable_floor_depth = depth
        self._save_settings()
        self.store.log("floor_set", actor, {"basin": basin_id, "depth": depth})

    def _settings_path(self) -> Path:
        return self.cfg.data_dir / "settings.json"

    def _save_settings(self) -> None:
        """Floor heights and modes survive a restart, so the system never wakes up believing the
        floor is lower than it is."""
        floors = {b.id: self.basins[b.id].movable_floor_depth for b in self.cfg.basins if b.movable_floor}
        data = {"floors": floors, "modes": {b: sorted(m) for b, m in self.modes.items() if m}}
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)

    def _restore_settings(self) -> None:
        try:
            data = json.loads(self._settings_path().read_text())
        except (FileNotFoundError, ValueError):
            return
        for bid, depth in data.get("floors", {}).items():
            if bid in self.basins and depth is not None:
                self.basins[bid].movable_floor_depth = float(depth)
        for bid, modes in data.get("modes", {}).items():
            if bid in self.modes:
                self.modes[bid] = set(modes) & set(self.cfg.basin(bid).turbulent_when)
        self.rules.turbulent = {b for b, m in self.modes.items() if m}

    # what the interface shows ------------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            persons = list(self.persons)
        now = time.time()
        return {
            "ts": now,
            "persons": [
                {
                    "id": p.id,
                    "x": round(float(p.xy[0]), 2),
                    "y": round(float(p.xy[1]), 2),
                    "basin": p.basin_id,
                    "under": round(p.p_under, 2),
                    "speed": round(p.kf.speed, 2),
                    "views": p.n_views,
                    "lost": p.lost,
                    "distress": round(p.distress, 2),
                    "depth": None if p.depth is None else round(p.depth, 1),
                }
                for p in persons
            ],
            "alarms": [a.public() for a in self.alarms.open_alarms()],
            "cameras": {c: h.status.public() for c, h in self.health.items()},
            "coverage": {b: c.public() for b, c in self.live_coverage.items()},
            "zones": [
                {
                    "id": z.id,
                    "basin": z.basin_id,
                    "polygon": z.polygon.tolist(),
                    "until": z.until,
                    "by": z.created_by,
                    "reason": z.reason,
                }
                for z in self.rules.zones.values()
            ],
            "closed": sorted(self.rules.closed),
            "modes": {b: sorted(m) for b, m in self.modes.items() if m},
            "floors": {b.id: self.basins[b.id].movable_floor_depth for b in self.cfg.basins if b.movable_floor},
            "stats": dict(self.stats),
        }
