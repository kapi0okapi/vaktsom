"""A camera source backed by the simulator: oracle detections every tick, rendered frames on the side.

Detections come from the ground truth at the full tick rate. Rendering is decoupled: a background
thread renders the cameras round-robin, the ones someone is watching (or that see an alarm) more
often, so the detection tick never waits for pictures.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from vaktsom.geometry import Basin, Camera
from vaktsom.perception.types import Frame
from vaktsom.runtime.engine import TickResult
from vaktsom.sim.oracle import GlareSpot, NoiseModel, OracleDetector, body_points
from vaktsom.sim.render import SceneRenderer
from vaktsom.sim.scene import Scene


class SimSource:
    simulated = True  # the engine never skips simulated time when it falls behind

    def __init__(
        self,
        scene: Scene,
        cameras: dict[str, Camera],
        basins: dict[str, Basin],
        render: bool = True,
        noise: NoiseModel | None = None,
        glare: dict[str, list[GlareSpot]] | None = None,
        size: tuple[int, int] = (960, 540),
        lane_ropes=None,
        seed: int = 0,
        background: bool = True,
    ):
        self.scene, self.cameras = scene, cameras
        self.oracle = OracleDetector(cameras, noise, glare, seed)
        self.renderer = (
            SceneRenderer(basins, cameras, size, glare=glare, lane_ropes=lane_ropes, seed=seed) if render else None
        )
        self.last_ts: float | None = None
        self.states = []
        self.n = 0
        self.frames: dict[str, Frame] = {}
        self.focus: dict[str, float] = {}  # camera -> wall time until which it renders often
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if self.renderer is not None and background:
            self._thread = threading.Thread(target=self._render_loop, name="sim-render", daemon=True)
            self._thread.start()

    def interest(self, camera_id: str, seconds: float = 3.0) -> None:
        self.focus[camera_id] = time.time() + seconds

    def _render_loop(self) -> None:
        order = list(self.cameras)
        k = 0
        while not self._stop.is_set():
            with self._lock:
                states, t, ts = list(self.states), self.scene.t, self.last_ts
            if ts is None:
                time.sleep(0.05)
                continue
            now = time.time()
            hot = [c for c, until in self.focus.items() if until > now]
            # focused cameras every other slot, the rest round-robin in between
            cid = hot[k // 2 % len(hot)] if hot and k % 2 == 0 else order[k % len(order)]
            k += 1
            image = self.renderer.render(cid, t, states)
            with self._lock:
                self.frames[cid] = Frame(cid, ts, image, k)

    def stop(self) -> None:
        self._stop.set()

    def tick(self, ts: float) -> TickResult:
        dt = 0.1 if self.last_ts is None else ts - self.last_ts
        states = self.scene.step(dt)
        with self._lock:
            self.states, self.last_ts = states, ts
            rendered = dict(self.frames)
            self.frames.clear()  # each picture is handed to the engine once
        self.n += 1
        frames, dets = {}, {}
        pts = np.stack([body_points(s) for s in states]) if states else None
        for cid in self.cameras:
            frames[cid] = rendered.get(cid) or Frame(cid, ts, None, self.n)
            frames[cid].truth = states
            dets[cid] = self.oracle.detect(cid, states, pts)
        if self.renderer is not None and self._thread is None:  # synchronous rendering (offline videos)
            for cid in self.cameras:
                frames[cid] = Frame(cid, ts, self.renderer.render(cid, self.scene.t, states), self.n, truth=states)
        return TickResult(ts, frames, dets)
