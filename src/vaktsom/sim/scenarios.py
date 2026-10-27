"""Named scenarios: the demo, the ISO 20380 tests, and long normal-use runs for false alarms."""

from __future__ import annotations

import numpy as np

from vaktsom.geometry import Basin
from vaktsom.sim.oracle import GlareSpot, NoiseModel
from vaktsom.sim.scene import (
    BreathHolder,
    Drowner,
    Floater,
    LapSwimmer,
    Mannequin,
    Player,
    Scene,
    Stander,
)


def _ids():
    n = 0
    while True:
        n += 1
        yield n


def demo(basins: dict[str, Basin], seed: int = 1) -> Scene:
    """A busy afternoon: lap swimmers, leisure bathers, one drowning in the deep end at ~30 s."""
    rng = np.random.default_rng(seed)
    ids = _ids()
    scene = Scene(basins)
    b = basins
    if "idrett" in b:
        for k in range(10):
            scene.bathers.append(LapSwimmer(next(ids), b["idrett"], rng, lane_y=1.25 + 2.5 * k))
        scene.bathers.append(Drowner(next(ids), b["idrett"], rng, start=25.0, idr_s=20.0))
        scene.bathers[-1].xy = np.array([38.0, 13.0])
        scene.bathers.append(Stander(next(ids), b["idrett"], rng, at=(6.0, 20.0)))
    if "stup" in b:
        for _ in range(3):
            scene.bathers.append(Player(next(ids), b["stup"], rng, dive_rate=1 / 25))
    if "bolge" in b:
        for _ in range(14):
            scene.bathers.append(Player(next(ids), b["bolge"], rng))
        scene.bathers.append(Floater(next(ids), b["bolge"], rng))
    if "barn" in b:
        for _ in range(6):
            scene.bathers.append(Player(next(ids), b["barn"], rng, length=float(rng.uniform(0.9, 1.3))))
    return scene


def demo_story(basins: dict[str, Basin], seed: int = 7) -> Scene:
    """Two minutes for the demo recording: normal bathing in every basin, and one drowning in lane 5
    of the sports pool (struggle from 15 s, under water at about 30 s, still on the bottom after)."""
    rng = np.random.default_rng(seed)
    ids = _ids()
    scene = Scene(basins)
    if "idrett" in basins:
        for k in (0, 1, 2, 3, 6, 7, 8, 9):
            scene.bathers.append(LapSwimmer(next(ids), basins["idrett"], rng, lane_y=1.25 + 2.5 * k))
        d = Drowner(next(ids), basins["idrett"], rng, start=15.0, idr_s=14.0)
        d.xy = np.array([30.0, 11.2])
        d.speed = 0.15
        scene.bathers.append(d)
        scene.bathers.append(Stander(next(ids), basins["idrett"], rng, at=(6.0, 20.0)))
    if "stup" in basins:
        for _ in range(2):
            scene.bathers.append(Player(next(ids), basins["stup"], rng, dive_rate=1 / 300))
    if "bolge" in basins:
        for _ in range(10):
            scene.bathers.append(Player(next(ids), basins["bolge"], rng, dive_rate=1 / 300))
        scene.bathers.append(Floater(next(ids), basins["bolge"], rng))
    if "barn" in basins:
        for _ in range(5):
            scene.bathers.append(
                Player(next(ids), basins["barn"], rng, length=float(rng.uniform(0.9, 1.3)), dive_rate=0.0)
            )
    return scene


def demo_glare() -> dict[str, list[GlareSpot]]:
    """Low winter sun through the west windows, seen on the water by two cameras."""
    return {"k04": [GlareSpot((14.0, 8.0), (3.5, 2.0), 0.9)], "k09": [GlareSpot((6.0, 38.0), (3.0, 2.0), 0.8)]}


def iso_detection(
    basin: Basin, grid: int = 4, dwell_s: float = 25.0, gap_s: float = 15.0, crowd: int = 6, seed: int = 0
) -> tuple[Scene, list[Mannequin]]:
    """ISO 20380 5.3 style: a mannequin on the bottom at positions spread over the whole basin, one
    after another, while other bathers keep swimming. Returns the scene and the mannequins."""
    rng = np.random.default_rng(seed)
    ids = _ids()
    scene = Scene({basin.id: basin})
    lo, hi = basin.outline.min(0), basin.outline.max(0)
    xs = np.linspace(lo[0], hi[0], grid + 2)[1:-1]
    ys = np.linspace(lo[1], hi[1], grid + 2)[1:-1]
    t = 5.0
    dummies = []
    for x in xs:
        for y in ys:
            m = Mannequin(next(ids), basin, rng, at=(float(x), float(y)), start=t, end=t + dwell_s)
            scene.bathers.append(m)
            dummies.append(m)
            t += dwell_s + gap_s
    for _ in range(crowd):
        scene.bathers.append(Player(next(ids), basin, rng, dive_rate=1 / 60))
    return scene, dummies


def iso_non_detection(basin: Basin, seed: int = 0) -> Scene:
    """ISO 20380 5.2: a swimmer standing still in each quadrant and the centre of the shallow area."""
    rng = np.random.default_rng(seed)
    lo, hi = basin.outline.min(0), basin.outline.max(0)
    c = (lo + hi) / 2
    q = (hi - lo) / 4
    spots = [c, c - q, c + q, c + [q[0], -q[1]], c + [-q[0], q[1]]]
    scene = Scene({basin.id: basin})
    for k, s in enumerate(spots):
        st = Stander(k + 1, basin, rng, at=(float(s[0]), float(s[1])))
        scene.bathers.append(st)
    return scene


def normal_use(
    basins: dict[str, Basin],
    bathers_per_100m2: float = 4.0,
    breath_holds: int = 0,
    seed: int = 0,
    rich: bool = False,
    duration_s: float = 3600.0,
) -> Scene:
    """Hours of ordinary bathing with no incident: every alarm here is a false alarm.

    ``rich`` adds what makes the job harder: people floating still on their backs, people standing
    still where they can stand, and a short breath-hold game (3 to 7 s on the bottom) every ten
    minutes, spread over ``duration_s``."""
    rng = np.random.default_rng(seed)
    ids = _ids()
    scene = Scene(basins)
    for b in basins.values():
        n = max(int(b.area / 100 * bathers_per_100m2), 1)
        lap = b.area >= 600
        for k in range(n):
            if lap and k % 2 == 0:
                lo, hi = b.outline.min(0), b.outline.max(0)
                scene.bathers.append(LapSwimmer(next(ids), b, rng, lane_y=float(rng.uniform(lo[1] + 1, hi[1] - 1))))
            else:
                scene.bathers.append(Player(next(ids), b, rng))
        for _ in range(breath_holds):
            scene.bathers.append(
                BreathHolder(next(ids), b, rng, start=float(rng.uniform(30, 600)), hold_s=float(rng.uniform(4, 9)))
            )
        if not rich:
            continue
        for _ in range(max(n // 8, 1)):
            scene.bathers.append(Floater(next(ids), b, rng))
        for _ in range(max(n // 8, 1)):
            s = Stander(next(ids), b, rng)
            for _try in range(50):  # only where one can stand
                if b.depth_at(s.xy[None])[0] <= 1.3:
                    scene.bathers.append(s)
                    break
                s.xy = s._random_point()
        for _ in range(int(duration_s / 600)):
            start = float(rng.uniform(10, max(duration_s - 30, 20)))
            scene.bathers.append(BreathHolder(next(ids), b, rng, start=start, hold_s=float(rng.uniform(3, 7))))
    return scene


def drowning_trials(basin: Basin, n: int, passive_share: float = 0.3, crowd: int = 8, seed: int = 0) -> list[Scene]:
    """Independent drowning incidents at random places, each in its own short scene."""
    rng = np.random.default_rng(seed)
    scenes = []
    for _k in range(n):
        ids = _ids()
        s = Scene({basin.id: basin})
        d = Drowner(next(ids), basin, rng, start=float(rng.uniform(8, 20)), passive=rng.random() < passive_share)
        s.bathers.append(d)
        for _ in range(crowd):
            s.bathers.append(Player(next(ids), basin, rng))
        scenes.append(s)
    return scenes


DEFAULT_NOISE = NoiseModel()
