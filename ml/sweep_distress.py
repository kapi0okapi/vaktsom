"""Operating point of the surface-distress warning. One simulation pass records every person's
distress score; the warning rule (score above a threshold for a hold time, short gaps forgiven) is
then replayed for a grid of thresholds and hold times. Each point gives the false warnings per
12-hour day in ordinary bathing (the rich mix) against the share of active drownings warned about
before the person went under.

    python ml/sweep_distress.py --config configs/demo.yaml --basin idrett --basin stup --basin bolge \\
        --hours 2 --trials 20 --workers 16 --out runs/sweep_distress.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vaktsom.analysis.rules import DistressRule
from vaktsom.config import load_config
from vaktsom.eval.run import _engine
from vaktsom.sim import scenarios
from vaktsom.sim.oracle import NoiseModel
from vaktsom.sim.scene import Drowner, Player, Scene

FLOOR = 0.6  # lowest threshold of the grid: scores below it are never kept
PROBS = [0.75, 0.8, 0.85, 0.9, 0.95]
HOLDS = [2.5, 3.5, 5.0, 7.0, 10.0]
GAP = DistressRule.gap_tolerance_s
REFRACTORY_S = 30.0  # the alarm manager folds repeated warnings for one person into one alarm


def _record(engine, seconds: float) -> dict[int, list[tuple[float, float, float, float]]]:
    """Per person: (ts, score, x, y) for every tick in which the person was visible and scored >= FLOOR."""
    period = 1.0 / engine.cfg.tick_hz
    t0 = engine.now
    series: dict[int, list] = {}
    for k in range(int(seconds / period)):
        ts = t0 + k * period
        engine.step(ts)
        for p in engine.world.persons.values():
            if not p.lost and p.distress >= FLOOR:
                series.setdefault(p.id, []).append((ts - t0, float(p.distress), float(p.xy[0]), float(p.xy[1])))
    return series


def normal(args) -> dict:
    cfg_path, basin_id, hours, seed = args
    cfg = load_config(cfg_path)
    basin = cfg.basin(basin_id).build()
    scene = scenarios.normal_use({basin_id: basin}, seed=seed, rich=True, duration_s=hours * 3600)
    engine, _ = _engine(cfg, scene, NoiseModel(), seed, {basin_id})
    return {"basin": basin_id, "hours": hours, "series": _record(engine, hours * 3600)}


def trial(args) -> dict:
    cfg_path, basin_id, seed = args
    cfg = load_config(cfg_path)
    basin = cfg.basin(basin_id).build()
    rng = np.random.default_rng(seed)
    scene = Scene({basin_id: basin})
    d = Drowner(1, basin, rng, start=float(rng.uniform(8, 15)), passive=False)
    scene.bathers.append(d)
    scene.bathers += [Player(k + 2, basin, rng) for k in range(8)]
    engine, _ = _engine(cfg, scene, NoiseModel(), seed, {basin_id})
    series = _record(engine, d.start + d.idr_s + 5.0)
    ev = d.events
    return {"basin": basin_id, "series": series, "xy": [float(v) for v in d.xy], "submerged": ev.submerged}


def fires(series: dict, prob: float, hold: float) -> list[tuple[float, float, float]]:
    """The rule replayed: (ts, x, y) of each warning, one per person per REFRACTORY_S."""
    out = []
    for rows in series.values():
        onset = last = last_fire = None
        for ts, score, x, y in rows:
            if score < prob:
                continue
            if last is None or ts - last > GAP:
                onset, fired = ts, False
            last = ts
            if not fired and ts - onset >= hold and (last_fire is None or ts - last_fire >= REFRACTORY_S):
                out.append((ts, x, y))
                fired, last_fire = True, ts
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--basin", action="append", required=True)
    ap.add_argument("--hours", type=float, default=2.0, help="ordinary bathing per basin, in 1-hour runs")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    chunks = max(int(round(args.hours)), 1)
    jobs_n = [(args.config, b, args.hours / chunks, 100 + s) for b in args.basin for s in range(chunks)]
    jobs_t = [(args.config, b, 500 + s) for b in args.basin for s in range(args.trials)]
    with ProcessPoolExecutor(args.workers) as ex:
        runs = list(ex.map(normal, jobs_n))
        trials = list(ex.map(trial, jobs_t))
    table = []
    for b in args.basin:
        hours = sum(r["hours"] for r in runs if r["basin"] == b)
        mine = [t for t in trials if t["basin"] == b and t["submerged"] is not None]
        for prob in PROBS:
            for hold in HOLDS:
                false = sum(len(fires(r["series"], prob, hold)) for r in runs if r["basin"] == b)
                warned = 0
                for t in mine:
                    early = [f for f in fires(t["series"], prob, hold) if f[0] < t["submerged"]]
                    warned += any(np.hypot(f[1] - t["xy"][0], f[2] - t["xy"][1]) <= 4.0 for f in early)
                table.append(
                    {
                        "basin": b,
                        "prob": prob,
                        "hold_s": hold,
                        "false_per_12h_day": round(12 * false / hours, 2),
                        "warned_before_under": round(warned / max(len(mine), 1), 3),
                        "trials": len(mine),
                    }
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2))
    for row in table:
        print(row)


if __name__ == "__main__":
    main()
