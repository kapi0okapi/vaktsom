"""System-level evaluation in simulation, following ISO 20380's test methods where they exist.

- ``iso``: detection test (a mannequin on the bottom at a grid of positions over each basin, other
  bathers swimming around it): detection rate, alarm time against the fixed set-off time (the
  standard allows 2 s), position error (the standard allows 2 m). Non-detection test: a person
  standing still in the four quadrants and the centre of every area <= 1.5 m deep for 40 s.
- ``false-alarms``: hours of ordinary bathing without incidents; every alarm is a false alarm.
  Lap swimmers and leisure bathers with duck dives, plus people floating still, people standing
  still in shallow water and a short breath-hold game every ten minutes. Reported per basin and
  opening day (12 h), the unit of ISO 20380 4.3.4 (at most 5).
- ``drowning``: full drowning sequences (active struggle and sudden passive submersion) among
  other bathers: share detected, time from reaching the bottom to the alarm, and time from the
  start of the struggle to the first pre-alarm.

Perception is the oracle with the noise model in ``vaktsom.sim.oracle``; the numbers therefore
measure tracking, fusion, rules and alarm handling under stated detector error rates, not the
detector itself (which is measured separately on images).
"""

from __future__ import annotations

import json
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vaktsom.config import SiteConfig, load_config
from vaktsom.runtime.engine import Engine
from vaktsom.sim import scenarios
from vaktsom.sim.oracle import NoiseModel
from vaktsom.sim.scene import Drowner, Scene
from vaktsom.sim.source import SimSource

T0 = 1_700_000_000.0


def _engine(cfg: SiteConfig, scene: Scene, noise: NoiseModel | None, seed: int, basins_only: set[str] | None = None):
    cfg = cfg.model_copy(deep=True)
    cfg.data_dir = Path(tempfile.mkdtemp(prefix="vaktsom-eval-"))
    if basins_only:
        cfg.cameras = [c for c in cfg.cameras if set(c.basins) & basins_only]
    basins = {b.id: b.build() for b in cfg.basins}
    cams = {c.id: c.build() for c in cfg.cameras}
    src = SimSource(scene, cams, basins, render=False, noise=noise, seed=seed, background=False)
    return Engine(cfg, src), src


def _run(engine: Engine, seconds: float, on_tick=None, respond_s: float | None = 20.0) -> None:
    """Advance the engine; lifeguards close each alarm ``respond_s`` after it was raised."""
    period = 1.0 / engine.cfg.tick_hz
    n = int(seconds / period)
    for k in range(n):
        ts = T0 + k * period
        engine.step(ts)
        if respond_s is not None:
            for a in engine.alarms.open_alarms():
                if ts - a.raised_at >= respond_s:
                    engine.alarms.resolve(a.id, "eval", "test")
        if on_tick is not None:
            on_tick(engine, ts)


def _alarms(engine: Engine) -> list[dict]:
    return engine.store.alarms(0, 100000)


# ISO detection test -------------------------------------------------------------------------------------
@dataclass
class DetectionTrial:
    basin: str
    x: float
    y: float
    depth: float
    detected: bool
    alarm_delay_s: float | None  # alarm time minus (bottom time + set-off time)
    position_error_m: float | None


def iso_detection(cfg_path: str, basin_id: str, grid: int, seed: int) -> list[dict]:
    cfg = load_config(cfg_path)
    basin_cfg = cfg.basin(basin_id)
    basin = basin_cfg.build()
    scene, dummies = scenarios.iso_detection(basin, grid=grid, seed=seed)
    engine, _ = _engine(cfg, scene, NoiseModel(), seed, {basin_id})
    duration = max(d.end for d in dummies) + 5
    _run(engine, duration)
    events = _body_alarm_events(engine, basin_id)
    trials = []
    for d in dummies:
        onset = T0 + d.start - 0.1  # scene time runs one tick ahead of the engine clock
        due = onset + basin_cfg.set_off_time_s
        at = d.at
        match = [
            e for e in events if onset - 1 <= e["ts"] <= T0 + d.end and np.hypot(e["x"] - at[0], e["y"] - at[1]) <= 4.0
        ]
        if match:
            e = min(match, key=lambda e: e["ts"])
            trials.append(
                DetectionTrial(
                    basin_id,
                    *map(float, at),
                    float(basin.depth_at(at[None])[0]),
                    True,
                    e["ts"] - due,
                    float(np.hypot(e["x"] - at[0], e["y"] - at[1])),
                )
            )
        else:
            trials.append(
                DetectionTrial(basin_id, *map(float, at), float(basin.depth_at(at[None])[0]), False, None, None)
            )
    return [asdict(t) for t in trials]


def _body_alarm_events(engine: Engine, basin_id: str) -> list[dict]:
    """Every moment an alarm became a bottom or missing alarm (raised as one, or upgraded), with its
    position at that moment."""
    out = []
    for a in _alarms(engine):
        if a["basin_id"] != basin_id:
            continue
        for h in engine.store.alarm_history(a["id"]):
            if h["event"] in ("raised", "upgraded") and h["detail"].get("kind") in ("bottom", "missing"):
                out.append(
                    {
                        "ts": h["ts"],
                        "x": h["detail"].get("x", a["x"]),
                        "y": h["detail"].get("y", a["y"]),
                        "alarm": a["id"],
                    }
                )
    return out


def iso_non_detection(cfg_path: str, basin_id: str, seed: int) -> dict:
    cfg = load_config(cfg_path)
    basin = cfg.basin(basin_id).build()
    scene = scenarios.iso_non_detection(basin, seed=seed)
    engine, _ = _engine(cfg, scene, NoiseModel(), seed, {basin_id})
    _run(engine, 45.0)
    alarms = [a for a in _alarms(engine) if a["severity"] == "alarm"]
    return {"basin": basin_id, "positions": 5, "alarms": len(alarms)}


# false alarms ----------------------------------------------------------------------------------------------
def false_alarms(cfg_path: str, basin_id: str, hours: float, seed: int) -> dict:
    cfg = load_config(cfg_path)
    basin = cfg.basin(basin_id).build()
    scene = scenarios.normal_use({basin_id: basin}, seed=seed, rich=True, duration_s=hours * 3600)
    engine, _ = _engine(cfg, scene, NoiseModel(), seed, {basin_id})
    _run(engine, hours * 3600)
    rows = _alarms(engine)
    return {
        "basin": basin_id,
        "hours": hours,
        "bathers": len(scene.bathers),
        "alarms": sum(a["kind"] in ("bottom", "missing") for a in rows),
        "pre_alarms": sum(a["kind"] in ("distress", "face_down") for a in rows),
        "kinds": [a["kind"] for a in rows],
    }


# drowning sequences ----------------------------------------------------------------------------------------
def drowning(cfg_path: str, basin_id: str, seed: int, passive: bool) -> dict:
    cfg = load_config(cfg_path)
    basin_cfg = cfg.basin(basin_id)
    basin = basin_cfg.build()
    rng = np.random.default_rng(seed)
    scene = Scene({basin_id: basin})
    d = Drowner(1, basin, rng, start=float(rng.uniform(8, 15)), passive=passive)
    scene.bathers.append(d)
    from vaktsom.sim.scene import Player

    for k in range(8):
        scene.bathers.append(Player(k + 2, basin, rng))
    engine, _ = _engine(cfg, scene, NoiseModel(), seed, {basin_id})
    _run(engine, d.start + (0 if passive else d.idr_s) + 60.0)
    rows = _alarms(engine)
    ev = d.events
    out = {"basin": basin_id, "passive": passive, "bottom_s": ev.bottom, "detected": False}
    if ev.bottom is None:
        return out
    hist = {a["id"]: engine.store.alarm_history(a["id"]) for a in rows}
    first_pre = None
    bottom_alarm = None
    for a in rows:
        if np.hypot(a["x"] - d.xy[0], a["y"] - d.xy[1]) > 4.0:
            continue
        for h in hist[a["id"]]:
            kind = h["detail"].get("kind")
            if h["event"] in ("raised", "upgraded") and kind in ("distress", "face_down"):
                first_pre = h["ts"] if first_pre is None else min(first_pre, h["ts"])
            if h["event"] in ("raised", "upgraded") and kind in ("bottom", "missing"):
                bottom_alarm = h["ts"] if bottom_alarm is None else min(bottom_alarm, h["ts"])
    out.update(
        detected=bottom_alarm is not None,
        bottom_to_alarm_s=None if bottom_alarm is None else bottom_alarm - (T0 + ev.bottom),
        set_off_s=basin_cfg.set_off_time_s,
        distress_to_pre_s=None if first_pre is None or ev.distress is None else first_pre - (T0 + ev.distress),
        pre_before_submersion=first_pre is not None and ev.submerged is not None and first_pre < T0 + ev.submerged,
    )
    return out


# runner ------------------------------------------------------------------------------------------------------
def run_suite(args) -> None:
    cfg = load_config(args.config)
    basins = args.basin or [b.id for b in cfg.basins]
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "config": str(args.config),
        "started": time.strftime("%Y-%m-%d %H:%M"),
        "noise": asdict(NoiseModel()),
    }
    with ProcessPoolExecutor(args.workers) as pool:
        if args.suite in ("iso", "all"):
            jobs = [pool.submit(iso_detection, str(args.config), b, 5, s) for b in basins for s in range(2)]
            trials = [t for j in jobs for t in j.result()]
            nd = [
                pool.submit(iso_non_detection, str(args.config), b, 0)
                for b in basins
                if cfg.basin(b).build().depth_at(cfg.basin(b).build().outline.mean(0)[None])[0] <= 1.5
            ]
            report["iso_detection"] = _summarise_iso(trials)
            report["iso_detection_trials"] = trials
            report["iso_non_detection"] = [j.result() for j in nd]
        if args.suite in ("false-alarms", "all"):
            per = max(args.hours / max(args.workers // len(basins), 1), 0.5)
            jobs = [
                pool.submit(false_alarms, str(args.config), b, per, s)
                for b in basins
                for s in range(max(int(round(args.hours / per)), 1))
            ]
            res = [j.result() for j in jobs]
            report["false_alarms"] = _summarise_false(res)
        if args.suite in ("drowning", "all"):
            jobs = [
                pool.submit(drowning, str(args.config), b, s, s % 3 == 0) for b in basins for s in range(args.trials)
            ]
            report["drowning"] = _summarise_drowning([j.result() for j in jobs])
    path = args.out / f"{args.suite}.json"
    path.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_trials")}, indent=2, default=float))
    print(f"wrote {path}")


def _summarise_iso(trials: list[dict]) -> dict:
    out = {}
    for b in sorted({t["basin"] for t in trials}):
        ts = [t for t in trials if t["basin"] == b]
        on_time = [t for t in ts if t["detected"] and abs(t["alarm_delay_s"]) <= 2.0 and t["position_error_m"] <= 2.0]
        delays = [t["alarm_delay_s"] for t in ts if t["detected"]]
        errs = [t["position_error_m"] for t in ts if t["detected"]]
        out[b] = {
            "trials": len(ts),
            "detected": sum(t["detected"] for t in ts),
            "iso_pass_rate": len(on_time) / len(ts),
            "delay_s_median": float(np.median(delays)) if delays else None,
            "delay_s_max": float(np.max(delays)) if delays else None,
            "position_error_m_median": float(np.median(errs)) if errs else None,
            "position_error_m_max": float(np.max(errs)) if errs else None,
        }
    return out


def _summarise_false(rows: list[dict]) -> dict:
    out = {}
    for b in sorted({r["basin"] for r in rows}):
        rs = [r for r in rows if r["basin"] == b]
        hours = sum(r["hours"] for r in rs)
        out[b] = {
            "hours": hours,
            "bathers": rs[0]["bathers"],
            "alarms": sum(r["alarms"] for r in rs),
            "pre_alarms": sum(r["pre_alarms"] for r in rs),
            "alarms_per_12h_day": 12 * sum(r["alarms"] for r in rs) / hours,
            "pre_alarms_per_12h_day": 12 * sum(r["pre_alarms"] for r in rs) / hours,
        }
    return out


def _summarise_drowning(rows: list[dict]) -> dict:
    out = {}
    for b in sorted({r["basin"] for r in rows}):
        for passive in (False, True):
            rs = [r for r in rows if r["basin"] == b and r["passive"] == passive and r["bottom_s"] is not None]
            if not rs:
                continue
            det = [r for r in rs if r["detected"]]
            lat = [r["bottom_to_alarm_s"] for r in det]
            pre = [r["distress_to_pre_s"] for r in rs if r.get("distress_to_pre_s") is not None]
            out[f"{b}/{'passive' if passive else 'active'}"] = {
                "trials": len(rs),
                "detected": len(det),
                "rate": len(det) / len(rs),
                "bottom_to_alarm_s_median": float(np.median(lat)) if lat else None,
                "bottom_to_alarm_s_p95": float(np.percentile(lat, 95)) if lat else None,
                "pre_alarm_rate": len(pre) / len(rs),
                "distress_to_pre_alarm_s_median": float(np.median(pre)) if pre else None,
                "pre_alarm_before_submersion": sum(r.get("pre_before_submersion", False) for r in rs) / len(rs),
            }
    return out
