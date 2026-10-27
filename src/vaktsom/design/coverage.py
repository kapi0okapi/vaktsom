"""Camera layout for the technical study (ISO 20380 clause 4.2): the fewest cameras that see every
part of every basin from at least two directions, sharply enough and steeply enough.

Candidates are generated from mounting positions the building offers (along the walls of the hall
at a few heights, and under the ceiling), aimed at a grid of points over the basin, with a few
lens angles. A cell counts as seen by a camera when it is in the picture, the ground resolution is
at least ``min_px_per_m`` and the line of sight is no flatter than ``max_view_deg`` from vertical
(flat views see reflections and are refracted most). A greedy multi-cover then picks cameras until
every cell has ``views`` cameras or no candidate adds coverage.

    python -m vaktsom.design.coverage --config configs/demo.yaml --basin idrett --views 2
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass

import numpy as np

from vaktsom.geometry import Basin, Camera


@dataclass
class Candidate:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    hfov_deg: float
    seen: np.ndarray  # bool per cell


def cells_of(basin: Basin, cell: float) -> np.ndarray:
    lo, hi = basin.outline.min(0), basin.outline.max(0)
    xs = np.arange(lo[0] + cell / 2, hi[0], cell)
    ys = np.arange(lo[1] + cell / 2, hi[1], cell)
    grid = np.array([(x, y) for x in xs for y in ys])
    return grid[basin.contains(grid)]


def seen_by(cam: Camera, cells: np.ndarray, min_px_per_m: float, max_view_deg: float) -> np.ndarray:
    vis = cam.sees(cells)
    if not vis.any():
        return vis
    res = cam.ground_resolution(cells)
    ray = np.hstack([cells, np.zeros((len(cells), 1))]) - cam.center
    angle = np.degrees(np.arccos(np.clip(-ray[:, 2] / np.linalg.norm(ray, axis=1), -1, 1)))
    return vis & (res >= min_px_per_m) & (angle <= max_view_deg)


def mounts(basin: Basin, wall_offset: float, heights: list[float], spacing: float, ceiling: bool) -> list[tuple]:
    lo, hi = basin.outline.min(0) - wall_offset, basin.outline.max(0) + wall_offset
    pts = []
    for h in heights:
        for x in np.arange(lo[0], hi[0] + 1e-6, spacing):
            pts += [(x, lo[1], h), (x, hi[1], h)]
        for y in np.arange(lo[1] + spacing, hi[1] - 1e-6, spacing):
            pts += [(lo[0], y, h), (hi[0], y, h)]
    if ceiling:
        h = max(heights)
        blo, bhi = basin.outline.min(0), basin.outline.max(0)
        for x in np.arange(blo[0] + spacing / 2, bhi[0], spacing * 2):
            for y in np.arange(blo[1] + spacing / 2, bhi[1], spacing * 2):
                pts.append((x, y, h))
    return pts


def design(
    basin: Basin,
    size: tuple[int, int] = (2688, 1512),
    views: int = 2,
    cell: float = 0.5,
    min_px_per_m: float = 40.0,
    max_view_deg: float = 65.0,
    heights: tuple = (6.0, 9.0, 12.0),
    wall_offset: float = 2.0,
    spacing: float = 4.0,
    ceiling: bool = False,
    fovs: tuple = (60.0, 80.0, 100.0),
    target_spacing: float = 5.0,
    min_separation: float = 4.0,
) -> dict:
    cells = cells_of(basin, cell)
    lo, hi = basin.outline.min(0), basin.outline.max(0)
    targets = [
        (x, y)
        for x in np.arange(lo[0] + target_spacing / 2, hi[0], target_spacing)
        for y in np.arange(lo[1] + target_spacing / 2, hi[1], target_spacing)
    ]
    targets = [t for t in targets if basin.contains(np.array([t]))[0]] or [tuple((lo + hi) / 2)]
    cands: list[Candidate] = []
    for pos, tgt, fov in itertools.product(mounts(basin, wall_offset, list(heights), spacing, ceiling), targets, fovs):
        if np.hypot(pos[0] - tgt[0], pos[1] - tgt[1]) < 0.5 and pos[2] < 8:
            continue
        cam = Camera.looking_at(np.array(pos), np.array([tgt[0], tgt[1], 0.0]), fov, size)
        s = seen_by(cam, cells, min_px_per_m, max_view_deg)
        if s.sum() >= 4:
            cands.append(Candidate(pos, (tgt[0], tgt[1], 0.0), fov, s))
    need = np.full(len(cells), views)
    chosen: list[Candidate] = []
    while (need > 0).any() and cands:
        gains = np.array([np.minimum(c.seen, need > 0).sum() for c in cands])
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        c = cands.pop(best)
        chosen.append(c)
        need = np.maximum(need - c.seen.astype(int), 0)
        # two cameras on one bracket are not two views: the next one must hang elsewhere
        cands = [k for k in cands if np.linalg.norm(np.subtract(k.position, c.position)) >= min_separation]
    count = sum(c.seen.astype(int) for c in chosen) if chosen else np.zeros(len(cells), int)
    return {
        "basin": basin.id,
        "area_m2": round(basin.area, 1),
        "cells": int(len(cells)),
        "cameras": len(chosen),
        "covered_1": float((count >= 1).mean()),
        "covered_2": float((count >= 2).mean()),
        "layout": [
            {
                "position": [round(float(v), 2) for v in c.position],
                "target": [round(float(v), 2) for v in c.target],
                "hfov_deg": c.hfov_deg,
            }
            for c in chosen
        ],
        "constraints": {
            "min_px_per_m": min_px_per_m,
            "max_view_deg": max_view_deg,
            "image": list(size),
            "views": views,
        },
    }


def main(argv: list[str] | None = None) -> None:
    from vaktsom.config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--basin", action="append")
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--min-px", type=float, default=40.0)
    ap.add_argument("--max-view", type=float, default=65.0)
    ap.add_argument("--ceiling", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    results = []
    for b in cfg.basins:
        if args.basin and b.id not in args.basin:
            continue
        r = design(
            b.build(), views=args.views, min_px_per_m=args.min_px, max_view_deg=args.max_view, ceiling=args.ceiling
        )
        results.append(r)
        print(
            f"{b.name:22s} {r['area_m2']:7.1f} m2  {r['cameras']:2d} cameras  >=1 view {r['covered_1']:.1%}  "
            f">=2 views {r['covered_2']:.1%}"
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
