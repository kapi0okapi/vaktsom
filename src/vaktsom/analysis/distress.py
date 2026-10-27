"""Early-distress score per person, 0..1, from the last seconds of the fused track.

The instinctive drowning response (Pia, 1974; Vittone and Pia, 2006): upright in deep water, no
forward progress, arms pressing down at the sides, the head sinking and resurfacing (mouth at the
waterline), lasting 20 to 60 seconds before submersion. Treading water looks similar but the head
stays clear of the water. The score below measures that signature.
"""

from __future__ import annotations

import numpy as np

from vaktsom.tracking.world import Person

# history row: (ts, x, y, speed, p_under, motion, p_vertical, n_views, depth)
TS, X, Y, SPEED, UNDER, MOTION, VERTICAL = range(7)


class HeuristicDistress:
    window_s = 6.0

    def __init__(self, standing_depth: dict[str, float]):
        self.standing_depth = standing_depth

    def score(self, p: Person, now: float) -> float:
        if p.lost or p.basin_id is None or len(p.history) < 10:
            return 0.0
        if (p.bottom_depth or 0.0) <= self.standing_depth.get(p.basin_id, 1.35):
            return 0.0  # a person who can stand is not in an instinctive drowning response
        h = np.array([r[:7] for r in p.history if now - r[0] <= self.window_s])
        if len(h) < 10 or h[-1, TS] - h[0, TS] < self.window_s * 0.7:
            return 0.0
        span = h[-1, TS] - h[0, TS]
        progress = np.hypot(h[-1, X] - h[0, X], h[-1, Y] - h[0, Y]) / span
        upright = h[:, VERTICAL].mean()
        under = h[:, UNDER]
        # head going under and resurfacing again and again: count crossings of the waterline
        crossings = np.count_nonzero(np.diff((under > 0.3).astype(int)) != 0) / span
        longest_under = _longest_run(under > 0.5, h[:, TS])
        if longest_under > 3.0:  # one long submersion is a dive (or worse), not surface struggle
            return 0.0
        active = h[:, MOTION].mean()
        z = 4.0 * (upright - 0.65) + 5.0 * (crossings - 0.35) + 2.0 * (active - 0.45) - 10.0 * (progress - 0.15)
        return float(1.0 / (1.0 + np.exp(-z)))


def _longest_run(flags: np.ndarray, ts: np.ndarray) -> float:
    best, start = 0.0, None
    for f, t in zip(flags, ts, strict=True):
        if f and start is None:
            start = t
        elif not f and start is not None:
            best, start = max(best, t - start), None
    if start is not None:
        best = max(best, ts[-1] - start)
    return float(best)
