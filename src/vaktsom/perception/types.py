"""What perception hands to tracking: one detected person in one camera frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    box: np.ndarray  # (4,) x0, y0, x1, y1 in pixels
    score: float
    p_under: float = 0.0  # probability the whole person, head included, is below the surface
    p_vertical: float | None = None  # upright body (standing, treading, drowning response); None = unknown
    motion: float | None = None  # limb motion relative to the water around the body, 0..1+
    keypoints: np.ndarray | None = None  # (17, 3) x, y, visibility
    extra: dict = field(default_factory=dict)

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    @property
    def size(self) -> float:
        return float(np.sqrt(max(self.box[2] - self.box[0], 1) * max(self.box[3] - self.box[1], 1)))


@dataclass
class Frame:
    camera_id: str
    ts: float  # seconds (wall clock of capture)
    image: np.ndarray | None  # BGR uint8, None for metadata-only sources
    index: int = 0
    truth: list | None = None  # ground truth from the simulator, never used by the runtime logic
