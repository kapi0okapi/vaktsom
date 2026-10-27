"""Site configuration: basins, cameras, alarm outputs and rules, loaded from one YAML file.

The file is the result of the technical study (ISO 20380 clause 4.2): where the cameras are, which
areas they cover, the fixed alarm set-off time per basin and how alarms reach the lifeguards.
Validation enforces the parts of the standard that a configuration can get wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from vaktsom.geometry import Basin, Camera, DepthRegion

Point = tuple[float, float]


class DepthRegionConfig(BaseModel):
    polygon: list[Point]
    depth: float = Field(gt=0, le=10)
    slope_to: tuple[Point, Point, float] | None = None
    movable: bool = False


class BasinConfig(BaseModel):
    id: str
    name: str
    outline: list[Point] = Field(min_length=3)
    default_depth: float = Field(gt=0, le=10)
    regions: list[DepthRegionConfig] = []
    movable_floor: bool = False
    # the movable floor's setting at start-up (later ones come from the floor controller or the
    # dashboard and are kept across restarts); while it is unknown, bottom contact is not judged there
    floor_depth: float | None = Field(default=None, ge=0, le=10)
    # ISO 20380 4.3.1: <= 15 s, fixed by the supplier, shown in the interface, not editable by staff
    set_off_time_s: float = Field(default=10.0, gt=0, le=15)
    # shallow enough to stand: a still person here is normal (ISO non-detection test, <= 1.5 m)
    standing_depth_m: float = 1.35
    covered: bool = True
    # operating modes that disturb the water, e.g. ["waves"], ["bubbles"]: while one is on, the basin
    # shows reduced detection and surface warnings pause; the bottom and missing rules keep running
    turbulent_when: list[str] = []

    def build(self) -> Basin:
        return Basin(
            id=self.id,
            name=self.name,
            outline=np.asarray(self.outline, float),
            regions=[DepthRegion(np.asarray(r.polygon, float), r.depth, r.slope_to, r.movable) for r in self.regions],
            default_depth=self.default_depth,
            movable_floor_depth=self.floor_depth,
        )


class CameraConfig(BaseModel):
    id: str
    name: str = ""
    url: str  # rtsp://..., file path, or sim://<scenario>/<camera>
    basins: list[str]
    size: tuple[int, int] = (1920, 1080)
    fps: float = 10.0
    # calibration: either full pinhole (K, rvec/R, t) or a design pose (position, target, hfov)
    K: list[list[float]] | None = None
    R: list[list[float]] | None = None
    t: list[float] | None = None
    dist: list[float] = [0, 0, 0, 0, 0]
    position: tuple[float, float, float] | None = None
    target: tuple[float, float, float] | None = None
    hfov_deg: float | None = None
    water_mask: list[Point] | None = None  # image polygon of visible water (pixels)
    privacy_masks: list[list[Point]] = []  # image polygons blacked out before storage and display

    @model_validator(mode="after")
    def _calibrated(self) -> CameraConfig:
        full = self.K is not None and self.R is not None and self.t is not None
        design = self.position is not None and self.target is not None and self.hfov_deg is not None
        if not (full or design):
            raise ValueError(f"camera {self.id}: give K, R, t or position, target, hfov_deg")
        return self

    def build(self) -> Camera:
        if self.K is not None and self.R is not None and self.t is not None:
            return Camera(
                K=np.asarray(self.K, float),
                R=np.asarray(self.R, float),
                t=np.asarray(self.t, float),
                dist=np.asarray(self.dist, float),
                size=self.size,
            )
        return Camera.looking_at(np.asarray(self.position), np.asarray(self.target), self.hfov_deg, self.size)


OutputKind = Literal["panel", "wearable", "siren", "webhook", "modbus_coil", "log"]


class OutputConfig(BaseModel):
    id: str
    kind: OutputKind
    wired: bool  # ISO 20380 4.4.2: every alarm reaches staff in >= 2 ways, at least one wired
    url: str | None = None
    basins: list[str] | None = None  # None = all basins
    severities: list[Literal["alarm", "pre_alarm", "system"]] = ["alarm", "pre_alarm", "system"]
    options: dict = {}  # sirens: {when: escalated} sounds only for alarms nobody acknowledged in time

    def reaches_staff_at_once(self) -> bool:
        """Counts as one of the ISO 20380 4.4.2 alarm paths: a log is not, an escalation-only siren is late."""
        return self.kind != "log" and "alarm" in self.severities and self.options.get("when") != "escalated"


class EscalationConfig(BaseModel):
    ack_timeout_s: float = 10.0  # unacknowledged alarm -> every output, siren included
    pre_alarm_to_alarm_s: float = 15.0  # a pre-alarm that persists this long becomes an alarm
    repeat_s: float = 20.0


class RetentionConfig(BaseModel):
    clip_days: int = Field(default=7, ge=1, le=30)  # alarm clips (crypto-shredded after)
    log_days: int = Field(default=30, ge=30)  # ISO 20380 4.4.1: at least the last 30 days
    pre_s: float = 10.0
    post_s: float = 10.0


class RuleConfig(BaseModel):
    still_speed_mps: float = 0.12
    # ISO 20380 detects a stationary immersed mass: the bottom rule judges stillness by position and
    # only rejects clear swimming on the spot; waves make a lying body shimmer, which is not motion
    still_motion_bottom: float = 0.9
    still_motion: float = 0.35  # stricter, for a body at the surface (face down)
    submerged_prob: float = 0.6
    missing_after_s: float | None = None  # default: the basin's set-off time
    # chosen with ml/sweep_distress.py and the real detector's scores on rendered video: from 0.8 up no
    # false warnings in 18 h of the rich bathing mix; up to 0.8 every rendered active drowning is warned
    # about before going under (at 0.85 the real detector's scores fall short for some)
    distress_prob: float = 0.8
    distress_hold_s: float = 2.5
    face_down_s: float = 20.0


class SiteConfig(BaseModel):
    site: str
    timezone: str = "Europe/Oslo"
    language: Literal["nb", "en"] = "nb"
    basins: list[BasinConfig]
    cameras: list[CameraConfig]
    outputs: list[OutputConfig]
    escalation: EscalationConfig = EscalationConfig()
    retention: RetentionConfig = RetentionConfig()
    rules: RuleConfig = RuleConfig()
    tick_hz: float = 10.0
    data_dir: Path = Path("data")

    @field_validator("basins")
    @classmethod
    def _unique_basins(cls, v: list[BasinConfig]) -> list[BasinConfig]:
        ids = [b.id for b in v]
        if len(ids) != len(set(ids)):
            raise ValueError("basin ids must be unique")
        return v

    @model_validator(mode="after")
    def _references(self) -> SiteConfig:
        basins = {b.id for b in self.basins}
        for cam in self.cameras:
            unknown = set(cam.basins) - basins
            if unknown:
                raise ValueError(f"camera {cam.id}: unknown basins {sorted(unknown)}")
        for out in self.outputs:
            if out.basins and set(out.basins) - basins:
                raise ValueError(f"output {out.id}: unknown basins {sorted(set(out.basins) - basins)}")
        for b in self.basins:
            if not b.covered:
                continue
            paths = [o for o in self.outputs if o.reaches_staff_at_once() and (not o.basins or b.id in o.basins)]
            if len(paths) < 2 or not any(o.wired for o in paths):
                raise ValueError(f"basin {b.id}: alarms need at least two outputs, one of them wired (ISO 20380 4.4.2)")
        return self

    def basin(self, basin_id: str) -> BasinConfig:
        return next(b for b in self.basins if b.id == basin_id)


def load_config(path: str | Path) -> SiteConfig:
    path = Path(path)
    cfg = SiteConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if not cfg.data_dir.is_absolute():
        cfg.data_dir = (path.parent / cfg.data_dir).resolve()
    return cfg
