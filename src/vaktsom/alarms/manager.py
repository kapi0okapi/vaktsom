"""Alarm life cycle: raise, deliver on every path, escalate, acknowledge, resolve.

- One alarm per incident: a new finding for a person who already has an open alarm upgrades that
  alarm (pre-alarm -> alarm) instead of raising a second one; findings within 2 m and 30 s of an
  open alarm in the same basin are merged into it.
- Every alarm goes to all matching outputs at once; ISO 20380 4.4.2 requires at least two ways to
  reach staff, one of them wired, which the configuration validates.
- An alarm nobody acknowledges within ``ack_timeout_s`` escalates to every output (sirens
  included) and repeats until someone acknowledges it. A pre-alarm that persists becomes an alarm.
- Resolving records the outcome (real incident, false alarm and why, drill, test): the
  statistics the supplier and the pool review every month.
"""

from __future__ import annotations

import itertools
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from vaktsom.analysis.rules import Finding
from vaktsom.config import EscalationConfig
from vaktsom.storage.db import EventStore

Outcome = Literal["real", "false_alarm", "drill", "test", "other"]
SEVERITY_RANK = {"system": 0, "pre_alarm": 1, "alarm": 2}
# a body under water outranks every early warning, even one already escalated to an alarm
KIND_RANK = {"test": 0, "system": 0, "face_down": 1, "distress": 1, "missing": 2, "bottom": 3}


@dataclass
class Alarm:
    id: str
    kind: str
    severity: str
    basin_id: str
    xy: tuple[float, float]
    person_id: int | None
    onset: float
    raised_at: float
    confidence: float = 0.0
    ack_at: float | None = None
    ack_by: str | None = None
    resolved_at: float | None = None
    resolved_by: str | None = None
    outcome: str | None = None
    note: str | None = None
    escalated: bool = False
    escalation_from: float | None = None  # escalation timer start (raise, or re-arm after an upgrade)
    last_sent: float = 0.0
    clip_id: str | None = None
    cameras: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def open(self) -> bool:
        return self.resolved_at is None

    def row(self) -> dict:
        d = asdict(self)
        d["x"], d["y"] = self.xy
        d["detail"] = {**self.detail, "confidence": self.confidence, "clip_id": self.clip_id, "cameras": self.cameras}
        return d

    def public(self) -> dict:
        """What panels and wearables receive."""
        return {
            "id": self.id,
            "kind": self.kind,
            "severity": self.severity,
            "basin_id": self.basin_id,
            "x": round(self.xy[0], 2),
            "y": round(self.xy[1], 2),
            "onset": self.onset,
            "raised_at": self.raised_at,
            "ack_at": self.ack_at,
            "ack_by": self.ack_by,
            "resolved_at": self.resolved_at,
            "escalated": self.escalated,
            "person_id": self.person_id,
            "label": self.detail.get("label", ""),
            "cameras": self.cameras,
        }


class AlarmManager:
    def __init__(self, store: EventStore, escalation: EscalationConfig, dispatch, basin_names: dict[str, str]):
        """``dispatch(alarm, event, everywhere)`` delivers to outputs (see vaktsom.alarms.outputs)."""
        self.store = store
        self.esc = escalation
        self.dispatch = dispatch
        self.basin_names = basin_names
        self.alarms: dict[str, Alarm] = {}
        self._lock = threading.RLock()
        self._seq = itertools.count(1)
        self.listeners: list = []  # callables(alarm, event) for the API's live updates
        self.clock = time.time  # the engine replaces this with its own clock (simulation time in tests)

    # helpers ------------------------------------------------------------------------------------------
    def _emit(
        self, alarm: Alarm, event: str, actor: str = "system", everywhere: bool = False, detail: dict | None = None
    ) -> None:
        now = self.clock()
        self.store.upsert_alarm(alarm.row())
        self.store.alarm_event(alarm.id, event, actor, detail, ts=now)
        if event in ("raised", "upgraded", "escalated", "repeat", "acknowledged", "resolved"):
            self.dispatch(alarm, event, everywhere)
            alarm.last_sent = now
        for fn in self.listeners:
            fn(alarm, event)

    def open_alarms(self) -> list[Alarm]:
        with self._lock:
            return [a for a in self.alarms.values() if a.open]

    def _match(self, f: Finding) -> Alarm | None:
        for a in self.open_alarms():
            if a.person_id is not None and a.person_id == f.person_id:
                return a
            near = np.hypot(a.xy[0] - f.xy[0], a.xy[1] - f.xy[1]) <= 2.0
            if a.basin_id == f.basin_id and near and f.ts - a.raised_at <= 30.0:
                return a
        return None

    # findings -> alarms ------------------------------------------------------------------------------
    def handle(self, findings: list[Finding], cameras_for, label_for) -> list[Alarm]:
        """``cameras_for(person_id)`` lists the cameras that saw the person; ``label_for(basin, xy)``
        turns a position into words for the lifeguards ("Idrettsbasseng, bane 4, dyp ende")."""
        raised = []
        with self._lock:
            for f in findings:
                existing = self._match(f)
                if existing is not None:
                    more_severe = SEVERITY_RANK[f.severity] > SEVERITY_RANK[existing.severity]
                    more_critical = KIND_RANK.get(f.kind, 0) > KIND_RANK.get(existing.kind, 0)
                    if more_severe or more_critical:
                        existing.severity = max(existing.severity, f.severity, key=SEVERITY_RANK.get)
                        existing.kind = f.kind
                        if more_critical and existing.ack_at is not None:
                            # acknowledged as a warning, now the person is under water: alarm again
                            existing.ack_at = existing.ack_by = None
                            existing.escalated = False
                            existing.escalation_from = f.ts
                        existing.xy = (float(f.xy[0]), float(f.xy[1]))
                        existing.detail["label"] = label_for(f.basin_id, f.xy)
                        self._emit(
                            existing, "upgraded", detail={"kind": f.kind, "x": existing.xy[0], "y": existing.xy[1]}
                        )
                    continue
                alarm = Alarm(
                    id=f"{time.strftime('%Y%m%d')}-{next(self._seq):04d}-{uuid.uuid4().hex[:4]}",
                    kind=f.kind,
                    severity=f.severity,
                    basin_id=f.basin_id,
                    xy=(float(f.xy[0]), float(f.xy[1])),
                    person_id=f.person_id,
                    onset=f.onset,
                    raised_at=f.ts,
                    confidence=f.confidence,
                    cameras=cameras_for(f.person_id),
                    detail={
                        **f.detail,
                        "label": label_for(f.basin_id, f.xy),
                        "basin": self.basin_names.get(f.basin_id, f.basin_id),
                    },
                )
                self.alarms[alarm.id] = alarm
                self._emit(
                    alarm,
                    "raised",
                    detail={"kind": f.kind, "set_off_s": round(f.ts - f.onset, 2), "x": alarm.xy[0], "y": alarm.xy[1]},
                )
                raised.append(alarm)
        return raised

    def raise_manual(
        self, kind: str, severity: str, basin_id: str, xy, actor: str, detail: dict | None = None
    ) -> Alarm:
        """Test alarms (daily test, ISO 6.1) and system alarms (camera down)."""
        now = self.clock()
        with self._lock:
            alarm = Alarm(
                f"{time.strftime('%Y%m%d')}-{next(self._seq):04d}-{uuid.uuid4().hex[:4]}",
                kind,
                severity,
                basin_id,
                (float(xy[0]), float(xy[1])),
                None,
                now,
                now,
                detail=detail or {},
            )
            self.alarms[alarm.id] = alarm
            self._emit(alarm, "raised", actor)
            return alarm

    def tick(self, now: float) -> None:
        with self._lock:
            for a in self.open_alarms():
                if a.ack_at is not None:
                    continue
                age = now - (a.escalation_from if a.escalation_from is not None else a.raised_at)
                if a.severity == "pre_alarm" and age >= self.esc.pre_alarm_to_alarm_s:
                    a.severity = "alarm"
                    a.escalation_from = now  # staff get the alarm first, the hall sirens after ack_timeout_s
                    self._emit(a, "upgraded", detail={"reason": "pre-alarm persisted"})
                elif a.severity == "alarm" and not a.escalated and age >= self.esc.ack_timeout_s:
                    a.escalated = True
                    self._emit(a, "escalated", everywhere=True)
                elif a.escalated and now - a.last_sent >= self.esc.repeat_s:
                    self._emit(a, "repeat", everywhere=True)

    def acknowledge(self, alarm_id: str, actor: str) -> Alarm:
        with self._lock:
            a = self.alarms[alarm_id]
            if a.ack_at is None:
                a.ack_at, a.ack_by = self.clock(), actor
                self._emit(a, "acknowledged", actor)
            return a

    def resolve(self, alarm_id: str, actor: str, outcome: Outcome, note: str = "") -> Alarm:
        with self._lock:
            a = self.alarms[alarm_id]
            if a.ack_at is None:
                a.ack_at, a.ack_by = self.clock(), actor
            a.resolved_at, a.resolved_by, a.outcome, a.note = self.clock(), actor, outcome, note
            self._emit(a, "resolved", actor, detail={"outcome": outcome, "note": note})
            return a

    def pinned_persons(self) -> set[int]:
        return {a.person_id for a in self.open_alarms() if a.person_id is not None}
