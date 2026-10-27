"""Period report: what the pool and the supplier review every month, and what the acceptance test
measures. False alarms per basin per day come from the outcomes lifeguards record when they close
an alarm (NS-EN ISO 20380 4.3.4: on average at most 5 per basin per day over 30 days). Test alarms
are counted only as daily tests (ISO 20380 6.1), system faults not at all.
"""

from __future__ import annotations

import statistics
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

from vaktsom.config import SiteConfig
from vaktsom.storage.db import EventStore

DAY = 86400.0
ISO_FALSE_ALARMS_PER_DAY = 5.0
WEAK = ("degraded", "down")  # coverage states in which part of the water is not watched
ISO_KINDS = ("bottom", "missing")  # a still body under water, and a submerged person lost from view


def period_report(store: EventStore, cfg: SiteConfig, start: float, end: float) -> dict:
    tz = ZoneInfo(cfg.timezone)
    days = max((end - start) / DAY, 1 / 24)
    alarms = [a for a in store.alarms(start, 10**6) if a["raised_at"] < end]
    oplog = [r for r in store.oplog(start, 10**6) if r["ts"] < end]
    health = sorted(store.health_events(0.0, 10**6), key=lambda h: h["ts"])
    out = {}
    for b in cfg.basins:
        mine = [a for a in alarms if a["basin_id"] == b.id and a["kind"] not in ("test", "system")]
        # the ISO figure counts the drowning-detection alarms; surface warnings are counted apart, also
        # when nobody acknowledged one in time and it became an alarm
        full = [a for a in mine if a["kind"] in ISO_KINDS]
        pre = [a for a in mine if a["kind"] not in ISO_KINDS]
        acks = [a["ack_at"] - a["raised_at"] for a in mine if a["ack_at"]]
        tests = {
            datetime.fromtimestamp(r["ts"], tz).date()
            for r in oplog
            if r["kind"] == "daily_test" and r["detail"].get("basin") == b.id
        }
        false_full = sum(a["outcome"] == "false_alarm" for a in full)
        out[b.id] = {
            "name": b.name,
            "alarms": len(full),
            "pre_alarms": len(pre),
            "by_kind": dict(Counter(a["kind"] for a in mine)),
            "by_outcome": dict(Counter(a["outcome"] or "open" for a in mine)),
            "false_alarms": false_full,
            "false_alarms_per_day": round(false_full / days, 2),
            "false_pre_alarms_per_day": round(sum(a["outcome"] == "false_alarm" for a in pre) / days, 2),
            "within_iso_limit": false_full / days <= ISO_FALSE_ALARMS_PER_DAY,
            "unresolved": sum(a["outcome"] is None for a in mine),
            "ack_median_s": round(statistics.median(acks), 1) if acks else None,
            "ack_max_s": round(max(acks), 1) if acks else None,
            "days_with_daily_test": len(tests),
            "reduced_coverage_h": round(_hours_in(health, f"basin:{b.id}", WEAK, start, end), 2),
        }
    cameras = {c.id: round(_hours_in(health, f"camera:{c.id}", ("down",), start, end), 2) for c in cfg.cameras}
    return {
        "site": cfg.site,
        "start": start,
        "end": end,
        "days": round(days, 2),
        "basins": out,
        "camera_down_h": {c: h for c, h in cameras.items() if h > 0},
    }


def _hours_in(health: list[dict], source: str, states: tuple[str, ...], start: float, end: float) -> float:
    """Hours of [start, end) that ``source`` spent in one of ``states``, from its logged changes."""
    state, since, total = "ok", start, 0.0
    for h in health:
        if h["source"] != source:
            continue
        if h["ts"] >= end:
            break
        if h["ts"] > start and state in states:
            total += h["ts"] - max(since, start)
        state, since = h["state"], max(h["ts"], start)
    if state in states:
        total += end - since
    return total / 3600
