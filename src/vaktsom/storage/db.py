"""Event store: alarms, their history, the operational log and health events (SQLite, WAL).

ISO 20380 4.4: date, time, images and position of each detection on the system's own disk, every
alarm logged, and a log of use (session start/stop, logins) for at least the last 30 days. Rows
are pruned after ``retention.log_days``; alarm images are handled by ``vaktsom.storage.clips``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS alarms (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, severity TEXT NOT NULL, basin_id TEXT NOT NULL,
    x REAL, y REAL, person_id INTEGER, onset REAL, raised_at REAL NOT NULL,
    ack_at REAL, ack_by TEXT, resolved_at REAL, resolved_by TEXT, outcome TEXT, note TEXT,
    escalated INTEGER DEFAULT 0, detail TEXT
);
CREATE INDEX IF NOT EXISTS alarms_raised ON alarms(raised_at);
CREATE TABLE IF NOT EXISTS alarm_events (
    alarm_id TEXT NOT NULL, ts REAL NOT NULL, event TEXT NOT NULL, actor TEXT, detail TEXT
);
CREATE INDEX IF NOT EXISTS alarm_events_alarm ON alarm_events(alarm_id);
CREATE TABLE IF NOT EXISTS oplog (ts REAL NOT NULL, kind TEXT NOT NULL, actor TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS oplog_ts ON oplog(ts);
CREATE TABLE IF NOT EXISTS health (ts REAL NOT NULL, source TEXT NOT NULL, state TEXT NOT NULL, detail TEXT);
CREATE INDEX IF NOT EXISTS health_ts ON health(ts);
"""

ALARM_FIELDS = (
    "id",
    "kind",
    "severity",
    "basin_id",
    "x",
    "y",
    "person_id",
    "onset",
    "raised_at",
    "ack_at",
    "ack_by",
    "resolved_at",
    "resolved_by",
    "outcome",
    "note",
    "escalated",
    "detail",
)


class EventStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)

    def _exec(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # alarms ---------------------------------------------------------------------------------------
    def upsert_alarm(self, row: dict) -> None:
        values = [row.get(k) for k in ALARM_FIELDS]
        values[ALARM_FIELDS.index("detail")] = json.dumps(row.get("detail") or {})
        values[ALARM_FIELDS.index("escalated")] = int(bool(row.get("escalated")))
        placeholders = ",".join("?" * len(ALARM_FIELDS))
        updates = ",".join(f"{k}=excluded.{k}" for k in ALARM_FIELDS[1:])
        self._exec(
            f"INSERT INTO alarms ({','.join(ALARM_FIELDS)}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            tuple(values),
        )

    def alarm_event(
        self, alarm_id: str, event: str, actor: str = "system", detail: dict | None = None, ts: float | None = None
    ) -> None:
        self._exec(
            "INSERT INTO alarm_events VALUES (?,?,?,?,?)",
            (alarm_id, time.time() if ts is None else ts, event, actor, json.dumps(detail or {})),
        )

    def alarms(self, since: float = 0.0, limit: int = 500) -> list[dict]:
        rows = self._exec(
            f"SELECT {','.join(ALARM_FIELDS)} FROM alarms WHERE raised_at >= ? ORDER BY raised_at DESC LIMIT ?",
            (since, limit),
        )
        out = []
        for r in rows:
            d = dict(zip(ALARM_FIELDS, r, strict=True))
            d["detail"] = json.loads(d["detail"] or "{}")
            out.append(d)
        return out

    def set_outcome(self, alarm_id: str, outcome: str, note: str, actor: str, ts: float | None = None) -> bool:
        """Outcome of a closed alarm, set or corrected later (the monthly review, ISO 20380 4.3.4)."""
        if not self._exec("SELECT 1 FROM alarms WHERE id=?", (alarm_id,)):
            return False
        now = time.time() if ts is None else ts
        self._exec(
            "UPDATE alarms SET outcome=?, note=?, resolved_by=COALESCE(resolved_by, ?), "
            "resolved_at=COALESCE(resolved_at, ?) WHERE id=?",
            (outcome, note, actor, now, alarm_id),
        )
        self.alarm_event(alarm_id, "outcome_set", actor, {"outcome": outcome, "note": note}, ts=now)
        return True

    def alarm_history(self, alarm_id: str) -> list[dict]:
        rows = self._exec("SELECT ts, event, actor, detail FROM alarm_events WHERE alarm_id=? ORDER BY ts", (alarm_id,))
        return [{"ts": ts, "event": e, "actor": a, "detail": json.loads(d or "{}")} for ts, e, a, d in rows]

    # operational log and health --------------------------------------------------------------------
    def log(self, kind: str, actor: str = "system", detail: dict | None = None, ts: float | None = None) -> None:
        self._exec(
            "INSERT INTO oplog VALUES (?,?,?,?)",
            (time.time() if ts is None else ts, kind, actor, json.dumps(detail or {})),
        )

    def oplog(self, since: float = 0.0, limit: int = 1000) -> list[dict]:
        rows = self._exec(
            "SELECT ts, kind, actor, detail FROM oplog WHERE ts >= ? ORDER BY ts DESC LIMIT ?", (since, limit)
        )
        return [{"ts": ts, "kind": k, "actor": a, "detail": json.loads(d or "{}")} for ts, k, a, d in rows]

    def health_event(self, source: str, state: str, detail: dict | None = None, ts: float | None = None) -> None:
        self._exec(
            "INSERT INTO health VALUES (?,?,?,?)",
            (time.time() if ts is None else ts, source, state, json.dumps(detail or {})),
        )

    def health_events(self, since: float = 0.0, limit: int = 1000) -> list[dict]:
        rows = self._exec(
            "SELECT ts, source, state, detail FROM health WHERE ts >= ? ORDER BY ts DESC LIMIT ?", (since, limit)
        )
        return [{"ts": ts, "source": s, "state": st, "detail": json.loads(d or "{}")} for ts, s, st, d in rows]

    def prune(self, keep_days: int, now: float | None = None) -> None:
        cutoff = (now or time.time()) - keep_days * 86400
        for table, col in (("alarm_events", "ts"), ("oplog", "ts"), ("health", "ts"), ("alarms", "raised_at")):
            self._exec(f"DELETE FROM {table} WHERE {col} < ?", (cutoff,))

    def close(self) -> None:
        with self._lock:
            self._db.close()
