from vaktsom.config import load_config
from vaktsom.storage.db import EventStore
from vaktsom.storage.report import period_report

DAY = 86400.0


def alarm(i, basin, severity, outcome, raised, ack=None, kind="bottom"):
    return {
        "id": f"a{i}",
        "kind": kind,
        "severity": severity,
        "basin_id": basin,
        "x": 1.0,
        "y": 1.0,
        "person_id": i,
        "onset": raised - 10,
        "raised_at": raised,
        "ack_at": ack,
        "outcome": outcome,
    }


def test_false_alarms_per_day_daily_tests_and_reduced_coverage(tmp_path):
    cfg = load_config("configs/demo.yaml")
    store = EventStore(tmp_path / "events.db")
    t0 = 1_800_000_000.0  # a fixed instant, so the report does not depend on today's date
    rows = [alarm(k, "stup", "alarm", "false_alarm", t0 + k * 3600, ack=t0 + k * 3600 + 4) for k in range(12)]
    rows += [alarm(20, "stup", "alarm", "real", t0 + 5000, ack=t0 + 5002)]
    rows += [alarm(21, "stup", "pre_alarm", "false_alarm", t0 + 6000, kind="distress")]
    rows += [alarm(22, "stup", "alarm", "test", t0 + 7000, kind="test")]
    rows += [alarm(23, "idrett", "alarm", None, t0 + 8000)]
    rows += [alarm(24, "stup", "alarm", "false_alarm", t0 + 3 * DAY)]  # after the period
    for r in rows:
        store.upsert_alarm(r)
    store.log("daily_test", "anne", {"basin": "stup"}, ts=t0 + 100)
    store.log("daily_test", "anne", {"basin": "stup"}, ts=t0 + DAY + 100)
    store.health_event("basin:bolge", "degraded", {"reasons": ["mode:waves"]}, ts=t0 - 3600)
    store.health_event("basin:bolge", "ok", {}, ts=t0 + 2 * 3600)
    store.health_event("camera:stup-01", "down", {}, ts=t0 + DAY)

    rep = period_report(store, cfg, t0, t0 + 2 * DAY)
    stup, bolge, idrett = rep["basins"]["stup"], rep["basins"]["bolge"], rep["basins"]["idrett"]
    assert stup["alarms"] == 13 and stup["pre_alarms"] == 1  # the test alarm is not an alarm here
    assert stup["false_alarms_per_day"] == 6.0 and not stup["within_iso_limit"]
    assert stup["false_pre_alarms_per_day"] == 0.5
    assert stup["ack_median_s"] == 4.0 and stup["days_with_daily_test"] == 2
    assert idrett["unresolved"] == 1 and idrett["within_iso_limit"]
    assert bolge["reduced_coverage_h"] == 2.0  # degraded from before the period until 2 h into it
    assert rep["camera_down_h"] == {"stup-01": 24.0}
