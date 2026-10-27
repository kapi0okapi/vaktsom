import time
from pathlib import Path

import numpy as np
import pytest

from vaktsom.alarms.manager import AlarmManager
from vaktsom.analysis.rules import Finding, NonDetectionZone, RuleEngine
from vaktsom.config import BasinConfig, EscalationConfig, RuleConfig, SiteConfig
from vaktsom.geometry import WATER_N, Basin, Camera, points_in_polygon, refract
from vaktsom.perception.types import Detection
from vaktsom.storage.clips import ClipStore, FrameRing
from vaktsom.storage.db import EventStore
from vaktsom.tracking.camera_tracker import CameraTracker
from vaktsom.tracking.world import Person, PointKalman


def pool() -> Basin:
    return BasinConfig(id="b", name="B", outline=[(0, 0), (25, 0), (25, 12), (0, 12)], default_depth=4.0).build()


def test_snell_law():
    d = np.array([[np.sin(0.7), 0.0, -np.cos(0.7)]])
    t = refract(d)[0]
    sin_t = np.hypot(t[0], t[1])
    assert np.isclose(np.sin(0.7), WATER_N * sin_t, atol=1e-9)


def test_underwater_round_trip():
    """Projecting a submerged point and casting the pixel back at that depth returns the point."""
    cam = Camera.looking_at([5, -4, 9], [12, 6, 0], 80)
    target = np.array([[13.0, 7.0, -3.5]])
    px = cam.project_underwater(target)
    back = cam.to_depth(px, 3.5)
    assert np.allclose(back[0], target[0, :2], atol=1e-3)
    # ignoring refraction would misplace the body by metres at this depth
    naive = cam.to_surface(px) + 0  # straight-line guess stops at the surface
    assert np.linalg.norm(naive[0] - target[0, :2]) > 1.0


def test_points_in_polygon():
    poly = np.array([[0, 0], [10, 0], [10, 5], [0, 5]])
    assert points_in_polygon(np.array([[5, 2], [11, 2], [-1, 1]]), poly).tolist() == [True, False, False]


def test_depth_regions():
    b = BasinConfig(
        id="w",
        name="W",
        outline=[(0, 0), (20, 0), (20, 10), (0, 10)],
        default_depth=1.6,
        regions=[{"polygon": [(0, 0), (20, 0), (20, 10), (0, 10)], "depth": 1.6, "slope_to": [(10, 0), (10, 10), 0.1]}],
    ).build()
    d = b.depth_at(np.array([[10, 0], [10, 5], [10, 10]]))
    assert np.allclose(d, [1.6, 0.85, 0.1])


def test_tracker_keeps_identity_through_a_missed_frame():
    tr = CameraTracker()
    box = np.array([100, 100, 140, 200], float)
    ids = set()
    for k in range(20):
        dets = [] if k == 10 else [Detection(box + [k * 3, 0, k * 3, 0], 0.9)]
        for t in tr.update(dets, k * 0.1):
            ids.add(t.id)
    assert ids == {1}


def _person(pid: int, under: float, speed: float, motion: float, lost: bool = False) -> Person:
    p = Person(pid, PointKalman(np.array([10.0, 6.0]), 0.0), 0.0, 0.0, basin_id="b")
    p.kf.x[2] = speed
    p.p_under, p.motion, p.lost, p.updates = under, motion, lost, 50
    p.bottom_depth = 4.0
    return p


def test_bottom_rule_fires_at_set_off_time():
    basin = BasinConfig(
        id="b", name="B", outline=[(0, 0), (25, 0), (25, 12), (0, 12)], default_depth=4.0, set_off_time_s=10
    )
    eng = RuleEngine([basin], RuleConfig())
    p = _person(1, 0.9, 0.02, 0.05)
    fired = []
    for k in range(200):
        ts = k * 0.1
        fired += [(ts, f) for f in eng.evaluate([p], ts)]
    assert len(fired) == 1
    ts, f = fired[0]
    assert f.kind == "bottom"
    # the onset is back-dated by the stillness window, so the alarm comes ~10 s after the true onset
    assert abs((ts - f.onset) - 10.0) < 0.2


def test_moving_or_surfaced_person_does_not_fire():
    basin = BasinConfig(id="b", name="B", outline=[(0, 0), (25, 0), (25, 12), (0, 12)], default_depth=4.0)
    eng = RuleEngine([basin], RuleConfig())
    swimmer = _person(1, 0.9, 0.8, 0.9)  # under water but swimming
    surfaced = _person(2, 0.1, 0.0, 0.1)  # still, head above water
    assert not any(eng.evaluate([swimmer, surfaced], k * 0.1) for k in range(300))


def test_non_detection_zone_suppresses():
    basin = BasinConfig(id="b", name="B", outline=[(0, 0), (25, 0), (25, 12), (0, 12)], default_depth=4.0)
    eng = RuleEngine([basin], RuleConfig())
    eng.zones["z"] = NonDetectionZone("z", "b", np.array([[8, 4], [12, 4], [12, 8], [8, 8]]), until=15.0)
    p = _person(1, 0.9, 0.0, 0.0)
    fired = [k * 0.1 for k in range(300) if eng.evaluate([p], k * 0.1)]
    assert fired and fired[0] >= 15.0 + 10.0 - 0.9  # detection resumes by itself when the zone expires


def test_missing_rule_uses_last_sighting():
    basin = BasinConfig(
        id="b", name="B", outline=[(0, 0), (25, 0), (25, 12), (0, 12)], default_depth=4.0, set_off_time_s=10
    )
    eng = RuleEngine([basin], RuleConfig())
    p = _person(1, 0.9, 0.5, 0.8, lost=True)
    p.first_ts, p.last_seen = -5.0, 0.0
    fired = [(k * 0.1, f) for k in range(200) for f in eng.evaluate([p], k * 0.1)]
    assert len(fired) == 1 and fired[0][1].kind == "missing" and abs(fired[0][0] - 10.0) < 0.15


def test_alarm_manager_one_alarm_per_person_and_escalation(tmp_path):
    store = EventStore(tmp_path / "e.db")
    sent = []
    mgr = AlarmManager(
        store,
        EscalationConfig(ack_timeout_s=10),
        lambda a, e, everywhere=False: sent.append((e, everywhere)),
        {"b": "B"},
    )
    now = [0.0]
    mgr.clock = lambda: now[0]
    f = Finding("distress", "pre_alarm", 7, "b", np.array([3.0, 4.0]), 0.0, 0.0, 0.9)
    mgr.handle([f], lambda pid: ["k1"], lambda b, xy: "B, midt")
    g = Finding("bottom", "alarm", 7, "b", np.array([3.2, 4.1]), 5.0, 0.0, 0.9)
    now[0] = 5.0
    mgr.handle([g], lambda pid: ["k1"], lambda b, xy: "B, midt")
    assert len(mgr.alarms) == 1
    a = next(iter(mgr.alarms.values()))
    assert a.kind == "bottom" and a.severity == "alarm"
    now[0] = 16.0
    mgr.tick(16.0)
    assert a.escalated and ("escalated", True) in sent
    mgr.acknowledge(a.id, "vakt1")
    mgr.resolve(a.id, "vakt1", "drill")
    assert not mgr.open_alarms()
    events = [h["event"] for h in store.alarm_history(a.id)]
    assert events[:2] == ["raised", "upgraded"] and "resolved" in events


def test_clip_is_unreadable_after_shredding(tmp_path):
    store = ClipStore(tmp_path)
    ring = FrameRing(seconds=30, fps=10)
    for k in range(50):
        ring.push(k * 0.1, np.full((48, 64, 3), k, np.uint8))
    cid = store.start("a1", ["c1"], onset=1.0, raised_at=3.0, pre_s=1.0, post_s=1.0, rings={"c1": ring})
    store.poll(5.0, {"c1": ring})
    header, frames = store.read(cid)
    assert header["alarm_id"] == "a1" and len(frames["c1"]) > 5
    blob = (tmp_path / "clips" / f"{cid}.bin").read_bytes()
    store.shred(cid)
    with pytest.raises(KeyError):
        store.read(cid)
    # a backup copy of the file is useless without the destroyed key
    assert b"JFIF" not in blob


def test_config_requires_wired_and_second_alarm_path():
    base = {
        "site": "x",
        "basins": [{"id": "b", "name": "B", "outline": [(0, 0), (1, 0), (1, 1)], "default_depth": 1.0}],
        "cameras": [
            {"id": "c", "url": "sim://", "basins": ["b"], "position": (0, 0, 5), "target": (1, 1, 0), "hfov_deg": 80}
        ],
    }
    with pytest.raises(ValueError):
        SiteConfig.model_validate({**base, "outputs": [{"id": "w", "kind": "wearable", "wired": False}]})
    ok = SiteConfig.model_validate(
        {
            **base,
            "outputs": [
                {"id": "w", "kind": "wearable", "wired": False},
                {"id": "s", "kind": "siren", "wired": True, "url": "http://x"},
            ],
        }
    )
    assert ok.basins[0].set_off_time_s <= 15


def test_set_off_time_cannot_exceed_iso_limit():
    with pytest.raises(ValueError):
        BasinConfig(id="b", name="B", outline=[(0, 0), (1, 0), (1, 1)], default_depth=1.0, set_off_time_s=20)


def test_engine_end_to_end_detects_drowning(tmp_path):
    from vaktsom.config import load_config
    from vaktsom.runtime.engine import Engine
    from vaktsom.sim.scene import Drowner, Player, Scene
    from vaktsom.sim.source import SimSource

    cfg = load_config("configs/demo.yaml")
    cfg.data_dir = tmp_path
    cfg.cameras = [c for c in cfg.cameras if "stup" in c.basins]
    basins = {b.id: b.build() for b in cfg.basins}
    rng = np.random.default_rng(3)
    scene = Scene({"stup": basins["stup"]})
    d = Drowner(1, basins["stup"], rng, start=5.0, idr_s=8.0)
    scene.bathers += [d] + [Player(k, basins["stup"], rng) for k in range(2, 6)]
    cams = {c.id: c.build() for c in cfg.cameras}
    eng = Engine(cfg, SimSource(scene, cams, basins, render=False, background=False))
    t0 = time.time()
    for k in range(700):
        eng.step(t0 + k * 0.1)
    kinds = {a["kind"] for a in eng.store.alarms()}
    assert "bottom" in kinds or "missing" in kinds


def test_acknowledged_warning_alarms_again_when_the_person_goes_under(tmp_path):
    store = EventStore(tmp_path / "e.db")
    sent = []
    mgr = AlarmManager(
        store, EscalationConfig(ack_timeout_s=10), lambda a, e, everywhere=False: sent.append(e), {"b": "B"}
    )
    now = [0.0]
    mgr.clock = lambda: now[0]
    mgr.handle(
        [Finding("distress", "pre_alarm", 3, "b", np.array([1.0, 1.0]), 0.0, 0.0, 0.9)], lambda p: [], lambda b, xy: ""
    )
    a = next(iter(mgr.alarms.values()))
    mgr.acknowledge(a.id, "vakt")
    now[0] = 20.0
    mgr.handle(
        [Finding("bottom", "alarm", 3, "b", np.array([1.0, 1.0]), 20.0, 9.0, 0.9)], lambda p: [], lambda b, xy: ""
    )
    assert a.kind == "bottom" and a.ack_at is None
    now[0] = 31.0
    mgr.tick(31.0)
    assert a.escalated and sent.count("escalated") == 1


def test_motion_probe_separates_a_still_body_from_a_swimmer():
    """A still body under shimmering water scores low, a moving one high."""
    from vaktsom.perception.motion import MotionProbe

    rng = np.random.default_rng(0)
    probe = MotionProbe(lag_s=0.5)
    base = np.full((240, 320, 3), 150, np.uint8)
    body = Detection(np.array([100, 100, 140, 180], float), 0.9)
    swimmer = Detection(np.array([200, 100, 240, 180], float), 0.9)
    for k in range(12):
        img = base.copy()
        img += rng.integers(0, 12, img.shape, dtype=np.uint8)  # water shimmer everywhere
        img[100:180, 100:140] = 60  # the still body
        y = 100 + (k % 3) * 12  # the swimmer kicks (not in step with the comparison lag)
        img[y : y + 55, 200:240] = 60
        dets = [Detection(body.box.copy(), 0.9), Detection(swimmer.box.copy(), 0.9)]
        probe.annotate(k * 0.25, img, dets)
    assert dets[0].motion is not None and dets[0].motion < 0.2
    assert dets[1].motion > 0.5


def test_wave_mode_pauses_surface_warnings_but_not_the_bottom_rule():
    basin = BasinConfig(id="b", name="B", outline=[(0, 0), (25, 0), (25, 12), (0, 12)], default_depth=4.0)
    for turbulent, expected in ((set(), {"distress", "bottom"}), ({"b"}, {"bottom"})):
        eng = RuleEngine([basin], RuleConfig())
        eng.turbulent = turbulent
        bobbing = _person(1, 0.2, 0.3, 0.6)
        bobbing.distress = 0.95
        body = _person(2, 0.9, 0.02, 0.05)
        kinds = {f.kind for k in range(200) for f in eng.evaluate([bobbing, body], k * 0.1)}
        assert kinds == expected


def test_unknown_movable_floor_does_not_block_the_bottom_rule():
    cfg = BasinConfig(
        id="b",
        name="B",
        outline=[(0, 0), (25, 0), (25, 12), (0, 12)],
        default_depth=2.0,
        movable_floor=True,
        regions=[{"polygon": [(0, 0), (10, 0), (10, 12), (0, 12)], "depth": 1.8, "movable": True}],
    )
    basin = cfg.build()
    assert basin.floor_unknown([[5.0, 6.0], [20.0, 6.0]]).tolist() == [True, False]
    basin.movable_floor_depth = 0.4  # floor raised for a swimming class
    assert basin.floor_unknown([[5.0, 6.0]]).tolist() == [False]
    assert basin.depth_at([[5.0, 6.0]])[0] == 0.4


def test_hall_siren_waits_for_escalation_and_a_log_is_not_an_alarm_path():
    from vaktsom.alarms.outputs import SirenOutput
    from vaktsom.config import OutputConfig

    hall = SirenOutput(OutputConfig(id="s", kind="siren", wired=True, url="http://x", options={"when": "escalated"}))
    alarm = type("A", (), {"basin_id": "b", "severity": "alarm"})()
    assert not hall.wants(alarm, "raised", False)
    assert hall.wants(alarm, "escalated", True) and hall.wants(alarm, "acknowledged", False)
    base = {
        "site": "x",
        "basins": [{"id": "b", "name": "B", "outline": [(0, 0), (1, 0), (1, 1)], "default_depth": 1.0}],
        "cameras": [
            {"id": "c", "url": "sim://", "basins": ["b"], "position": (0, 0, 5), "target": (1, 1, 0), "hfov_deg": 80}
        ],
    }
    late = {"id": "s", "kind": "siren", "wired": True, "url": "http://x", "options": {"when": "escalated"}}
    for second in (late, {"id": "l", "kind": "log", "wired": True}):
        with pytest.raises(ValueError):
            SiteConfig.model_validate({**base, "outputs": [{"id": "w", "kind": "wearable", "wired": False}, second]})


def test_housekeeping_shreds_old_clips_and_prunes_logs(tmp_path):
    from vaktsom.config import load_config
    from vaktsom.runtime.engine import Engine
    from vaktsom.sim.scene import Scene
    from vaktsom.sim.source import SimSource

    cfg = load_config("configs/demo.yaml")
    cfg.data_dir = tmp_path
    cfg.cameras = [c for c in cfg.cameras if "stup" in c.basins]
    basins = {b.id: b.build() for b in cfg.basins}
    cams = {c.id: c.build() for c in cfg.cameras}
    eng = Engine(cfg, SimSource(Scene({"stup": basins["stup"]}), cams, basins, render=False, background=False))
    now = time.time()
    eng.store.log("old", ts=now - 40 * 86400)
    eng.store.log("recent", ts=now - 86400)
    eng.housekeeping(now)
    kinds = {r["kind"] for r in eng.store.oplog(0)}
    assert "old" not in kinds and "recent" in kinds
    ring = FrameRing(seconds=30, fps=10)
    for k in range(50):
        ring.push(k * 0.1, np.full((48, 64, 3), k, np.uint8))
    cid = eng.clips.start("a1", ["stup-01"], onset=1.0, raised_at=3.0, pre_s=1.0, post_s=1.0, rings={"stup-01": ring})
    eng.clips.poll(5.0, {"stup-01": ring})
    eng.housekeeping(now + 6 * 86400)
    assert eng.clips.read(cid)[0]["alarm_id"] == "a1"  # kept for the review within the 7 days
    eng.housekeeping(now + 8 * 86400)
    with pytest.raises(KeyError):
        eng.clips.read(cid)
    assert any(r["kind"] == "clips_shredded" for r in eng.store.oplog(0))


def test_detection_loop_survives_an_error_and_the_watchdog_warns_when_it_stops(tmp_path):
    from vaktsom.config import load_config
    from vaktsom.runtime.engine import Engine
    from vaktsom.sim.scene import Scene
    from vaktsom.sim.source import SimSource

    cfg = load_config("configs/demo.yaml")
    cfg.data_dir = tmp_path
    cfg.cameras = [c for c in cfg.cameras if "stup" in c.basins]
    basins = {b.id: b.build() for b in cfg.basins}
    cams = {c.id: c.build() for c in cfg.cameras}
    eng = Engine(cfg, SimSource(Scene({"stup": basins["stup"]}), cams, basins, render=False, background=False))
    update, calls = eng.world.update, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("bad frame")
        return update(*a, **k)

    eng.world.update = flaky
    eng.run(realtime=False, duration=1.0, start=time.time())
    assert len(calls) >= 10 and eng.stats["ticks"] == len(calls) - 1  # it went on after the error
    assert any(r["kind"] == "engine_error" for r in eng.store.oplog(0))

    now = time.time()
    eng.last_tick_wall = now - 10  # the loop has stopped
    eng.check_alive(now)
    down = [a for a in eng.alarms.open_alarms() if a.kind == "system"]
    assert len(down) == 1 and down[0].severity == "system"
    eng.last_tick_wall = now  # running again
    eng.check_alive(now)
    assert not [a for a in eng.alarms.open_alarms() if a.kind == "system"]


def test_an_ignored_warning_becomes_an_alarm_before_the_hall_sirens():
    import tempfile

    store = EventStore(Path(tempfile.mkdtemp()) / "e.db")
    sent = []
    mgr = AlarmManager(
        store,
        EscalationConfig(ack_timeout_s=10, pre_alarm_to_alarm_s=15),
        lambda a, e, everywhere=False: sent.append(e),
        {"b": "B"},
    )
    now = [0.0]
    mgr.clock = lambda: now[0]
    mgr.handle(
        [Finding("distress", "pre_alarm", 7, "b", np.array([3.0, 4.0]), 0.0, 0.0, 0.9)],
        lambda pid: [],
        lambda b, xy: "B",
    )
    times = {}
    for t in np.arange(0.0, 40.0, 0.5):
        now[0] = float(t)
        mgr.tick(float(t))
        for e in ("upgraded", "escalated"):
            if e in sent and e not in times:
                times[e] = float(t)
    assert times == {"upgraded": 15.0, "escalated": 25.0}


def test_a_single_camera_ghost_lost_under_water_is_folded_but_a_real_missing_person_is_not():
    from vaktsom.tracking.world import WorldTracker

    def person(pid, xy, first, links, under=0.1):
        p = Person(pid, PointKalman(np.array(xy, float), first), first, 100.0, basin_id="b")
        p.links, p.p_under = dict(links), under
        p.history.append((99.0, xy[0], xy[1], 0.0, 0.8, 0.5, 0.3, len(links), None))  # under water a second ago
        return p

    from vaktsom.tracking.world import Observation

    def seen_at(xy, step=(0.0, -1.2)):  # one camera's view: on the surface at xy, deeper along step
        z = np.zeros(2)
        return Observation("c3", 1, 100.0, z, np.array(xy, float), 0.85, 0.3, 0.2, 1.0, np.zeros(4), np.array(step))

    world = WorldTracker({}, {"b": pool()})
    seen = person(1, (10.0, 6.0), 0.0, {"c1": 1, "c2": 2})
    ghost = person(2, (10.4, 8.2), 90.0, {}, under=0.85)  # 2.2 m off at the surface, on the ray at 2 m depth
    ghost.lost, ghost.observations = True, {"c3": seen_at((10.4, 8.2))}
    real = person(3, (14.0, 6.0), 20.0, {}, under=0.85)  # seen from two cameras before it vanished
    real.lost, real.observations = True, {"c1": seen_at((14.0, 6.0)), "c2": seen_at((14.0, 6.0))}
    world.persons = {1: seen, 2: ghost, 3: real}
    world._merge_ghosts(100.5)
    assert set(world.persons) == {1, 3}
    beside = person(4, (14.4, 6.0), 50.0, {}, under=0.85)  # one camera, but older than everyone near it
    beside.lost, beside.observations = True, {"c3": seen_at((14.4, 6.0))}
    world.persons[4] = beside
    world.persons[5] = person(5, (14.2, 6.0), 60.0, {"c1": 7, "c2": 8})
    world._merge_ghosts(100.5)
    assert 4 in world.persons


def _view(cam, track, xy, under=0.1, weight=1.0, step=(0.0, -0.7), ts=100.0):
    """One camera's view as the world tracker gets it: on the surface at ``xy``, deeper along ``step``."""
    from vaktsom.tracking.world import Observation

    z = np.zeros(2)
    return Observation(cam, track, ts, z, np.array(xy, float), under, 0.3, 0.3, weight, np.zeros(4), np.array(step))


def test_a_view_of_a_body_deep_under_a_swimmer_does_not_join_the_swimmer():
    """3.4 m down, this view's ray passes right under a swimmer at the surface: the view may join a
    person only where that person can be, so it becomes a person of its own."""
    from vaktsom.tracking.world import WorldTracker

    world = WorldTracker({}, {"b": pool()})
    swimmer = Person(101, PointKalman(np.array([10.0, 6.0]), 99.9), 0.0, 99.9, basin_id="b")
    swimmer.depth, swimmer.links, swimmer.updates = 0.0, {"c1": 1, "c2": 2}, 50
    world.persons = {101: swimmer}
    deep = _view("c3", 1, (10.0, 8.4), under=0.85)
    assert deep.ray_distance(swimmer.xy, 0.0, 3.9) < 0.1
    world.update([_view("c1", 1, (10.0, 6.0)), _view("c2", 2, (10.0, 6.0)), deep], 100.0)
    assert world.persons[101].links == {"c1": 1, "c2": 2}
    assert any(p.links == {"c3": 1} for p in world.persons.values())
    diver = Person(9, PointKalman(np.array([10.0, 6.0]), 99.9), 0.0, 99.9, basin_id="b")
    diver.depth = None  # under water, one view, depth unknown: anywhere along the ray
    assert world._depths(diver, 3.9, 100.0) == (0.0, 3.9)


def test_a_camera_that_swapped_two_people_is_let_go_and_not_taken_back():
    from vaktsom.tracking.world import WorldTracker

    world = WorldTracker({}, {"b": pool()})
    p = Person(101, PointKalman(np.array([10.6, 6.0]), 99.9), 0.0, 99.9, basin_id="b")
    p.depth, p.links, p.updates = 0.0, {"c1": 1, "c2": 2}, 50
    world.persons = {101: p}
    for k in range(13):  # c2 now follows someone 2 m away
        world.update([_view("c1", 1, (10.0, 6.0)), _view("c2", 2, (12.0, 6.0), weight=0.5)], 100.0 + 0.1 * k)
    assert world.persons[101].links == {"c1": 1}  # the poorer view went after a second
    other = [q for q in world.persons.values() if q.id != 101]
    assert len(other) == 1 and other[0].links == {"c2": 2}


def test_rays_that_never_meet_give_no_depth():
    from vaktsom.tracking.world import WorldTracker

    world = WorldTracker({}, {"b": pool()})
    a = _view("c1", 1, (10.0, 4.0), under=0.85, step=(0.0, 0.7))
    b = _view("c2", 1, (10.0, 8.0), under=0.85, step=(0.0, -0.7))
    xy, depth = world._triangulate([a, b])  # one body 2.86 m down at (10, 6)
    assert np.allclose(xy, [10.0, 6.0], atol=0.1) and abs(depth - 2.86) < 0.1
    c = _view("c2", 1, (13.0, 8.0), under=0.85, step=(0.0, -0.7))  # 3 m apart at every depth
    assert world._triangulate([a, c]) is None


def test_a_diver_seen_by_one_camera_continues_as_the_newcomer_who_surfaces_on_its_ray():
    from vaktsom.tracking.world import WorldTracker

    world = WorldTracker({}, {"b": pool()})
    diver = Person(1, PointKalman(np.array([12.8, 6.0]), 0.0), 0.0, 100.0, basin_id="b")
    diver.lost, diver.p_under, diver.depth, diver.depth_est = True, 0.85, None, 3.7
    diver.observations = {"c1": _view("c1", 1, (10.2, 6.0), under=0.85, step=(0.7, 0.0))}
    newcomer = Person(2, PointKalman(np.array([10.0, 6.1]), 100.5), 100.5, 100.5, basin_id="b")
    newcomer.links = {"c2": 5}
    world.persons = {1: diver, 2: newcomer}
    world._merge_splits(100.5)  # 2.8 m from the working-depth point, but on the camera's ray
    assert set(world.persons) == {2} and world.persons[2].first_ts == 0.0


def test_a_breath_hold_in_the_diving_pool_takes_seconds_down_and_up():
    from vaktsom.sim.scene import BreathHolder

    deep = BasinConfig(id="s", name="S", outline=[(0, 0), (20, 0), (20, 15), (0, 15)], default_depth=5.0).build()
    b = BreathHolder(1, deep, np.random.default_rng(0), start=10.0, hold_s=5.0)
    z = {t: b.step(t, 0.1).z for t in (9.0, 11.0, 14.0, 17.0, 19.0, 22.0, 25.0)}
    assert z[9.0] == 0.15 and 0.3 < z[11.0] < 2.0  # going down at 1.2 m/s
    assert z[14.0] == z[17.0] == 4.7  # on the bottom
    assert 2.5 < z[19.0] < 4.7 and 0.3 <= z[22.0] < 1.5 and z[25.0] == 0.15  # back up at 1 m/s


def test_a_moment_of_a_wrong_position_does_not_restart_the_stillness_of_a_body():
    from vaktsom.analysis.rules import still_since

    def body(away_from, away_s):
        p = _person(1, 0.9, 0.0, 0.05)
        for k in range(80):  # 8 s on the bottom at (10, 6), triangulated 3.8 m down
            t = 100.0 + 0.1 * k
            x = 10.8 if away_from <= t < away_from + away_s - 1e-9 else 10.0
            p.history.append((t, x, 6.0, 0.0, 0.9, 0.05, 0.3, 2, 3.8))
        return p

    assert still_since(body(103.0, 0.6), 107.9) == 100.0  # one camera followed a wrong detection
    assert abs(still_since(body(103.0, 1.5), 107.9) - 104.5) < 1e-6  # the body moved


def test_a_body_by_the_wall_stays_in_its_basin_when_its_estimate_strays_past_the_edge():
    from vaktsom.tracking.world import WorldTracker

    world = WorldTracker({}, {"b": pool()})
    near, gap = pool().clamp(np.array([[25.2, 6.0], [10.0, 6.0]]))
    assert np.allclose(near, [[25.0, 6.0], [10.0, 6.0]]) and np.allclose(gap, [0.2, 0.0])
    basin, at = world._in_basin(np.array([25.2, 6.0]))  # 20 cm past the wall: still the basin
    assert basin is not None and basin.id == "b" and np.allclose(at, [25.0, 6.0])
    assert world._in_basin(np.array([26.0, 6.0]))[0] is None  # on the deck


def test_a_still_body_keeps_its_time_when_a_crossing_swimmer_swaps_identities_with_it():
    basin = BasinConfig(
        id="b", name="B", outline=[(0, 0), (25, 0), (25, 12), (0, 12)], default_depth=4.0, set_off_time_s=10
    )
    eng = RuleEngine([basin], RuleConfig())
    body, swimmer = _person(1, 0.9, 0.0, 0.05), _person(2, 0.9, 1.0, 0.8)
    swimmer.kf.x[:2] = [13.0, 6.0]
    fired = []
    for k in range(250):
        ts = k * 0.1
        if k == 60:  # a swimmer crosses over the body: the cameras swap the two identities
            body.kf.x[:3], body.motion = [13.0, 6.0, 1.0], 0.8
            swimmer.kf.x[:3], swimmer.motion = [10.0, 6.0, 0.0], 0.05
        for p in (body, swimmer):
            p.history.append((ts, *p.xy, p.kf.speed, p.p_under, p.motion, 0.3, 2, 3.8))
        fired += [(ts, f.person_id) for f in eng.evaluate([body, swimmer], ts)]
    assert len(fired) == 1 and fired[0][1] == 2 and fired[0][0] < 10.0  # not 10 s after the swap
