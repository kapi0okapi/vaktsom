import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vaktsom.api.auth import Accounts
from vaktsom.api.hub import Hub
from vaktsom.api.server import create_app


def engine(tmp_path):
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
    eng.step(time.time())
    return eng


@pytest.fixture
def site(tmp_path):
    accounts = Accounts(tmp_path / "accounts.json")
    accounts.add_user("anne", "badevakt-passord", "badevakt")
    accounts.add_user("lise", "leder-passord", "leder")
    accounts.add_user("lev", "leverandor-passord", "leverandor")
    eng = engine(tmp_path)
    with TestClient(create_app(eng, Hub(), accounts)) as client:
        yield client, eng, accounts


def login(client, name, password):
    return client.post("/api/login", json={"name": name, "password": password})


def test_everything_but_pages_needs_a_login(site):
    client, _, _ = site
    assert client.get("/").status_code == 200  # static page; it sends people to the login page
    for path in ("/api/state", "/api/config", "/api/alarms", "/api/log", "/api/cameras/stup-01/snapshot.jpg"):
        assert client.get(path).status_code == 401, path
    assert login(client, "anne", "wrong-password").status_code == 401
    assert login(client, "anne", "badevakt-passord").status_code == 200
    assert client.get("/api/state").status_code == 200


def test_roles(site):
    client, _, _ = site
    login(client, "anne", "badevakt-passord")
    assert client.get("/api/clips/none").status_code == 403  # clips are for leaders only
    assert client.post("/api/basins/stup/close").status_code == 403
    assert client.get("/api/log").status_code == 403
    login(client, "lise", "leder-passord")
    assert client.get("/api/clips/none").status_code == 404  # allowed, but there is no such clip
    assert client.post("/api/basins/stup/close").status_code == 200
    assert client.post("/api/users", json={"name": "x", "password": "0123456789", "role": "admin"}).status_code == 403
    login(client, "lev", "leverandor-passord")
    assert client.get("/api/state").status_code == 200  # service access to health and logs
    assert client.get("/api/log").status_code == 200
    assert client.get("/api/cameras/stup-01/snapshot.jpg").status_code == 403  # never images
    assert client.post("/api/test/alarm", params={"basin": "stup"}).status_code == 403


def test_actions_are_logged_under_the_signed_in_name(site):
    client, eng, _ = site
    login(client, "anne", "badevakt-passord")
    a = client.post("/api/test/alarm", params={"basin": "stup"}).json()
    client.post(f"/api/alarms/{a['id']}/ack", params={"actor": "someone-else"})
    assert eng.alarms.alarms[a["id"]].ack_by == "anne"
    login(client, "lise", "leder-passord")
    client.get("/api/clips/none")
    kinds = {(r["kind"], r["actor"]) for r in client.get("/api/log").json()}
    assert ("login", "anne") in kinds and ("daily_test", "anne") in kinds


def test_device_token_until_revoked(site):
    client, _, accounts = site
    token = accounts.add_device("Post stup", "panel")
    assert client.post("/api/device-session", json={"token": "guess"}).status_code == 401
    assert client.post("/api/device-session", json={"token": token}).status_code == 200
    assert client.get("/api/me").json()["device"] is True
    assert client.get("/api/state").status_code == 200
    assert client.get("/api/clips/none").status_code == 403
    Accounts(accounts.path).revoke_device("Post stup")  # e.g. from the command line, server running
    assert client.get("/api/state").status_code == 401


def test_repeated_failures_lock_the_name(site):
    client, _, _ = site
    for _ in range(5):
        login(client, "anne", "wrong-password")
    assert login(client, "anne", "badevakt-passord").status_code == 401


def test_websocket_needs_a_session_from_this_site(site):
    client, _, accounts = site
    with client.websocket_connect("/ws?role=panel") as ws, pytest.raises(WebSocketDisconnect) as e:
        ws.receive_json()
    assert e.value.code == 4401
    token = accounts.add_device("Klokke 1", "wearable")
    with client.websocket_connect(f"/ws?role=wearable&token={token}") as ws:
        assert ws.receive_json()["type"] == "state"
    login(client, "anne", "badevakt-passord")
    with (
        pytest.raises(WebSocketDisconnect) as e,
        client.websocket_connect("/ws", headers={"origin": "http://elsewhere.example"}),
    ):
        pass
    assert e.value.code == 4403


def test_revoking_a_device_ends_its_live_connection(site, monkeypatch):
    from vaktsom.api import server

    monkeypatch.setattr(server, "RECHECK_S", 0.05)
    client, _, accounts = site
    token = accounts.add_device("Klokke 2", "wearable")
    with client.websocket_connect(f"/ws?role=wearable&token={token}") as ws:
        assert ws.receive_json()["type"] == "state"
        accounts.revoke_device("Klokke 2")
        with pytest.raises(WebSocketDisconnect) as e:
            while True:
                ws.receive_json()
    assert e.value.code == 4401


def test_wave_mode_and_floor_height_are_kept_across_restarts(site, tmp_path):
    client, eng, _ = site
    login(client, "anne", "badevakt-passord")
    assert client.post("/api/basins/stup/modes/waves", json={"on": True}).status_code == 404  # no wave machine
    assert client.post("/api/basins/bolge/modes/waves", json={"on": True}).status_code == 200
    assert client.post("/api/basins/idrett/floor", json={"depth": 9.0}).status_code == 422
    assert client.post("/api/basins/idrett/floor", json={"depth": 0.8}).status_code == 200
    state = client.get("/api/state").json()
    assert state["modes"] == {"bolge": ["waves"]} and state["floors"]["idrett"] == 0.8
    assert eng.rules.turbulent == {"bolge"}
    again = engine(tmp_path)  # same data directory: a restart
    assert again.rules.turbulent == {"bolge"} and again.basins["idrett"].movable_floor_depth == 0.8
    again.step(time.time() + 2)
    assert again.live_coverage["bolge"].state in ("degraded", "down")
    assert "mode:waves" in again.live_coverage["bolge"].reasons
    login(client, "lise", "leder-passord")
    assert {"mode_on", "floor_set"} <= {r["kind"] for r in client.get("/api/log").json()}


def test_outcome_can_be_set_later_by_a_leader_only(site):
    client, eng, _ = site
    login(client, "anne", "badevakt-passord")
    a = client.post("/api/test/alarm", params={"basin": "stup"}).json()
    assert client.post(f"/api/alarms/{a['id']}/resolve", json={"outcome": "nonsense"}).status_code == 422
    client.post(f"/api/alarms/{a['id']}/resolve", json={"outcome": "test"})
    assert client.post(f"/api/alarms/{a['id']}/outcome", json={"outcome": "drill"}).status_code == 403
    login(client, "lise", "leder-passord")
    r = client.post(f"/api/alarms/{a['id']}/outcome", json={"outcome": "drill", "note": "brannøvelse"})
    assert r.status_code == 200
    row = client.get(f"/api/alarms/{a['id']}").json()
    assert row["outcome"] == "drill" and row["note"] == "brannøvelse"
    assert row["history"][-1]["event"] == "outcome_set" and row["history"][-1]["actor"] == "lise"
    assert client.post("/api/alarms/nope/outcome", json={"outcome": "drill"}).status_code == 404
    assert client.get("/alarms").status_code == 200


def test_demo_mode_needs_no_login(tmp_path):
    with TestClient(create_app(engine(tmp_path), Hub())) as client:
        assert client.get("/api/state").status_code == 200
        assert client.get("/api/me").json()["auth"] is False
