"""HTTP and WebSocket API, and the three web clients: dashboard, lifeguard post panel, wearable."""

from __future__ import annotations

import asyncio
import io
import json
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vaktsom.api.auth import Accounts, Principal
from vaktsom.api.hub import Hub
from vaktsom.runtime.engine import Engine
from vaktsom.storage.report import period_report

WEB = Path(__file__).resolve().parent.parent / "web"
STATE_COLOURS = {"alarm": (40, 40, 230), "under": (230, 160, 40), "normal": (90, 200, 90), "lost": (160, 160, 160)}


COOKIE = "vaktsom_session"
RECHECK_S = 5.0  # open live connections re-check their sign-in this often


OUTCOMES = ("real", "false_alarm", "drill", "test", "other")


class ResolveBody(BaseModel):
    outcome: Literal["real", "false_alarm", "drill", "test", "other"]
    note: str = Field(default="", max_length=500)


class ZoneBody(BaseModel):
    basin: str
    polygon: list[tuple[float, float]]
    minutes: float
    reason: str = ""


class LoginBody(BaseModel):
    name: str
    password: str


class UserBody(BaseModel):
    name: str
    password: str
    role: str


class DeviceBody(BaseModel):
    name: str
    kind: str  # panel | wearable


class TokenBody(BaseModel):
    token: str


class FloorBody(BaseModel):
    depth: float


class ModeBody(BaseModel):
    on: bool


def overlay(engine: Engine, camera_id: str, image: np.ndarray) -> np.ndarray:
    """Tracked people drawn on a camera frame: green normal, amber submerged, red in an alarm."""
    out = image.copy()
    cam_cfg = next(c for c in engine.cfg.cameras if c.id == camera_id)
    sx, sy = out.shape[1] / cam_cfg.size[0], out.shape[0] / cam_cfg.size[1]
    alarm_people = {a.person_id for a in engine.alarms.open_alarms()}
    owner = {}
    for p in engine.world.persons.values():
        tid = p.links.get(camera_id)
        if tid is not None:
            owner[tid] = p
    frame = engine.latest.get(camera_id)
    now = frame.ts if frame is not None else time.time()
    for t in engine.trackers[camera_id].tracks:
        if not t.confirmed or now - t.last_ts > 1.0:
            continue
        p = owner.get(t.id)
        state = "normal"
        if p is not None and p.id in alarm_people:
            state = "alarm"
        elif t.p_under >= 0.6:
            state = "under"
        x0, y0, x1, y1 = (t.box * [sx, sy, sx, sy]).astype(int)
        col = STATE_COLOURS[state]
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2 if state != "alarm" else 3, cv2.LINE_AA)
        if p is not None:
            label = f"{p.id}"
            cv2.putText(out, label, (x0, max(y0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out


def create_app(engine: Engine, hub: Hub, accounts: Accounts | None = None) -> FastAPI:
    """``accounts=None`` runs without login, for demonstrations on simulated cameras only."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        hub.attach(asyncio.get_running_loop())
        yield

    app = FastAPI(title="Vaktsom", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
    cfg = engine.cfg

    def identify(cookie: str | None, authorization: str | None, token: str | None = None) -> Principal | None:
        """Browsers carry a session cookie; native watch apps may send their device token instead."""
        if accounts is None:
            return Principal("demo", "admin")
        who = accounts.from_session(cookie)
        if who is None and authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
        if who is None and token:
            who = accounts.device(token)
        return who

    def need(role: str, *also: str):
        """Route dependency: the caller, signed in with ``role`` or higher, or with one of ``also``."""

        def check(request: Request) -> Principal:
            who = identify(request.cookies.get(COOKIE), request.headers.get("authorization"))
            if who is None:
                raise HTTPException(401, "login required")
            if not (who.at_least(role) or who.role in also):
                raise HTTPException(403, "not allowed for this role")
            return who

        return Depends(check)

    signed_in = need("leverandor")
    viewer = need("badevakt", "leverandor")  # the supplier reads state for service, never images or clips
    lifeguard = need("badevakt")
    manager = need("leder")
    service = need("leder", "leverandor")
    admin = need("admin")

    def set_session(request: Request, response: Response, who: Principal) -> None:
        value, life = accounts.session(who)
        response.set_cookie(
            COOKIE, value, max_age=life, httponly=True, samesite="strict", secure=request.url.scheme == "https"
        )

    # pages: static, all data behind the API -------------------------------------------------------------
    @app.get("/")
    async def index():
        return FileResponse(WEB / "index.html")

    @app.get("/panel")
    async def panel():
        return FileResponse(WEB / "panel.html")

    @app.get("/watch")
    async def watch():
        return FileResponse(WEB / "watch.html")

    @app.get("/login")
    async def login_page():
        return FileResponse(WEB / "login.html")

    @app.get("/report")
    async def report_page():
        return FileResponse(WEB / "report.html")

    @app.get("/alarms")
    async def history_page():
        return FileResponse(WEB / "history.html")

    # sign-in ---------------------------------------------------------------------------------------------
    @app.post("/api/login")
    async def login(body: LoginBody, request: Request, response: Response):
        if accounts is None:
            return {"name": "demo", "role": "admin"}
        who = accounts.check(body.name, body.password)
        if who is None:
            engine.store.log("login_failed", body.name[:64])
            raise HTTPException(401, "wrong name or password")
        set_session(request, response, who)
        engine.store.log("login", who.name, {"role": who.role})
        return {"name": who.name, "role": who.role}

    @app.post("/api/device-session")
    async def device_session(body: TokenBody, request: Request, response: Response):
        """A panel or watch opened with ``#token=...`` trades the token for a session cookie once."""
        if accounts is None:
            return {"name": "demo", "role": "admin"}
        who = accounts.device(body.token)
        if who is None:
            engine.store.log("device_rejected", "unknown")
            raise HTTPException(401, "unknown or revoked device")
        set_session(request, response, who)
        engine.store.log("device_signed_in", who.name)
        return {"name": who.name, "role": who.role}

    @app.post("/api/logout")
    async def logout(response: Response, who: Principal = signed_in):
        response.delete_cookie(COOKIE)
        engine.store.log("logout", who.name)
        return {"ok": True}

    @app.get("/api/me")
    async def me(request: Request, response: Response, who: Principal = signed_in):
        if who.device and accounts is not None:
            set_session(request, response, who)  # devices stay signed in while they are in use
        return {"name": who.name, "role": who.role, "device": bool(who.device), "auth": accounts is not None}

    if accounts is not None:

        @app.get("/api/accounts")
        async def list_accounts(who: Principal = admin):
            return accounts.overview()

        @app.post("/api/users")
        async def add_user(body: UserBody, who: Principal = admin):
            try:
                accounts.add_user(body.name, body.password, body.role)
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
            engine.store.log("user_added", who.name, {"user": body.name, "role": body.role})
            return {"ok": True}

        @app.delete("/api/users/{name}")
        async def remove_user(name: str, who: Principal = admin):
            accounts.remove_user(name)
            engine.store.log("user_removed", who.name, {"user": name})
            return {"ok": True}

        @app.post("/api/device-tokens")
        async def add_device(body: DeviceBody, who: Principal = admin):
            token = accounts.add_device(body.name, body.kind)
            engine.store.log("device_added", who.name, {"device": body.name, "kind": body.kind})
            return {"name": body.name, "token": token}

        @app.delete("/api/device-tokens/{name}")
        async def revoke_device(name: str, who: Principal = admin):
            accounts.revoke_device(name)
            engine.store.log("device_revoked", who.name, {"device": name})
            return {"ok": True}

    app.mount("/static", StaticFiles(directory=WEB), name="static")

    # configuration and state -----------------------------------------------------------------------
    @app.get("/api/config", dependencies=[viewer])
    async def config():
        return {
            "site": cfg.site,
            "language": cfg.language,
            "basins": [
                {
                    "id": b.id,
                    "name": b.name,
                    "outline": b.outline,
                    "depth": b.default_depth,
                    "regions": [r.model_dump() for r in b.regions],
                    "set_off_time_s": b.set_off_time_s,
                    "covered": b.covered,
                    "modes": b.turbulent_when,
                    "movable_floor": b.movable_floor,
                }
                for b in cfg.basins
            ],
            "cameras": [
                {"id": c.id, "name": c.name, "basins": c.basins, "position": c.position, "target": c.target}
                for c in cfg.cameras
            ],
            "outputs": [{"id": o.id, "kind": o.kind, "wired": o.wired} for o in cfg.outputs],
            "escalation": cfg.escalation.model_dump(),
            "retention": cfg.retention.model_dump(),
            "design_coverage": engine.coverage.design_report(),
        }

    @app.get("/api/state", dependencies=[viewer])
    async def state():
        return engine.snapshot()

    @app.get("/api/devices", dependencies=[service])
    async def devices():
        return hub.devices()

    # alarms ------------------------------------------------------------------------------------------
    @app.get("/api/alarms", dependencies=[viewer])
    async def alarms(since: float = 0.0, limit: int = 500):
        return engine.store.alarms(since, limit)

    @app.get("/api/alarms/{alarm_id}", dependencies=[viewer])
    async def alarm(alarm_id: str):
        rows = [a for a in engine.store.alarms(0, 5000) if a["id"] == alarm_id]
        if not rows:
            raise HTTPException(404)
        return {**rows[0], "history": engine.store.alarm_history(alarm_id), "clips": engine.clips.clips_for(alarm_id)}

    @app.post("/api/alarms/{alarm_id}/ack")
    async def ack(alarm_id: str, who: Principal = lifeguard):
        if alarm_id not in engine.alarms.alarms:
            raise HTTPException(404)
        return engine.alarms.acknowledge(alarm_id, who.name).public()

    @app.post("/api/alarms/{alarm_id}/resolve")
    async def resolve(alarm_id: str, body: ResolveBody, who: Principal = lifeguard):
        if alarm_id not in engine.alarms.alarms:
            raise HTTPException(404)
        return engine.alarms.resolve(alarm_id, who.name, body.outcome, body.note).public()

    @app.post("/api/alarms/{alarm_id}/outcome")
    async def set_outcome(alarm_id: str, body: ResolveBody, who: Principal = manager):
        """Outcome set or corrected after the fact, for the monthly review."""
        live = engine.alarms.alarms.get(alarm_id)
        if live is not None and live.open:
            return engine.alarms.resolve(alarm_id, who.name, body.outcome, body.note).public()
        if not engine.store.set_outcome(alarm_id, body.outcome, body.note, who.name):
            raise HTTPException(404)
        return {"ok": True}

    @app.post("/api/test/alarm")
    async def test_alarm(basin: str, who: Principal = lifeguard):
        """Daily test (ISO 20380 6.1): a test alarm on every output; deliveries are logged."""
        b = engine.basins.get(basin)
        if b is None:
            raise HTTPException(404)
        centre = b.outline.mean(0)
        a = engine.alarms.raise_manual("test", "alarm", basin, centre, who.name, {"label": f"{b.name}, test"})
        engine.store.log("daily_test", who.name, {"alarm": a.id, "basin": basin})
        return a.public()

    # clips ---------------------------------------------------------------------------------------------
    @app.get("/api/clips/{clip_id}")
    async def clip_index(clip_id: str, who: Principal = manager):
        try:
            header, frames = engine.clips.read(clip_id)
        except KeyError as e:
            raise HTTPException(404, "clip deleted or not finished") from e
        engine.store.log("clip_viewed", who.name, {"clip": clip_id})
        return {
            "clip_id": clip_id,
            "alarm_id": header["alarm_id"],
            "cameras": {cam: [t for t, _ in fr] for cam, fr in frames.items()},
        }

    @app.get("/api/clips/{clip_id}/{camera}/{index}.jpg", dependencies=[manager])
    async def clip_frame(clip_id: str, camera: str, index: int):
        try:
            _, frames = engine.clips.read(clip_id)
            return Response(frames[camera][index][1], media_type="image/jpeg")
        except (KeyError, IndexError) as e:
            raise HTTPException(404) from e

    @app.get("/api/clips/{clip_id}.zip")
    async def clip_zip(clip_id: str, who: Principal = manager):
        """Export for an incident review (logged: who exported what, ISO 4.4 and the DPA)."""
        try:
            header, frames = engine.clips.read(clip_id)
        except KeyError as e:
            raise HTTPException(404) from e
        engine.store.log("clip_exported", who.name, {"clip": clip_id})
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("alarm.json", json.dumps(header, indent=2))
            for cam, fr in frames.items():
                for k, (t, data) in enumerate(fr):
                    z.writestr(f"{cam}/{k:04d}_{t:.2f}.jpg", data)
        return Response(
            buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{clip_id}.zip"'},
        )

    # cameras -------------------------------------------------------------------------------------------
    def _jpeg(camera_id: str, draw: bool, width: int) -> bytes | None:
        frame = engine.latest.get(camera_id)
        if frame is None or frame.image is None:
            return None
        img = overlay(engine, camera_id, frame.image) if draw else frame.image
        if img.shape[1] > width:
            img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
        return buf.tobytes() if ok else None

    @app.get("/api/cameras/{camera_id}/snapshot.jpg", dependencies=[lifeguard])
    async def snapshot(camera_id: str, overlay: bool = True, width: int = 960):
        data = _jpeg(camera_id, overlay, width)
        if data is None:
            raise HTTPException(404)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{camera_id}/mjpeg", dependencies=[lifeguard])
    async def mjpeg(request: Request, camera_id: str, overlay: bool = True, width: int = 640, fps: float = 5.0):
        if camera_id not in engine.trackers:
            raise HTTPException(404)
        credentials = (request.cookies.get(COOKIE), request.headers.get("authorization"))

        async def frames():
            checked = time.monotonic()
            while True:
                if time.monotonic() - checked > RECHECK_S:
                    if identify(*credentials) is None:
                        return  # signed out or revoked: the stream ends
                    checked = time.monotonic()
                if fps >= 4 and hasattr(engine.source, "interest"):
                    engine.source.interest(camera_id)
                data = _jpeg(camera_id, overlay, width)
                if data is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                await asyncio.sleep(1.0 / fps)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    # zones and basins ---------------------------------------------------------------------------------
    @app.post("/api/zones")
    async def add_zone(body: ZoneBody, who: Principal = lifeguard):
        if body.basin not in engine.basins:
            raise HTTPException(404)
        if not 1 <= body.minutes <= 240:
            raise HTTPException(422, "1-240 minutes")
        z = engine.add_zone(body.basin, body.polygon, body.minutes, who.name, body.reason)
        return {"id": z.id, "until": z.until}

    @app.delete("/api/zones/{zone_id}")
    async def remove_zone(zone_id: str, who: Principal = lifeguard):
        engine.remove_zone(zone_id, who.name)
        return {"ok": True}

    @app.post("/api/basins/{basin_id}/floor")
    async def set_floor(basin_id: str, body: FloorBody, who: Principal = lifeguard):
        b = next((b for b in cfg.basins if b.id == basin_id and b.movable_floor), None)
        if b is None:
            raise HTTPException(404)
        if not 0 <= body.depth <= b.default_depth:
            raise HTTPException(422, f"0-{b.default_depth} m")
        engine.set_floor(basin_id, round(body.depth, 2), who.name)
        return {"ok": True}

    @app.post("/api/basins/{basin_id}/modes/{mode}")
    async def set_mode(basin_id: str, mode: str, body: ModeBody, who: Principal = lifeguard):
        if basin_id not in engine.basins or mode not in cfg.basin(basin_id).turbulent_when:
            raise HTTPException(404)
        engine.set_mode(basin_id, mode, body.on, who.name)
        return {"ok": True}

    @app.post("/api/basins/{basin_id}/{action}")
    async def basin_action(basin_id: str, action: str, who: Principal = manager):
        if basin_id not in engine.basins or action not in ("open", "close"):
            raise HTTPException(404)
        engine.set_basin_open(basin_id, action == "open", who.name)
        return {"ok": True}

    # logs ----------------------------------------------------------------------------------------------
    @app.get("/api/report", dependencies=[service])
    async def report(start: float | None = None, end: float | None = None):
        """Alarms, outcomes, false alarms per basin per day (ISO 20380 4.3.4), daily tests, coverage."""
        end = end or time.time()
        return period_report(engine.store, cfg, start or end - 30 * 86400, end)

    @app.get("/api/log", dependencies=[service])
    async def oplog(since: float = 0.0):
        return engine.store.oplog(since)

    # live ------------------------------------------------------------------------------------------------
    @app.websocket("/ws")
    async def ws(socket: WebSocket, role: str = "dashboard", name: str = "", token: str | None = None):
        origin = socket.headers.get("origin")
        if origin and urlsplit(origin).netloc != socket.headers.get("host"):
            await socket.close(code=4403)  # another site's page must not borrow the lifeguards' session
            return
        credentials = (socket.cookies.get(COOKIE), socket.headers.get("authorization"), token)
        who = identify(*credentials)
        await socket.accept()
        if who is None or not who.at_least("badevakt"):
            await socket.close(code=4401)  # after accept, so the page sees the code and signs in again
            return
        if accounts is not None:
            name = who.name
        client = hub.connect(role if role in ("dashboard", "panel", "wearable") else "dashboard", name)
        engine.store.log("device_connected", name or role, {"role": role})
        await socket.send_text(json.dumps({"type": "state", **engine.snapshot()}, default=float))

        async def sender():
            try:
                while True:
                    await socket.send_text(await client.queue.get())
            except RuntimeError:  # closed by recheck()
                pass

        async def recheck():  # a revoked device or an expired session also loses the live feed
            while identify(*credentials) is not None:
                await asyncio.sleep(RECHECK_S)
            await socket.close(code=4401)

        tasks = [asyncio.create_task(sender()), asyncio.create_task(recheck())]
        try:
            while True:
                msg = json.loads(await socket.receive_text())
                client.last_seen = time.time()
                kind, alarm_id = msg.get("type"), msg.get("alarm_id")
                if kind == "ack" and alarm_id in engine.alarms.alarms:
                    engine.alarms.acknowledge(alarm_id, name or role)
                elif kind == "received" and alarm_id:
                    engine.store.alarm_event(alarm_id, "received", name or role, {"role": role})
                elif kind == "resolve" and alarm_id in engine.alarms.alarms and msg.get("outcome", "other") in OUTCOMES:
                    engine.alarms.resolve(
                        alarm_id, name or role, msg.get("outcome", "other"), str(msg.get("note", ""))[:500]
                    )
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            hub.disconnect(client)
            engine.store.log("device_disconnected", name or role, {"role": role})

    return app
