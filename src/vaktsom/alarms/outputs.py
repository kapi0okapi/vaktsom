"""Alarm outputs: lifeguard panels and wearables (WebSocket), network sirens and strobes, relays,
webhooks. Each delivery is logged with its result, so a failed path is visible at once.

Sirens follow the alarm: they start on an alarm (not on a pre-alarm unless configured) and stop
when someone acknowledges it. Wired paths (siren, relay, post panel on a cable) and wireless paths
(wearables) are configured per output; the site configuration guarantees both kinds exist.
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from vaktsom.config import OutputConfig
from vaktsom.storage.db import EventStore

log = logging.getLogger(__name__)

STARTS = {"raised", "upgraded", "escalated", "repeat"}
STOPS = {"acknowledged", "resolved"}


@dataclass
class Delivery:
    ok: bool
    error: str = ""


class Output:
    def __init__(self, cfg: OutputConfig):
        self.cfg = cfg

    def wants(self, alarm, event: str, everywhere: bool) -> bool:
        if self.cfg.basins and alarm.basin_id not in self.cfg.basins and alarm.kind != "system":
            return False
        return everywhere or alarm.severity in self.cfg.severities

    def send(self, alarm, event: str) -> Delivery:  # pragma: no cover - interface
        raise NotImplementedError


class HubOutput(Output):
    """Panels and wearables: pushed over the API's WebSocket hub."""

    def __init__(self, cfg: OutputConfig, hub):
        super().__init__(cfg)
        self.hub = hub

    def send(self, alarm, event: str) -> Delivery:
        n = self.hub.publish(self.cfg.kind, {"type": "alarm", "event": event, "alarm": alarm.public()})
        return Delivery(n > 0, "" if n else "no connected device")


class SirenOutput(Output):
    """AXIS strobe sirens / horn speakers through the VAPIX siren-and-light API."""

    def wants(self, alarm, event: str, everywhere: bool) -> bool:
        if event in STOPS:
            return True
        if self.cfg.options.get("when") == "escalated" and event not in ("escalated", "repeat"):
            return False  # hall sirens: only when nobody has acknowledged in time
        pre = self.cfg.options.get("pre_alarm", False)
        return super().wants(alarm, event, everywhere) and (alarm.severity == "alarm" or pre)

    def send(self, alarm, event: str) -> Delivery:
        method = "stop" if event in STOPS else "start"
        body = json.dumps(
            {
                "apiVersion": "1.0",
                "method": method,
                "params": {"profile": self.cfg.options.get("profile", "Drukningsalarm")},
            }
        ).encode()
        url = self.cfg.url.rstrip("/") + "/axis-cgi/siren_and_light.cgi"
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        opener = urllib.request.build_opener()
        user, pw = self.cfg.options.get("user"), self.cfg.options.get("password")
        if user:
            mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            mgr.add_password(None, url, user, pw)
            opener = urllib.request.build_opener(urllib.request.HTTPDigestAuthHandler(mgr))
        try:
            with opener.open(req, timeout=2.0) as r:
                return Delivery(200 <= r.status < 300)
        except OSError as e:
            return Delivery(False, str(e))


class WebhookOutput(Output):
    def send(self, alarm, event: str) -> Delivery:
        body = json.dumps({"event": event, "alarm": alarm.public()}).encode()
        req = urllib.request.Request(self.cfg.url, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=2.0) as r:
                return Delivery(200 <= r.status < 300)
        except OSError as e:
            return Delivery(False, str(e))


class ModbusCoilOutput(Output):
    """A relay on a Modbus TCP I/O module (light, horn, or an input of the building's alarm system)."""

    _tid = 0

    def send(self, alarm, event: str) -> Delivery:
        host, _, port = (self.cfg.url or "").removeprefix("modbus://").partition(":")
        coil = int(self.cfg.options.get("coil", 0))
        unit = int(self.cfg.options.get("unit", 1))
        value = 0x0000 if event in STOPS else 0xFF00
        ModbusCoilOutput._tid = (ModbusCoilOutput._tid + 1) & 0xFFFF
        pdu = struct.pack(">BHH", 0x05, coil, value)  # write single coil
        adu = struct.pack(">HHHB", ModbusCoilOutput._tid, 0, len(pdu) + 1, unit) + pdu
        try:
            with socket.create_connection((host, int(port or 502)), timeout=2.0) as s:
                s.sendall(adu)
                reply = s.recv(64)
            return Delivery(len(reply) >= 12 and reply[7] == 0x05)
        except OSError as e:
            return Delivery(False, str(e))

    def wants(self, alarm, event: str, everywhere: bool) -> bool:
        return event in STOPS or (super().wants(alarm, event, everywhere) and alarm.severity == "alarm")


class LogOutput(Output):
    def send(self, alarm, event: str) -> Delivery:
        log.warning("ALARM %s %s %s %s at (%.1f, %.1f)", event, alarm.id, alarm.kind, alarm.basin_id, *alarm.xy)
        return Delivery(True)


def build_outputs(configs: list[OutputConfig], hub) -> list[Output]:
    kinds = {"siren": SirenOutput, "webhook": WebhookOutput, "modbus_coil": ModbusCoilOutput, "log": LogOutput}
    out: list[Output] = []
    for c in configs:
        out.append(HubOutput(c, hub) if c.kind in ("panel", "wearable") else kinds[c.kind](c))
    return out


class Dispatcher:
    """Sends each alarm event to every output that wants it, in parallel, and logs the results."""

    def __init__(self, outputs: list[Output], store: EventStore, workers: int = 8):
        self.outputs = outputs
        self.store = store
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="alarm-out")

    def __call__(self, alarm, event: str, everywhere: bool = False) -> None:
        for out in self.outputs:
            if out.wants(alarm, event, everywhere):
                self.pool.submit(self._send, out, alarm, event)

    def _send(self, out: Output, alarm, event: str) -> None:
        result = Delivery(False, "not sent")
        for _ in range(3):
            try:
                result = out.send(alarm, event)
            except Exception as e:  # an output must never take the alarm path down
                result = Delivery(False, repr(e))
            if result.ok:
                break
        self.store.alarm_event(
            alarm.id,
            "delivered" if result.ok else "delivery_failed",
            "system",
            {"output": out.cfg.id, "kind": out.cfg.kind, "wired": out.cfg.wired, "event": event, "error": result.error},
        )
