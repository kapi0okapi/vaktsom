"""Live updates to browsers, panels and wearables over WebSocket, fed from the engine thread."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Client:
    id: int
    role: str  # dashboard | panel | wearable
    name: str
    queue: asyncio.Queue
    connected: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


# which message topics reach which kind of client
TOPICS = {
    "state": {"dashboard", "panel", "wearable"},
    "panel": {"panel", "dashboard"},
    "wearable": {"wearable", "dashboard"},
    "event": {"dashboard"},
}


class Hub:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: dict[int, Client] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._last_state = 0.0
        self.state_hz = 5.0  # browsers do not need every engine tick

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def connect(self, role: str, name: str) -> Client:
        c = Client(next(self._ids), role, name, asyncio.Queue(maxsize=200))
        with self._lock:
            self.clients[c.id] = c
        return c

    def disconnect(self, client: Client) -> None:
        with self._lock:
            self.clients.pop(client.id, None)

    def devices(self) -> list[dict]:
        with self._lock:
            return [
                {"id": c.id, "role": c.role, "name": c.name, "connected": c.connected, "last_seen": c.last_seen}
                for c in self.clients.values()
            ]

    def publish(self, topic: str, message: dict) -> int:
        """Thread-safe. Returns how many clients the message was queued for."""
        if self.loop is None:
            return 0
        if topic == "state":
            now = time.time()
            if now - self._last_state < 1.0 / self.state_hz:
                return 0
            self._last_state = now
            message = {"type": "state", **message}
        roles = TOPICS.get(topic, {"dashboard"})
        payload = json.dumps(message, default=float)
        with self._lock:
            targets = [c for c in self.clients.values() if c.role in roles]
        for c in targets:
            self.loop.call_soon_threadsafe(_offer, c.queue, payload)
        return len(targets)


def _offer(queue: asyncio.Queue, payload: str) -> None:
    if queue.full():  # a slow client loses old state, never the newest message
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    queue.put_nowait(payload)
