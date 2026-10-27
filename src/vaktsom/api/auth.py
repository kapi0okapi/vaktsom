"""Who may do what: user accounts with roles, device tokens for panels and watches, sessions.

Roles, from least to most: ``badevakt`` (acknowledge and close alarms, non-detection zones, daily
test), ``leder`` (also alarm clips and exports, closing basins, logs), ``admin`` (users, devices),
``leverandor`` (supplier service access: health and logs, no clips). Passwords are stored as scrypt
hashes; sessions are HMAC-signed cookies (HttpOnly, SameSite=Strict, 12 h for people). Panels and watches
authenticate with long random device tokens that admins can revoke; a device exchanges its token
once for a session that lasts until the token is revoked. Every login, failed login and access to
clips goes to the operational log.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROLES = ["badevakt", "leder", "admin"]
RANK = {"leverandor": 0, "badevakt": 1, "leder": 2, "admin": 3}
SESSION_S = 12 * 3600
DEVICE_SESSION_S = 400 * 24 * 3600  # the longest cookie browsers keep; renewed whenever the device loads


@dataclass
class Principal:
    name: str
    role: str
    device: str = ""  # panels and watches: the key of their device token

    def at_least(self, role: str) -> bool:
        if self.role == "leverandor":
            return role == "leverandor"
        return RANK.get(self.role, -1) >= RANK[role]


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt, digest = stored.split("$")
    except ValueError:
        return False
    test = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1, dklen=32)
    return hmac.compare_digest(test.hex(), digest)


class Accounts:
    """Users and device tokens in a JSON file (0600), the session secret beside it."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.data = {"users": {}, "devices": {}}
        self._mtime = 0
        self._load()
        secret_path = self.path.with_suffix(".secret")
        if not secret_path.exists():
            secret_path.write_bytes(secrets.token_bytes(32))
            secret_path.chmod(0o600)
        self.secret = secret_path.read_bytes()
        self._failures: dict[str, list[float]] = {}

    def _load(self) -> None:
        """Re-reads the file when another process has changed it, so a device revoked from the
        command line is locked out of the running server at once."""
        try:
            mtime = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return
        if mtime != self._mtime:
            self.data = json.loads(self.path.read_text())
            self._mtime = mtime

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.chmod(0o600)
        tmp.replace(self.path)
        self._mtime = self.path.stat().st_mtime_ns

    # users -------------------------------------------------------------------------------------------
    def add_user(self, name: str, password: str, role: str) -> None:
        if role not in RANK:
            raise ValueError(f"unknown role {role}")
        if len(password) < 10:
            raise ValueError("password must have at least 10 characters")
        with self._lock:
            self._load()
            self.data["users"][name] = {"hash": hash_password(password), "role": role, "created": time.time()}
            self._save()

    def remove_user(self, name: str) -> None:
        with self._lock:
            self._load()
            self.data["users"].pop(name, None)
            self._save()

    def check(self, name: str, password: str) -> Principal | None:
        self._load()
        now = time.time()
        recent = [t for t in self._failures.get(name, []) if now - t < 300]
        if len(recent) >= 5:  # five failures in five minutes: wait
            return None
        user = self.data["users"].get(name)
        if user is None:
            hash_password(password)  # same cost as a real check, so timing does not reveal names
        elif verify_password(password, user["hash"]):
            self._failures.pop(name, None)
            return Principal(name, user["role"])
        self._failures[name] = recent + [now]
        return None

    # devices -----------------------------------------------------------------------------------------
    def add_device(self, name: str, kind: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._load()
            self.data["devices"][hashlib.sha256(token.encode()).hexdigest()] = {
                "name": name,
                "kind": kind,
                "created": time.time(),
            }
            self._save()
        return token  # shown once, stored only as a hash

    def revoke_device(self, name: str) -> None:
        with self._lock:
            self._load()
            self.data["devices"] = {k: v for k, v in self.data["devices"].items() if v["name"] != name}
            self._save()

    def device(self, token: str) -> Principal | None:
        self._load()
        key = hashlib.sha256(token.encode()).hexdigest()
        d = self.data["devices"].get(key)
        return Principal(d["name"], "badevakt", device=key) if d else None

    def overview(self) -> dict:
        self._load()
        users = [{"name": n, "role": u["role"]} for n, u in self.data["users"].items()]
        devices = [{"name": d["name"], "kind": d["kind"]} for d in self.data["devices"].values()]
        return {"users": users, "devices": devices}

    # sessions ----------------------------------------------------------------------------------------
    def session(self, principal: Principal) -> tuple[str, int]:
        """A signed cookie value and how long it lives."""
        life = DEVICE_SESSION_S if principal.device else SESSION_S
        payload = json.dumps({"n": principal.name, "d": principal.device, "e": int(time.time()) + life})
        sig = hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()
        return payload.encode().hex() + "." + sig, life

    def from_session(self, cookie: str | None) -> Principal | None:
        if not cookie or "." not in cookie:
            return None
        body, sig = cookie.split(".", 1)
        try:
            payload = bytes.fromhex(body).decode()
        except ValueError:
            return None
        if not hmac.compare_digest(hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest(), sig):
            return None
        data = json.loads(payload)
        if data["e"] < time.time():
            return None
        self._load()
        if data["d"]:  # a device stays signed in until its token is revoked
            d = self.data["devices"].get(data["d"])
            return Principal(d["name"], "badevakt", device=data["d"]) if d else None
        user = self.data["users"].get(data["n"])  # the role is read now, so changes apply at once
        return Principal(data["n"], user["role"]) if user else None
