"""Alarm clips: a few seconds of video around each alarm, encrypted, and deleted beyond recovery.

No continuous recording. Each camera keeps a short in-memory ring of JPEG frames (privacy masks
already applied). When an alarm is raised, the frames from ``pre_s`` before its onset to ``post_s``
after it are written as one file, encrypted with AES-256-GCM under a key used for that clip only.
Keys live in a separate key store with SQLite secure_delete. Deleting a clip destroys its key
(crypto-shredding), so the clip cannot be read from any copy or backup afterwards, backups
included.
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"VKCLIP1\0"


def apply_privacy_masks(image: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        return image
    out = image.copy()
    for poly in masks:
        cv2.fillPoly(out, [np.asarray(poly, np.int32)], (0, 0, 0))
    return out


class FrameRing:
    """Last ``seconds`` of one camera as JPEG bytes, decimated to ``fps`` and ``width`` pixels."""

    def __init__(self, seconds: float = 30.0, fps: float = 5.0, width: int = 1280, quality: int = 80):
        self.seconds, self.period, self.width, self.quality = seconds, 1.0 / fps, width, quality
        self.frames: deque[tuple[float, bytes]] = deque()
        self._last = -1e9
        self._lock = threading.Lock()

    def push(self, ts: float, image: np.ndarray, masks: list[np.ndarray] | None = None) -> None:
        if ts - self._last < self.period:
            return
        self._last = ts
        img = apply_privacy_masks(image, masks or [])
        h, w = img.shape[:2]
        if w > self.width:
            img = cv2.resize(img, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self._lock:
            self.frames.append((ts, buf.tobytes()))
            while self.frames and ts - self.frames[0][0] > self.seconds:
                self.frames.popleft()

    def between(self, t0: float, t1: float) -> list[tuple[float, bytes]]:
        with self._lock:
            return [(t, b) for t, b in self.frames if t0 <= t <= t1]


@dataclass
class PendingClip:
    clip_id: str
    alarm_id: str
    cameras: list[str]
    t0: float
    t1: float
    frames: dict[str, list[tuple[float, bytes]]] = field(default_factory=dict)


class ClipStore:
    def __init__(self, root: Path | str, retention_days: int = 7):
        self.root = Path(root)
        (self.root / "clips").mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._keys = sqlite3.connect(self.root / "keys.db", check_same_thread=False, isolation_level=None)
        self._keys.execute("PRAGMA secure_delete=ON")  # deleted keys are overwritten on disk
        self._keys.execute("PRAGMA journal_mode=DELETE")  # no WAL copies of key pages
        self._keys.execute(
            "CREATE TABLE IF NOT EXISTS clips (clip_id TEXT PRIMARY KEY, alarm_id TEXT, created REAL, "
            "cameras TEXT, key BLOB)"
        )
        self.pending: list[PendingClip] = []

    def start(
        self,
        alarm_id: str,
        cameras: list[str],
        onset: float,
        raised_at: float,
        pre_s: float,
        post_s: float,
        rings: dict[str, FrameRing],
    ) -> str:
        clip_id = f"{alarm_id}-{int(raised_at)}"
        clip = PendingClip(clip_id, alarm_id, cameras, onset - pre_s, raised_at + post_s)
        for cam in cameras:
            if cam in rings:
                clip.frames[cam] = rings[cam].between(clip.t0, raised_at)
        self.pending.append(clip)
        return clip_id

    def poll(self, now: float, rings: dict[str, FrameRing]) -> list[str]:
        """Add post-roll frames; finalise clips whose window has ended. Returns finished clip ids."""
        done = []
        for clip in list(self.pending):
            for cam in clip.cameras:
                if cam not in rings:
                    continue
                have = {t for t, _ in clip.frames.get(cam, [])}
                extra = [(t, b) for t, b in rings[cam].between(clip.t0, clip.t1) if t not in have]
                clip.frames.setdefault(cam, []).extend(extra)
            if now >= clip.t1:
                self._write(clip)
                self.pending.remove(clip)
                done.append(clip.clip_id)
        return done

    def _write(self, clip: PendingClip) -> None:
        header, chunks = {"clip_id": clip.clip_id, "alarm_id": clip.alarm_id, "frames": []}, []
        offset = 0
        for cam, frames in clip.frames.items():
            for t, data in sorted(frames):
                header["frames"].append({"camera": cam, "ts": t, "offset": offset, "size": len(data)})
                chunks.append(data)
                offset += len(data)
        head = json.dumps(header).encode()
        plain = struct.pack("<I", len(head)) + head + b"".join(chunks)
        key = AESGCM.generate_key(bit_length=256)
        nonce = os.urandom(12)
        blob = MAGIC + nonce + AESGCM(key).encrypt(nonce, plain, clip.clip_id.encode())
        # key first: retention removes clip files that have no key
        with self._lock:
            self._keys.execute(
                "INSERT OR REPLACE INTO clips VALUES (?,?,?,?,?)",
                (clip.clip_id, clip.alarm_id, time.time(), json.dumps(clip.cameras), key),
            )
        path = self.root / "clips" / f"{clip.clip_id}.bin"
        tmp = path.with_suffix(".part")
        tmp.write_bytes(blob)
        tmp.rename(path)

    def read(self, clip_id: str) -> tuple[dict, dict[str, list[tuple[float, bytes]]]]:
        """Decrypt a clip: (header, camera -> [(ts, jpeg bytes)]). KeyError once shredded."""
        with self._lock:
            row = self._keys.execute("SELECT key FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
        if row is None:
            raise KeyError(clip_id)
        blob = (self.root / "clips" / f"{clip_id}.bin").read_bytes()
        if not blob.startswith(MAGIC):
            raise ValueError("not a clip file")
        nonce, ct = blob[len(MAGIC) : len(MAGIC) + 12], blob[len(MAGIC) + 12 :]
        plain = AESGCM(row[0]).decrypt(nonce, ct, clip_id.encode())
        n = struct.unpack("<I", plain[:4])[0]
        header = json.loads(plain[4 : 4 + n])
        body = plain[4 + n :]
        frames: dict[str, list[tuple[float, bytes]]] = {}
        for f in header["frames"]:
            frames.setdefault(f["camera"], []).append((f["ts"], body[f["offset"] : f["offset"] + f["size"]]))
        return header, frames

    def clips_for(self, alarm_id: str) -> list[str]:
        with self._lock:
            return [r[0] for r in self._keys.execute("SELECT clip_id FROM clips WHERE alarm_id=?", (alarm_id,))]

    def shred(self, clip_id: str) -> None:
        with self._lock:
            self._keys.execute("DELETE FROM clips WHERE clip_id=?", (clip_id,))
        (self.root / "clips" / f"{clip_id}.bin").unlink(missing_ok=True)

    def enforce_retention(self, now: float | None = None) -> list[str]:
        cutoff = (now or time.time()) - self.retention_days * 86400
        with self._lock:
            old = [r[0] for r in self._keys.execute("SELECT clip_id FROM clips WHERE created < ?", (cutoff,))]
        for clip_id in old:
            self.shred(clip_id)
        # files without a key (e.g. restored from a backup) are unreadable; remove them as well
        with self._lock:
            known = {r[0] for r in self._keys.execute("SELECT clip_id FROM clips")}
        for path in (self.root / "clips").glob("*.bin"):
            if path.stem not in known:
                path.unlink(missing_ok=True)
        return old
