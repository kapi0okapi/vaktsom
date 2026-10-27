"""Live camera sources: one decoding thread per camera, always handing over the newest frame.

RTSP over TCP from ONVIF cameras (H.264/H.265), decoded with FFmpeg (PyAV; on the site server the
same streams decode on the GPU's NVDEC). A stream that stalls reconnects with back-off; the health
monitor sees the missing frames and reports the camera, so a dead stream is never silent. Video
files work the same way (looping), for tests and for replaying recorded incidents.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from vaktsom.config import SiteConfig
from vaktsom.perception.types import Frame
from vaktsom.runtime.engine import TickResult

log = logging.getLogger(__name__)


class StreamReader(threading.Thread):
    def __init__(self, camera_id: str, url: str, fps: float, loop: bool = False):
        super().__init__(name=f"cam-{camera_id}", daemon=True)
        self.camera_id, self.url, self.fps, self.loop = camera_id, url, fps, loop
        self.latest: Frame | None = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.n = 0
        self.errors = 0

    def run(self) -> None:
        import av  # imported here: on macOS PyAV and OpenCV ship clashing FFmpeg builds

        backoff = 1.0
        while not self.stop_event.is_set():
            try:
                opts = {"rtsp_transport": "tcp", "stimeout": "5000000"} if self.url.startswith("rtsp") else {}
                with av.open(self.url, options=opts, timeout=5.0) as container:
                    stream = container.streams.video[0]
                    stream.thread_type = "AUTO"
                    period = 1.0 / self.fps
                    last = 0.0
                    file_t0 = time.time()
                    for packet_frame in container.decode(stream):
                        if self.stop_event.is_set():
                            return
                        now = time.time()
                        if not self.url.startswith("rtsp"):  # files play at their own pace
                            pts = float(packet_frame.time or 0.0)
                            delay = file_t0 + pts - now
                            if delay > 0:
                                time.sleep(delay)
                            now = time.time()
                        if now - last < period * 0.9:
                            continue
                        last = now
                        img = packet_frame.to_ndarray(format="bgr24")
                        self.n += 1
                        with self.lock:
                            self.latest = Frame(self.camera_id, now, img, self.n)
                        backoff = 1.0
                if not self.loop:
                    return
            except Exception as e:  # network, codec, camera reboot: log and try again
                self.errors += 1
                log.warning("camera %s: %s (retry in %.0f s)", self.camera_id, e, backoff)
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 30.0)

    def take(self) -> Frame | None:
        with self.lock:
            f, self.latest = self.latest, None
            return f


class RtspSource:
    """The engine's source for real cameras (and video files)."""

    def __init__(self, cfg: SiteConfig, loop_files: bool = True):
        self.readers = {c.id: StreamReader(c.id, c.url, c.fps, loop=loop_files) for c in cfg.cameras}
        for r in self.readers.values():
            r.start()

    def tick(self, ts: float) -> TickResult:
        frames = {}
        for cid, r in self.readers.items():
            f = r.take()
            frames[cid] = f if f is not None else Frame(cid, ts, None)
        return TickResult(ts, frames)

    def stop(self) -> None:
        for r in self.readers.values():
            r.stop_event.set()


def blank(size: tuple[int, int]) -> np.ndarray:
    return np.zeros((size[1], size[0], 3), np.uint8)


class FolderSource:
    """Pre-recorded frames (one folder of JPEGs per camera), one frame per tick: deterministic replays
    of test videos, independent of how fast the machine is."""

    def __init__(self, folders: dict[str, str], start_index: int = 0):
        import glob

        self.files = {cid: sorted(glob.glob(f"{folder}/*.jpg")) for cid, folder in folders.items()}
        self.k = start_index

    @property
    def length(self) -> int:
        return min(len(v) for v in self.files.values())

    def tick(self, ts: float) -> TickResult:
        import cv2

        frames = {}
        for cid, files in self.files.items():
            img = cv2.imread(files[self.k]) if self.k < len(files) else None
            frames[cid] = Frame(cid, ts, img, self.k)
        self.k += 1
        return TickResult(ts, frames)
