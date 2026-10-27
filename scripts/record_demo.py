"""Record the demo: dashboard, lifeguard post panel and a lifeguard's watch, side by side.

Needs a running server with the story scenario (``vaktsom serve --sim demo_story --no-auth``). A
scripted lifeguard acknowledges on the watch a few seconds after each alarm reaches it, and closes
the incident on the dashboard at the end. The three browser recordings are composed into one video.

    python scripts/record_demo.py --url http://127.0.0.1:8096 --seconds 80 --out demo.mp4
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

LAYOUT = {"dashboard": (1440, 900), "panel": (1280, 720), "watch": (420, 420)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8096")
    ap.add_argument("--seconds", type=float, default=80)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ack-delay", type=float, default=4.0)
    ap.add_argument("--resolve-after", type=float, default=10.0)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="vaktsom-rec-"))
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"]
        )
        pages = {}
        for name, (w, h) in LAYOUT.items():
            ctx = browser.new_context(
                viewport={"width": w, "height": h},
                record_video_dir=str(tmp / name),
                record_video_size={"width": w, "height": h},
                device_scale_factor=1,
            )
            pages[name] = ctx.new_page()
        pages["dashboard"].goto(args.url + "/", wait_until="domcontentloaded")
        pages["panel"].goto(args.url + "/panel?name=Badevaktpost%20idrett", wait_until="domcontentloaded")
        pages["watch"].goto(args.url + "/watch?name=Badevakt%201", wait_until="domcontentloaded")
        pages["panel"].click("#unlock")
        t0 = time.time()
        first_seen: dict[str, float] = {}
        resolved: set[str] = set()
        while time.time() - t0 < args.seconds:
            state = json.loads(urllib.request.urlopen(args.url + "/api/state", timeout=2).read())
            now = time.time()
            for a in state["alarms"]:
                first_seen.setdefault(f"{a['id']}:{a['kind']}", now)
            waiting = [
                a
                for a in state["alarms"]
                if a["ack_at"] is None and now - first_seen[f"{a['id']}:{a['kind']}"] >= args.ack_delay
            ]
            if waiting:  # the watch shows the most urgent open alarm; the lifeguard acknowledges it there
                with contextlib.suppress(Exception):
                    pages["watch"].click("#ack", timeout=1000)
            for a in state["alarms"]:
                if (
                    a["kind"] == "bottom"
                    and a["ack_at"]
                    and now - a["ack_at"] > args.resolve_after
                    and a["id"] not in resolved
                ):
                    card = pages["dashboard"].locator(f'.alarm[data-id="{a["id"]}"]')
                    card.locator('[data-act="outcome"]').select_option("real")
                    card.locator('[data-act="resolve"]').click()
                    resolved.add(a["id"])
            time.sleep(0.25)
        videos = {}
        for name, page in pages.items():
            path = page.video.path()
            page.context.close()
            videos[name] = path
        browser.close()
    # compose: dashboard left, panel top right, watch bottom right, on the bar colour
    out = args.out
    filt = (
        "color=c=0x0f1720:s=1920x1080:d=1000[bg];"
        "[0:v]scale=1344:840[d];[1:v]scale=544:306[p];[2:v]scale=300:300[w];"
        "[bg][d]overlay=16:120:shortest=1[a];[a][p]overlay=1376:120[b];[b][w]overlay=1498:470,format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(videos["dashboard"]),
        "-i",
        str(videos["panel"]),
        "-i",
        str(videos["watch"]),
        "-filter_complex",
        filt,
        "-r",
        "25",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "medium",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    for name, path in videos.items():
        shutil.copy(path, out.with_name(f"{out.stem}-{name}.webm"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
