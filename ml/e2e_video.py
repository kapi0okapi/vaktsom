"""End-to-end test on rendered video: the trained detector, tracking, fusion and rules on
photorealistic frames from two calibrated cameras, compared with the ground-truth timeline.

    python ml/e2e_video.py --video $DATA/video_demo --model det.onnx [--attributes attr.onnx] --out e2e.json \
        [--overlay out.mp4]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from vaktsom.config import SiteConfig
from vaktsom.perception.detector import OnnxDetector
from vaktsom.runtime.engine import Engine
from vaktsom.runtime.sources import FolderSource

T0 = 1_700_000_000.0


def site(video: Path, truth: dict) -> SiteConfig:
    pool = truth["pool"]
    cams = []
    for f in sorted(video.glob("cam*.json")):
        c = json.loads(f.read_text())
        cams.append(
            {
                "id": c["id"],
                "url": str(video / c["id"]),
                "basins": ["pool"],
                "size": c["size"],
                "K": c["K"],
                "R": c["R"],
                "t": c["t"],
            }
        )
    return SiteConfig.model_validate(
        {
            "site": "Rendered test pool",
            "tick_hz": truth["fps"],
            "data_dir": tempfile.mkdtemp(prefix="vaktsom-e2e-"),
            "basins": [
                {
                    "id": "pool",
                    "name": "Testbasseng",
                    "outline": [(0, 0), (pool["L"], 0), (pool["L"], pool["W"]), (0, pool["W"])],
                    "default_depth": pool["depth"],
                    "set_off_time_s": 10,
                }
            ],
            "cameras": cams,
            "outputs": [
                {"id": "panel", "kind": "panel", "wired": True},
                {"id": "watch", "kind": "wearable", "wired": False},
            ],
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--attributes", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overlay", type=Path)
    ap.add_argument("--score", type=float, default=0.25)
    ap.add_argument("--scores", type=Path, help="write every person's distress score per frame, for threshold replays")
    args = ap.parse_args()
    truth = json.loads((args.video / "truth.json").read_text())
    cfg = site(args.video, truth)
    source = FolderSource({c.id: c.url for c in cfg.cameras})
    attributes = None
    if args.attributes:
        from vaktsom.perception.attributes import OnnxAttributes

        attributes = OnnxAttributes(args.attributes)
    eng = Engine(cfg, source, detector=OnnxDetector(args.model, score=args.score), attributes=attributes, min_fps=1.0)
    period = 1.0 / truth["fps"]
    frames_dir = Path(tempfile.mkdtemp(prefix="vaktsom-overlay-")) if args.overlay else None
    victim = np.array([truth["victim"]["x"], truth["victim"]["y"]])
    track = []
    scores: dict[int, list] = {}
    for k in range(source.length):
        ts = T0 + k * period
        eng.step(ts)
        if args.scores is not None:
            for p in eng.world.persons.values():
                if not p.lost:
                    scores.setdefault(p.id, []).append(
                        [round(k * period, 2), round(float(p.distress), 4), *p.xy.round(2).tolist()]
                    )
        near = min(eng.world.persons.values(), key=lambda p: np.linalg.norm(p.xy - victim), default=None)
        if near is not None:
            track.append(
                {
                    "t": round(k * period, 1),
                    "dist": float(np.linalg.norm(near.xy - victim)),
                    "under": round(near.p_under, 2),
                    "motion": round(near.motion, 2),
                    "speed": round(near.kf.speed, 2),
                    "views": near.n_views,
                    "depth": near.depth,
                    "lost": near.lost,
                    "distress": round(near.distress, 2),
                }
            )
        if frames_dir is not None:
            from vaktsom.api.server import overlay

            tiles = [overlay(eng, cid, eng.latest[cid].image) for cid in sorted(eng.latest)]
            panel = np.hstack([cv2.resize(t, (960, 540)) for t in tiles])
            status = "Overvaking aktiv"
            colour = (60, 160, 60)
            for a in eng.alarms.open_alarms():
                status = f"{'ALARM' if a.severity == 'alarm' else 'FORVARSEL'}: {a.kind} - {a.detail.get('label', '')}"
                colour = (40, 40, 220) if a.severity == "alarm" else (0, 140, 230)
            bar = np.full((70, panel.shape[1], 3), 24, np.uint8)
            cv2.putText(
                bar,
                f"t = {k * period:5.1f} s   {status}",
                (20, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                colour,
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(frames_dir / f"{k:05d}.jpg"), np.vstack([bar, panel]))
    rows = eng.store.alarms()
    events = []
    for a in rows:
        for h in eng.store.alarm_history(a["id"]):
            if h["event"] in ("raised", "upgraded"):
                events.append(
                    {
                        "t": round(h["ts"] - T0, 2),
                        "event": h["event"],
                        "kind": h["detail"].get("kind"),
                        "x": h["detail"].get("x"),
                        "y": h["detail"].get("y"),
                    }
                )
    v = truth["victim"]
    bottom_alarm = next(
        (e["t"] for e in sorted(events, key=lambda e: e["t"]) if e["kind"] in ("bottom", "missing")), None
    )
    pre = next((e["t"] for e in sorted(events, key=lambda e: e["t"]) if e["kind"] in ("distress", "face_down")), None)
    report = {
        "truth": v,
        "set_off_s": 10,
        "alarms": len(rows),
        "events": events,
        "bottom_alarm_s": bottom_alarm,
        "alarm_after_bottom_s": None if bottom_alarm is None else round(bottom_alarm - v["bottom_s"], 2),
        "pre_alarm_s": pre,
        "position_error_m": None if not events else float(np.hypot(events[-1]["x"] - v["x"], events[-1]["y"] - v["y"])),
        "victim_track": track[:: max(len(track) // 60, 1)],
    }
    args.out.write_text(json.dumps(report, indent=2))
    if args.scores is not None:
        args.scores.write_text(json.dumps({"victim": truth["victim"], "series": scores}))
    print(json.dumps({k: v for k, v in report.items() if k != "victim_track"}, indent=2))
    if frames_dir is not None and shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(truth["fps"]),
                "-i",
                str(frames_dir / "%05d.jpg"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                str(args.overlay),
            ],
            check=True,
        )
        print("wrote", args.overlay)


if __name__ == "__main__":
    main()
