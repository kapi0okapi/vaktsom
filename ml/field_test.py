"""Field test on real video: footage from a camera or phone at a pool, calibrated from a few points
whose pool coordinates are known, run through the whole chain (detector, tracking, rules, alarms).
The result is an annotated video, every alarm with its time and place, and a per-second summary to
compare with what happened in the water.

1. Film from a high, fixed point (a stand, a balcony, a tripod at 2 m or more), 10 to 30 fps.
   Only people who have agreed to be filmed; the footage stays on this machine.
2. Choose at least four points on the water line whose pool coordinates you know: corners, the ends
   of lane ropes, lane marks on the wall. x runs along the pool and y across it, in metres.
3. Click them in a frame (``--pick``) and type their coordinates; then run the test.

    python ml/field_test.py --video swim.mp4 --pick --pool 25 12.5
    python ml/field_test.py --video swim.mp4 --pool 25 12.5 --depth 2.0 --model det.onnx \\
        --attributes attr.onnx --out runs/field/swim [--start 30 --seconds 120]

The focal length is found from the points when ``--hfov`` is not given (four points are enough,
six or more make it robust). ``calibration.jpg`` shows the pool outline and a 1 m grid drawn with
the result: check that the lines sit on the real pool edges before trusting anything else.
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
from vaktsom.geometry import Camera
from vaktsom.perception.detector import OnnxDetector
from vaktsom.runtime.engine import Engine
from vaktsom.runtime.sources import FolderSource

T0 = 1_700_000_000.0
MAX_WIDTH = 1920


def extract(video: Path, out: Path, fps: float, start: float, seconds: float | None) -> tuple[int, tuple[int, int]]:
    """Frames at ``fps`` from ``start`` s, at most MAX_WIDTH wide, as numbered JPEGs."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    out.mkdir(parents=True, exist_ok=True)
    n, size = 0, None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if seconds is not None and t >= start + seconds:
            break
        if t < start + n / fps - 1e-3:  # keep one frame per tick
            continue
        if img.shape[1] > MAX_WIDTH:
            img = cv2.resize(
                img, (MAX_WIDTH, round(img.shape[0] * MAX_WIDTH / img.shape[1])), interpolation=cv2.INTER_AREA
            )
        size = (img.shape[1], img.shape[0])
        cv2.imwrite(str(out / f"{n:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n += 1
    if size is None:
        raise SystemExit("no frames in the chosen window")
    return n, size


def first_frame(video: Path, start: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    ok, img = cap.read()
    if not ok:
        raise SystemExit(f"cannot read {video}")
    if img.shape[1] > MAX_WIDTH:
        img = cv2.resize(img, (MAX_WIDTH, round(img.shape[0] * MAX_WIDTH / img.shape[1])), interpolation=cv2.INTER_AREA)
    return img


def pick(img: np.ndarray, path: Path) -> None:
    """Click points in the frame (Enter when done), then type each point's pool coordinates."""
    scale = min(1.0, 1500 / img.shape[1])
    shown = cv2.resize(img, None, fx=scale, fy=scale)
    clicks: list[tuple[float, float]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x / scale, y / scale))
            cv2.circle(shown, (x, y), 6, (0, 0, 255), 2)
            cv2.putText(shown, str(len(clicks)), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.namedWindow("points")
    cv2.setMouseCallback("points", on_mouse)
    while True:
        cv2.imshow("points", shown)
        if cv2.waitKey(30) in (13, 10, 27):
            break
    cv2.destroyAllWindows()
    pool = []
    for i, (u, v) in enumerate(clicks, 1):
        x, y = (float(s) for s in input(f"point {i} at pixel ({u:.0f}, {v:.0f}): pool x y in metres? ").split())
        pool.append((x, y))
    path.write_text(json.dumps({"size": [img.shape[1], img.shape[0]], "image": clicks, "pool": pool}, indent=2))
    print("wrote", path)


def calibrate(points: dict, hfov: float | None) -> tuple[Camera, float, float]:
    """Camera pose (and focal length, unless given) from the point pairs; returns the RMS error in pixels."""
    w, h = points["size"]
    img = np.asarray(points["image"], float)
    pool = np.hstack([np.asarray(points["pool"], float), np.zeros((len(img), 1))])
    if len(img) < 4:
        raise SystemExit("at least four points are needed")

    def fit(fov: float) -> tuple[Camera, float]:
        f = (w / 2) / np.tan(np.radians(fov) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        cam = Camera.from_pnp(img, pool[:, :2], K, size=(w, h))
        err = np.sqrt(np.mean(np.sum((cam.project(pool) - img) ** 2, axis=1)))
        return cam, float(err)

    if hfov is not None:
        cam, err = fit(hfov)
        return cam, hfov, err
    best = min((fit(fov) + (fov,) for fov in np.arange(30.0, 130.0, 0.25)), key=lambda r: r[1])
    return best[0], best[2], best[1]


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def draw_check(img: np.ndarray, cam: Camera, L: float, W: float) -> np.ndarray:
    out = img.copy()
    for x in np.arange(0, L + 1e-6, 1.0):
        line = cam.project(np.array([[x, 0, 0], [x, W, 0]], float))
        cv2.line(out, _pt(line[0]), _pt(line[1]), (0, 200, 255), 1, cv2.LINE_AA)
    for y in np.arange(0, W + 1e-6, 1.0):
        line = cam.project(np.array([[0, y, 0], [L, y, 0]], float))
        cv2.line(out, _pt(line[0]), _pt(line[1]), (0, 200, 255), 1, cv2.LINE_AA)
    corners = cam.project(np.array([[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0]], float)).astype(int)
    cv2.polylines(out, [corners.reshape(-1, 1, 2)], True, (0, 0, 255), 3, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--pool", type=float, nargs=2, metavar=("LENGTH", "WIDTH"), required=True)
    ap.add_argument("--depth", type=float, default=2.0, help="water depth in metres")
    ap.add_argument("--points", type=Path, help="default: <video>.points.json")
    ap.add_argument("--pick", action="store_true", help="click the calibration points and stop")
    ap.add_argument("--frame", type=Path, help="save the first frame (as the points must refer to it) and stop")
    ap.add_argument("--hfov", type=float, help="horizontal field of view in degrees, if known")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--attributes", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--score", type=float, default=0.25)
    args = ap.parse_args()
    points_path = args.points or args.video.with_suffix(".points.json")
    if args.frame:
        cv2.imwrite(str(args.frame), first_frame(args.video, args.start))
        return
    if args.pick:
        pick(first_frame(args.video, args.start), points_path)
        return
    if args.model is None or args.out is None:
        raise SystemExit("--model and --out are needed for the test (or --pick to calibrate)")
    L, W = args.pool
    points = json.loads(points_path.read_text())
    cam, hfov, err = calibrate(points, args.hfov)
    args.out.mkdir(parents=True, exist_ok=True)
    frame0 = first_frame(args.video, args.start)
    if (frame0.shape[1], frame0.shape[0]) != tuple(points["size"]):
        raise SystemExit(f"the points were clicked on a {points['size']} frame, the video gives {frame0.shape[1::-1]}")
    cv2.imwrite(str(args.out / "calibration.jpg"), draw_check(frame0, cam, L, W))
    print(f"calibration: hfov {hfov:.1f} deg, reprojection error {err:.1f} px; check calibration.jpg")

    frames = Path(tempfile.mkdtemp(prefix="vaktsom-field-"))
    n, size = extract(args.video, frames / "cam0", args.fps, args.start, args.seconds)
    cfg = SiteConfig.model_validate(
        {
            "site": "Field test",
            "tick_hz": args.fps,
            "data_dir": str(frames / "data"),
            "basins": [
                {
                    "id": "pool",
                    "name": "Testbasseng",
                    "outline": [(0, 0), (L, 0), (L, W), (0, W)],
                    "default_depth": args.depth,
                }
            ],
            "cameras": [
                {
                    "id": "cam0",
                    "url": str(frames / "cam0"),
                    "basins": ["pool"],
                    "size": list(size),
                    "K": cam.K.tolist(),
                    "R": cam.R.tolist(),
                    "t": cam.t.tolist(),
                }
            ],
            "outputs": [
                {"id": "panel", "kind": "panel", "wired": True},
                {"id": "watch", "kind": "wearable", "wired": False},
            ],
        }
    )
    attributes = None
    if args.attributes:
        from vaktsom.perception.attributes import OnnxAttributes

        attributes = OnnxAttributes(args.attributes)
    source = FolderSource({"cam0": str(frames / "cam0")})
    eng = Engine(cfg, source, detector=OnnxDetector(args.model, score=args.score), attributes=attributes, min_fps=1.0)
    from vaktsom.api.server import overlay

    shots = args.out / "frames"
    shots.mkdir(exist_ok=True)
    seconds = []
    for k in range(n):
        ts = T0 + k / args.fps
        eng.step(ts)
        img = overlay(eng, "cam0", eng.latest["cam0"].image)
        status, colour = "Overvaking aktiv", (60, 160, 60)
        for a in eng.alarms.open_alarms():
            status = f"{'ALARM' if a.severity == 'alarm' else 'FORVARSEL'}: {a.kind} {a.detail.get('label', '')}"
            colour = (40, 40, 220) if a.severity == "alarm" else (0, 140, 230)
        bar = np.full((60, img.shape[1], 3), 24, np.uint8)
        cv2.putText(
            bar,
            f"t = {args.start + k / args.fps:6.1f} s   {status}",
            (16, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            colour,
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(shots / f"{k:05d}.jpg"), np.vstack([bar, img]))
        if k % round(args.fps) == 0:
            people = [p for p in eng.world.persons.values() if p.basin_id and not p.lost]
            seconds.append(
                {
                    "t": round(args.start + k / args.fps, 1),
                    "in_water": len(people),
                    "under": int(sum(p.p_under >= 0.6 for p in people)),
                    "detections": int(eng.stats["detections"]),
                }
            )
    alarms = []  # every raise and upgrade: a warning that becomes an alarm shows both moments
    for a in eng.store.alarms():
        for h in eng.store.alarm_history(a["id"]):
            if h["event"] in ("raised", "upgraded"):
                alarms.append(
                    {
                        "t": round(args.start + h["ts"] - T0, 1),
                        "event": h["event"],
                        "kind": (kind := h["detail"].get("kind", a["kind"])),
                        "severity": "pre_alarm" if kind in ("distress", "face_down") else "alarm",
                        "x": round(float(h["detail"].get("x", a["x"])), 2),
                        "y": round(float(h["detail"].get("y", a["y"])), 2),
                    }
                )
    alarms.sort(key=lambda e: e["t"])
    report = {
        "video": str(args.video),
        "calibration": {"hfov_deg": hfov, "rms_px": round(err, 2), "points": len(points["image"])},
        "frames": n,
        "fps": args.fps,
        "alarms": alarms,
        "per_second": seconds,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_second"}, indent=2))
    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(args.fps),
                "-i",
                str(shots / "%05d.jpg"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                str(args.out / "overlay.mp4"),
            ],
            check=True,
        )
        shutil.rmtree(shots)
        print("wrote", args.out / "overlay.mp4")
    shutil.rmtree(frames, ignore_errors=True)


if __name__ == "__main__":
    main()
