"""Run the end-to-end video test over a set of rendered videos and summarise.

Expected outcome per scenario: active and passive drownings must raise a bottom (or missing) alarm
within the set-off time + 2 s of the body reaching the bottom; a breath hold shorter than the set-off
time and normal swimming must raise no alarm.

    python ml/e2e_set.py --videos data/video_set --model det.onnx [--attributes attr.onnx] --out e2e_set.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--attributes", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overlays", type=Path)
    ap.add_argument("--scores", type=Path, help="directory for per-video distress score series")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in sorted(p for p in args.videos.iterdir() if (p / "truth.json").exists()):
        out = args.out.parent / f"e2e_{v.name}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("e2e_video.py")),
            "--video",
            str(v),
            "--model",
            str(args.model),
            "--out",
            str(out),
        ]
        if args.attributes:
            cmd += ["--attributes", str(args.attributes)]
        if args.overlays:
            args.overlays.mkdir(parents=True, exist_ok=True)
            cmd += ["--overlay", str(args.overlays / f"{v.name}.mp4")]
        if args.scores:
            args.scores.mkdir(parents=True, exist_ok=True)
            cmd += ["--scores", str(args.scores / f"{v.name}.json")]
        subprocess.run(cmd, check=True, capture_output=True)
        r = json.loads(out.read_text())
        truth = json.loads((v / "truth.json").read_text())
        scen = truth.get("scenario", "active")
        should_alarm = scen in ("active", "passive")
        delay = r["alarm_after_bottom_s"]
        ok = delay is not None and delay <= r["set_off_s"] + 2.0 if should_alarm else r["alarms"] == 0
        rows.append(
            {
                "video": v.name,
                "scenario": scen,
                "depth": truth["pool"]["depth"],
                "alarms": r["alarms"],
                "alarm_after_bottom_s": delay,
                "pre_alarm_s": r["pre_alarm_s"],
                "submerged_s": truth["victim"].get("submerged_s"),
                "position_error_m": r["position_error_m"],
                "pass": ok,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    summary = {"videos": len(rows), "passed": sum(r["pass"] for r in rows), "rows": rows}
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"{summary['passed']}/{summary['videos']} passed")


if __name__ == "__main__":
    main()
