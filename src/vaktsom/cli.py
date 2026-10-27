"""Command line: run the system (live cameras or the simulator) and the evaluations.

vaktsom serve --config configs/demo.yaml --sim demo --no-auth # demo with simulated cameras
vaktsom serve --config configs/site.yaml --model det.onnx      # real cameras
vaktsom user add anne --role leder --config configs/site.yaml  # asks for the password
vaktsom device add "Post idrett" --kind panel --url https://vaktsom.local --config configs/site.yaml
vaktsom report --config configs/site.yaml --month 2026-11      # alarms and false alarms per basin
vaktsom eval iso --config configs/demo.yaml --basin stup       # ISO 20380 style tests
"""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path


def serve(args: argparse.Namespace) -> None:
    import uvicorn

    from vaktsom.alarms.outputs import Dispatcher, build_outputs
    from vaktsom.api.hub import Hub
    from vaktsom.api.server import create_app
    from vaktsom.config import load_config
    from vaktsom.runtime.engine import Engine
    from vaktsom.storage.db import EventStore

    cfg = load_config(args.config)
    if args.data:
        cfg.data_dir = Path(args.data)
    hub = Hub()
    store = EventStore(cfg.data_dir / "events.db")
    dispatch = Dispatcher(build_outputs(cfg.outputs, hub), store)
    basins = {b.id: b.build() for b in cfg.basins}
    cams = {c.id: c.build() for c in cfg.cameras}
    if args.sim:
        from vaktsom.sim import scenarios
        from vaktsom.sim.source import SimSource

        scene = getattr(scenarios, args.sim)(basins)
        glare = None
        ropes = [((0.5, 2.5 * k), (49.5, 2.5 * k)) for k in range(1, 10)] if "idrett" in basins else None
        source = SimSource(scene, cams, basins, render=True, glare=glare, lane_ropes=ropes, seed=args.seed)
        detector = None
    attributes = None
    if not args.sim:
        from vaktsom.perception.detector import OnnxDetector
        from vaktsom.runtime.sources import RtspSource

        source = RtspSource(cfg)
        detector = OnnxDetector(args.model)
        if args.attributes:
            from vaktsom.perception.attributes import OnnxAttributes

            attributes = OnnxAttributes(args.attributes)
    # simulated cameras are rendered round-robin, a few frames per second each
    engine = Engine(
        cfg,
        source,
        detector=detector,
        dispatch=dispatch,
        store=store,
        hub=hub,
        min_fps=0.3 if args.sim else 5.0,
        attributes=attributes,
    )
    accounts = None
    if not args.no_auth:
        from vaktsom.api.auth import Accounts

        accounts = Accounts(cfg.data_dir / "accounts.json")
        if not accounts.data["users"]:
            logging.getLogger("vaktsom").warning(
                "no user accounts yet: vaktsom user add NAME --role admin --config %s", args.config
            )
    thread = threading.Thread(target=engine.run, name="engine", daemon=True)
    thread.start()
    threading.Thread(target=engine.watchdog, name="watchdog", daemon=True).start()
    app = create_app(engine, hub, accounts)
    tls = {"ssl_certfile": str(args.certfile), "ssl_keyfile": str(args.keyfile)} if args.certfile else {}
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", **tls)
    engine.stop()


def manage(args: argparse.Namespace) -> None:
    """User accounts and device tokens, kept in the data directory; every change is logged."""
    import getpass

    from vaktsom.api.auth import Accounts
    from vaktsom.config import load_config
    from vaktsom.storage.db import EventStore

    data = args.data or load_config(args.config).data_dir
    accounts = Accounts(Path(data) / "accounts.json")
    store = EventStore(Path(data) / "events.db")
    if args.action == "list":
        for row in accounts.overview()["users" if args.cmd == "user" else "devices"]:
            print(row["name"], row.get("role") or row.get("kind"))
        return
    if not args.name:
        raise SystemExit("a name is needed")
    if args.cmd == "user" and args.action == "add":
        password = getpass.getpass(f"Password for {args.name}: ")
        if password != getpass.getpass("Repeat: "):
            raise SystemExit("the passwords differ")
        accounts.add_user(args.name, password, args.role)
        store.log("user_added", "cli", {"user": args.name, "role": args.role})
    elif args.cmd == "user":
        accounts.remove_user(args.name)
        store.log("user_removed", "cli", {"user": args.name})
    elif args.action == "add":
        token = accounts.add_device(args.name, args.kind)
        store.log("device_added", "cli", {"device": args.name, "kind": args.kind})
        page = "panel" if args.kind == "panel" else "watch"
        print(f"Open once on the device (the token is shown only now):\n{args.url.rstrip('/')}/{page}#token={token}")
    else:
        accounts.revoke_device(args.name)
        store.log("device_revoked", "cli", {"device": args.name})


def report(args: argparse.Namespace) -> None:
    import json
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from vaktsom.config import load_config
    from vaktsom.storage.db import EventStore
    from vaktsom.storage.report import period_report

    cfg = load_config(args.config)
    tz = ZoneInfo(cfg.timezone)
    if args.month:
        y, m = map(int, args.month.split("-"))
        start, end = datetime(y, m, 1, tzinfo=tz), datetime(y + m // 12, m % 12 + 1, 1, tzinfo=tz)
    else:
        end = datetime.now(tz)
        start = end - timedelta(days=args.days)
    store = EventStore(Path(args.data or cfg.data_dir) / "events.db")
    rep = period_report(store, cfg, start.timestamp(), min(end, datetime.now(tz)).timestamp())
    if args.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return
    print(f"{rep['site']}, {start:%Y-%m-%d} to {end:%Y-%m-%d}, {rep['days']} days")
    print(f"{'basin':<22}{'alarms':>8}{'pre':>6}{'false':>7}{'false/day':>11}{'open':>6}{'tests':>7}{'weak h':>8}")
    for b in rep["basins"].values():
        flag = "" if b["within_iso_limit"] else "  over the ISO limit"
        print(
            f"{b['name']:<22}{b['alarms']:>8}{b['pre_alarms']:>6}{b['false_alarms']:>7}"
            f"{b['false_alarms_per_day']:>11.2f}{b['unresolved']:>6}{b['days_with_daily_test']:>7}"
            f"{b['reduced_coverage_h']:>8.1f}{flag}"
        )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        prog="vaktsom", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--sim", help="scenario name from vaktsom.sim.scenarios (simulated cameras)")
    s.add_argument("--model", type=Path, help="detector ONNX for live cameras")
    s.add_argument("--attributes", type=Path, help="posture/submersion ONNX (optional)")
    s.add_argument("--data", type=Path)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--no-auth", action="store_true", help="no login; only with --sim, for demonstrations")
    s.add_argument("--certfile", type=Path, help="HTTPS certificate (PEM), e.g. from the site's own CA")
    s.add_argument("--keyfile", type=Path, help="its private key (PEM)")
    u = sub.add_parser("user", help="user accounts for people")
    u.add_argument("action", choices=["add", "remove", "list"])
    u.add_argument("name", nargs="?")
    u.add_argument("--role", choices=["badevakt", "leder", "admin", "leverandor"], default="badevakt")
    d = sub.add_parser("device", help="device tokens for alarm panels and watches")
    d.add_argument("action", choices=["add", "revoke", "list"])
    d.add_argument("name", nargs="?")
    d.add_argument("--kind", choices=["panel", "wearable"], default="panel")
    d.add_argument("--url", default="http://localhost:8080", help="the address the device opens")
    r = sub.add_parser("report", help="alarms, outcomes and false alarms per basin for a period")
    r.add_argument("--month", help="YYYY-MM; default the last --days days")
    r.add_argument("--days", type=float, default=30)
    r.add_argument("--json", action="store_true")
    for p in (u, d, r):
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--data", type=Path)
    e = sub.add_parser("eval")
    e.add_argument("suite", choices=["iso", "false-alarms", "drowning", "all"])
    e.add_argument("--config", type=Path, required=True)
    e.add_argument("--out", type=Path, default=Path("runs/eval"))
    e.add_argument("--basin", action="append")
    e.add_argument("--hours", type=float, default=2.0)
    e.add_argument("--trials", type=int, default=40)
    e.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    if args.cmd == "serve" and args.no_auth and not args.sim:
        ap.error("--no-auth is only for simulated demonstrations (--sim)")
    if args.cmd == "serve" and bool(args.certfile) != bool(args.keyfile):
        ap.error("--certfile and --keyfile go together")
    if args.cmd == "serve":
        serve(args)
    elif args.cmd in ("user", "device"):
        manage(args)
    elif args.cmd == "report":
        report(args)
    else:
        from vaktsom.eval.run import run_suite

        run_suite(args)


if __name__ == "__main__":
    main()
