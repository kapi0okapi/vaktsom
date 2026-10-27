"""Throughput of the detector on one GPU: how many camera frames per second, batched over cameras.

python ml/bench_detector.py --weights $RUNS/det_medium/checkpoint_best_total.pth --batch 1 8 16 32
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="RFDETRMedium")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 16, 32])
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    import rfdetr

    wrapper = getattr(rfdetr, args.model)(pretrain_weights=args.weights)
    net = wrapper.model.model.cuda().eval().half()
    res = wrapper.model_config.resolution
    gpu = torch.cuda.get_device_name()
    out = {"gpu": gpu, "resolution": res, "fp16": True, "results": []}
    with torch.inference_mode():
        for b in args.batch:
            x = torch.randn(b, 3, res, res, device="cuda", dtype=torch.half)
            for _ in range(5):
                net(x)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(args.iters):
                net(x)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t) / args.iters
            out["results"].append({"batch": b, "ms_per_batch": dt * 1000, "frames_per_s": b / dt})
            print(f"{gpu} batch {b:3d}: {dt * 1000:7.2f} ms, {b / dt:7.1f} frames/s", flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
