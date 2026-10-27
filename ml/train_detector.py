"""Fine-tune RF-DETR (Apache 2.0) to find people in and around pool water.

python ml/train_detector.py --data $DATA/det --model medium --out $RUNS/det_medium --epochs 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "seg-small": "RFDETRSegSmall",
    "seg-medium": "RFDETRSegMedium",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", choices=MODELS, default="medium")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--weights", type=Path, default=None, help="start from these weights")
    args = ap.parse_args(argv)

    import rfdetr

    cls = getattr(rfdetr, MODELS[args.model])
    kwargs = {"pretrain_weights": str(args.weights)} if args.weights else {}
    if args.resolution:
        kwargs["resolution"] = args.resolution
    model = cls(**kwargs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    resume = args.out / "checkpoint.pth"
    model.train(
        dataset_dir=str(args.data),
        output_dir=str(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        early_stopping=True,
        early_stopping_patience=12,
        resume=str(resume) if resume.exists() else None,
    )


if __name__ == "__main__":
    main()
