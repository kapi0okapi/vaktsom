"""Export a trained RF-DETR checkpoint to ONNX for the runtime (``vaktsom.perception.detector``):
dynamic batch, outputs ``dets`` and ``labels``, with the class list and input size beside it.

    python ml/export_detector.py --model large --weights $RUNS/det_large/checkpoint_best_total.pth \\
        --data $DATA/det --out $RUNS/export/det_large
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rfdetr

MODELS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="the training set, for the class names")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cats = json.loads((args.data / "train" / "_annotations.coco.json").read_text())["categories"]
    classes = [c["name"] for c in sorted(cats, key=lambda c: c["id"])]
    model = getattr(rfdetr, MODELS[args.model])(pretrain_weights=str(args.weights.resolve()))  # absolute path
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"rfdetr-{args.model}"
    path = Path(model.export(output_dir=str(args.out), dynamic_batch=True, output_name=f"{name}.onnx"))
    if path.name != f"{name}.onnx":
        path = path.rename(args.out / f"{name}.onnx")
    resolution = model.model_config.resolution
    (args.out / f"{name}.json").write_text(json.dumps({"classes": classes, "resolution": resolution}))
    print("wrote", path, classes, resolution)


if __name__ == "__main__":
    main()
