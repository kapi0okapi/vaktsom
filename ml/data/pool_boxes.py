"""Swimming-pool boxes (pixels) for the selected Open Images pool photos, from the cached CSVs.

The compositor only puts people into water inside these boxes.

    python ml/data/pool_boxes.py --root $DATA/openimages_pool
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args(argv)
    cache = args.root / "cache"
    with (cache / "classes.csv").open(newline="", encoding="utf-8") as f:
        pool = {name: mid for mid, name in csv.reader(f)}["Swimming pool"]
    sizes = {}
    for split in ("train", "val"):
        coco = json.loads((args.root / f"{split}.json").read_text())
        for im in coco["images"]:
            sizes[Path(im["file_name"]).stem] = (im["width"], im["height"])
    boxes: dict[str, list[list[float]]] = {}
    for split in ("train", "validation", "test"):
        with (cache / f"{split}-bbox.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["LabelName"] != pool or row["ImageID"] not in sizes:
                    continue
                w, h = sizes[row["ImageID"]]
                x0, x1 = float(row["XMin"]) * w, float(row["XMax"]) * w
                y0, y1 = float(row["YMin"]) * h, float(row["YMax"]) * h
                boxes.setdefault(row["ImageID"], []).append([x0, y0, x1 - x0, y1 - y0])
    (args.root / "pool_boxes.json").write_text(json.dumps(boxes))
    print(f"pool boxes for {len(boxes)} images")


if __name__ == "__main__":
    main()
