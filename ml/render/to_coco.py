"""Collect the per-task label files of the Blender renders into COCO train/val sets.

python ml/render/to_coco.py --root $DATA/renders --val 0.08
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CATEGORIES = [{"id": 0, "name": "person"}, {"id": 1, "name": "head"}, {"id": 2, "name": "person_under"}]
IDS = {c["name"]: c["id"] for c in CATEGORIES}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--val", type=float, default=0.08)
    args = ap.parse_args(argv)
    items = []
    for f in sorted(args.root.glob("labels_*.json")):
        items += [it for it in json.loads(f.read_text()) if (args.root / "images" / it["file_name"]).exists()]
    random.Random(0).shuffle(items)
    n_val = int(len(items) * args.val)
    for split, part in (("val", items[:n_val]), ("train", items[n_val:])):
        coco = {"images": [], "annotations": [], "categories": CATEGORIES}
        for i, it in enumerate(part):
            coco["images"].append(
                {
                    "id": i,
                    "file_name": it["file_name"],
                    "width": it["width"],
                    "height": it["height"],
                    "scene_depth": it["scene"]["depth"],
                }
            )
            for a in it["annotations"]:
                x, y, w, h = a["bbox"]
                coco["annotations"].append(
                    {
                        "id": len(coco["annotations"]),
                        "image_id": i,
                        "category_id": IDS[a["category"]],
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "state": a.get("state"),
                        "pixels": a.get("pixels"),
                    }
                )
        (args.root / f"{split}.json").write_text(json.dumps(coco))
        n_under = sum(a["category_id"] == 2 for a in coco["annotations"])
        print(f"{split}: {len(part)} images, {len(coco['annotations'])} boxes ({n_under} under water)")


if __name__ == "__main__":
    main()
