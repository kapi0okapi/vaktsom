"""Cut-outs of things that end up on a pool bottom (balls, boards, bags, bottles, toys) from COCO,
used as unlabelled distractors by the compositor.

    python ml/data/coco_objects.py --coco $DATA/coco --out $DATA/coco_objects
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NAMES = {
    "sports ball",
    "frisbee",
    "surfboard",
    "backpack",
    "handbag",
    "suitcase",
    "umbrella",
    "bottle",
    "cup",
    "teddy bear",
    "kite",
    "skateboard",
    "book",
    "cell phone",
    "remote",
    "skis",
    "snowboard",
    "tie",
    "bowl",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=150)
    args = ap.parse_args(argv)
    import numpy as np
    from PIL import Image
    from pycocotools import mask as mask_utils

    root = args.coco / "images" if (args.coco / "images").is_dir() else args.coco
    inst = json.loads((args.coco / "annotations" / "instances_train2017.json").read_text())
    cats = {c["id"]: c["name"] for c in inst["categories"] if c["name"] in NAMES}
    images = {im["id"]: im for im in inst["images"]}
    args.out.mkdir(parents=True, exist_ok=True)
    count = {c: 0 for c in cats}
    for a in inst["annotations"]:
        c = a["category_id"]
        if c not in cats or a["iscrowd"] or a["area"] < 40 * 40 or count[c] >= args.per_class:
            continue
        im = images[a["image_id"]]
        x, y, w, h = a["bbox"]
        if x < 2 or y < 2 or x + w > im["width"] - 2 or y + h > im["height"] - 2:
            continue
        img = np.asarray(Image.open(root / "train2017" / im["file_name"]).convert("RGB"))
        rle = mask_utils.frPyObjects(a["segmentation"], im["height"], im["width"])
        m = mask_utils.decode(mask_utils.merge(rle) if isinstance(rle, list) else rle)
        x0, y0, x1, y1 = int(x), int(y), int(x + w) + 1, int(y + h) + 1
        Image.fromarray(np.dstack([img[y0:y1, x0:x1], m[y0:y1, x0:x1] * 255]).astype(np.uint8)).save(
            args.out / f"{cats[c].replace(' ', '_')}_{a['id']}.png"
        )
        count[c] += 1
    print(f"{sum(count.values())} object cut-outs: " + ", ".join(f"{cats[c]} {n}" for c, n in count.items()))


if __name__ == "__main__":
    main()
