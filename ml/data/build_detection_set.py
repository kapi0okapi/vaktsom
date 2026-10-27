"""Merge COCO-style sources into the folder layout RF-DETR trains on.

    out/train/_annotations.coco.json + images (symlinks), out/valid/..., out/test/...

Categories are remapped to one list (person, head). Sources: Open Images pool subset (its own
train/val split; val is split again into valid and test), COCO pool images (train only), and any
number of synthetic sets (``--synthetic dir`` with train.json/val.json and images/).

    python ml/data/build_detection_set.py --out $DATA/det \
        --openimages $DATA/openimages_pool --coco-pool $DATA/coco_pool
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

CATEGORIES = [{"id": 0, "name": "person"}, {"id": 1, "name": "head"}, {"id": 2, "name": "person_under"}]
NAME_TO_ID = {c["name"]: c["id"] for c in CATEGORIES}


def load(path: Path, image_dir: Path | None) -> list[tuple[Path, dict, list[dict]]]:
    coco = json.loads(path.read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_image: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    out = []
    for im in coco["images"]:
        f = Path(im["file_name"])
        src = f if f.is_absolute() else (image_dir / f)
        anns = []
        for a in by_image.get(im["id"], []):
            name = cats[a["category_id"]]
            if name not in NAME_TO_ID:
                continue
            anns.append({**a, "category_id": NAME_TO_ID[name]})
        out.append((src, im, anns))
    return out


def write(split_dir: Path, items: list[tuple[Path, dict, list[dict]]]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    for n, (src, im, anns) in enumerate(items):
        name = f"{n:07d}{src.suffix.lower() or '.jpg'}"
        link = split_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(src.resolve(), link)
        coco["images"].append({"id": n, "file_name": name, "width": im["width"], "height": im["height"]})
        for a in anns:
            coco["annotations"].append(
                {**a, "id": len(coco["annotations"]), "image_id": n, "segmentation": a.get("segmentation", [])}
            )
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))
    print(f"{split_dir}: {len(coco['images'])} images, {len(coco['annotations'])} boxes")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--openimages", type=Path)
    ap.add_argument("--coco-pool", type=Path)
    ap.add_argument("--synthetic", type=Path, action="append", default=[])
    args = ap.parse_args(argv)
    train, valid, test = [], [], []
    if args.openimages:
        oi = args.openimages
        train += load(oi / "train.json", oi / "images")
        val = load(oi / "val.json", oi / "images")
        random.Random(0).shuffle(val)
        valid += val[: len(val) // 2]
        test += val[len(val) // 2 :]
    if args.coco_pool:
        for split in ("train2017", "val2017"):
            f = args.coco_pool / f"pool_{split}.json"
            if f.exists():
                train += load(f, None)
    for syn in args.synthetic:
        train += load(syn / "train.json", syn / "images")
        if (syn / "val.json").exists():
            valid += load(syn / "val.json", syn / "images")
    for name, items in (("train", train), ("valid", valid), ("test", test)):
        write(args.out / name, items)


if __name__ == "__main__":
    main()
