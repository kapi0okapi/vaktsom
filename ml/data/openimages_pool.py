"""People in and around swimming pools from Open Images V7, as a COCO-style detection set.

Selects every image that has a "Swimming pool" box and at least one person box (Person, Man, Woman,
Boy, Girl), downloads it from the public bucket and writes COCO json with two categories: person and
head ("Human head" boxes). Group-of boxes become iscrowd=1 so they are ignored by the matcher.
Images are CC BY 2.0 (as listed by Open Images), annotations CC BY 4.0; the per-image licence,
author and source URL are kept in ``attribution.csv``.

Standard library only, so it runs on a login node with the bare Python module:

    python3 ml/data/openimages_pool.py --out $DATA/openimages_pool
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://storage.googleapis.com/openimages"
BOXES = {
    "train": f"{BASE}/v6/oidv6-train-annotations-bbox.csv",
    "validation": f"{BASE}/v5/validation-annotations-bbox.csv",
    "test": f"{BASE}/v5/test-annotations-bbox.csv",
}
IMAGE_INFO = {
    "train": f"{BASE}/2018_04/train/train-images-boxable-with-rotation.csv",
    "validation": f"{BASE}/2018_04/validation/validation-images-with-rotation.csv",
    "test": f"{BASE}/2018_04/test/test-images-with-rotation.csv",
}
CLASSES = f"{BASE}/v7/oidv7-class-descriptions-boxable.csv"
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"

POOL = "Swimming pool"
PERSON_NAMES = {"Person", "Man", "Woman", "Boy", "Girl"}
HEAD_NAME = "Human head"


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def class_ids(cache: Path) -> dict[str, str]:
    """Display name -> MID for the boxable classes."""
    path = fetch(CLASSES, cache / "classes.csv")
    with path.open(newline="", encoding="utf-8") as f:
        return {name: mid for mid, name in csv.reader(f)}


def scan_boxes(path: Path, pool: str, person: set[str], head: str) -> dict[str, list[dict]]:
    """ImageID -> its person/head boxes, for images that also contain a pool box."""
    boxes: dict[str, list[dict]] = defaultdict(list)
    has_pool: set[str] = set()
    wanted = person | {head, pool}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mid = row["LabelName"]
            if mid not in wanted:
                continue
            image_id = row["ImageID"]
            if mid == pool:
                has_pool.add(image_id)
                continue
            boxes[image_id].append(
                {
                    "category": "person" if mid in person else "head",
                    "box": [float(row[k]) for k in ("XMin", "YMin", "XMax", "YMax")],
                    "group": row["IsGroupOf"] == "1",
                    "depiction": row["IsDepiction"] == "1",
                }
            )
    return {
        i: b
        for i, b in boxes.items()
        if i in has_pool and any(x["category"] == "person" for x in b) and not any(x["depiction"] for x in b)
    }


def image_info(path: Path, ids: set[str]) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["ImageID"]: row for row in csv.DictReader(f) if row["ImageID"] in ids}


def download(split: str, image_id: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        with urllib.request.urlopen(IMAGE_URL.format(split=split, image_id=image_id), timeout=60) as r:
            data = r.read()
    except OSError as e:
        print(f"skip {image_id}: {e}", file=sys.stderr)
        return False
    dest.write_bytes(data)
    return True


def jpeg_size(path: Path) -> tuple[int, int]:
    """(width, height) from the JPEG header, without decoding."""
    data = path.read_bytes()
    stream = io.BytesIO(data)
    stream.read(2)
    while True:
        marker = stream.read(2)
        if len(marker) < 2 or marker[0] != 0xFF:
            raise ValueError(f"not a JPEG: {path}")
        length = int.from_bytes(stream.read(2), "big")
        if marker[1] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            stream.read(1)
            h = int.from_bytes(stream.read(2), "big")
            w = int.from_bytes(stream.read(2), "big")
            return w, h
        stream.read(length - 2)


def to_coco(items: list[tuple[str, str, list[dict]]], images_dir: Path) -> dict:
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "head"}],
    }
    cat = {"person": 1, "head": 2}
    for n, (_split, image_id, boxes) in enumerate(items):
        path = images_dir / f"{image_id}.jpg"
        try:
            w, h = jpeg_size(path)
        except (ValueError, OSError):
            continue
        coco["images"].append({"id": n, "file_name": path.name, "width": w, "height": h})
        for b in boxes:
            x0, y0, x1, y1 = b["box"]
            bw, bh = (x1 - x0) * w, (y1 - y0) * h
            coco["annotations"].append(
                {
                    "id": len(coco["annotations"]),
                    "image_id": n,
                    "category_id": cat[b["category"]],
                    "bbox": [x0 * w, y0 * h, bw, bh],
                    "area": bw * bh,
                    "iscrowd": int(b["group"]),
                }
            )
    return coco


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args(argv)

    cache = args.out / "cache"
    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    names = class_ids(cache)
    pool, head = names[POOL], names[HEAD_NAME]
    person = {names[n] for n in PERSON_NAMES}

    selected: list[tuple[str, str, list[dict]]] = []
    attribution: list[dict] = []
    for split, url in BOXES.items():
        found = scan_boxes(fetch(url, cache / f"{split}-bbox.csv"), pool, person, head)
        info = image_info(fetch(IMAGE_INFO[split], cache / f"{split}-info.csv"), set(found))
        print(f"{split}: {len(found)} pool images with people", flush=True)
        for image_id, boxes in sorted(found.items()):
            selected.append((split, image_id, boxes))
            row = info.get(image_id, {})
            attribution.append(
                {
                    "image_id": image_id,
                    "license": row.get("License", ""),
                    "author": row.get("Author", ""),
                    "url": row.get("OriginalLandingURL", ""),
                }
            )

    with ThreadPoolExecutor(args.workers) as pool_exec:
        ok = list(pool_exec.map(lambda it: download(it[0], it[1], images_dir / f"{it[1]}.jpg"), selected))
    selected = [it for it, good in zip(selected, ok, strict=True) if good]
    print(f"downloaded {len(selected)} images", flush=True)

    random.Random(0).shuffle(selected)
    n_val = int(len(selected) * args.val_fraction)
    for name, part in (("val", selected[:n_val]), ("train", selected[n_val:])):
        (args.out / f"{name}.json").write_text(json.dumps(to_coco(part, images_dir)))
    with (args.out / "attribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "license", "author", "url"])
        writer.writeheader()
        writer.writerows(attribution)
    print(f"wrote {args.out}/train.json, val.json, attribution.csv", flush=True)


if __name__ == "__main__":
    main()
