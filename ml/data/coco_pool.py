"""Pool images and swimwear person cut-outs from COCO 2017.

1. Images whose captions describe a swimming pool become a COCO detection subset (person, with
   the original instance masks), added to the pool training data.
2. Unoccluded, untruncated people from beach/pool/swimming images are cut out with their masks
   (RGBA PNG, cropped) for the compositing engine, which places them under and on the water.

    python3 ml/data/coco_pool.py --coco $DATA/coco --out $DATA/coco_pool
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

POOL = re.compile(r"\b(swimming pool|pool)\b")
POOL_CONTEXT = re.compile(r"\b(swim\w*|water|diving|dive|splash\w*|float\w*|lap|lane)\b")
NOT_POOL = re.compile(r"\b(pool (table|cue|stick|ball)|billiard\w*|carpool|snooker)\b")
SWIMWEAR = re.compile(r"\b(beach|swim\w*|surf\w*|bikini|shirtless|pool|wading|ocean|lake)\b")


def fetch_annotations(coco: Path) -> Path:
    ann = coco / "annotations"
    if not (ann / "instances_train2017.json").exists() or not (ann / "captions_train2017.json").exists():
        zip_path = coco / "annotations_trainval2017.zip"
        if not zip_path.exists():
            print(f"downloading {ANNOTATIONS}", flush=True)
            urllib.request.urlretrieve(ANNOTATIONS, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(coco)
    return ann


def captions_by_image(path: Path) -> dict[int, str]:
    caps: dict[int, list[str]] = defaultdict(list)
    for a in json.loads(path.read_text())["annotations"]:
        caps[a["image_id"]].append(a["caption"].lower())
    return {k: " ".join(v) for k, v in caps.items()}


def is_pool(text: str) -> bool:
    return bool(POOL.search(text) and POOL_CONTEXT.search(text) and not NOT_POOL.search(text))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco", type=Path, required=True, help="folder with train2017/ and val2017/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-cutout-area", type=float, default=96 * 96)
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image
    from pycocotools import mask as mask_utils

    ann_dir = fetch_annotations(args.coco)
    image_root = args.coco / "images" if (args.coco / "images").is_dir() else args.coco
    args.out.mkdir(parents=True, exist_ok=True)
    cut_dir = args.out / "cutouts"
    cut_dir.mkdir(exist_ok=True)
    n_cut = 0
    for split in ("train2017", "val2017"):
        inst = json.loads((ann_dir / f"instances_{split}.json").read_text())
        caps = captions_by_image(ann_dir / f"captions_{split}.json")
        images = {im["id"]: im for im in inst["images"]}
        people = defaultdict(list)
        for a in inst["annotations"]:
            if a["category_id"] == 1:
                people[a["image_id"]].append(a)

        pool_ids = sorted(i for i in people if is_pool(caps.get(i, "")))
        subset = {
            "images": [{**images[i], "file_name": str(image_root / split / images[i]["file_name"])} for i in pool_ids],
            "annotations": [a for i in pool_ids for a in people[i]],
            "categories": [{"id": 1, "name": "person"}],
        }
        (args.out / f"pool_{split}.json").write_text(json.dumps(subset))
        print(f"{split}: {len(pool_ids)} pool images, {len(subset['annotations'])} people", flush=True)

        # cut-outs: whole, single people in swimwear contexts
        for image_id, anns in people.items():
            if not SWIMWEAR.search(caps.get(image_id, "")):
                continue
            im = images[image_id]
            good = [a for a in anns if not a["iscrowd"] and a["area"] >= args.min_cutout_area]
            if not good:
                continue
            img = None
            for a in good:
                x, y, w, h = a["bbox"]
                # people cut by the image border would teach the compositor half bodies
                if x < 2 or y < 2 or x + w > im["width"] - 2 or y + h > im["height"] - 2:
                    continue
                if img is None:
                    img = np.asarray(Image.open(image_root / split / im["file_name"]).convert("RGB"))
                rle = mask_utils.frPyObjects(a["segmentation"], im["height"], im["width"])
                m = mask_utils.decode(mask_utils.merge(rle) if isinstance(rle, list) else rle)
                x0, y0, x1, y1 = int(x), int(y), int(x + w) + 1, int(y + h) + 1
                rgba = np.dstack([img[y0:y1, x0:x1], m[y0:y1, x0:x1] * 255]).astype(np.uint8)
                Image.fromarray(rgba).save(cut_dir / f"coco_{a['id']}.png")
                n_cut += 1
    print(f"{n_cut} cut-outs in {cut_dir}", flush=True)


if __name__ == "__main__":
    main()
