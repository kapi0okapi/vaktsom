"""Physically motivated compositing: people (and objects) placed under and on the water of real
pool photographs, with labels.

Real images of fully submerged people seen from above barely exist, and a detector that never saw
one will miss exactly the case that matters. Each composite takes a real pool photo (with its own
labelled people), finds the water, and adds cut-out people:

- under water (class person_under) at a random depth: per-channel absorption towards the colour
  of the surrounding water (red is lost first), blur growing with depth, refraction warp from a
  random wave field, caustic light, and the photo's own surface glints and reflections laid back
  on top, because the surface is above the body;
- at the surface (class person): head and shoulders dry, the rest seen through the water with the
  same processing at a shallow depth, so the two classes differ by the water and not by the cut;
- objects (toys, boards, bags) on the bottom as unlabelled distractors, so a still shape on the
  bottom is not automatically a person.

    python ml/synth/composite.py --backgrounds $DATA/openimages_pool --cutouts $DATA/coco_pool/cutouts \
        --objects $DATA/coco_objects --out $DATA/composites --n 20000 --workers 16
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

CATEGORIES = [{"id": 0, "name": "person"}, {"id": 1, "name": "head"}, {"id": 2, "name": "person_under"}]
# absorption per metre for (B, G, R): clear pool water swallows red within a few metres
ABSORB = np.array([0.05, 0.09, 0.42], np.float32)


def water_mask(img: np.ndarray, pool_boxes: list[list[float]]) -> np.ndarray:
    """Pool water: blue-cyan, saturated pixels, inside the pool boxes when they are known."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 80) & (h <= 125) & (s >= 40) & (v >= 40)).astype(np.uint8)
    if pool_boxes:
        roi = np.zeros_like(m)
        for x, y, w, bh in pool_boxes:
            roi[int(y) : int(y + bh), int(x) : int(x + w)] = 1
        m &= roi
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return m.astype(bool)


def smooth_field(h: int, w: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    small = rng.normal(0, 1, (max(int(h / scale), 2), max(int(w / scale), 2))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def caustics(h: int, w: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    c = np.zeros((h, w), np.float32)
    for k in range(2):
        n = smooth_field(h, w, scale * (k + 1), rng)
        c += np.exp(-np.abs(n) * 5)
    return c / max(c.max(), 1e-6)


def underwater(
    rgba: np.ndarray, depth: float, water_bgr: np.ndarray, wave: float, rng: np.random.Generator, dry_rows: int = 0
) -> np.ndarray:
    """Make a cut-out look ``depth`` metres under water. Rows above ``dry_rows`` stay dry."""
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3].astype(np.float32) / 255
    t = np.exp(-ABSORB * depth)
    wet = rgb * t + water_bgr * (1 - t)
    # caustics bring light back in shallow water
    c = caustics(h, w, rng, max(min(h, w) / 6, 4))
    wet *= (1 + 0.35 * np.exp(-depth / 2) * c)[..., None]
    sigma = 0.6 + 0.9 * depth
    wet = cv2.GaussianBlur(wet, (0, 0), sigma)
    a_wet = cv2.GaussianBlur(alpha, (0, 0), sigma * 0.8) * (0.95 - 0.08 * min(depth, 4))
    if dry_rows > 0:
        ramp = np.clip((np.arange(h) - dry_rows) / max(h * 0.06, 2), 0, 1)[:, None]
        rgb = rgb * (1 - ramp[..., None]) + wet * ramp[..., None]
        alpha = alpha * (1 - ramp) + a_wet * ramp
    else:
        rgb, alpha = wet, a_wet
    # refraction: the whole sheet of water bends the body
    amp = wave * (1.5 + 2.5 * min(depth, 3)) * max(h, w) / 120
    dx = smooth_field(h, w, max(min(h, w) / 5, 3), rng) * amp
    dy = smooth_field(h, w, max(min(h, w) / 5, 3), rng) * amp
    if dry_rows > 0:
        dx *= ramp
        dy *= ramp
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    rgb = cv2.remap(rgb, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    alpha = cv2.remap(alpha, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return np.dstack([np.clip(rgb, 0, 255), np.clip(alpha, 0, 1) * 255]).astype(np.float32)


def rotate_rgba(rgba: np.ndarray, angle: float, scale: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def paste(img: np.ndarray, layer: np.ndarray, x: int, y: int, glints: np.ndarray | None) -> tuple | None:
    """Alpha-blend ``layer`` (float RGBA) at (x, y); surface glints of the photo stay on top.
    Returns the tight box of the visible part, or None if almost nothing is visible."""
    H, W = img.shape[:2]
    h, w = layer.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    sub = layer[y0 - y : y1 - y, x0 - x : x1 - x]
    a = sub[..., 3:4] / 255
    roi = img[y0:y1, x0:x1].astype(np.float32)
    out = roi * (1 - a) + sub[..., :3] * a
    if glints is not None:
        g = glints[y0:y1, x0:x1][..., None]
        out = out * (1 - g) + roi * g  # the bright reflections lie on the surface, above the body
    img[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    vis = sub[..., 3] > 60
    if glints is not None:
        vis &= glints[y0:y1, x0:x1] < 0.5
    if vis.sum() < 0.25 * (sub[..., 3] > 60).sum() or vis.sum() < 30:
        return None
    ys, xs = np.nonzero(vis)
    return float(x0 + xs.min()), float(y0 + ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)


def glint_map(img: np.ndarray, water: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = ((hsv[..., 2] > 225) & (hsv[..., 1] < 60) & water).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 1.5)


def make_one(task: tuple) -> tuple[str, dict, list[dict]] | None:
    k, bg, anns, pool_boxes, cutouts, objects, out_dir, seed = task
    rng = np.random.default_rng(seed)
    pyr = random.Random(seed)
    img = cv2.imread(bg)
    if img is None:
        return None
    H, W = img.shape[:2]
    water = water_mask(img, pool_boxes)
    if water.sum() < 0.04 * H * W:
        return None
    glints = glint_map(img, water)
    people = [a for a in anns if a["category_id"] == 0 and not a.get("iscrowd")]
    ref_h = float(np.median([a["bbox"][3] for a in people])) if people else 0.18 * H
    ys, xs = np.nonzero(water)
    new_anns = [dict(a) for a in anns]
    taken = [a["bbox"] for a in anns if a["category_id"] == 0]

    def free(x: int, y: int, w: int, h: int) -> bool:
        for bx, by, bw, bh in taken:
            ix = max(0, min(x + w, bx + bw) - max(x, bx))
            iy = max(0, min(y + h, by + bh) - max(y, by))
            if ix * iy > 0.1 * min(w * h, bw * bh):
                return False
        return True

    for _ in range(int(rng.integers(1, 4))):
        mode = "under" if rng.random() < 0.6 else "surface"
        cut = cv2.imread(pyr.choice(cutouts), cv2.IMREAD_UNCHANGED)
        if cut is None or cut.shape[2] != 4:
            continue
        i = int(rng.integers(len(xs)))
        cx, cy = int(xs[i]), int(ys[i])
        size = ref_h * float(rng.uniform(0.6, 1.3)) * (0.75 if mode == "under" else 1.0)
        size = max(size, 24.0)
        scale = size / max(cut.shape[:2])
        if mode == "under":
            depth = float(rng.uniform(0.6, 4.5))
            layer = rotate_rgba(cut, float(rng.uniform(0, 360)), scale)
            layer = underwater(layer, depth, _local_water(img, water, cx, cy), float(rng.uniform(0.3, 1.0)), rng)
            cat = 2
        else:
            layer = rotate_rgba(cut, float(rng.normal(0, 25)), scale)
            dry = int(layer.shape[0] * rng.uniform(0.15, 0.35))
            layer = underwater(
                layer,
                float(rng.uniform(0.3, 0.9)),
                _local_water(img, water, cx, cy),
                float(rng.uniform(0.2, 0.8)),
                rng,
                dry_rows=dry,
            )
            cat = 0
        x0, y0 = cx - layer.shape[1] // 2, cy - layer.shape[0] // 2
        if not free(x0, y0, layer.shape[1], layer.shape[0]):
            continue
        box = paste(img, layer, x0, y0, glints if mode == "under" else None)
        if box is not None:
            new_anns.append({"category_id": cat, "bbox": list(box), "area": box[2] * box[3], "iscrowd": 0})
            taken.append(list(box))
    for _ in range(int(rng.integers(0, 3))):  # distractors on the bottom, never labelled
        if not objects:
            break
        obj = cv2.imread(pyr.choice(objects), cv2.IMREAD_UNCHANGED)
        if obj is None or obj.shape[2] != 4:
            continue
        i = int(rng.integers(len(xs)))
        cx, cy = int(xs[i]), int(ys[i])
        scale = ref_h * float(rng.uniform(0.15, 0.6)) / max(obj.shape[:2])
        layer = rotate_rgba(obj, float(rng.uniform(0, 360)), scale)
        layer = underwater(layer, float(rng.uniform(0.5, 4.5)), _local_water(img, water, cx, cy), 0.6, rng)
        paste(img, layer, cx - layer.shape[1] // 2, cy - layer.shape[0] // 2, glints)
    name = f"syn_{k:07d}.jpg"
    cv2.imwrite(str(Path(out_dir) / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 95))])
    return name, {"width": W, "height": H}, new_anns


def _local_water(img: np.ndarray, water: np.ndarray, cx: int, cy: int, r: int = 40) -> np.ndarray:
    H, W = img.shape[:2]
    y0, y1, x0, x1 = max(cy - r, 0), min(cy + r, H), max(cx - r, 0), min(cx + r, W)
    patch, m = img[y0:y1, x0:x1], water[y0:y1, x0:x1]
    if m.sum() < 10:
        return np.array([190, 160, 80], np.float32)
    return patch[m].astype(np.float32).mean(0)


def skin_fraction(path: str) -> float:
    """Share of the cut-out that looks like skin (YCrCb rule), a proxy for swimwear."""
    rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[2] != 4:
        return 0.0
    m = rgba[..., 3] > 127
    ycc = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2YCrCb)
    skin = (ycc[..., 1] >= 135) & (ycc[..., 1] <= 180) & (ycc[..., 2] >= 85) & (ycc[..., 2] <= 135)
    return float((skin & m).sum() / max(m.sum(), 1))


def load_backgrounds(root: Path) -> list[tuple[str, list[dict], list[list[float]]]]:
    """Images with their person/head labels and the pool boxes (from Open Images, when present)."""
    out = []
    for split in ("train.json",):
        coco = json.loads((root / split).read_text())
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        by_img: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        pools = json.loads((root / "pool_boxes.json").read_text()) if (root / "pool_boxes.json").exists() else {}
        for im in coco["images"]:
            anns = []
            for a in by_img.get(im["id"], []):
                name = cats[a["category_id"]]
                cid = {"person": 0, "head": 1}.get(name)
                if cid is not None:
                    anns.append(
                        {"category_id": cid, "bbox": a["bbox"], "area": a["area"], "iscrowd": a.get("iscrowd", 0)}
                    )
            out.append((str(root / "images" / im["file_name"]), anns, pools.get(Path(im["file_name"]).stem, [])))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backgrounds", type=Path, required=True)
    ap.add_argument("--cutouts", type=Path, required=True)
    ap.add_argument("--objects", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-skin", type=float, default=0.25, help="prefer people in swimwear")
    args = ap.parse_args(argv)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    bgs = load_backgrounds(args.backgrounds)
    cutouts = [p for p in sorted(str(p) for p in args.cutouts.glob("*.png")) if skin_fraction(p) >= args.min_skin]
    print(f"{len(cutouts)} cut-outs with at least {args.min_skin:.0%} skin")
    objects = sorted(str(p) for p in args.objects.glob("*.png")) if args.objects else []
    rnd = random.Random(args.seed)
    tasks = []
    for k in range(args.n):
        bg, anns, pools = bgs[rnd.randrange(len(bgs))]
        tasks.append((k, bg, anns, pools, cutouts, objects, str(args.out), args.seed * 1_000_003 + k))
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(make_one, tasks, chunksize=32):
            if r is not None:
                results.append(r)
    rnd.shuffle(results)
    n_val = int(len(results) * args.val)
    for split, part in (("val", results[:n_val]), ("train", results[n_val:])):
        coco = {"images": [], "annotations": [], "categories": CATEGORIES}
        for i, (name, size, anns) in enumerate(part):
            coco["images"].append({"id": i, "file_name": name, **size})
            for a in anns:
                coco["annotations"].append({**a, "id": len(coco["annotations"]), "image_id": i})
        (args.out / f"{split}.json").write_text(json.dumps(coco))
        print(
            f"{split}: {len(part)} images, {len(coco['annotations'])} boxes "
            f"({sum(a['category_id'] == 2 for a in coco['annotations'])} under water)"
        )


if __name__ == "__main__":
    main()
