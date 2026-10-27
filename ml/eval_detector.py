"""Compare detectors on held-out sets: what matters for drowning detection is finding every body,
above all the fully submerged ones, and telling the two apart.

Reported per test set:
- person AP50 and AP50:95, class-agnostic (a submerged body found as "person" still counts),
- recall of fully submerged bodies at the operating threshold and at 0.95 precision on all people,
- submersion accuracy: for matched bodies, is the larger of the two person scores the right one,
- per-class AP (COCO) where the model has the class.

    python ml/eval_detector.py --model medium --weights $RUNS/det_medium/checkpoint_best_total.pth \
        --test $DATA/renders_test/val.json:$DATA/renders_test/images --name det_medium --out $RUNS/eval_det
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-9)


def agnostic_ap(gts: list[np.ndarray], preds: list[tuple[np.ndarray, np.ndarray]], thr: float) -> tuple[float, list]:
    """AP at one IoU threshold for 'any person'; also returns (score, tp) pairs."""
    scored, n_gt = [], sum(len(g) for g in gts)
    for g, (boxes, scores) in zip(gts, preds, strict=True):
        order = np.argsort(-scores)
        used = np.zeros(len(g), bool)
        m = iou(boxes[order], g) if len(g) and len(boxes) else np.zeros((len(boxes), len(g)))
        for k, i in enumerate(order):
            j = int(np.argmax(m[k])) if len(g) else -1
            tp = j >= 0 and m[k, j] >= thr and not used[j]
            if tp:
                used[j] = True
            scored.append((float(scores[i]), tp))
    scored.sort(key=lambda s: -s[0])
    tp = np.cumsum([s[1] for s in scored])
    fp = np.cumsum([not s[1] for s in scored])
    rec = tp / max(n_gt, 1)
    prec = tp / np.maximum(tp + fp, 1)
    ap = 0.0
    for r in np.linspace(0, 1, 101):  # COCO-style 101-point interpolation
        p = prec[rec >= r].max() if (rec >= r).any() else 0.0
        ap += p / 101
    return float(ap), scored


def _by_depth(rows: list[tuple]) -> dict:
    out = {}
    for lo, hi in ((0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 6.0)):
        sel = [f for d, _, f in rows if d is not None and lo <= d < hi]
        if sel:
            out[f"{lo:.0f}-{hi:.0f} m"] = {"bodies": len(sel), "recall": float(np.mean(sel))}
    return out


def run(args) -> dict:
    import rfdetr
    from PIL import Image

    cls = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "large": "RFDETRLarge",
        "seg-medium": "RFDETRSegMedium",
    }[args.model]
    model = getattr(rfdetr, cls)(pretrain_weights=str(args.weights))
    report = {"name": args.name, "weights": str(args.weights), "sets": {}}
    for spec in args.test:
        ann_path, img_dir = spec.split(":")
        coco = json.loads(Path(ann_path).read_text())
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        by_img: dict[int, list] = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        gts, gts_under, preds, sub_ok, sub_n = [], [], [], 0, 0
        under_rows = []  # (scene depth, state, found at threshold) for every body under water
        fp_per_image = []
        model_names = getattr(model, "class_names", None)
        for im in coco["images"][: args.limit]:
            img = Image.open(Path(img_dir) / im["file_name"]).convert("RGB")
            det = model.predict(img, threshold=0.05)
            boxes, scores, labels = np.asarray(det.xyxy), np.asarray(det.confidence), np.asarray(det.class_id)
            names = [model_names[int(c)] if model_names and int(c) < len(model_names) else str(int(c)) for c in labels]
            person = (
                np.array([n in ("person", "person_under") for n in names], bool) if len(names) else np.zeros(0, bool)
            )
            anns = by_img.get(im["id"], [])
            g = np.array(
                [
                    [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                    for a in anns
                    if cats[a["category_id"]] in ("person", "person_under")
                ]
            ).reshape(-1, 4)
            g_under = np.array(
                [
                    cats[a["category_id"]] == "person_under"
                    for a in anns
                    if cats[a["category_id"]] in ("person", "person_under")
                ],
                bool,
            )
            gts.append(g)
            gts_under.append(g_under)
            preds.append((boxes[person], scores[person]))
            keep = scores[person] >= args.threshold
            body_anns = [a for a in anns if cats[a["category_id"]] in ("person", "person_under")]
            m_all = iou(g, boxes[person][keep]) if len(g) and keep.any() else np.zeros((len(g), int(keep.sum())))
            for gi, a in enumerate(body_anns):
                if g_under[gi]:
                    found = bool(m_all.shape[1] and m_all[gi].max() >= 0.5)
                    under_rows.append((im.get("scene_depth"), a.get("state"), found))
            fp_per_image.append(int(((m_all.max(0) if len(g) else np.zeros(m_all.shape[1])) < 0.5).sum()))
            # submersion: matched bodies, predicted class vs truth
            if len(g) and person.any():
                m = iou(g, boxes[person])
                for gi in range(len(g)):
                    j = int(np.argmax(m[gi]))
                    if m[gi, j] >= 0.5:
                        sub_n += 1
                        sub_ok += int((np.array(names)[person][j] == "person_under") == g_under[gi])
        ap50, scored = agnostic_ap(gts, preds, 0.5)
        aps = [agnostic_ap(gts, preds, t)[0] for t in np.arange(0.5, 0.96, 0.05)]
        # recall of submerged bodies at the operating threshold
        under_found, under_total = 0, 0
        for g, gu, (boxes, scores) in zip(gts, gts_under, preds, strict=True):
            keep = scores >= args.threshold
            if not gu.any():
                continue
            under_total += int(gu.sum())
            if keep.any():
                m = iou(g[gu], boxes[keep])
                under_found += int((m.max(1) >= 0.5).sum())
        report["sets"][Path(ann_path).parent.name + "/" + Path(ann_path).stem] = {
            "images": min(len(coco["images"]), args.limit),
            "people": int(sum(len(g) for g in gts)),
            "under": under_total,
            "person_AP50": ap50,
            "person_AP50_95": float(np.mean(aps)),
            "under_recall@thr": under_found / max(under_total, 1),
            "threshold": args.threshold,
            "submersion_accuracy": sub_ok / max(sub_n, 1) if sub_n else None,
            "false_positives_per_image@thr": float(np.mean(fp_per_image)) if fp_per_image else None,
            "under_recall_by_depth": _by_depth(under_rows),
            "under_recall_by_state": {
                st: float(np.mean([f for _, s_, f in under_rows if s_ == st]))
                for st in sorted({s_ for _, s_, _ in under_rows if s_})
            },
        }
        print(args.name, json.dumps(report["sets"], indent=1), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.name}.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--test", action="append", required=True, help="annotations.json:image_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=100000)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
