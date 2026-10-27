"""Per-person attribute model: upright or lying (posture) and fully under water or not, from a crop.

The detector finds the body; this small network looks at the crop and says how it lies. Posture
labels come from the ceiling-camera renders (standing, treading water and the instinctive drowning
response are upright; swimming, floating, sinking and lying on the bottom are not). Submersion
labels come from both the renders and the composites. Missing labels are masked out of the loss.

    python ml/train_attributes.py --renders $DATA/renders --composites $DATA/composites --out $RUNS/attributes
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

UPRIGHT = {"stand", "tread", "idr"}
SIZE = 128


def items_from_renders(root: Path) -> list[dict]:
    out = []
    for f in sorted(root.glob("labels_*.json")):
        for it in json.loads(f.read_text()):
            for a in it["annotations"]:
                if a["category"] == "head" or a["bbox"][2] < 12 or a["bbox"][3] < 12:
                    continue
                out.append(
                    {
                        "image": str(root / "images" / it["file_name"]),
                        "box": a["bbox"],
                        "under": float(a["category"] == "person_under"),
                        "upright": float(a.get("pose") in UPRIGHT),
                    }
                )
    return out


def items_from_coco(root: Path, split: str = "train") -> list[dict]:
    coco = json.loads((root / f"{split}.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    files = {im["id"]: im["file_name"] for im in coco["images"]}
    out = []
    for a in coco["annotations"]:
        n = names[a["category_id"]]
        if n not in ("person", "person_under") or a["bbox"][2] < 12 or a["bbox"][3] < 12:
            continue
        out.append(
            {
                "image": str(root / "images" / files[a["image_id"]]),
                "box": a["bbox"],
                "under": float(n == "person_under"),
                "upright": -1.0,
            }
        )  # posture unknown
    return out


class Crops(torch.utils.data.Dataset):
    def __init__(self, items: list[dict], train: bool):
        self.items, self.train = items, train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        img = Image.open(it["image"]).convert("RGB")
        x, y, w, h = it["box"]
        pad = 0.15 * max(w, h)
        if self.train:  # box jitter like a real detector's
            x += random.uniform(-0.08, 0.08) * w
            y += random.uniform(-0.08, 0.08) * h
            pad *= random.uniform(0.7, 1.4)
        side = max(w, h) + 2 * pad
        cx, cy = x + w / 2, y + h / 2
        crop = img.crop((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)).resize((SIZE, SIZE))
        a = np.asarray(crop, np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:
                a = a[:, ::-1]
            a = np.clip(a * random.uniform(0.75, 1.25) + random.uniform(-0.08, 0.08), 0, 1)
        a = (a - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.from_numpy(a.transpose(2, 0, 1).astype(np.float32))
        return t, torch.tensor([it["under"], it["upright"]], dtype=torch.float32)


def model() -> nn.Module:
    import torchvision

    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, 2)  # logits: under water, upright
    return net


def run(args) -> None:
    renders = items_from_renders(args.renders)
    comps = items_from_coco(args.composites) if args.composites else []
    random.Random(1).shuffle(comps)
    items = renders + comps[: args.max_composites]
    random.Random(0).shuffle(items)
    print(f"{len(renders)} render crops, {min(len(comps), args.max_composites)} composite crops", flush=True)
    n_val = max(int(len(items) * 0.05), 200)
    val, train = items[:n_val], items[n_val:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = model().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(
        Crops(train, True), batch_size=128, shuffle=True, num_workers=args.workers, drop_last=True
    )
    vl = torch.utils.data.DataLoader(Crops(val, False), batch_size=256, num_workers=args.workers)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=args.epochs * len(dl))
    best = 0.0
    args.out.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        net.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            logits = net(x)
            mask = (y >= 0).float()
            loss = (
                nn.functional.binary_cross_entropy_with_logits(logits, y.clamp(min=0), reduction="none") * mask
            ).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        ok = np.zeros(2)
        n = np.zeros(2)
        with torch.no_grad():
            for x, y in vl:
                p = (torch.sigmoid(net(x.to(dev))).cpu() > 0.5).float()
                for k in range(2):
                    m = y[:, k] >= 0
                    ok[k] += (p[m, k] == y[m, k]).sum().item()
                    n[k] += m.sum().item()
        acc = ok / np.maximum(n, 1)
        print(
            f"epoch {ep + 1}: under-water accuracy {acc[0]:.3f} (n={int(n[0])}), "
            f"upright accuracy {acc[1]:.3f} (n={int(n[1])})",
            flush=True,
        )
        if acc.mean() > best:
            best = acc.mean()
            torch.save(net.state_dict(), args.out / "best.pt")
            (args.out / "metrics.json").write_text(
                json.dumps(
                    {
                        "epoch": ep + 1,
                        "under_acc": acc[0],
                        "upright_acc": acc[1],
                        "val_under": int(n[0]),
                        "val_upright": int(n[1]),
                    }
                )
            )
    net.load_state_dict(torch.load(args.out / "best.pt", map_location=dev))
    net.eval().cpu()
    torch.onnx.export(
        net,
        torch.zeros(1, 3, SIZE, SIZE),
        args.out / "attributes.onnx",
        input_names=["crop"],
        output_names=["logits"],
        dynamic_axes={"crop": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    print("exported", args.out / "attributes.onnx")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--composites", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-composites", type=int, default=40000)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
