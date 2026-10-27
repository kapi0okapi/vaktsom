"""The trained detector at run time: ONNX Runtime (TensorRT or CUDA on the site server, CoreML or CPU
on a laptop), batched over cameras.

The model (RF-DETR, fine-tuned) has three classes: a person not fully under water, a head, and a
fully submerged person. Per detected person the two person classes give the probability of full
submersion; a head box inside a person box is direct evidence of a head above water.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from vaktsom.perception.types import Detection
from vaktsom.tracking.camera_tracker import iou_matrix

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def cpu_budget() -> int:
    """Threads for ONNX Runtime: the CPUs this process may use (Slurm allocation, container limits),
    not every core of the machine, which makes it crawl on shared nodes."""
    import os

    env = os.environ.get("VAKTSOM_THREADS") or os.environ.get("SLURM_CPUS_PER_TASK")
    if env:
        return max(int(env), 1)
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except AttributeError:  # macOS
        return max((os.cpu_count() or 2) // 2, 1)


class OnnxDetector:
    def __init__(
        self,
        path: str | Path,
        classes: list[str] | None = None,
        score: float = 0.15,
        providers: list[str] | None = None,
    ):
        import onnxruntime as ort

        path = Path(path)
        meta = path.with_suffix(".json")
        info = json.loads(meta.read_text()) if meta.exists() else {}
        self.classes = classes or info.get("classes", ["person", "head", "person_under"])
        self.score = score
        if providers is None:
            available = ort.get_available_providers()
            # CoreML cannot run the transformer decoder (zero-size slices); laptops use the CPU
            providers = [
                p
                for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                if p in available
            ]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = cpu_budget()
        self.session = ort.InferenceSession(str(path), opts, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(info.get("resolution") or inp.shape[-1])
        self.batch_fixed = isinstance(inp.shape[0], int) and inp.shape[0] > 0
        outs = [o.name for o in self.session.get_outputs()]
        self.box_out = next((o for o in outs if "det" in o or "box" in o), outs[0])
        self.logit_out = next((o for o in outs if "label" in o or "logit" in o), outs[1])

    def _prep(self, img: np.ndarray) -> np.ndarray:
        x = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((x - MEAN) / STD).transpose(2, 0, 1)

    def __call__(self, images: list[np.ndarray]) -> list[list[Detection]]:
        if not images:
            return []
        batch = np.stack([self._prep(im) for im in images])
        if self.batch_fixed:
            outs = [self.session.run([self.box_out, self.logit_out], {self.input_name: b[None]}) for b in batch]
            boxes = np.concatenate([o[0] for o in outs])
            logits = np.concatenate([o[1] for o in outs])
        else:
            boxes, logits = self.session.run([self.box_out, self.logit_out], {self.input_name: batch})
        return [self._post(boxes[i], logits[i], images[i].shape[:2]) for i in range(len(images))]

    def _post(self, boxes: np.ndarray, logits: np.ndarray, hw: tuple[int, int]) -> list[Detection]:
        prob = 1.0 / (1.0 + np.exp(-logits))  # (Q, C) sigmoid scores, one per class
        cls = {name: k for k, name in enumerate(self.classes)}
        p_surface = prob[:, cls["person"]]
        p_under = prob[:, cls["person_under"]] if "person_under" in cls else np.zeros(len(prob))
        p_head = prob[:, cls["head"]] if "head" in cls else np.zeros(len(prob))
        h, w = hw
        xyxy = np.stack(
            [
                (boxes[:, 0] - boxes[:, 2] / 2) * w,
                (boxes[:, 1] - boxes[:, 3] / 2) * h,
                (boxes[:, 0] + boxes[:, 2] / 2) * w,
                (boxes[:, 1] + boxes[:, 3] / 2) * h,
            ],
            1,
        )
        person = np.maximum(p_surface, p_under)
        keep = np.flatnonzero(person >= self.score)
        keep = keep[np.argsort(-person[keep])]
        # one query per person: the two person classes can fire on two queries for one body
        chosen: list[int] = []
        for i in keep:
            if chosen and iou_matrix(xyxy[i][None], xyxy[chosen]).max() > 0.7:
                continue
            chosen.append(int(i))
        heads = np.flatnonzero(p_head >= 0.35)
        dets = []
        for i in chosen:
            under = float(p_under[i] / max(p_surface[i] + p_under[i], 1e-6))
            if len(heads):  # a head box inside the person's box: the head is above the water
                b = xyxy[i]
                inside = [
                    j
                    for j in heads
                    if b[0] <= (xyxy[j, 0] + xyxy[j, 2]) / 2 <= b[2] and b[1] <= (xyxy[j, 1] + xyxy[j, 3]) / 2 <= b[3]
                ]
                if inside:
                    under *= 1.0 - 0.8 * float(max(p_head[j] for j in inside))
            dets.append(
                Detection(
                    box=np.clip(xyxy[i], 0, [w - 1, h - 1, w - 1, h - 1]),
                    score=float(person[i]),
                    p_under=under,
                    extra={"under_from_detector": "person_under" in cls},
                )
            )
        return dets
