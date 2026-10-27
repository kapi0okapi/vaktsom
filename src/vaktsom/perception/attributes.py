"""Posture and submersion per detected person, from a crop (ONNX, see ml/train_attributes.py).

Fills ``p_vertical`` (upright: standing, treading water, instinctive drowning response) and refines
``p_under`` by averaging with the detector's own estimate.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vaktsom.perception.types import Detection

SIZE = 128
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class OnnxAttributes:
    def __init__(self, path: str | Path, providers: list[str] | None = None):
        import onnxruntime as ort

        if providers is None:
            providers = [
                p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()
            ]
        from vaktsom.perception.detector import cpu_budget

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = cpu_budget()
        self.session = ort.InferenceSession(str(path), opts, providers=providers)

    @staticmethod
    def crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        side = max(w, h) * 1.3
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        src = np.float32(
            [[cx - side / 2, cy - side / 2], [cx + side / 2, cy - side / 2], [cx - side / 2, cy + side / 2]]
        )
        dst = np.float32([[0, 0], [SIZE, 0], [0, SIZE]])
        patch = cv2.warpAffine(
            image,
            cv2.getAffineTransform(src, dst),
            (SIZE, SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((rgb - MEAN) / STD).transpose(2, 0, 1)

    def annotate(self, image: np.ndarray, detections: list[Detection]) -> None:
        if not detections:
            return
        batch = np.stack([self.crop(image, d.box) for d in detections]).astype(np.float32)
        logits = self.session.run(None, {"crop": batch})[0]
        p = 1.0 / (1.0 + np.exp(-logits))
        for d, (under, upright) in zip(detections, p, strict=True):
            # average with the detector when it has a submersion class, otherwise this is the estimate
            known = d.extra.get("under_from_detector", True)
            d.p_under = float(0.5 * (d.p_under + under)) if known else float(under)
            d.p_vertical = float(upright)
