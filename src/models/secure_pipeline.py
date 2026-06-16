"""
Secure OCR pipeline for high-integrity (banking) documents.

  PaddleOCR detection  ->  Top-K CTC recognition  ->  whitelist logic gate

No text is ever generated or edited. Each field is either:
  - accepted as one of the model's own Top-K hypotheses that passed the gate, or
  - returned as the verbatim top-1 prediction flagged [REVIEW_REQUIRED].

Requires the exported recognition model and its char dict at:
    models/paddle_rec/inference.pdmodel
    models/paddle_rec/inference.pdiparams
    models/paddle_rec/en_dict.txt
"""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from paddleocr import PaddleOCR

from src.models.topk_recognizer import TopKRecognizer
from src.bank_validation import select_prediction

REC_MODEL_DIR = "models/paddle_rec"
DICT_PATH     = "models/paddle_rec/en_dict.txt"


def _four_point_crop(img_bgr, box):
    """Perspective-rectify a 4-point detection box into an upright crop."""
    box = np.array(box, dtype="float32")
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    if w < 1 or h < 1:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    M = cv2.getPerspectiveTransform(box, dst)
    return cv2.warpPerspective(img_bgr, M, (w, h))


class SecurePipeline:
    def __init__(self, use_gpu: bool = False, top_k: int = 5):
        if not Path(REC_MODEL_DIR).exists():
            raise FileNotFoundError(
                f"{REC_MODEL_DIR} not found — export the trained model there first."
            )
        self.top_k = top_k
        # PaddleOCR is used for DETECTION only; recognition is replaced by our
        # Top-K recognizer so we can apply the validation gate.
        self.detector = PaddleOCR(use_angle_cls=False, lang="en",
                                  show_log=False, use_gpu=use_gpu)
        self.recognizer = TopKRecognizer(REC_MODEL_DIR, DICT_PATH, use_gpu=use_gpu)

    def extract(self, image: Image.Image, mode: str = "any") -> list:
        """
        Returns a list of per-field result dicts, top-to-bottom:
            {text, accepted, matched_rank, score, candidates}
        `candidates` is the raw Top-K list (text, score) for full auditability.
        """
        img_bgr = np.ascontiguousarray(
            np.array(image.convert("RGB"))[:, :, ::-1])        # RGB -> BGR

        # Use the low-level detector directly — PaddleOCR 2.7's high-level
        # .ocr(rec=False) has a numpy-truthiness bug.
        dt_boxes, _ = self.detector.text_detector(img_bgr)
        boxes = [] if dt_boxes is None else list(dt_boxes)
        boxes = sorted(boxes, key=lambda b: min(p[1] for p in b))  # reading order

        results = []
        for box in boxes:
            crop = _four_point_crop(img_bgr, box)
            if crop is None or crop.size == 0:
                continue
            candidates = self.recognizer.recognize_topk(crop, top_k=self.top_k)
            decision = select_prediction(candidates, mode=mode)
            decision["candidates"] = candidates     # keep full hypothesis list
            decision["box"] = [[float(p[0]), float(p[1])] for p in box]
            results.append(decision)
        return results

    def extract_text(self, image: Image.Image, mode: str = "any") -> str:
        """Convenience: join the per-field decisions into a single text block."""
        return "\n".join(r["text"] for r in self.extract(image, mode=mode)
                         if r["text"].strip())


# ──────────────────────────────────────────────────────────────────────────────
# Visualization: draw color-coded detection boxes on the document
# ──────────────────────────────────────────────────────────────────────────────
# RGB colors
_COLOR_ACCEPTED = (0, 160, 0)      # green  — passed the whitelist gate
_COLOR_REVIEW   = (230, 70, 30)    # red    — flagged [REVIEW_REQUIRED]


def render_annotated(image: Image.Image, results: list) -> Image.Image:
    """Return a copy of the image with each field's box drawn and numbered."""
    img = np.array(image.convert("RGB")).copy()
    for i, r in enumerate(results, start=1):
        if not r.get("box"):
            continue
        box   = np.array(r["box"], dtype=np.int32)
        color = _COLOR_ACCEPTED if r["accepted"] else _COLOR_REVIEW
        cv2.polylines(img, [box], isClosed=True, color=color, thickness=2)
        # field number tag at the top-left corner of the box
        x, y = int(box[0][0]), int(box[0][1])
        cv2.putText(img, str(i), (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return Image.fromarray(img)
