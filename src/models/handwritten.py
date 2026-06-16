"""
Hybrid handwritten text recognition.

  PaddleOCR  →  detection (finds line/word boxes on arbitrary images)
  TrOCR      →  recognition (reads each crop — strong on free/cursive writing)

PaddleOCR alone was the weak link on handwriting recognition; TrOCR-large is
purpose-built for it. PaddleOCR is kept only as the detector, where it's strong.
TrOCR runs on GPU in fp16 (see trocr_recognizer); detection stays on CPU.
"""

# torch / TrOCR MUST be imported before paddleocr — loading torch after paddle
# triggers a Windows DLL conflict (shm.dll fails to load).
import torch  # noqa: F401
from src.models.trocr_recognizer import TrOCRRecognizer

import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
from paddleocr import PaddleOCR

_MAX_SIDE = 1800     # downscale huge photos before detection
_CHUNK    = 8        # crops per TrOCR batch (bounded for 6 GB VRAM)


def _downscale(image: Image.Image) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= _MAX_SIDE:
        return image
    s = _MAX_SIDE / float(longest)
    return image.resize((int(w * s), int(h * s)))


def _four_point_crop(img_bgr, box):
    """Perspective-rectify a 4-point detection box into an upright BGR crop."""
    box = np.array(box, dtype="float32")
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    if w < 2 or h < 2:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    M = cv2.getPerspectiveTransform(box, dst)
    return cv2.warpPerspective(img_bgr, M, (w, h))


def _line_groups(boxes) -> list:
    """Group detection boxes into reading order: lines top-to-bottom, each line
    sorted left-to-right. Returns a list of lines (each a list of boxes)."""
    def top(b):    return min(p[1] for p in b)
    def left(b):   return min(p[0] for p in b)
    def height(b): return max(p[1] for p in b) - min(p[1] for p in b)

    boxes = sorted(boxes, key=top)
    med_h = float(np.median([height(b) for b in boxes])) or 10.0

    lines, current, anchor = [], [boxes[0]], top(boxes[0])
    for b in boxes[1:]:
        if top(b) - anchor < med_h * 0.6:
            current.append(b)
        else:
            lines.append(current)
            current, anchor = [b], top(b)
    lines.append(current)

    for line in lines:
        line.sort(key=left)
    return lines


class HandwrittenOCR:
    def __init__(self, use_gpu: bool = False):
        # PaddleOCR is used for DETECTION only.
        print("[Handwritten] Loading PaddleOCR detector...")
        # Higher detection resolution + lower thresholds so faint/thin ink is
        # found (safe now that the pyarrow segfault — not this — is fixed).
        self.detector = PaddleOCR(use_angle_cls=False, lang="en",
                                  show_log=False, use_gpu=use_gpu,
                                  det_limit_side_len=1920,
                                  det_db_box_thresh=0.3, det_db_thresh=0.2)
        # TrOCR does the recognition.
        self.recognizer = TrOCRRecognizer()
        print("[Handwritten] Hybrid ready (Paddle detect + TrOCR read).")

    def read_image(self, image: Image.Image) -> str:
        """Detect lines with PaddleOCR, read each with TrOCR, return ordered text."""
        image   = _downscale(image)
        img_bgr = np.ascontiguousarray(np.array(image.convert("RGB"))[:, :, ::-1])

        boxes, _ = self.detector.text_detector(img_bgr)
        if boxes is None or len(boxes) == 0:
            return ""

        lines = _line_groups(boxes)

        # Crop every box in reading order, tracking which line it belongs to.
        items = []   # (line_index, PIL crop)
        for li, line in enumerate(lines):
            for box in line:
                crop = _four_point_crop(img_bgr, box)
                if crop is not None and crop.size:
                    items.append((li, Image.fromarray(crop[:, :, ::-1])))  # BGR->RGB

        if not items:
            return ""

        # Read crops with TrOCR in memory-bounded chunks.
        texts = []
        for i in range(0, len(items), _CHUNK):
            chunk = [c for _, c in items[i:i + _CHUNK]]
            texts.extend(self.recognizer.read_batch(chunk))

        # Reassemble: words within a line joined by space, lines by newline.
        line_words = defaultdict(list)
        for (li, _), txt in zip(items, texts):
            if txt.strip():
                line_words[li].append(txt.strip())

        return "\n".join(" ".join(line_words[li])
                         for li in range(len(lines)) if line_words[li])

    def read_crop(self, image: Image.Image) -> str:
        """Recognize a single pre-cropped region (used by the mixed pipeline)."""
        return self.recognizer.read_crop(image)
