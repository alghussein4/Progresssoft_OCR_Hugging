"""
Handwritten text recognition using PaddleOCR.

PaddleOCR performs its own robust text detection (locating each line/word)
and recognition, so no manual line segmentation is required — this is what
makes it read arbitrary images reliably.

If a fine-tuned recognition model is present at models/paddle_rec/, it is
used automatically. Otherwise PaddleOCR's pretrained English recognizer is
used as a fallback.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from paddleocr import PaddleOCR

REC_MODEL_DIR = "models/paddle_rec"
_MAX_SIDE = 1800   # downscale images whose longest side exceeds this


def _downscale(image: Image.Image) -> Image.Image:
    """Shrink very large photos so CPU inference stays fast and responsive."""
    w, h = image.size
    longest = max(w, h)
    if longest <= _MAX_SIDE:
        return image
    scale = _MAX_SIDE / float(longest)
    return image.resize((int(w * scale), int(h * scale)))


def _reading_order_lines(items) -> list:
    """
    Order detected word-boxes into natural reading order:
    group boxes into text lines by vertical position, then sort each line
    left-to-right. Returns a list of line strings.
    """
    if not items:
        return []

    def top(it):    return min(p[1] for p in it[0])
    def left(it):   return min(p[0] for p in it[0])
    def height(it): return max(p[1] for p in it[0]) - min(p[1] for p in it[0])

    items = [it for it in items if it[1][0].strip()]
    if not items:
        return []

    med_h = float(np.median([height(it) for it in items])) or 10.0
    items.sort(key=top)

    lines, current, anchor = [], [], top(items[0])
    for it in items:
        if it is items[0] or top(it) - anchor < med_h * 0.6:
            current.append(it)
        else:
            lines.append(current)
            current = [it]
            anchor = top(it)
    if current:
        lines.append(current)

    out = []
    for line in lines:
        line.sort(key=left)                       # left-to-right within the line
        out.append(" ".join(it[1][0].strip() for it in line))
    return out


class HandwrittenOCR:
    def __init__(self, use_gpu: bool = False):
        kwargs = dict(
            use_angle_cls=True,
            lang="en",
            show_log=False,
            use_gpu=use_gpu,
        )

        if Path(REC_MODEL_DIR).exists():
            kwargs["rec_model_dir"] = REC_MODEL_DIR
            print(f"[Handwritten] Using fine-tuned recognizer: {REC_MODEL_DIR}")
        else:
            print("[Handwritten] Fine-tuned recognizer not found — "
                  "using PaddleOCR pretrained English model.")

        print("[Handwritten] Loading PaddleOCR...")
        self.ocr = PaddleOCR(**kwargs)
        print("[Handwritten] Ready.")

    def read_image(self, image: Image.Image) -> str:
        """
        Detect and read all text in a full image.
        Returns the recognized lines joined top-to-bottom in reading order.
        """
        image = _downscale(image)          # cap huge phone photos for speed
        img    = np.array(image.convert("RGB"))
        result = self.ocr.ocr(img, cls=True)

        # result is [ [ [box, (text, conf)], ... ] ] — None if nothing found
        if not result or result[0] is None:
            return ""

        return "\n".join(_reading_order_lines(result[0]))

    def read_crop(self, image: Image.Image) -> str:
        """
        Recognize a pre-cropped single line/word (detection skipped).
        Used by the mixed-document pipeline for regions already located.
        """
        img    = np.array(image.convert("RGB"))
        result = self.ocr.ocr(img, det=False, cls=True)

        if not result or not result[0]:
            return ""

        text, _ = result[0][0]
        return text.strip()
