"""
Line segmentation for OCR preprocessing.
Splits a multi-line image into individual line images
using horizontal projection profiling.

Works for both handwritten and printed text.
"""

import cv2
import numpy as np
from PIL import Image


def segment_lines(image: Image.Image, printed: bool = False) -> list:
    """
    Splits an image into individual line images.

    Args:
        image   : Input PIL image (any size, any ink color)
        printed : Use lower threshold for printed text (smaller characters)

    Returns:
        List of PIL Images, one per detected line.
        Falls back to [image] if no lines detected.
    """
    img_array   = np.array(image.convert("RGB"))
    red_channel = img_array[:, :, 0]

    # Invert red channel — blue/black ink becomes bright, white background dark
    inverted = (255 - red_channel).astype(np.uint8)

    # Blur to connect nearby ink pixels
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)

    # Otsu threshold — automatically finds the right ink/background cutoff
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Horizontal projection: count ink pixels per row
    h_proj    = np.sum(binary, axis=1) / 255

    # Printed text has smaller characters — needs lower threshold
    threshold = binary.shape[1] * (0.01 if printed else 0.05)

    # Find row ranges where text exists
    in_line    = False
    line_start = 0
    lines      = []

    for i, val in enumerate(h_proj):
        if not in_line and val > threshold:
            in_line    = True
            line_start = i
        elif in_line and val <= threshold:
            in_line = False
            lines.append((line_start, i))

    if in_line:
        lines.append((line_start, len(h_proj)))

    if not lines:
        return [image]

    # Merge lines that are too close (< 5px gap between them)
    merged = [lines[0]]
    for start, end in lines[1:]:
        if start - merged[-1][1] < 5:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # Crop each line from original image with padding
    padding     = 6
    h, w        = img_array.shape[:2]
    line_images = []

    for start, end in merged:
        y1 = max(0, start - padding)
        y2 = min(h, end + padding)
        if y2 - y1 > 5:   # skip tiny noise rows
            line_images.append(image.crop((0, y1, w, y2)))

    return line_images if line_images else [image]
