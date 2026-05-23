"""
Printed text recognition model.
Uses EasyOCR — a pure Python OCR engine with excellent accuracy
on printed documents, books, and forms.

EasyOCR handles its own internal segmentation so the full image
is passed directly without pre-splitting into lines.

Models are stored inside the project at models/easyocr/
to keep everything self-contained.
"""

import easyocr
import numpy as np
from PIL import Image


class PrintedOCR:
    def __init__(self, device_type: str = "cpu"):
        use_gpu = (device_type == "cuda")

        print(f"[Printed] Loading EasyOCR (GPU={use_gpu})...")
        self.reader = easyocr.Reader(
            ["en"],
            gpu=use_gpu,
            model_storage_directory="models/easyocr",
            download_enabled=True,
            verbose=False,
        )
        print("[Printed] Ready.")

    def read_image(self, image: Image.Image) -> str:
        """
        Run EasyOCR on the full image.
        Returns the full extracted text as a single string.
        """
        img_array = np.array(image.convert("RGB"))

        results = self.reader.readtext(
            img_array,
            detail=0,           # return text only, no bounding boxes
            paragraph=True,     # merge nearby text into paragraphs
        )

        return "\n".join(r.strip() for r in results if r.strip())
