"""
TrOCR recognizer — handwriting recognition for the hybrid pipeline.

PaddleOCR handles detection (finding line/word boxes); this module handles
recognition (reading each crop) using microsoft/trocr-large-handwritten, which
is purpose-built for handwriting and far stronger than PaddleOCR's recognizer
on free/cursive writing.

Runs on GPU in fp16 to fit the 6 GB laptop card alongside EasyOCR; falls back
to CPU automatically if CUDA is unavailable.
"""

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_ID = "microsoft/trocr-large-handwritten"


class TrOCRRecognizer:
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if self.use_cuda else "cpu")

        print(f"[TrOCR] Loading {MODEL_ID} on {self.device}"
              f"{' (fp16)' if self.use_cuda else ''}...")
        self.processor = TrOCRProcessor.from_pretrained(MODEL_ID)
        model = VisionEncoderDecoderModel.from_pretrained(
            MODEL_ID, device_map=None, low_cpu_mem_usage=False,
        )
        if self.use_cuda:
            model = model.half()          # fp16 to halve VRAM + speed up
        self.model = model.to(self.device).eval()
        print("[TrOCR] Ready.")

    @torch.no_grad()
    def read_crop(self, image: Image.Image) -> str:
        """Read a single cropped line/word image and return the text."""
        pixel_values = self.processor(
            images=image.convert("RGB"), return_tensors="pt"
        ).pixel_values.to(self.device)
        if self.use_cuda:
            pixel_values = pixel_values.half()

        generated_ids = self.model.generate(pixel_values, max_new_tokens=64)
        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    @torch.no_grad()
    def read_batch(self, images: list) -> list:
        """
        Read several crops in one forward pass (much faster than one-by-one).
        Returns a list of strings aligned with the input order.
        """
        if not images:
            return []
        pixel_values = self.processor(
            images=[im.convert("RGB") for im in images], return_tensors="pt"
        ).pixel_values.to(self.device)
        if self.use_cuda:
            pixel_values = pixel_values.half()

        generated_ids = self.model.generate(pixel_values, max_new_tokens=64)
        return [t.strip() for t in self.processor.batch_decode(
            generated_ids, skip_special_tokens=True)]
