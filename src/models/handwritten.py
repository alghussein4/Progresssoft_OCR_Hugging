"""
Handwritten text recognition model.
Uses our fine-tuned TrOCR Large model trained on IAM handwriting dataset.
Falls back to Microsoft's pretrained model if checkpoint not found.

CER: 1.92% on IAM validation set.
"""

import torch
from pathlib import Path
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

CHECKPOINT = "outputs/checkpoints/best_model"
FALLBACK   = "microsoft/trocr-large-handwritten"


class HandwrittenOCR:
    def __init__(self, device: torch.device):
        self.device   = device
        model_src     = CHECKPOINT if Path(CHECKPOINT).exists() else FALLBACK

        print(f"[Handwritten] Loading from: {model_src}")
        self.processor = TrOCRProcessor.from_pretrained(model_src)
        self.model     = VisionEncoderDecoderModel.from_pretrained(
            model_src,
            device_map=None,
            low_cpu_mem_usage=False,
        ).to(device)
        self.model.eval()
        print("[Handwritten] Ready.")

    def read_line(self, image: Image.Image) -> str:
        """Run inference on a single line image."""
        pixel_values = self.processor(
            images=image.convert("RGB"),
            return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=128
            )

        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
