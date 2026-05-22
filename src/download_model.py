"""
Downloads the pretrained TrOCR models from Hugging Face and caches them locally.
Run once before anything else.

Usage:
    python src/download_model.py
"""

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODELS = [
    "microsoft/trocr-base-handwritten",
    "microsoft/trocr-base-printed",
    "microsoft/trocr-large-handwritten",
]

def download(model_id: str):
    print(f"\n{'='*60}")
    print(f"Downloading: {model_id}")
    print(f"{'='*60}")

    print("  [1/2] Downloading processor...")
    TrOCRProcessor.from_pretrained(model_id)

    print("  [2/2] Downloading model weights...")
    VisionEncoderDecoderModel.from_pretrained(model_id)

    print(f"  Done. Cached at ~/.cache/huggingface/hub")

if __name__ == "__main__":
    for model_id in MODELS:
        download(model_id)
    print("\nAll models downloaded successfully.")
