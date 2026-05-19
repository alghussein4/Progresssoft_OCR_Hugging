"""
Tests TrOCR inference on a sample image using your GPU.
Verifies the model loads correctly and can read both handwritten and printed text.

Usage:
    python src/test_inference.py
    python src/test_inference.py --image path/to/your/image.jpg
"""

import argparse
import time
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Sample images for testing (publicly available)
SAMPLE_IMAGES = {
    "handwritten": "https://fki.iam.unibe.ch/static/img/sample-page.jpg",
    "printed": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
}

# Fallback: use a locally generated test image if download fails
def create_test_image(text: str = "Hello OCR World") -> Image.Image:
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", (400, 80), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, 20), text, fill="black", font=font)
    return img


def load_model(model_id: str, device: torch.device):
    print(f"  Loading processor...")
    processor = TrOCRProcessor.from_pretrained(model_id)

    print(f"  Loading model weights onto {device}...")
    model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def run_inference(image: Image.Image, processor, model, device: torch.device) -> tuple[str, float]:
    image = image.convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    elapsed = time.perf_counter() - start

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text, elapsed


def get_test_image(path: str | None) -> Image.Image:
    if path:
        return Image.open(path)

    # Try downloading a real handwritten sample
    cache = Path("data/test_sample.jpg")
    cache.parent.mkdir(exist_ok=True)

    if not cache.exists():
        print("  Downloading test image...")
        try:
            url = "https://raw.githubusercontent.com/microsoft/unilm/master/trocr/data/test.jpg"
            urllib.request.urlretrieve(url, cache)
            print("  Downloaded.")
        except Exception:
            print("  Download failed — generating a synthetic test image instead.")
            img = create_test_image("The quick brown fox jumps over the lazy dog")
            img.save(cache)

    return Image.open(cache)


def main(image_path: str | None = None):
    print("\n" + "="*60)
    print("TrOCR Inference Test")
    print("="*60)

    # GPU check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice  : {device}")
    if device.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING : CUDA not found — running on CPU (slow)")

    # Load test image
    print("\nLoading test image...")
    image = get_test_image(image_path)
    print(f"  Image size: {image.size}")

    # Test both models
    models_to_test = [
        ("microsoft/trocr-base-handwritten", "Handwritten"),
        ("microsoft/trocr-base-printed",     "Printed"),
    ]

    for model_id, label in models_to_test:
        print(f"\n[{label}] {model_id}")
        print("-" * 50)
        try:
            processor, model = load_model(model_id, device)
            text, elapsed = run_inference(image, processor, model, device)
            print(f"  Result  : {text!r}")
            print(f"  Time    : {elapsed:.3f}s")
            del model  # free VRAM before loading next model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR   : {e}")
            print(f"  Have you run 'python src/download_model.py' first?")

    print("\n" + "="*60)
    print("Test complete.")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a custom test image (optional)")
    args = parser.parse_args()
    main(args.image)
