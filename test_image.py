"""
Direct test script — runs OCR on a specific image without the UI.
Usage: python test_image.py
"""
import torch
import numpy as np
import easyocr
from PIL import Image

IMAGE_PATH = r"X:\Downloads\extract-scan1.png"
MODE       = "printed"   # "printed" or "handwritten"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

image = Image.open(IMAGE_PATH)
print(f"Image size: {image.size}")

if MODE == "printed":
    print("Loading EasyOCR...")
    reader = easyocr.Reader(
        ["en"],
        gpu=(device.type == "cuda"),
        model_storage_directory="models/easyocr",
        verbose=False,
    )
    img_array = np.array(image.convert("RGB"))
    results   = reader.readtext(img_array, detail=0, paragraph=True)
    full_text = "\n".join(r.strip() for r in results if r.strip())
    print("\n--- FULL OUTPUT ---")
    print(full_text)

else:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import cv2

    MODEL_ID = "outputs/checkpoints/best_model"
    print(f"Loading {MODEL_ID}...")
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_ID, device_map=None, low_cpu_mem_usage=False
    ).to(device)
    model.eval()

    def segment_lines(image, printed=False):
        img_array   = np.array(image.convert("RGB"))
        red_channel = img_array[:, :, 0]
        inverted    = (255 - red_channel).astype(np.uint8)
        blurred     = cv2.GaussianBlur(inverted, (5, 5), 0)
        _, binary   = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h_proj      = np.sum(binary, axis=1) / 255
        threshold   = binary.shape[1] * (0.01 if printed else 0.05)
        in_line = False; line_start = 0; lines = []
        for i, val in enumerate(h_proj):
            if not in_line and val > threshold:
                in_line = True; line_start = i
            elif in_line and val <= threshold:
                in_line = False; lines.append((line_start, i))
        if in_line: lines.append((line_start, len(h_proj)))
        if not lines: return [image]
        merged = [lines[0]]
        for start, end in lines[1:]:
            if start - merged[-1][1] < 5: merged[-1] = (merged[-1][0], end)
            else: merged.append((start, end))
        h, w = np.array(image).shape[:2]
        return [image.crop((0, max(0, s-6), w, min(h, e+6))) for s, e in merged if e-s > 5] or [image]

    lines = segment_lines(image)
    print(f"Detected {len(lines)} lines")
    results = []
    for i, line in enumerate(lines):
        pv = processor(images=line.convert("RGB"), return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pv, max_new_tokens=128)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        print(f"Line {i+1}: {text}")
        results.append(text)
    print("\n--- FULL OUTPUT ---")
    print("\n".join(results))
