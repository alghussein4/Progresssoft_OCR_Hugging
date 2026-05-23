"""
Progresssoft OCR — Gradio Web UI
Combines handwritten and printed OCR models with line segmentation.

Usage:
    python -m src.app
"""

import torch
import gradio as gr
import tempfile
from PIL import Image

from src.models.handwritten import HandwrittenOCR
from src.models.printed import PrintedOCR
from src.segmentation import segment_lines

# ── Load both models ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

handwritten = HandwrittenOCR(device)
printed     = PrintedOCR(device.type)   # EasyOCR takes "cuda" or "cpu" string


# ── Text cleanup ──────────────────────────────────────────────────────────────
def clean_line(text: str) -> str:
    text = text.strip()
    # Remove trailing isolated period
    if text.endswith(" ."):
        text = text[:-2].strip()
    # Remove trailing ellipsis artifacts
    if text.endswith(" ..."):
        text = text[:-4].strip()
    # Remove hash artifacts
    text = text.replace("# ", "").replace(" #", "").replace("#", "")
    # Remove leading/trailing tilde artifacts
    text = text.strip("~ ").strip()
    return text.strip()


# ── Main inference ────────────────────────────────────────────────────────────
def recognize_text(image: Image.Image, mode: str):
    if image is None:
        return "Please upload an image.", None

    try:
        is_printed = (mode == "Printed")

        if is_printed:
            # EasyOCR handles its own segmentation — pass the full image
            full_text = printed.read_image(image)
        else:
            # TrOCR needs line-by-line segmentation
            lines     = segment_lines(image, printed=False)
            results   = [clean_line(handwritten.read_line(line)) for line in lines]
            full_text = "\n".join(r for r in results if r.strip())

        if not full_text.strip():
            return "No text detected.", None

        # Save extracted text to a temp .txt file for download
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", mode="w", encoding="utf-8"
        )
        tmp.write(full_text)
        tmp.close()

        return full_text, tmp.name

    except Exception as e:
        return f"Error: {str(e)}", None


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Progresssoft OCR") as demo:

    gr.Markdown("""
    # Progresssoft OCR
    ### Handwritten & Printed Text Recognition
    Upload an image, select the text type, and extract the text.
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                height=350,
            )
            mode_selector = gr.Radio(
                choices=["Handwritten", "Printed"],
                value="Handwritten",
                label="Text Type",
            )
            submit_btn = gr.Button("Extract Text", variant="primary", size="lg")

        with gr.Column():
            text_output = gr.Textbox(
                label="Extracted Text",
                lines=15,
                placeholder="Extracted text will appear here...",
            )
            download_btn = gr.File(
                label="Download as .txt",
                interactive=False,
            )

    submit_btn.click(
        fn=recognize_text,
        inputs=[image_input, mode_selector],
        outputs=[text_output, download_btn],
    )

    gr.Markdown("""
    ---
    **Handwritten:** TrOCR Large — Fine-tuned on IAM — CER: 1.92%
    **Printed:** TrOCR Large Printed — Microsoft Pretrained
    """)


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
