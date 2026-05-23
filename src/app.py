"""
Progresssoft OCR — Gradio Web UI
Three-tab interface: Handwritten, Printed, and Mixed Document recognition.

Usage:
    python -m src.app
"""

import torch
import gradio as gr
import tempfile
import numpy as np
from PIL import Image

from src.models.handwritten import HandwrittenOCR
from src.models.printed import PrintedOCR
from src.segmentation import segment_lines

# ── Load models ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

handwritten = HandwrittenOCR(device)
printed     = PrintedOCR(device.type)


# ── Text cleanup ──────────────────────────────────────────────────────────────
def clean_line(text: str) -> str:
    text = text.strip()
    if text.endswith(" ."):
        text = text[:-2].strip()
    if text.endswith(" ..."):
        text = text[:-4].strip()
    text = text.replace("# ", "").replace(" #", "").replace("#", "")
    text = text.strip("~ ").strip()
    return text.strip()


# ── Inference ─────────────────────────────────────────────────────────────────
def extract_handwritten(image: Image.Image):
    if image is None:
        return "Please upload an image.", None
    try:
        lines     = segment_lines(image, printed=False)
        results   = [clean_line(handwritten.read_line(line)) for line in lines]
        full_text = "\n".join(r for r in results if r.strip())
        if not full_text.strip():
            return "No text detected.", None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
        tmp.write(full_text)
        tmp.close()
        return full_text, tmp.name
    except Exception as e:
        return f"Error: {str(e)}", None


def extract_printed(image: Image.Image):
    if image is None:
        return "Please upload an image.", None
    try:
        full_text = printed.read_image(image)
        if not full_text.strip():
            return "No text detected.", None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
        tmp.write(full_text)
        tmp.close()
        return full_text, tmp.name
    except Exception as e:
        return f"Error: {str(e)}", None


def extract_mixed(image: Image.Image):
    """
    Mixed document extraction.
    Uses EasyOCR to detect all text regions with confidence scores.
    For high-confidence regions (printed text) → keeps EasyOCR result.
    For low-confidence regions (likely handwritten) → runs TrOCR instead.
    Combines both in reading order (top to bottom).
    """
    if image is None:
        return "Please upload an image.", None
    try:
        CONFIDENCE_THRESHOLD = 0.6
        img_array = np.array(image.convert("RGB"))

        # EasyOCR with detail=1 returns [([[bbox]], text, confidence), ...]
        detections = printed.reader.readtext(img_array, detail=1, paragraph=False)

        if not detections:
            return "No text detected.", None

        # Sort detections top to bottom by y-coordinate of bounding box
        detections = sorted(detections, key=lambda d: d[0][0][1])

        lines = []
        for bbox, text, confidence in detections:
            text = text.strip()
            if not text:
                continue

            if confidence < CONFIDENCE_THRESHOLD:
                # Low confidence → likely handwritten → run TrOCR on this region
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x1, y1 = max(0, int(min(x_coords)) - 4), max(0, int(min(y_coords)) - 4)
                x2, y2 = min(image.width, int(max(x_coords)) + 4), min(image.height, int(max(y_coords)) + 4)

                crop      = image.crop((x1, y1, x2, y2))
                hw_result = clean_line(handwritten.read_line(crop))
                lines.append(hw_result if hw_result.strip() else text)
            else:
                # High confidence → printed text → keep EasyOCR result
                lines.append(clean_line(text))

        full_text = "\n".join(l for l in lines if l.strip())
        if not full_text.strip():
            return "No text detected.", None

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
        tmp.write(full_text)
        tmp.close()
        return full_text, tmp.name

    except Exception as e:
        return f"Error: {str(e)}", None


# ── Custom CSS ────────────────────────────────────────────────────────────────
css = """
    /* Tab styling */
    .tab-nav button {
        font-size: 16px;
        font-weight: 600;
        padding: 12px 32px;
    }

    /* Output text area wrapper — relative so we can anchor the button */
    .output-wrap {
        position: relative;
    }

    /* Small download button pinned to bottom-right of the text box */
    .download-corner {
        position: absolute;
        bottom: 10px;
        right: 10px;
        z-index: 10;
    }

    .download-corner button {
        padding: 4px 12px !important;
        font-size: 12px !important;
        min-width: unset !important;
    }

    /* Upload area */
    .upload-box {
        min-height: 260px;
    }

    /* Extract button */
    .extract-btn {
        margin-top: 8px;
    }

    footer { display: none !important; }
"""


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Progresssoft OCR") as demo:

    gr.Markdown("""
    # Progresssoft OCR
    Select the document type, upload your image, and extract the text.
    """)

    with gr.Tabs():

        # ── Handwritten Tab ───────────────────────────────────────────────────
        with gr.Tab("✍️  Handwritten"):
            hw_image = gr.Image(
                type="pil",
                label="Upload Handwritten Document",
                elem_classes=["upload-box"],
            )
            hw_btn = gr.Button(
                "Extract Text",
                variant="primary",
                elem_classes=["extract-btn"],
            )
            with gr.Group(elem_classes=["output-wrap"]):
                hw_text = gr.Textbox(
                    label="Extracted Text",
                    lines=14,
                    placeholder="Extracted text will appear here...",
                                    )
                with gr.Row(elem_classes=["download-corner"]):
                    hw_download = gr.DownloadButton(
                        label="⬇ Download",
                        visible=False,
                        size="sm",
                    )

            hw_btn.click(
                fn=extract_handwritten,
                inputs=hw_image,
                outputs=[hw_text, hw_download],
            )

            hw_text.change(
                fn=lambda t: gr.DownloadButton(visible=bool(t and t.strip() and not t.startswith("Error") and not t.startswith("Please"))),
                inputs=hw_text,
                outputs=hw_download,
            )

        # ── Printed Tab ───────────────────────────────────────────────────────
        with gr.Tab("🖨️  Printed"):
            pr_image = gr.Image(
                type="pil",
                label="Upload Printed Document",
                elem_classes=["upload-box"],
            )
            pr_btn = gr.Button(
                "Extract Text",
                variant="primary",
                elem_classes=["extract-btn"],
            )
            with gr.Group(elem_classes=["output-wrap"]):
                pr_text = gr.Textbox(
                    label="Extracted Text",
                    lines=14,
                    placeholder="Extracted text will appear here...",
                                    )
                with gr.Row(elem_classes=["download-corner"]):
                    pr_download = gr.DownloadButton(
                        label="⬇ Download",
                        visible=False,
                        size="sm",
                    )

            pr_btn.click(
                fn=extract_printed,
                inputs=pr_image,
                outputs=[pr_text, pr_download],
            )

            pr_text.change(
                fn=lambda t: gr.DownloadButton(visible=bool(t and t.strip() and not t.startswith("Error") and not t.startswith("Please"))),
                inputs=pr_text,
                outputs=pr_download,
            )

        # ── Mixed Tab ─────────────────────────────────────────────────────────
        with gr.Tab("📄  Mixed Document"):
            gr.Markdown(
                "For documents that contain **both printed and handwritten** text, "
                "such as filled forms or annotated documents."
            )
            mx_image = gr.Image(
                type="pil",
                label="Upload Mixed Document",
                elem_classes=["upload-box"],
            )
            mx_btn = gr.Button(
                "Extract Text",
                variant="primary",
                elem_classes=["extract-btn"],
            )
            with gr.Group(elem_classes=["output-wrap"]):
                mx_text = gr.Textbox(
                    label="Extracted Text",
                    lines=14,
                    placeholder="Extracted text will appear here...",
                )
                with gr.Row(elem_classes=["download-corner"]):
                    mx_download = gr.DownloadButton(
                        label="⬇ Download",
                        visible=False,
                        size="sm",
                    )

            mx_btn.click(
                fn=extract_mixed,
                inputs=mx_image,
                outputs=[mx_text, mx_download],
            )

            mx_text.change(
                fn=lambda t: gr.DownloadButton(visible=bool(t and t.strip() and not t.startswith("Error") and not t.startswith("Please"))),
                inputs=mx_text,
                outputs=mx_download,
            )

    gr.Markdown("""
    ---
    **Handwritten model:** TrOCR Large — Fine-tuned on IAM Handwriting Dataset — CER 1.92%
    **Printed model:** EasyOCR — Optimized for printed documents
    **Mixed model:** EasyOCR + TrOCR — Auto-detects text type per region
    """)


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft(), css=css)
