"""
Progresssoft OCR — Gradio Web UI
Three-tab interface: Handwritten, Printed, and Mixed Document recognition.

  Handwritten : PaddleOCR (fine-tuned on IAM handwriting)
  Printed     : EasyOCR
  Mixed       : EasyOCR detection + per-region routing to the right engine
  Post-OCR    : SymSpell spelling correction on every result

Usage:
    python -m src.app
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# CRITICAL load order: pandas/pyarrow and gradio MUST be imported BEFORE torch
# and paddle. pyarrow throws a fatal access-violation (segfault) if it loads
# into a process that already has torch + paddle native libraries in it —
# which is exactly what happened when Gradio lazily imported it during a
# request. Loading it first, into a clean process, avoids the conflict.
import pandas  # noqa: F401  — pulls in pyarrow early
import gradio as gr
import tempfile
import html as _html
import numpy as np
from PIL import Image

import torch
from src.models.handwritten import HandwrittenOCR
from src.models.printed import PrintedOCR
from src.correction import correct_text
from src.models.secure_pipeline import SecurePipeline, render_annotated

# ── Load models ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

handwritten = HandwrittenOCR(use_gpu=False)   # CPU PaddlePaddle build
printed     = PrintedOCR(device.type)         # EasyOCR (uses CUDA if available)


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_line(text: str) -> str:
    text = text.strip()
    if text.endswith(" ."):
        text = text[:-2].strip()
    if text.endswith(" ..."):
        text = text[:-4].strip()
    text = text.replace("# ", "").replace(" #", "").replace("#", "")
    text = text.strip("~ ").strip()
    return text.strip()


def _finish(full_text: str):
    """Apply SymSpell correction, then save a downloadable .txt file."""
    full_text = correct_text(full_text)
    if not full_text.strip():
        return "No text detected.", None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt",
                                      mode="w", encoding="utf-8")
    tmp.write(full_text)
    tmp.close()
    return full_text, tmp.name


# ── Inference ─────────────────────────────────────────────────────────────────
def extract_handwritten(image: Image.Image):
    if image is None:
        return "Please upload an image.", None
    try:
        return _finish(handwritten.read_image(image))
    except Exception as e:
        return f"Error: {str(e)}", None


def extract_printed(image: Image.Image):
    if image is None:
        return "Please upload an image.", None
    try:
        return _finish(printed.read_image(image))
    except Exception as e:
        return f"Error: {str(e)}", None


def extract_mixed(image: Image.Image):
    """
    EasyOCR locates every text region with a confidence score.
      - High confidence (>= 0.6)  → printed → keep EasyOCR result
      - Low confidence  (< 0.6)   → handwritten → re-read region with PaddleOCR
    Regions are combined in top-to-bottom reading order.
    """
    if image is None:
        return "Please upload an image.", None
    try:
        CONFIDENCE_THRESHOLD = 0.6
        img_array  = np.array(image.convert("RGB"))
        detections = printed.reader.readtext(img_array, detail=1, paragraph=False)

        if not detections:
            return "No text detected.", None

        detections = sorted(detections, key=lambda d: d[0][0][1])

        lines = []
        for bbox, text, confidence in detections:
            text = text.strip()
            if not text:
                continue

            if confidence < CONFIDENCE_THRESHOLD:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x1, y1 = max(0, int(min(xs)) - 4), max(0, int(min(ys)) - 4)
                x2 = min(image.width,  int(max(xs)) + 4)
                y2 = min(image.height, int(max(ys)) + 4)

                crop = image.crop((x1, y1, x2, y2))
                hw   = handwritten.read_crop(crop)
                lines.append(clean_line(hw) if hw.strip() else clean_line(text))
            else:
                lines.append(clean_line(text))

        return _finish("\n".join(l for l in lines if l.strip()))
    except Exception as e:
        return f"Error: {str(e)}", None


# ── Bank Contract (high-integrity) ────────────────────────────────────────────
# Lazy-loaded so the app still starts before the trained model is in place.
_secure_pipe = None
_secure_error = None


def _get_secure_pipe():
    global _secure_pipe, _secure_error
    if _secure_pipe is None and _secure_error is None:
        try:
            _secure_pipe = SecurePipeline(use_gpu=False, top_k=5)
        except Exception as e:
            _secure_error = str(e)
    return _secure_pipe


def extract_bank(image: Image.Image):
    """Run the secure pipeline; return (annotated image, audit HTML table)."""
    if image is None:
        return None, "<p>Please upload a bank contract.</p>"

    pipe = _get_secure_pipe()
    if pipe is None:
        return None, (
            "<p style='color:#c0392b'><b>Secure model not available yet.</b><br>"
            f"{_html.escape(_secure_error or '')}<br>Place the trained model at "
            "<code>models/paddle_rec/</code> (inference.pdmodel, inference.pdiparams, "
            "en_dict.txt).</p>"
        )
    try:
        results = pipe.extract(image)
    except Exception as e:
        return None, f"<p style='color:#c0392b'>Error: {_html.escape(str(e))}</p>"

    if not results:
        return image, "<p>No text fields detected.</p>"

    annotated = render_annotated(image, results)

    accepted = sum(1 for r in results if r["accepted"])
    review   = len(results) - accepted

    rows = []
    for i, r in enumerate(results, start=1):
        ok     = r["accepted"]
        bg     = "#e7f6e7" if ok else "#fdecea"
        status = "✅ Accepted" if ok else "⚠ Review"
        text   = _html.escape(r["text"].replace(" [REVIEW_REQUIRED]", ""))
        rank   = r["matched_rank"] if ok else "—"
        score  = f"{r['score']:.2f}"
        alts   = " &nbsp;·&nbsp; ".join(_html.escape(t) for t, _ in r["candidates"])
        rows.append(
            f"<tr style='background:{bg}'>"
            f"<td style='text-align:center'>{i}</td>"
            f"<td style='white-space:nowrap'>{status}</td>"
            f"<td><b>{text}</b></td>"
            f"<td style='text-align:center'>{rank}</td>"
            f"<td style='text-align:center'>{score}</td>"
            f"<td style='color:#555;font-size:12px'>{alts}</td></tr>"
        )

    table = (
        f"<div style='margin:6px 0 10px'><b style='color:#1a7a1a'>{accepted} accepted</b>"
        f" &nbsp;·&nbsp; <b style='color:#c0392b'>{review} need review</b>"
        f" &nbsp;·&nbsp; {len(results)} fields total</div>"
        "<table style='width:100%;border-collapse:collapse;font-size:13px' "
        "border='1' cellpadding='6'>"
        "<tr style='background:#f0f0f0'>"
        "<th>#</th><th>Status</th><th>Selected text</th>"
        "<th>Matched&nbsp;rank</th><th>Score</th>"
        "<th>Top-K hypotheses (audit trail)</th></tr>"
        + "".join(rows) + "</table>"
    )
    return annotated, table


# ── Custom CSS ────────────────────────────────────────────────────────────────
css = """
    .tab-nav button { font-size: 16px; font-weight: 600; padding: 12px 32px; }
    .output-wrap { position: relative; }
    .download-corner {
        position: absolute; bottom: 10px; right: 10px; z-index: 10;
    }
    .download-corner button {
        padding: 4px 12px !important; font-size: 12px !important; min-width: unset !important;
    }
    .upload-box { min-height: 260px; }
    .extract-btn { margin-top: 8px; }
    footer { display: none !important; }
"""

_visible = lambda t: gr.DownloadButton(
    visible=bool(t and t.strip()
                 and not t.startswith("Error")
                 and not t.startswith("Please")
                 and t != "No text detected.")
)


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Progresssoft OCR") as demo:

    gr.Markdown("""
    # Progresssoft OCR
    Select the document type, upload your image, and extract the text.
    """)

    with gr.Tabs():

        # ── Handwritten ───────────────────────────────────────────────────────
        with gr.Tab("✍️  Handwritten"):
            hw_image = gr.Image(type="pil", label="Upload Handwritten Document",
                                elem_classes=["upload-box"])
            hw_btn   = gr.Button("Extract Text", variant="primary",
                                 elem_classes=["extract-btn"])
            with gr.Group(elem_classes=["output-wrap"]):
                hw_text = gr.Textbox(label="Extracted Text", lines=14,
                                     placeholder="Extracted text will appear here...")
                with gr.Row(elem_classes=["download-corner"]):
                    hw_download = gr.DownloadButton(label="⬇ Download",
                                                    visible=False, size="sm")
            hw_btn.click(extract_handwritten, hw_image, [hw_text, hw_download])
            hw_text.change(_visible, hw_text, hw_download)

        # ── Printed ───────────────────────────────────────────────────────────
        with gr.Tab("🖨️  Printed"):
            pr_image = gr.Image(type="pil", label="Upload Printed Document",
                                elem_classes=["upload-box"])
            pr_btn   = gr.Button("Extract Text", variant="primary",
                                 elem_classes=["extract-btn"])
            with gr.Group(elem_classes=["output-wrap"]):
                pr_text = gr.Textbox(label="Extracted Text", lines=14,
                                     placeholder="Extracted text will appear here...")
                with gr.Row(elem_classes=["download-corner"]):
                    pr_download = gr.DownloadButton(label="⬇ Download",
                                                    visible=False, size="sm")
            pr_btn.click(extract_printed, pr_image, [pr_text, pr_download])
            pr_text.change(_visible, pr_text, pr_download)

        # ── Mixed ─────────────────────────────────────────────────────────────
        with gr.Tab("📄  Mixed Document"):
            gr.Markdown(
                "For documents that contain **both printed and handwritten** "
                "text, such as filled forms or annotated documents."
            )
            mx_image = gr.Image(type="pil", label="Upload Mixed Document",
                                elem_classes=["upload-box"])
            mx_btn   = gr.Button("Extract Text", variant="primary",
                                 elem_classes=["extract-btn"])
            with gr.Group(elem_classes=["output-wrap"]):
                mx_text = gr.Textbox(label="Extracted Text", lines=14,
                                     placeholder="Extracted text will appear here...")
                with gr.Row(elem_classes=["download-corner"]):
                    mx_download = gr.DownloadButton(label="⬇ Download",
                                                    visible=False, size="sm")
            mx_btn.click(extract_mixed, mx_image, [mx_text, mx_download])
            mx_text.change(_visible, mx_text, mx_download)

        # ── Bank Contract (high-integrity) ────────────────────────────────────
        with gr.Tab("🏦  Bank Contract"):
            gr.Markdown(
                "**High-integrity mode** for financial documents. Each field is read "
                "with Top-K hypotheses and validated against a banking whitelist — "
                "**nothing is auto-corrected or generated.** "
                "<span style='color:#1a7a1a'>**Green**</span> boxes passed the gate; "
                "<span style='color:#c0392b'>**red**</span> boxes are flagged "
                "`[REVIEW_REQUIRED]` for a human operator."
            )
            bk_image = gr.Image(type="pil", label="Upload Bank Contract",
                                elem_classes=["upload-box"])
            bk_btn   = gr.Button("Analyze Document", variant="primary",
                                 elem_classes=["extract-btn"])
            with gr.Row():
                bk_annotated = gr.Image(
                    label="Detected fields — green = accepted, red = needs review",
                    height=420,
                )
            bk_table = gr.HTML()
            bk_btn.click(extract_bank, bk_image, [bk_annotated, bk_table])

    gr.Markdown("""
    ---
    **Handwritten:** PaddleOCR — Fine-tuned on IAM Handwriting Dataset
    **Printed:** EasyOCR — Optimized for printed documents
    **Mixed:** EasyOCR + PaddleOCR — Auto-detects text type per region
    **Bank Contract:** Top-K CTC + whitelist gate — zero-hallucination, human-in-the-loop
    **Correction:** SymSpell spelling correction (standard tabs only)
    """)


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft(), css=css)
