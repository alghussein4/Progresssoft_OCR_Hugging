# Progresssoft OCR — Setup & Run Guide

## Requirements

- Python 3.12
- Git
- NVIDIA GPU (recommended) — CPU works but is significantly slower
- ~10 GB free disk space

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/alghussein4/Progresssoft_OCR_Hugging.git
cd Progresssoft_OCR_Hugging
```

---

## Step 2 — Create the Virtual Environment

**Windows:**
```powershell
python -m venv .ProgresssoftOCRHuggingVenv
.\.ProgresssoftOCRHuggingVenv\Scripts\activate
```

**Linux / Mac:**
```bash
python3.12 -m venv .ProgresssoftOCRHuggingVenv
source .ProgresssoftOCRHuggingVenv/bin/activate
```

You should see `(.ProgresssoftOCRHuggingVenv)` appear in your terminal prompt.

---

## Step 3 — Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Step 4 — Install PyTorch with CUDA

> Skip this step if you do not have an NVIDIA GPU. PyTorch will run on CPU automatically.

```bash
python install_torch.py
```

This installs PyTorch with CUDA 12.4 support, compatible with RTX 30xx and 40xx series GPUs.

---

## Step 5 — Add the Trained Model

The fine-tuned handwriting model weights are **not included** in the repository due to their size (~1.2 GB).

1. Download `best_model.zip` from the provided Google Drive link
2. Extract it into the project at this exact path:

```
Progresssoft_OCR_Hugging/
└── outputs/
    └── checkpoints/
        └── best_model/
            ├── config.json
            ├── model.safetensors
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── generation_config.json
            └── processor_config.json
```

---

## Step 6 — Run the Application

```bash
python -m src.app
```

Then open your browser and go to:

```
http://127.0.0.1:7860
```

> The first time you run the app, EasyOCR will automatically download its models (~500 MB). This is a one-time download.

---

## Using the Application

The app has three tabs:

| Tab | Use for |
|-----|---------|
| ✍️ Handwritten | Documents written by hand |
| 🖨️ Printed | Typed or printed documents |
| 📄 Mixed Document | Forms or documents with both printed and handwritten text |

**Steps:**
1. Select the appropriate tab
2. Upload your image
3. Click **Extract Text**
4. Read the extracted text in the output box
5. Click **⬇ Download** to save the result as a `.txt` file

---

## Troubleshooting

**Virtual environment not activating on Windows:**
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**"No module named pip" error:**
```bash
python -m ensurepip --upgrade
```

**CUDA not detected:**
- Make sure your NVIDIA drivers are up to date
- Run `nvidia-smi` to confirm CUDA is available
- Re-run `python install_torch.py`

**EasyOCR download fails:**
- Check your internet connection
- Re-run `python -m src.app` — it will retry automatically
