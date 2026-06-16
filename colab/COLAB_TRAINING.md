# Fine-tuning PaddleOCR on IAM Handwriting — Google Colab Guide

This trains a PaddleOCR **text-recognition** model on the IAM handwriting
dataset (20 epochs). Detection uses PaddleOCR's pretrained English detector —
we only fine-tune recognition.

> Before starting: in Colab go to **Runtime → Change runtime type → T4 GPU**.

Run each cell below in order. If a cell errors, stop and send me the full error.

---

### Cell 1 — Install PaddlePaddle + PaddleOCR

```python
# PaddlePaddle GPU build for CUDA 12.x — note the ".post120" suffix,
# the plain "==2.6.2" does NOT exist on this index.
!python -m pip install paddlepaddle-gpu==2.6.2.post120 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/
!python -m pip install datasets pillow

# PaddleOCR training code (stable release with PP-OCRv4 configs)
!git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git
%cd PaddleOCR
!python -m pip install -r requirements.txt

# PaddlePaddle 2.6.2 is compiled against numpy 1.x — Colab ships numpy 2.x,
# which causes an ABI crash on `import paddle`. Pin it down.
!python -m pip install "numpy==1.26.4"
```

Verify the GPU is visible to Paddle (run as a **separate cell**):

```python
import paddle
print("Paddle version:", paddle.__version__)
print("GPU available:", paddle.device.is_compiled_with_cuda())
```

> If this throws a **numpy ABI error** (`module compiled against ABI version...`),
> go to **Runtime → Restart session**, then re-run *only this import cell* — not the
> install cell. The numpy downgrade needs a restart to take effect.

---

### Cell 2 — Prepare the IAM dataset in PaddleOCR format

Upload `prepare_iam_paddle.py` (from the `colab/` folder of the repo) to the
Colab file panel first, then:

```python
!python /content/prepare_iam_paddle.py --out_dir /content/iam_paddle
```

This writes (using an absolute path so it works regardless of which folder
the PaddleOCR repo ended up in):
- `/content/iam_paddle/images/` — every line image
- `/content/iam_paddle/rec_gt_train.txt` — 6,482 training labels
- `/content/iam_paddle/rec_gt_val.txt` — 976 validation labels

---

### Cell 3 — Download the pretrained English recognition model

We fine-tune **from** this model rather than training from scratch:

```python
!mkdir -p pretrain
!wget -q https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar -O pretrain/en_rec.tar
!tar -xf pretrain/en_rec.tar -C pretrain/
!ls pretrain/en_PP-OCRv4_rec_train
```

---

### Cell 4 — Train (20 epochs)

```python
!python tools/train.py \
  -c configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml \
  -o Global.pretrained_model=./pretrain/en_PP-OCRv4_rec_train/best_accuracy \
     Global.epoch_num=20 \
     Global.save_model_dir=./output/iam_rec \
     Global.character_dict_path=ppocr/utils/en_dict.txt \
     Global.eval_batch_step=[0,500] \
     Global.save_epoch_step=5 \
     Train.dataset.data_dir=/content/iam_paddle \
     Train.dataset.label_file_list=[/content/iam_paddle/rec_gt_train.txt] \
     Train.loader.batch_size_per_card=64 \
     Train.loader.num_workers=0 \
     Eval.dataset.data_dir=/content/iam_paddle \
     Eval.dataset.label_file_list=[/content/iam_paddle/rec_gt_val.txt] \
     Eval.loader.num_workers=0 \
     Optimizer.lr.learning_rate=0.0005
```

> `num_workers=0` is important on Colab — PaddleOCR's multi-worker data loading
> often deadlocks there (training hangs right after "load pretrain successful").

- The trainer prints accuracy + `norm_edit_dis` (1 − CER) on the val set during training.
- It auto-saves the **best** model to `./output/iam_rec/best_accuracy`.
- Lower `Optimizer.lr.learning_rate` to `0.0002` if it overfits, raise to `0.001` if it learns too slowly.

---

### Cell 5 — Export to inference format

The training checkpoint can't be used directly for inference — export it:

```python
!python tools/export_model.py \
  -c configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml \
  -o Global.pretrained_model=./output/iam_rec/best_accuracy \
     Global.character_dict_path=ppocr/utils/en_dict.txt \
     Global.save_inference_dir=./inference/iam_rec
```

---

### Cell 6 — Zip the model and save it

```python
import shutil
shutil.make_archive("/content/iam_rec_inference", "zip", "./inference/iam_rec")
print("Saved to /content/iam_rec_inference.zip")

# Option A — download directly:
from google.colab import files
files.download("/content/iam_rec_inference.zip")

# Option B — save to Google Drive instead:
# from google.colab import drive
# drive.mount("/content/drive")
# shutil.copy("/content/iam_rec_inference.zip", "/content/drive/MyDrive/")
```

---

## What you send back

The `iam_rec_inference.zip` — it contains:
```
inference/iam_rec/
├── inference.pdmodel
├── inference.pdiparams
└── inference.yml
```

Extract that into the project at `models/paddle_rec/` and the app will use it.
I'll wire that path in during the app rewrite.
