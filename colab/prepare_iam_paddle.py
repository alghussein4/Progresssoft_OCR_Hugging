"""
Export the Teklia/IAM-line dataset (Hugging Face) into the format
PaddleOCR's text-recognition trainer expects.

Output structure:
    <out_dir>/
    ├── images/
    │   ├── train_000000.jpg
    │   ├── train_000001.jpg
    │   └── ...
    ├── rec_gt_train.txt      # lines: images/train_000000.jpg\t<text>
    └── rec_gt_val.txt        # lines: images/val_000000.jpg\t<text>

The label files use a TAB between the image path and the transcription,
which is the format PaddleOCR's SimpleDataSet reader requires.

Usage (run on Colab after `pip install datasets pillow`):
    python prepare_iam_paddle.py --out_dir ./iam_paddle
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from PIL import Image


def export_split(dataset, split_name: str, images_dir: Path, label_path: Path) -> int:
    """Save every image to disk and write one label line per sample."""
    count = 0
    with open(label_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(dataset):
            text = (sample["text"] or "").strip()
            if not text:
                continue  # skip empty transcriptions

            image = sample["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = image.convert("RGB")

            fname    = f"{split_name}_{i:06d}.jpg"
            rel_path = f"images/{fname}"
            image.save(images_dir / fname, quality=95)

            # PaddleOCR rec format: <relative_image_path>\t<text>
            f.write(f"{rel_path}\t{text}\n")
            count += 1

            if count % 500 == 0:
                print(f"  [{split_name}] {count} images written...")

    return count


def main(out_dir: str):
    out      = Path(out_dir)
    img_dir  = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Teklia/IAM-line from Hugging Face...")
    train_ds = load_dataset("Teklia/IAM-line", split="train")
    val_ds   = load_dataset("Teklia/IAM-line", split="validation")

    print("\nExporting train split...")
    n_train = export_split(train_ds, "train", img_dir, out / "rec_gt_train.txt")

    print("\nExporting validation split...")
    n_val = export_split(val_ds, "val", img_dir, out / "rec_gt_val.txt")

    print("\n" + "=" * 50)
    print(f"Done.")
    print(f"  Train samples : {n_train}")
    print(f"  Val samples   : {n_val}")
    print(f"  Output dir    : {out.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./iam_paddle",
                        help="Where to write images and label files")
    args = parser.parse_args()
    main(args.out_dir)
