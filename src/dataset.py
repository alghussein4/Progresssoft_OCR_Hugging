"""
IAM Line Dataset loader for TrOCR fine-tuning.
Loads directly from Hugging Face — no manual downloading needed.
Applies data augmentation on the training split to prevent overfitting.

Usage:
    python -m src.dataset   # runs a quick sanity check
"""

from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import TrOCRProcessor
from PIL import Image
import torchvision.transforms as T


# ── Augmentation pipeline (training only) ────────────────────────────────────
# Applied BEFORE the TrOCR processor so images still get properly normalized.
# Strong enough to prevent overfitting, subtle enough to not destroy text.
TRAIN_AUGMENT = T.Compose([
    # Slight rotation — handwriting is never perfectly straight
    T.RandomApply([T.RandomRotation(degrees=4, fill=255)],            p=0.6),
    # Blur — simulates low quality scans
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],  p=0.4),
    # Brightness/contrast — different scanner settings
    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)],      p=0.6),
    # Affine — slight shear and translation
    T.RandomApply([T.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        shear=4,
        fill=255
    )], p=0.5),
    # Random erase — simulates ink smudges or torn paper
    T.RandomApply([T.RandomErasing(
        p=1.0,
        scale=(0.01, 0.03),
        ratio=(0.3, 3.0),
        value=255
    )], p=0.3),
])


class IAMDataset(Dataset):
    def __init__(self, split: str, processor: TrOCRProcessor, max_target_length: int = 128):
        """
        Args:
            split            : "train", "validation", or "test"
            processor        : TrOCRProcessor instance
            max_target_length: max number of tokens for the label
        """
        assert split in ("train", "validation", "test"), \
            f"Invalid split '{split}'. Choose from: train, validation, test"

        print(f"Loading IAM-line [{split}] from Hugging Face...")
        self.dataset           = load_dataset("Teklia/IAM-line", split=split)
        self.processor         = processor
        self.max_target_length = max_target_length
        self.augment           = (split == "train")   # only augment training data
        print(f"  Loaded {len(self.dataset)} samples.  Augmentation: {self.augment}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # ---- Image ----
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        # Apply augmentation on training split only
        if self.augment:
            image = TRAIN_AUGMENT(image)

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)   # (3, H, W)

        # ---- Label ----
        text = sample["text"]
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)      # (max_target_length,)

        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    from src.config import get_config

    cfg       = get_config()
    processor = TrOCRProcessor.from_pretrained(cfg["model_id"])

    for split in ("train", "validation", "test"):
        ds = IAMDataset(split=split, processor=processor,
                        max_target_length=cfg["max_target_length"])

        sample = ds[0]
        raw_text = ds.dataset[0]["text"]

        print(f"\n[{split}]")
        print(f"  Samples          : {len(ds)}")
        print(f"  pixel_values     : {sample['pixel_values'].shape}")
        print(f"  labels shape     : {sample['labels'].shape}")
        print(f"  Sample text      : {raw_text!r}")
        print(f"  Augmentation     : {ds.augment}")

    print("\nDataset check passed.")
