"""
CER (Character Error Rate) evaluation for TrOCR.
Called automatically during training after each epoch.

Lower CER = better. Target: < 3%

Usage:
    python -m src.evaluate   # runs standalone evaluation on test split
"""

import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import cer
from tqdm import tqdm


def compute_cer(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    dataloader: DataLoader,
    device: torch.device,
    max_new_tokens: int = 128,
) -> float:
    """
    Runs inference on the full dataloader and returns CER.

    Args:
        model          : fine-tuned or pretrained TrOCR model
        processor      : TrOCRProcessor
        dataloader     : DataLoader yielding (pixel_values, labels)
        device         : cuda or cpu
        max_new_tokens : max tokens to generate per image

    Returns:
        cer_score (float) — e.g. 0.035 means 3.5% CER
    """
    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"]

            # Decode ground truth — replace -100 with pad token
            labels_clean = labels.clone()
            labels_clean[labels_clean == -100] = processor.tokenizer.pad_token_id
            gt_texts = processor.tokenizer.batch_decode(
                labels_clean, skip_special_tokens=True
            )

            # Generate predictions
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens
            )
            pred_texts = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            all_preds.extend(pred_texts)
            all_targets.extend(gt_texts)

    # Guard against empty strings which crash jiwer
    pairs = [(t, p) for t, p in zip(all_targets, all_preds) if t.strip()]
    if not pairs:
        return 1.0  # worst possible CER if no valid targets
    targets_clean, preds_clean = zip(*pairs)

    cer_score = cer(list(targets_clean), list(preds_clean))
    return cer_score


# ── Standalone evaluation ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from src.dataset import IAMDataset
    from src.config import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned checkpoint folder. "
                             "Uses pretrained model if not provided.")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "validation", "test"])
    args = parser.parse_args()

    cfg       = get_config()
    device    = torch.device(cfg["device"])
    model_src = args.checkpoint or cfg["model_id"]

    print(f"\nModel  : {model_src}")
    print(f"Split  : {args.split}")
    print(f"Device : {device}")

    processor = TrOCRProcessor.from_pretrained(cfg["model_id"])
    model     = VisionEncoderDecoderModel.from_pretrained(model_src).to(device)

    dataset    = IAMDataset(args.split, processor, cfg["max_target_length"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"],
                            num_workers=cfg["num_workers"])

    cer_score = compute_cer(model, processor, dataloader, device)

    print(f"\nCER on [{args.split}] : {cer_score * 100:.2f}%")
