"""
Fine-tuning TrOCR on the IAM handwriting dataset.
Automatically adapts batch size based on available GPU VRAM.
Saves the best checkpoint based on validation CER.

Usage:
    python -m src.train                        # full training
    python -m src.train --epochs 1             # dry run (1 epoch)
    python -m src.train --resume outputs/checkpoints/best_model
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
)

from src.config import get_config
from src.dataset import IAMDataset
from src.evaluate import compute_cer


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=None,
                        help="Override number of epochs from config")
    parser.add_argument("--resume",   type=str,   default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg     = get_config()
    device  = torch.device(cfg["device"])
    epochs  = args.epochs or cfg["epochs"]

    out_dir = Path(cfg["output_dir"])
    log_dir = Path(cfg["log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("TrOCR Fine-tuning — IAM Handwriting Dataset")
    print("="*60)
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Epochs       : {epochs}")
    print(f"Batch size   : {cfg['batch_size']}")
    print(f"Mixed prec.  : {cfg['mixed_precision']}")
    print(f"Output dir   : {out_dir}")
    print("="*60)

    # ── Processor & Model ─────────────────────────────────────────────────────
    model_src = args.resume or cfg["model_id"]
    print(f"\nLoading model from: {model_src}")

    processor = TrOCRProcessor.from_pretrained(cfg["model_id"])
    model     = VisionEncoderDecoderModel.from_pretrained(model_src).to(device)

    # Required TrOCR config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size

    # Generation config (new transformers API)
    model.generation_config.max_new_tokens       = cfg["max_target_length"]
    model.generation_config.early_stopping       = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty       = 2.0
    model.generation_config.num_beams            = 4

    # ── Datasets & Dataloaders ────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_dataset = IAMDataset("train",      processor, cfg["max_target_length"])
    val_dataset   = IAMDataset("validation", processor, cfg["max_target_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    total_steps   = len(train_loader) * epochs
    scheduler     = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["warmup_steps"],
        num_training_steps=total_steps,
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda") if cfg["mixed_precision"] else None

    # ── Training loop ─────────────────────────────────────────────────────────
    best_cer    = float("inf")
    history     = []

    print(f"\nStarting training — {epochs} epochs\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, 1):
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)

            optimizer.zero_grad()

            if cfg["mixed_precision"]:
                with torch.amp.autocast("cuda"):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss    = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss    = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()

            # Progress every 50 steps
            if step % 50 == 0 or step == len(train_loader):
                avg_loss = epoch_loss / step
                print(f"  Epoch {epoch}/{epochs} | Step {step}/{len(train_loader)} "
                      f"| Loss: {avg_loss:.4f}")

        # ── Validation ────────────────────────────────────────────────────────
        avg_train_loss = epoch_loss / len(train_loader)
        val_cer        = compute_cer(model, processor, val_loader, device)
        elapsed        = time.time() - epoch_start

        print(f"\nEpoch {epoch}/{epochs} Summary")
        print(f"  Train Loss  : {avg_train_loss:.4f}")
        print(f"  Val CER     : {val_cer * 100:.2f}%")
        print(f"  Time        : {elapsed / 60:.1f} min")

        # Log history
        entry = {
            "epoch":      epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_cer":    round(val_cer * 100, 2),
            "time_min":   round(elapsed / 60, 1),
        }
        history.append(entry)

        with open(log_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Save best model
        if val_cer < best_cer:
            best_cer = val_cer
            best_path = out_dir / "best_model"
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)
            print(f"  New best model saved → {best_path}  (CER: {val_cer*100:.2f}%)")

        # Save periodic checkpoint
        if epoch % cfg["save_every_n_epochs"] == 0:
            ckpt_path = out_dir / f"checkpoint_epoch_{epoch}"
            model.save_pretrained(ckpt_path)
            processor.save_pretrained(ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

        print()

    # ── Done ──────────────────────────────────────────────────────────────────
    print("="*60)
    print("Training complete.")
    print(f"Best Val CER : {best_cer * 100:.2f}%")
    print(f"Best model   : {out_dir / 'best_model'}")
    print("="*60)


if __name__ == "__main__":
    train()
