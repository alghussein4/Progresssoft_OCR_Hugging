"""
Central training configuration.
Automatically selects settings based on available GPU VRAM.
No manual changes needed when switching between machines.
"""

import sys
import torch


def get_gpu_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0.0


def get_config() -> dict:
    vram = get_gpu_vram_gb()

    # num_workers > 0 can cause issues on Windows — safe to use on Linux/Mac
    is_windows = sys.platform == "win32"

    if vram >= 14:
        # RTX 4070 Ti Super (16GB)
        # Large model is heavy — batch size 4 + gradient checkpointing stays safe
        model_id    = "microsoft/trocr-large-handwritten"
        batch_size  = 4
        num_workers = 0 if is_windows else 4
    elif vram >= 6:
        # RTX 4050 Laptop (6GB) — use base model, large won't fit
        model_id    = "microsoft/trocr-base-handwritten"
        batch_size  = 4
        num_workers = 0 if is_windows else 2
    else:
        # CPU fallback
        model_id    = "microsoft/trocr-base-handwritten"
        batch_size  = 2
        num_workers = 0

    return {
        # Model
        "model_id":          model_id,

        # Training
        "epochs":            50,
        "batch_size":        batch_size,
        "learning_rate":     5e-6,
        "weight_decay":      0.01,
        "warmup_steps":      200,

        # Data
        "max_target_length": 128,

        # Hardware
        "device":            "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers":       num_workers,
        "mixed_precision":   torch.cuda.is_available(),   # fp16 on GPU

        # Saving
        "output_dir":        "outputs/checkpoints",
        "save_every_n_epochs": 5,
        "log_dir":           "outputs/logs",
    }


if __name__ == "__main__":
    cfg  = get_config()
    vram = get_gpu_vram_gb()
    print(f"\nDetected VRAM : {vram:.1f} GB")
    print(f"Model         : {cfg['model_id']}")
    print(f"Device        : {cfg['device']}")
    print(f"Batch size    : {cfg['batch_size']}")
    print(f"Mixed prec.   : {cfg['mixed_precision']}")
    print(f"Epochs        : {cfg['epochs']}")
    print(f"Warmup steps  : {cfg['warmup_steps']}")
    print(f"Workers       : {cfg['num_workers']}")
