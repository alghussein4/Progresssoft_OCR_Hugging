"""
Run this script once after activating the venv to install PyTorch with CUDA 12.4
(correct version for RTX 4050).

Usage:
    python install_torch.py
"""
import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "torch", "torchvision",
    "--index-url", "https://download.pytorch.org/whl/cu124"
])

print("\nVerifying CUDA availability...")
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
