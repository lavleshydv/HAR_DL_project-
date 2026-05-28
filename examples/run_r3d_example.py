"""
Example: build and run a forward pass with r3d_18 (ResNet3D-18)
Usage:
    python examples/run_r3d_example.py

The script will print helpful instructions if PyTorch is not installed.
"""

import sys
from pathlib import Path

try:
    import torch
except Exception:
    print("PyTorch not installed. Install dependencies with:\n\n    pip install -r requirements.txt\n")
    sys.exit(0)

# Add project root to path and import builder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src_model_builder import build_model_r3d18
from src_config import NUM_CLASSES


def main():
    print("Building r3d_18 model (pretrained=False, device='cpu')...")
    model = build_model_r3d18(pretrained=False, device='cpu')
    model.eval()

    # Create dummy input: (N, C, T, H, W) - PyTorch video convention
    batch = 2
    C = 3
    T = 16
    H = 112
    W = 112

    dummy = torch.randn(batch, C, T, H, W)
    print(f"Dummy input shape: {dummy.shape}")

    with torch.no_grad():
        out = model(dummy)

    print(f"Output shape: {out.shape}")
    print(f"Num classes expected: {NUM_CLASSES}")


if __name__ == '__main__':
    main()
