from __future__ import annotations

import argparse
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(description="Vision-Language HOI Theft Detection")
    p.add_argument("--check", action="store_true", help="run environment/model smoke checks")
    args = p.parse_args()
    print("Project root:", Path.cwd())
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
    if args.check:
        from src.models.classifier import VideoClassifier
        model = VideoClassifier()
        x = torch.randn(1, 16, 3, 224, 224)
        with torch.no_grad(): y = model(x)
        print("Smoke-test logits shape:", tuple(y.shape))

if __name__ == "__main__": main()
