from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from src.dataset.video_dataset import TheftVideoDataset, records_from_csv
from src.models.classifier import VideoClassifier
from src.evaluation.metrics import classification_metrics

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    ap.add_argument("--dataset-config", default="configs/dataset.yaml")
    args = ap.parse_args()
    with open(args.dataset_config, encoding="utf-8") as f: dc = yaml.safe_load(f)
    records = records_from_csv(dc["dataset"]["test"]["annotations"], dc["dataset"]["test"]["videos"])
    if not records: raise RuntimeError("No test videos found.")
    ds = TheftVideoDataset(dc["dataset"]["root"], records, dc["video"]["frames"], dc["video"]["image_size"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoClassifier(2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ys, ps = [], []
    for b in loader:
        prob = torch.softmax(model(b["video"].to(device)), 1)[:, 1]
        ys.append(int(b["label"].item())); ps.append(float(prob.item()))
    print(classification_metrics(ys, ps))

if __name__ == "__main__": main()
