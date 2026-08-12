from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset.video_dataset import TheftVideoDataset, records_from_csv
from src.models.classifier import VideoClassifier
from src.training.trainer import Trainer


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-config", default="configs/dataset.yaml")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    dc, tc = load_yaml(args.dataset_config), load_yaml(args.train_config)
    vcfg, tcfg = dc["video"], tc["training"]
    train_records = records_from_csv(dc["dataset"]["train"]["annotations"], dc["dataset"]["train"]["videos"])
    val_records = records_from_csv(dc["dataset"]["validation"]["annotations"], dc["dataset"]["validation"]["videos"])
    if not train_records:
        raise RuntimeError("No training videos found. Populate datasets/theft_dataset and annotation CSVs first.")
    train_ds = TheftVideoDataset(dc["dataset"]["root"], train_records, vcfg["frames"], vcfg["image_size"])
    val_ds = TheftVideoDataset(dc["dataset"]["root"], val_records, vcfg["frames"], vcfg["image_size"]) if val_records else None
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()) if val_ds else None
    model = VideoClassifier(tc["model"]["num_classes"])
    trainer = Trainer(model, tc["hardware"]["device"], tcfg["learning_rate"])
    best, epochs = -1.0, args.epochs or tcfg["epochs"]
    for epoch in range(1, epochs + 1):
        tr = trainer.train_epoch(train_loader)
        va = trainer.evaluate(val_loader) if val_loader else tr
        print(f"epoch={epoch:03d} train_loss={tr.loss:.4f} train_acc={tr.accuracy:.4f} val_loss={va.loss:.4f} val_acc={va.accuracy:.4f}")
        if va.accuracy > best:
            best = va.accuracy
            trainer.save(Path(tc["checkpoint"]["directory"]) / tc["checkpoint"]["filename"], epoch, {"val_accuracy": best})

if __name__ == "__main__":
    main()
