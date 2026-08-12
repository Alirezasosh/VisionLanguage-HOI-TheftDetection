from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochResult:
    loss: float
    accuracy: float


class Trainer:
    def __init__(self, model: nn.Module, device: str | torch.device = "auto", lr: float = 1e-4):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader: DataLoader) -> EpochResult:
        self.model.train()
        total_loss = correct = total = 0
        for batch in loader:
            x = batch["video"].to(self.device)
            y = batch["label"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        return EpochResult(total_loss / max(total, 1), correct / max(total, 1))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EpochResult:
        self.model.eval()
        total_loss = correct = total = 0
        for batch in loader:
            x = batch["video"].to(self.device)
            y = batch["label"].to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        return EpochResult(total_loss / max(total, 1), correct / max(total, 1))

    def save(self, path: str | Path, epoch: int, metrics: dict):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "metrics": metrics}, path)
