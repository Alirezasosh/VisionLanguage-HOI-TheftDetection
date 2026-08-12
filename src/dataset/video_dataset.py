from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

@dataclass(frozen=True)
class VideoRecord:
    path: Path
    label: int


def discover_videos(root: str | Path) -> list[VideoRecord]:
    root = Path(root)
    if not root.exists():
        return []
    records = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            label = int(class_dir.name)
        except ValueError:
            label = 1 if class_dir.name.lower() in {"theft", "steal", "anomaly", "positive"} else 0
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                records.append(VideoRecord(path, label))
    return records


def records_from_csv(csv_path: str | Path, video_root: str | Path) -> list[VideoRecord]:
    csv_path, video_root = Path(csv_path), Path(video_root)
    if not csv_path.exists():
        return discover_videos(video_root)
    df = pd.read_csv(csv_path)
    path_col = next((c for c in ("video", "path", "filename", "file") if c in df.columns), None)
    label_col = next((c for c in ("label", "target", "class", "y") if c in df.columns), None)
    if path_col is None or label_col is None:
        raise ValueError(f"CSV must contain a video/path/filename column and label/target/class column: {csv_path}")
    records = []
    for _, row in df.iterrows():
        p = Path(str(row[path_col]))
        if not p.is_absolute():
            p = video_root / p
        value = row[label_col]
        try:
            label = int(value)
        except (TypeError, ValueError):
            label = 1 if str(value).lower() in {"theft", "steal", "anomaly", "positive", "1"} else 0
        records.append(VideoRecord(p, label))
    return records


def read_video_clip(path: str | Path, num_frames: int = 16, size: int = 224) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = torch.linspace(0, total - 1, num_frames).round().long().tolist()
    wanted = set(indices)
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())
    return torch.stack(frames[:num_frames])

class TheftVideoDataset(Dataset):
    def __init__(self, root: str | Path, records: Sequence[VideoRecord] | None = None, num_frames: int = 16, size: int = 224):
        self.root = Path(root)
        self.records = list(records) if records is not None else discover_videos(self.root)
        self.num_frames, self.size = num_frames, size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        return {"video": read_video_clip(r.path, self.num_frames, self.size), "label": torch.tensor(r.label, dtype=torch.long), "path": str(r.path)}
