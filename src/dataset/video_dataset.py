from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import torch
from torch.utils.data import Dataset


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class VideoRecord:
    path: Path
    label: int


def discover_videos(root: str | Path) -> list[VideoRecord]:
    root = Path(root)
    records: list[VideoRecord] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            label = int(class_dir.name)
        except ValueError:
            label = 1 if class_dir.name.lower() in {"theft", "steal", "anomaly", "positive"} else 0
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                records.append(VideoRecord(path, label))
    return records


def read_video_clip(path: str | Path, num_frames: int = 16, size: int = 224) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = torch.linspace(0, total - 1, num_frames).round().long().tolist()
    frames = []
    wanted = set(indices)
    i = 0
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
    """Minimal filesystem video dataset.

    Expected layout: root/0/*.mp4 and root/1/*.mp4, or folders named
    normal/theft (folder names are mapped to 0/1).
    """

    def __init__(self, root: str | Path, records: Sequence[VideoRecord] | None = None,
                 num_frames: int = 16, size: int = 224):
        self.root = Path(root)
        self.records = list(records) if records is not None else discover_videos(self.root)
        self.num_frames = num_frames
        self.size = size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        clip = read_video_clip(record.path, self.num_frames, self.size)
        return {"video": clip, "label": torch.tensor(record.label, dtype=torch.long), "path": str(record.path)}
