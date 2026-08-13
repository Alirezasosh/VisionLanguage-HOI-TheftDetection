from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
from ultralytics import YOLOWorld


DEFAULT_CLASSES = [
    "person",
    "backpack",
    "handbag",
    "suitcase",
    "cell phone",
    "laptop",
    "bottle",
]


def _box_center_xyxy(box: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def run_video_demo(
    video_path: str,
    output_path: str,
    csv_path: str,
    model_name: str = "yolov8s-worldv2.pt",
    confidence: float = 0.25,
    classes: list[str] | None = None,
) -> None:
    """Run a fast open-vocabulary surveillance screening demo.

    This is a *screening/demonstration* pipeline, not a trained theft classifier.
    It detects user-defined object categories and writes an interaction-oriented
    suspicion score based on object proximity. It must not be reported as a
    validated theft metric without labeled ground-truth evaluation.
    """
    source = Path(video_path)
    output = Path(output_path)
    csv_file = Path(csv_path)

    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    model = YOLOWorld(model_name, verbose=False)
    model.set_classes(classes or DEFAULT_CLASSES)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Could not read video dimensions.")

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create output video: {output}")

    rows: list[dict[str, object]] = []
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            conf=confidence,
            verbose=False,
            device=0 if __import__("torch").cuda.is_available() else "cpu",
        )
        result = results[0]

        detections: list[tuple[str, float, tuple[float, float, float, float]]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            classes_idx = result.boxes.cls.cpu().tolist()

            for box, score, cls_id in zip(xyxy, confs, classes_idx):
                name = result.names[int(cls_id)]
                detections.append((name, float(score), tuple(map(float, box))))

        counts = Counter(name for name, _, _ in detections)
        person_centers = [_box_center_xyxy(box) for name, _, box in detections if name == "person"]
        object_centers = [
            _box_center_xyxy(box)
            for name, _, box in detections
            if name in {"backpack", "handbag", "suitcase", "cell phone", "laptop"}
        ]

        close_pairs = 0
        if person_centers and object_centers:
            # Pixel-distance heuristic normalized by image diagonal.
            diag = (width * width + height * height) ** 0.5
            threshold = 0.18 * diag
            for pc in person_centers:
                for oc in object_centers:
                    if _distance(pc, oc) <= threshold:
                        close_pairs += 1

        suspicious_score = min(
            1.0,
            0.15 * min(counts.get("cell phone", 0) + counts.get("laptop", 0), 2)
            + 0.20 * min(counts.get("backpack", 0) + counts.get("handbag", 0) + counts.get("suitcase", 0), 2)
            + 0.25 * min(close_pairs, 2),
        )

        annotated = result.plot()

        cv2.putText(
            annotated,
            f"screening_score={suspicious_score:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(annotated)

        rows.append(
            {
                "frame": frame_index,
                "time_sec": frame_index / fps,
                "person_count": counts.get("person", 0),
                "backpack_count": counts.get("backpack", 0),
                "handbag_count": counts.get("handbag", 0),
                "suitcase_count": counts.get("suitcase", 0),
                "cell_phone_count": counts.get("cell phone", 0),
                "laptop_count": counts.get("laptop", 0),
                "close_person_object_pairs": close_pairs,
                "screening_score": round(suspicious_score, 6),
            }
        )

        frame_index += 1

    capture.release()
    writer.release()

    fieldnames = list(rows[0].keys()) if rows else [
        "frame", "time_sec", "person_count", "backpack_count",
        "handbag_count", "suitcase_count", "cell_phone_count",
        "laptop_count", "close_person_object_pairs", "screening_score"
    ]
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"Input:  {source}")
    print(f"Video:  {output}")
    print(f"Log:    {csv_file}")
    print(f"Frames: {frame_index}")
    print("Note: screening_score is a heuristic demo signal, not a validated theft metric.")
