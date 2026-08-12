# VisionLanguage-HOI-TheftDetection

Research-grade baseline for suspicious/theft behavior detection in surveillance video.

## Current implementation
- CSV-backed surveillance video dataset loader
- Fixed-length RGB clip decoding
- 3D CNN video baseline
- GPU/CPU training loop
- Checkpointing
- Accuracy, precision, recall, F1, specificity, AP and ROC-AUC evaluation
- Reproducible YAML configuration

## Dataset layout
```text
datasets/theft_dataset/
├── train.csv
├── val.csv
├── test.csv
└── videos/
    ├── train/
    ├── val/
    └── test/
```

CSV columns: `video,label`. Labels may be `0/1` or `normal/theft`.

## Windows
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py --check
python train.py --epochs 1
python evaluate.py
```

## GPU
The project automatically selects CUDA when available. Large VLM/HOI components will be added as optional modules so the baseline remains runnable on modest GPUs.
