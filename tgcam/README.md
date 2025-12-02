# TGCam - Text-Guided Gradient-based CAM

Folder chứa code cho TGCam (Text-Guided Class Activation Mapping).

## Cấu trúc

```
tgcam/
├── train_tgcam_fusion.py       # Training script
└── train_tgcam_all.bat         # Batch training for all datasets
```

## Quick Start

### Training

```bash
cd tgcam
.\train_tgcam_all.bat
```

Hoặc train từng dataset:

```bash
python train_tgcam_fusion.py --dataset breast_tumors --epochs 100
```

## Core Components

TGCam sử dụng `saliency_maps/scripts/tgcam_components.py`:
- `TGCAMPipeline` - Text-guided CAM generation
- Attention-based fusion mechanism

## Import từ ngoài folder

```python
from saliency_maps.scripts.tgcam_components import TGCAMPipeline
```
