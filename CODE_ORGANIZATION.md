# Code Organization Guide

Project đã được tổ chức lại theo modules riêng biệt.

## Cấu trúc mới

```
MedCLIP-SAMv2/
│
├── 📁 freqmedclip/              # FreqMedCLIP (Smart Single-Stream)
│   ├── scripts/
│   │   ├── freq_components.py   # DWT + SmartFusionBlock
│   │   └── postprocess.py       # Postprocessing utils
│   ├── train_freq_fusion.py     # Main training
│   ├── evaluate_model.py        # Evaluation + visualization
│   ├── train_both_clean.bat     # ⭐ Best training script
│   └── README.md
│
├── 📁 tgcam/                    # TGCam (Text-Guided CAM)
│   ├── train_tgcam_fusion.py
│   ├── train_tgcam_all.bat
│   └── README.md
│
├── 📁 documentation/            # All markdown docs
│   ├── PIPELINE_-FreqMedCLIP-Smart-Single-Stream.md
│   ├── FreqMedCLIP_Implementation_Guide.md
│   ├── CODE_LOCATION_MAP.md
│   ├── POSTPROCESSING_GUIDE.md
│   └── PROMPT_VALIDATION_REPORT.md
│
├── 📁 utilities/                # Testing & debugging
│   ├── test_training.py
│   ├── inspect_checkpoint.py
│   ├── compare_epochs.py
│   └── README.md
│
├── 📁 data/                     # Datasets (unchanged)
├── 📁 checkpoints/              # Model checkpoints (unchanged)
├── 📁 saliency_maps/            # Original MedCLIP code (unchanged)
├── 📁 scripts/                  # Shared scripts (unchanged)
└── 📁 visualizations/           # Output visualizations
```

## Quick Start

### Training FreqMedCLIP

```bash
cd freqmedclip
.\train_both_clean.bat
```

### Training TGCam

```bash
cd tgcam
.\train_tgcam_all.bat
```

### Documentation

Tất cả tài liệu trong `documentation/`:
- Architecture overview
- Implementation guides
- Code location map
- Postprocessing guide

## Import từ root project

```python
# FreqMedCLIP
from freqmedclip.scripts.freq_components import SmartFusionBlock, DWTForward
from freqmedclip.train_freq_fusion import FreqMedCLIPDataset

# TGCam
from saliency_maps.scripts.tgcam_components import TGCAMPipeline
```

## Changes Summary

**✅ Organized:**
- FreqMedCLIP files → `freqmedclip/`
- TGCam files → `tgcam/`
- Documentation → `documentation/`
- Utilities → `utilities/`

**✅ Updated:**
- All import paths fixed
- Relative paths for data/checkpoints
- Added `__init__.py` for module imports
- README in each folder

**✅ Unchanged:**
- `data/` - Dataset location
- `checkpoints/` - Checkpoint save location
- `saliency_maps/` - Original MedCLIP code
- `scripts/` - Shared utilities (methods.py, etc.)

## Migration Notes

### Nếu train trước đó:

**Old command:**
```bash
python train_freq_fusion.py --dataset breast_tumors
```

**New command:**
```bash
cd freqmedclip
python train_freq_fusion.py --dataset breast_tumors
```

**Hoặc từ root:**
```bash
python freqmedclip/train_freq_fusion.py --dataset breast_tumors
```

### Checkpoints paths

Tất cả checkpoints vẫn lưu ở `../checkpoints/` từ trong folder freqmedclip:
- `../checkpoints/fusion_breast_tumors_epoch100.pth`
- `../checkpoints/fusion_brain_tumors_epoch100.pth`

### Data paths

Data paths tự động trỏ đến `../data/` từ trong folder freqmedclip.

## Không ảnh hưởng training

Tổ chức lại code **KHÔNG ảnh hưởng** training vì:
- ✅ Tất cả imports đã được fix
- ✅ Paths sử dụng relative paths (`../data`, `../checkpoints`)
- ✅ Batch scripts vẫn work như cũ
- ✅ Model architecture không đổi
- ✅ Checkpoints compatible với code cũ

Bạn có thể train ngay bây giờ với:
```bash
cd freqmedclip
.\train_both_clean.bat
```
