# Utilities

Folder chứa các công cụ testing và debugging.

## Files

- **test_training.py** - Quick training test (1 batch)
- **test_forward.py** - Forward pass test
- **inspect_checkpoint.py** - View checkpoint contents
- **compare_epochs.py** - Compare checkpoints
- **create_overlays.py** - Generate overlay images
- **reference_fusion.py** - Fusion block reference implementation
- **reference_wavelet.py** - Wavelet reference implementation

## Usage

```bash
cd utilities
python test_training.py
python inspect_checkpoint.py ../checkpoints/fusion_breast_tumors_epoch100.pth
```
