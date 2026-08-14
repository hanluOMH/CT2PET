# PSMA CT-to-PET Pix2Pix

This project trains a 2D slice-based Pix2Pix model to synthesize PET/SUV images
from CT images.

The original data stay as `.nii.gz`. Training uses a disk-budgeted staging cache:
only the current chunk and a prefetched next chunk are prepared, then completed
chunks are deleted. This avoids full-dataset decompression on servers with
limited disk space.

## Data Layout

Expected paths are configured in `configs/default.yaml`:

- train CT: `../data/ct_clipped_no_normed`
- train PET/SUV: `../data/suv_clipped_no_normed`
- validation CT: `../data/ct_val`
- validation PET/SUV: `../data/pet_val`

Cases are paired by numeric id in the filename, for example `401.nii.gz`.

## Normalization

- CT is clipped to `[-1000, 2000]` and mapped to `[-1, 1]`.
- PET/SUV is clipped to `[0, 50]` and mapped to `[-1, 1]`.
- Metrics and TensorBoard PET visualizations are converted back to SUV scale.

## Install

```bash
cd /Users/hanlufeng/Desktop/Pix2pix/CT2PET
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On a CUDA server, install the PyTorch build that matches the server CUDA version
if the default `pip install torch` is not appropriate.

## Local CPU Smoke Test

The default config is intentionally small and CPU-compatible:

```bash
python scripts/train.py --config configs/default.yaml
```

It runs one epoch with `batch_size=1`, `chunk_cases=1`, and `image_size=256`.

## Server Training Notes

Recommended config changes for GPU training:

```yaml
data:
  image_size: 512
  chunk_cases: 1
  max_cache_gb: 40
  max_train_slices: null
  max_val_slices: null

train:
  epochs: 100
  batch_size: 4
  device: auto
  amp: true
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4
```

If disk pressure is high, keep `chunk_cases=1`. The trainer prepares the next
chunk in the background while the current chunk is training, then removes
completed chunk caches.

## Outputs

- TensorBoard logs: `runs/psma_ct_to_pet_pix2pix`
- Metrics CSV: `metrics.csv`
- Checkpoints: `checkpoints/latest.pt`, `checkpoints/best.pt`, and per-epoch files

Resume is enabled by default. If `checkpoints/latest.pt` exists, training starts
from the next epoch after the saved checkpoint.

TensorBoard:

```bash
tensorboard --logdir runs
```
