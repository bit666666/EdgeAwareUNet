# EdgeAwareUNet

Reference implementation for **EdgeAwareUNet**, a lightweight encoder–decoder network for pixel-level wind-turbine extraction from satellite imagery. The code corresponds to the historical implementation used for the manuscript experiments.

## Repository status

This repository contains model, training, and evaluation code. The WTA and WTS image patches and masks are distributed separately at [GL0925/Wind-Turbine-Dataset](https://github.com/GL0925/Wind-Turbine-Dataset). Source imagery and derived annotations remain subject to the original providers’ terms.

## Environment

Python 3.8 or newer is recommended. Install a compatible PyTorch build for your CPU/GPU, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The minimal requirements file lists dependencies for the standard EdgeAwareUNet training and evaluation workflow. The optional EfficientNet-B4 encoder also requires `segmentation-models-pytorch` and its `timm` dependency.

## Dataset layout

Training and test data should use this layout:

```text
DATA_ROOT/
  train/img/
  train/label_1D/
  test/img/
  test/label_1D/
```

Images may be JPEG or PNG. Masks are PNG files with matching stems. The loader resizes inputs to 512 × 256 pixels and normalizes RGB channels with ImageNet statistics.

## Train

From the repository root:

```bash
python scripts/train.py --data /path/to/DATA_ROOT --model edgeaware --run-dir runs/wts_edgeaware
```

The script creates a validation split from the training directory, writes `config.json`, `split.json`, and `history.csv`, and saves the best checkpoint as `best.pt`. Use `--help` for all options.

## Evaluate

```bash
python scripts/evaluate.py --data /path/to/DATA_ROOT --model edgeaware --checkpoint runs/wts_edgeaware/best.pt --output runs/wts_edgeaware/test_metrics.json
```

## Reproducibility

Set the random seed and split seed explicitly when reproducing a run. The repository does not include trained weights or the WTS/WTA data files.

## License

Code is released under the MIT License. Dataset files are not covered by this code license; consult the source dataset licenses and attribution requirements before reuse.
