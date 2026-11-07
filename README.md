# EdgeAwareUNet

Reference implementation for **EdgeAwareUNet**, a lightweight encoder–decoder network for pixel-level wind-turbine extraction from satellite imagery.

## Repository status

This repository contains a cleaned, runnable binary-segmentation implementation derived from the historical project files. The WTA and WTS image patches and masks are distributed separately at [bit666666/Wind-Turbine-Dataset](https://github.com/bit666666/Wind-Turbine-Dataset). Source imagery and derived annotations remain subject to the original providers’ terms.

The manuscript reports experiments using the historical project configuration: 256 x 512 inputs, Adam, dataset-specific learning rates, a maximum of 350 epochs, patience 25, and repeated runs. This public reference release does not include the historical checkpoints, run logs, the private data-preparation utilities, or the complete multi-run orchestration. Therefore it should not be described as a byte-for-byte reproduction of every reported number.

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

Set the random seed and split seed explicitly when reproducing a run. The repository does not include trained weights or the WTS/WTA data files. The evaluation script imports its dataset and metric helpers from `scripts/train.py` and is runnable with a checkpoint produced by the released training script.

The historical WTS training script used two classes and the settings above. The historical WTA configuration used three classes, a learning rate of 1e-3, and batch size 8. Those historical settings are documented here for transparency; the released command-line training script is the maintained binary reference workflow.

## License

Code is released under the MIT License. Dataset files are not covered by this code license; consult the source dataset licenses and attribution requirements before reuse.
