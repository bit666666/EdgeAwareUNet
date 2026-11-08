# EdgeAwareUNet

EdgeAwareUNet is a lightweight encoder-decoder network for wind-turbine segmentation. This release adds the recovered historical WTA three-class and WTS binary workflows, plus ten 2025 state dictionaries. All recovered model, utility, training, evaluation, and checkpoint files predate **8 September 2026**. No recent experiment directory was used to reconstruct these sources.

Data: https://github.com/bit666666/Wind-Turbine-Dataset

## Available materials

- WTA: 189 training and 35 test images; labels 0 (background), 1 (blade), and 2 (tower).
- WTS: 871 training and 110 test images; labels 0 (background) and 1 (wind turbine).
- `historical/`: recovered model, losses, data loader, training loop, and evaluation source snapshots.
- `scripts/train_historical.py`: selects the dataset and training settings without collapsing WTA's classes.
- `scripts/evaluate_historical.py`: loads historical state dictionaries and reports both historical and global IoU.
- Ten state dictionaries: five WTA and five WTS checkpoints in the [historical-2025 release](https://github.com/bit666666/EdgeAwareUNet/releases/tag/historical-2025).
- `docs/HISTORICAL_PROVENANCE.json`: original dates and SHA-256 values.
- `docs/historical_experiment_record_2025.txt`: recovered experiment-summary record.
- `docs/checkpoint_verification_20261108.json`: complete checkpoint re-evaluation results.

## Environment

The historical checkpoint verification used Linux, Python 3.8.20, PyTorch 1.8.1+cu111, and torchvision 0.9.1+cu111. For that software family:

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements-historical.txt
```

Install a PyTorch build appropriate to your operating system and hardware. The older CUDA build is optional for CPU inspection. The previous cleaned binary reference workflow remains in `scripts/train.py` and `scripts/evaluate.py`; its broader dependency constraints are in `requirements.txt`. That reference workflow is not the historical three-class experiment entry point.

## Prepare data and weights

```bash
git clone https://github.com/bit666666/Wind-Turbine-Dataset.git
python Wind-Turbine-Dataset/WTS_dataset/download_training.py
python scripts/download_weights.py
```

Each dataset directory must contain `train/img`, `train/label_1D`, `train/label_3D`, and the corresponding `test` directories. The download script verifies every state dictionary against `docs/WEIGHTS.json`.

## Evaluate WTA and WTS

```bash
python scripts/evaluate_historical.py --dataset wta --data Wind-Turbine-Dataset/WTA_dataset --output results/wta.json
python scripts/evaluate_historical.py --dataset wts --data Wind-Turbine-Dataset/WTS_dataset --output results/wts.json
```

By default all five checkpoints for the selected dataset are evaluated. Use `--checkpoint path/to/state_dict.pth` for a single run or `--device cpu` for CPU evaluation. Input tensors are resized to height 256, width 512, preserving the recovered preprocessing and class indices.

## Recovered training configuration

WTA uses three classes, learning rate 1e-3, and batch size 8. WTS uses two classes, learning rate 1e-4, and batch size 2. Both use Adam, up to 350 epochs, patience 25, and deep-supervision output weights `[1.0, 0.2, 0.2, 0.2, 0.2]`. The loss is CE + 0.2 Jaccard + 0.2 Dice, with an additional 0.4 focal term for WTS. These settings are taken from the recovered summary and manuscript; the retained snapshot's active configuration is WTS.

The old training loop used the `test` directory during training for early stopping and checkpoint selection. To inspect that historical protocol explicitly:

```bash
python scripts/train_historical.py --dataset wta --data Wind-Turbine-Dataset/WTA_dataset --run-dir runs/wta_historical --selection-split test
python scripts/train_historical.py --dataset wts --data Wind-Turbine-Dataset/WTS_dataset --run-dir runs/wts_historical --selection-split test
```

For new experiments, supply a separately constructed `val` directory and use the default `--selection-split val`; keep the test set outside checkpoint selection. Neither release path claims an untouched test protocol for the historical results. The release adapter configures the old source and saves state dictionaries; it is a packaging addition, not evidence of a newly run paper experiment.

## Verification and limits

Five recovered WTA checkpoints reproduce the manuscript's historical mean class IoUs `[0.9957, 0.9402, 0.9155]` and mean IoU **0.9505** under the original metric convention. The recovered WTS checkpoints produce `[0.9990, 0.6313]`, mean IoU **0.8152**, in the verification environment; the original experiment record and manuscript report `[0.9991, 0.6321]`, mean IoU **0.8156**. The remaining WTS discrepancy is unresolved. The release does not claim exact reproduction of every numerical result or recovery of all comparison and ablation runs.

The historical IoU routine skips an image's contribution for a class absent from its ground truth. The evaluator also outputs standard global confusion-matrix IoU, including false positives on all images. These conventions can give different numbers and should be identified when comparing results. The recovered Jaccard loss uses hard argmax labels; that historical implementation is retained rather than silently replaced. See [historical source notes](historical/README.md).

No retraining or new benchmark result was generated during source recovery. Original random seeds and all training trajectories are not completely mapped to the released checkpoints. Dataset georeferencing, per-patch GSD, acquisition dates, and geographically independent splits are not established by the available archive.

## License

Author-contributed code and checkpoints are provided under the MIT License. Dataset files are distributed separately under their applicable licenses: WTA retains Apache 2.0, and the WTS authors' annotations use CC BY 4.0 with upstream attribution requirements. The code license does not override image-provider rights.
