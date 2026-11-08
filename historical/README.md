# Historical source snapshot

These model, utility, training, and evaluation sources were recovered from the original project archive. Every recovered source file predates 8 September 2026; model and state-dictionary files are from 2025, while the retained training/evaluation script timestamps are from July 2026. See `../docs/HISTORICAL_PROVENANCE.json` for dates and hashes.

Non-English code comments and documentation strings were removed or replaced with English text. Parsed computational code is unchanged. The small `Net/__init__.py` exports only the EdgeAwareUNet family, avoiding imports of unrelated third-party comparison models. No code from recent experimental directories is included here.

The snapshot `train_ds.py` is preserved with its active WTS configuration. Use `../scripts/train_historical.py` to select WTA three-class or WTS binary configuration from the published dataset settings. Use `../scripts/evaluate_historical.py` with the released state dictionaries for evaluation. The snapshot evaluation scripts retain their original hard-coded example model names and full-model-loading approach; the maintained evaluation entry point loads state dictionaries and selects the dataset explicitly.

## Historical selection and metric behavior

The recovered training loop evaluates `test` during training and uses its mean IoU for early stopping and checkpoint selection. This historical behavior is exposed explicitly by `--selection-split test`; it is not an untouched held-out-test protocol. The convenience runner defaults to a separately provided `val` directory for new experiments. It does not create new reported paper results.

The historical IoU function excludes an image's contribution for a class absent from that image's ground truth. The evaluation entry point reports both that historical convention and a global confusion-matrix IoU, which counts false positives on all images. They must not be silently interchanged.

The Jaccard component in the recovered loss uses hard argmax predictions. This archived implementation is retained for inspection; it is not presented as a newly corrected differentiable loss. No training run was performed or invented during recovery.
