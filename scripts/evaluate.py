import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from Net.EdgeAwareUnet import EdgeAwareUnet
from train_edgeaware_clean import (
    EdgeAwareEfficientNetB4,
    TurbineBinaryDataset,
    update_confusion,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="turbine_dataset_2")
    parser.add_argument("--model", choices=("edgeaware", "edgeaware_b4"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TurbineBinaryDataset(args.data, mode="test")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.model == "edgeaware":
        model = EdgeAwareUnet(classes=2)
    else:
        model = EdgeAwareEfficientNetB4(classes=2, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()

    confusion = torch.zeros((2, 2), dtype=torch.int64, device=device)
    target_present_images = 0
    false_positive_pixels_empty_images = 0
    with torch.no_grad():
        for image, target in loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(image)
            update_confusion(confusion, logits, target)
            if bool(target.any().item()):
                target_present_images += 1
            else:
                false_positive_pixels_empty_images += int((logits.argmax(dim=1) == 1).sum().item())

    matrix = confusion.cpu()
    tp = int(matrix[1, 1].item())
    fp = int(matrix[0, 1].item())
    fn = int(matrix[1, 0].item())
    total = int(matrix.sum().item())
    class_iou, target_iou = _ious(matrix)
    result = {
        "model": args.model,
        "checkpoint": os.path.abspath(args.checkpoint),
        "test_samples": len(dataset),
        "target_present_images": target_present_images,
        "empty_target_images": len(dataset) - target_present_images,
        "confusion_matrix_true_rows_pred_columns": matrix.tolist(),
        "background_iou": class_iou[0],
        "target_iou": target_iou,
        "mean_iou": sum(class_iou) / 2.0,
        "target_precision": tp / max(tp + fp, 1),
        "target_recall": tp / max(tp + fn, 1),
        "target_dice": (2 * tp) / max(2 * tp + fp + fn, 1),
        "pixel_accuracy": (int(matrix[0, 0]) + tp) / max(total, 1),
        "false_positive_pixels_on_empty_target_images": false_positive_pixels_empty_images,
        "selected_epoch": checkpoint.get("epoch"),
        "validation_target_iou": checkpoint.get("val_target_iou"),
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _ious(confusion):
    confusion = confusion.double()
    intersection = confusion.diag()
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
    values = (intersection / union.clamp_min(1.0)).tolist()
    return [float(value) for value in values], float(values[1])


if __name__ == "__main__":
    main()
