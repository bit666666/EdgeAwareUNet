import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from Net.EdgeAwareUnet import (
    ConvBlock,
    DDPU,
    EEM,
    EdgeEncoder,
    EdgeAwareUnet,
)


class TurbineBinaryDataset(Dataset):
    def __init__(self, root, mode="train", height=256, width=512):
        self.root = root
        self.mode = mode
        self.height = height
        self.width = width
        self.image_dir = os.path.join(root, mode, "img")
        self.mask_dir = os.path.join(root, mode, "label_1D")
        extensions = {".jpg", ".jpeg", ".png"}
        names = []
        for filename in sorted(os.listdir(self.image_dir)):
            stem, extension = os.path.splitext(filename)
            if extension.lower() in extensions:
                mask_path = os.path.join(self.mask_dir, stem + ".png")
                if not os.path.isfile(mask_path):
                    raise FileNotFoundError("Missing mask: " + mask_path)
                names.append((filename, stem + ".png"))
        if not names:
            raise RuntimeError("No image/mask pairs found under " + self.image_dir)
        self.names = names

    def __len__(self):
        return len(self.names)

    def has_foreground(self, index):
        _, mask_name = self.names[index]
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")
        return bool(np.asarray(mask).any())

    def __getitem__(self, index):
        image_name, mask_name = self.names[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")
        image = image.resize((self.width, self.height), Image.BILINEAR)
        mask = mask.resize((self.width, self.height), Image.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - np.array([0.485, 0.456, 0.406], dtype=np.float32))
        image_array = image_array / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())

        mask_array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_array.copy())
        return image_tensor, mask_tensor


def stratified_split(dataset, val_fraction, seed):
    groups = defaultdict(list)
    for index in range(len(dataset)):
        groups[dataset.has_foreground(index)].append(index)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []
    for indices in groups.values():
        rng.shuffle(indices)
        if len(indices) < 2:
            val_count = 0
        else:
            val_count = max(1, int(round(len(indices) * val_fraction)))
            val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    if not val_indices:
        raise RuntimeError("The train split produced no validation samples")
    return train_indices, val_indices


class EdgeAwareEfficientNetB4(nn.Module):
    def __init__(self, classes=2, pretrained=True):
        super().__init__()
        from segmentation_models_pytorch.encoders import get_encoder

        weights = "imagenet" if pretrained else None
        self.encoder = get_encoder(
            "timm-efficientnet-b4", in_channels=3, depth=4, weights=weights
        )
        channels = list(self.encoder.out_channels)
        if len(channels) != 5:
            raise RuntimeError("Expected input plus four encoder feature maps, got " + str(channels))

        c = [16, 32, 64, 128, 256]
        self.encoderConv1 = EdgeEncoder(3, c[0], 2)
        self.encoderConv2 = EdgeEncoder(channels[1], c[1], 2)
        self.encoderConv3 = EdgeEncoder(channels[2], c[2], 2)
        self.encoderConv4 = EdgeEncoder(channels[3], c[3], 2)
        self.bottleneck = ConvBlock(channels[4], c[4])

        self.ddpu1 = DDPU(c[4], c[4])
        self.eem1 = EEM(c[3])
        self.decoderConv1 = ConvBlock(c[4] + c[3], c[3])
        self.ddpu2 = DDPU(c[3], c[3])
        self.eem2 = EEM(c[2])
        self.decoderConv2 = ConvBlock(c[3] + c[2], c[2])
        self.ddpu3 = DDPU(c[2], c[2])
        self.eem3 = EEM(c[1])
        self.decoderConv3 = ConvBlock(c[2] + c[1], c[1])
        self.ddpu4 = DDPU(c[1], c[1])
        self.eem4 = EEM(c[0])
        self.decoderConv4 = ConvBlock(c[1] + c[0], c[0])
        self.out = nn.Conv2d(c[0], classes, kernel_size=1)
        self.sup1 = self._make_supervision_layer(c[3], classes)
        self.sup2 = self._make_supervision_layer(c[2], classes)
        self.sup3 = self._make_supervision_layer(c[1], classes)
        self.sup4 = self._make_supervision_layer(c[0], classes)

    @staticmethod
    def _make_supervision_layer(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, image):
        features = self.encoder(image)
        en_x1 = self.encoderConv1(image)
        en_x2 = self.encoderConv2(features[1])
        en_x3 = self.encoderConv3(features[2])
        en_x4 = self.encoderConv4(features[3])
        bottleneck = self.bottleneck(features[4])

        de_x1 = self.ddpu1(bottleneck)
        cat_1 = self.eem1(en_x4)
        de_x1 = self.decoderConv1(torch.cat([cat_1, de_x1], dim=1))
        de_x2 = self.ddpu2(de_x1)
        cat_2 = self.eem2(en_x3)
        de_x2 = self.decoderConv2(torch.cat([cat_2, de_x2], dim=1))
        de_x3 = self.ddpu3(de_x2)
        cat_3 = self.eem3(en_x2)
        de_x3 = self.decoderConv3(torch.cat([cat_3, de_x3], dim=1))
        de_x4 = self.ddpu4(de_x3)
        cat_4 = self.eem4(en_x1)
        de_x4 = self.decoderConv4(torch.cat([cat_4, de_x4], dim=1))
        output = self.out(de_x4)

        if self.training:
            size = image.shape[-2:]
            sup1 = self.sup1(F.interpolate(cat_1, size=size, mode="bilinear", align_corners=False))
            sup2 = self.sup2(F.interpolate(cat_2, size=size, mode="bilinear", align_corners=False))
            sup3 = self.sup3(F.interpolate(cat_3, size=size, mode="bilinear", align_corners=False))
            sup4 = self.sup4(F.interpolate(cat_4, size=size, mode="bilinear", align_corners=False))
            return [output, sup4, sup3, sup2, sup1]
        return output


def segmentation_loss(outputs, target):
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    output_weights = [1.0, 0.2, 0.2, 0.2, 0.2]
    total = outputs[0].new_tensor(0.0)
    one_hot = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
    for weight, logits in zip(output_weights, outputs):
        probabilities = F.softmax(logits, dim=1)
        ce_map = F.cross_entropy(logits, target, reduction="none")
        focal = (0.8 * (1.0 - torch.exp(-ce_map)).pow(2) * ce_map).mean()

        intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
        probability_sum = probabilities.sum(dim=(0, 2, 3))
        target_sum = one_hot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + 1e-5) / (probability_sum + target_sum + 1e-5)
        union = probability_sum + target_sum - intersection
        iou = (intersection + 1e-5) / (union + 1e-5)
        loss = (
            F.cross_entropy(logits, target)
            + 0.4 * focal
            + 0.2 * (1.0 - dice.mean())
            + 0.2 * (1.0 - iou.mean())
        )
        total = total + weight * loss
    return total


def update_confusion(confusion, logits, target):
    prediction = logits.argmax(dim=1)
    encoded = (target.reshape(-1) * 2 + prediction.reshape(-1)).long()
    confusion += torch.bincount(encoded, minlength=4).reshape(2, 2)


def metrics_from_confusion(confusion):
    confusion = confusion.double()
    intersection = confusion.diag()
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
    iou = intersection / union.clamp_min(1.0)
    return [float(value) for value in iou.cpu()], float(iou[1].cpu())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, optimizer, device, training, max_batches=0):
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    confusion = torch.zeros((2, 2), dtype=torch.int64, device=device)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (image, target) in enumerate(loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad()
            outputs = model(image)
            loss = segmentation_loss(outputs, target)
            if training:
                loss.backward()
                optimizer.step()
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            total_loss += float(loss.item()) * image.size(0)
            total_samples += image.size(0)
            update_confusion(confusion, logits, target)
            if max_batches and batch_index + 1 >= max_batches:
                break
    iou, foreground_iou = metrics_from_confusion(confusion)
    return total_loss / max(total_samples, 1), iou, foreground_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="turbine_dataset_2")
    parser.add_argument("--model", choices=("edgeaware", "edgeaware_b4"), required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=20261013)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=False)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TurbineBinaryDataset(args.data, mode="train")
    train_indices, val_indices = stratified_split(dataset, args.val_fraction, args.split_seed)
    split = {
        "split_seed": args.split_seed,
        "train": [dataset.names[i][0] for i in train_indices],
        "validation": [dataset.names[i][0] for i in val_indices],
        "test_used_during_training": False,
    }
    with open(os.path.join(args.run_dir, "split.json"), "w", encoding="utf-8") as handle:
        json.dump(split, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=torch.cuda.is_available(), drop_last=False
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=torch.cuda.is_available(), drop_last=False
    )

    if args.model == "edgeaware":
        model = EdgeAwareUnet(classes=2)
    else:
        model = EdgeAwareEfficientNetB4(classes=2, pretrained=not args.no_pretrained)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: (1.0 - epoch / max(args.epochs, 1)) ** 0.9
    )

    best_iou = -1.0
    stale_epochs = 0
    history_path = os.path.join(args.run_dir, "history.csv")
    start_time = time.time()
    with open(history_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "train_loss", "train_target_iou", "val_loss", "val_background_iou", "val_target_iou", "val_miou"))
        for epoch in range(1, args.epochs + 1):
            train_loss, _, train_target_iou = run_epoch(
                model, train_loader, optimizer, device, True, args.max_train_batches
            )
            val_loss, val_ious, val_target_iou = run_epoch(
                model, val_loader, optimizer, device, False, args.max_val_batches
            )
            scheduler.step()
            val_miou = sum(val_ious) / len(val_ious)
            writer.writerow((epoch, train_loss, train_target_iou, val_loss, val_ious[0], val_target_iou, val_miou))
            handle.flush()
            print(
                "epoch={}/{} train_loss={:.5f} train_target_iou={:.4f} "
                "val_loss={:.5f} val_target_iou={:.4f} val_miou={:.4f}".format(
                    epoch, args.epochs, train_loss, train_target_iou,
                    val_loss, val_target_iou, val_miou
                ),
                flush=True,
            )

            if val_target_iou > best_iou:
                best_iou = val_target_iou
                stale_epochs = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "seed": args.seed,
                        "split_seed": args.split_seed,
                        "val_target_iou": best_iou,
                        "model_name": args.model,
                    },
                    os.path.join(args.run_dir, "best.pt"),
                )
            else:
                stale_epochs += 1
            if stale_epochs >= args.patience:
                print("early_stop epoch={}".format(epoch), flush=True)
                break

    print(
        "finished best_val_target_iou={:.4f} elapsed_minutes={:.1f} run_dir={}".format(
            best_iou, (time.time() - start_time) / 60.0, args.run_dir
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
