"""Evaluate recovered WTA three-class or WTS binary state dictionaries."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from historical.Net.EdgeAwareUnet import EdgeAwareUnet
from historical.utils import WindTurbineDataset, calculate_classiou_batch


def historical_resize():
    try:
        return transforms.Resize((256, 512), antialias=False)
    except TypeError:
        return transforms.Resize((256, 512))


def evaluate(data, checkpoints, classes, device):
    image_transform = transforms.Compose([
        transforms.ToTensor(), historical_resize(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    mask_transform = transforms.Compose([transforms.PILToTensor(), historical_resize()])
    dataset = WindTurbineDataset(str(data), 'test', image_transform, mask_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    for checkpoint in checkpoints:
        try:
            state = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(str(checkpoint), map_location='cpu')
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        if all(key.startswith('module.') for key in state):
            state = {key[7:]: value for key, value in state.items()}
        model = EdgeAwareUnet(classes).to(device).eval()
        model.load_state_dict(state, strict=True)
        intersection, union = [0] * classes, [0] * classes
        confusion = torch.zeros(classes, classes, dtype=torch.int64, device=device)
        with torch.no_grad():
            for image, target, _ in loader:
                image, target = image.to(device), target.to(device)
                if int(target.min()) < 0 or int(target.max()) >= classes:
                    raise ValueError('Mask contains an invalid class index')
                prediction = model(image).argmax(1)
                a, b = calculate_classiou_batch(prediction, target, num_classes=classes)
                intersection = [x + y for x, y in zip(intersection, a)]
                union = [x + y for x, y in zip(union, b)]
                encoded = target.reshape(-1) * classes + prediction.reshape(-1)
                confusion += torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)
        old = [round(a / b if b else 0., 4) for a, b in zip(intersection, union)]
        cm = confusion.cpu().numpy()
        denominator = cm.sum(0) + cm.sum(1) - cm.diagonal()
        standard = (cm.diagonal() / np.maximum(denominator, 1)).tolist()
        row = {'checkpoint': checkpoint.name, 'samples': len(dataset),
               'historical_class_iou': old, 'historical_mean_iou': round(float(np.mean(old)), 4),
               'global_class_iou': standard, 'global_mean_iou': float(np.mean(standard))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    average = np.round(np.mean([row['historical_class_iou'] for row in rows], axis=0), 4).tolist()
    return {'torch': torch.__version__, 'runs': rows, 'mean_class_iou': average,
            'mean_iou': round(float(np.mean(average)), 4),
            'metric_note': 'Historical IoU excludes contributions for classes absent from each ground-truth image. Global IoU includes all pixels.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['wta', 'wts'], required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    classes = 3 if args.dataset == 'wta' else 2
    checkpoints = args.checkpoint or sorted((REPO / 'weights' / args.dataset).glob('*_dict.pth'))
    if not checkpoints:
        parser.error('No checkpoint was supplied or found')
    result = evaluate(args.data.resolve(), checkpoints, classes, torch.device(args.device))
    result['dataset'] = args.dataset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'dataset': args.dataset, 'mean_class_iou': result['mean_class_iou'], 'mean_iou': result['mean_iou']}))


if __name__ == '__main__':
    main()
