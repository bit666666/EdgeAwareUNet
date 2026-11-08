"""Configure the recovered pre-September training loop for WTA or WTS.

This adapter changes paths, configuration, and checkpoint serialization. Model,
loss, optimization, and early-stopping calculations come from the old source.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['wta', 'wts'], required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--selection-split', choices=['val', 'test'], default='val',
                        help='Use test only to inspect the historical selection protocol.')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--seed', type=int, default=2356)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if min(args.runs, args.epochs, args.patience) < 1:
        parser.error('runs, epochs, and patience must be positive')
    data = args.data.resolve()
    for split in ['train', args.selection_split]:
        for folder in ['img', 'label_1D', 'label_3D']:
            if not (data / split / folder).is_dir():
                parser.error('Missing directory: ' + str(data / split / folder))
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    repo = Path(__file__).resolve().parents[1]
    historical = repo / 'historical'
    sys.path.insert(0, str(historical))
    source = historical / 'train_ds.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    constants = {
        'dataset': 'turbine_dataset_1' if args.dataset == 'wta' else 'turbine_dataset_2',
        'lr': 1e-3 if args.dataset == 'wta' else 1e-4,
        'batch_size': 8 if args.dataset == 'wta' else 2,
        'num_epochs': args.epochs, 'max_patience': args.patience,
        'model_path': 'weights', 'output_path': 'logs',
    }

    class Configure(ast.NodeTransformer):
        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript)
                    and isinstance(node.value, ast.Constant) and node.value.value == '2'):
                target = ast.dump(node.targets[0])
                if 'CUDA_VISIBLE_DEVICES' in target:
                    return None
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in constants):
                node.value = ast.Constant(constants[node.targets[0].id])
            return self.generic_visit(node)

        def visit_While(self, node):
            if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == 'n':
                node.test.comparators = [ast.Constant(args.runs)]
            return self.generic_visit(node)

        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name):
                if node.func.id == 'WindTurbineDataset':
                    for keyword in node.keywords:
                        if keyword.arg == 'root_dir':
                            keyword.value = ast.Constant(str(data))
                        if keyword.arg == 'mode' and isinstance(keyword.value, ast.Constant) and keyword.value.value == 'test':
                            keyword.value = ast.Constant(args.selection_split)
                if node.func.id == 'Dsloss':
                    for keyword in node.keywords:
                        if keyword.arg == 'focal':
                            keyword.value = ast.Constant(args.dataset == 'wts')
                if node.func.id == 'get_and_save_random_seeds':
                    node.args = [ast.BinOp(ast.Constant(args.seed - 1), ast.Add(), ast.Name(id='n', ctx=ast.Load()))]
            if (isinstance(node.func, ast.Attribute) and node.func.attr == 'save'
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch'):
                node.args[0] = ast.Call(ast.Attribute(ast.Name(id='model', ctx=ast.Load()), 'state_dict', ast.Load()), [], [])
            return node

    tree = ast.fix_missing_locations(Configure().visit(tree))
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'weights').mkdir()
    (run_dir / 'logs').mkdir()
    config = dict(constants, dataset_name=args.dataset, data=str(data),
                  selection_split=args.selection_split, runs=args.runs, seed=args.seed,
                  historical_source=str(source), restored_training_adapter=True)
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    if args.selection_split == 'test':
        print('Historical protocol: test images are used for early stopping and checkpoint selection.', flush=True)
    os.chdir(run_dir)
    exec(compile(tree, str(source), 'exec'), {'__name__': '__main__', '__file__': str(source)})


if __name__ == '__main__':
    main()
