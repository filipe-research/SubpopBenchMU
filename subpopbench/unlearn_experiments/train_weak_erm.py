"""Train a weak ERM for AWM-FBC by wrapping subpopbench.train with reduced steps.

The output checkpoint is byte-compatible with --erm_checkpoint / --weak_checkpoint
in run_unlearn.py (same dict layout: model_dict, model_hparams, model_input_shape,
num_labels, num_attributes).

Default policy: reduce training to N_STEPS // 4 of the dataset's full schedule.
Override via --steps. The architecture defaults to 'resnet18_sup_in1k' for fairness
with a ResNet-18 main ERM; pass --image_arch to match a different main ERM arch
(e.g. resnet_sup_in1k for ResNet-50).

Usage:
    python -m subpopbench.unlearn_experiments.train_weak_erm \\
        --dataset Waterbirds \\
        --data_dir /path/to/datasets \\
        --output_folder_name weak_erm \\
        --store_name run0 \\
        [--steps 750] \\
        [--image_arch resnet18_sup_in1k]

The resulting checkpoint will live at:
    <output_dir>/<output_folder_name>_attr<Yes|No>/<store_name>/model.best.pkl
(same path layout produced by subpopbench.train).
"""
import argparse
import subprocess
import sys

from subpopbench.dataset import datasets


def main():
    parser = argparse.ArgumentParser(
        description='Train a weak ERM via subpopbench.train wrapper.'
    )
    parser.add_argument('--dataset', type=str, required=True, choices=datasets.DATASETS)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--output_folder_name', type=str, default='weak_erm')
    parser.add_argument('--store_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image_arch', type=str, default='resnet18_sup_in1k',
                        help='Default: resnet18_sup_in1k. Override to match the main ERM '
                             'arch if it differs (e.g. resnet_sup_in1k for ResNet-50).')
    parser.add_argument('--train_attr', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--steps', type=int, default=None,
                        help='Total training steps. Default: dataset.N_STEPS // 4.')
    # Anything else (e.g. --cmnist_*, --hparams) gets forwarded to subpopbench.train.
    args, extra = parser.parse_known_args()

    if args.steps is None:
        ds_class = vars(datasets)[args.dataset]
        n_steps_full = getattr(ds_class, 'N_STEPS', 5001)
        args.steps = n_steps_full // 4

    cmd = [
        sys.executable, '-m', 'subpopbench.train',
        '--algorithm', 'ERM',
        '--dataset', args.dataset,
        '--data_dir', args.data_dir,
        '--output_dir', args.output_dir,
        '--output_folder_name', args.output_folder_name,
        '--store_name', args.store_name,
        '--seed', str(args.seed),
        '--image_arch', args.image_arch,
        '--train_attr', args.train_attr,
        '--steps', str(args.steps),
    ] + extra

    suffix = '_attrYes' if args.train_attr == 'yes' else '_attrNo'
    expected_path = f"{args.output_dir}/{args.output_folder_name}{suffix}/{args.store_name}/model.best.pkl"

    print(f"=== Train Weak ERM ===")
    print(f"  Dataset:       {args.dataset}")
    print(f"  Steps:         {args.steps}  (default: N_STEPS // 4)")
    print(f"  Arch:          {args.image_arch}")
    print(f"  train_attr:    {args.train_attr}")
    print(f"  Seed:          {args.seed}")
    print(f"  Expected ckpt: {expected_path}")
    print(f"  Forwarded extras: {extra}")
    print(f"  Command:       {' '.join(cmd)}")
    print()

    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
