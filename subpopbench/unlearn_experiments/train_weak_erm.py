"""Train a weak ERM for AWM-FBC by wrapping subpopbench.train with reduced steps.

The output checkpoint is byte-compatible with --erm_checkpoint / --weak_checkpoint
in run_unlearn.py (same dict layout: model_dict, model_hparams, model_input_shape,
num_labels, num_attributes).

Default policy: reduce training to N_STEPS // 4 of the dataset's full schedule.
Override via --steps. The architecture defaults to 'resnet18_sup_in1k' for fairness
with a ResNet-18 main ERM; pass --image_arch to match a different main ERM arch.

subpopbench.train auto-derives the inner store folder as
    "{dataset}_{algorithm}_hparams{hparams_seed}_seed{seed}"
(plus a CMNIST hparam prefix for CMNIST), and appends "_attrYes" / "_attrNo" to
output_folder_name. So we don't pass --store_name; we just locate the checkpoint
after the run.

Usage:
    python -m subpopbench.unlearn_experiments.train_weak_erm \\
        --dataset Waterbirds \\
        --data_dir /path/to/datasets \\
        [--output_folder_name weak_waterbirds] \\
        [--steps 1250] \\
        [--image_arch resnet18_sup_in1k]
"""
import argparse
import subprocess
import sys
from pathlib import Path

from subpopbench.dataset import datasets


def _locate_checkpoint(output_dir, output_folder_name, train_attr):
    """Find the weak ERM checkpoint under {output_dir}/{output_folder_name}*.

    subpopbench.train appends _attrYes / _attrNo to output_folder_name and creates
    an auto-named subfolder. Prefer model.best.pkl; fall back to model.pkl.
    Returns Path or None.
    """
    root = Path(output_dir)
    # Most precise match first (with attr suffix), then fall back to any prefix match.
    suffix = '_attrYes' if train_attr == 'yes' else '_attrNo'
    candidates = (
        list(root.glob(f"{output_folder_name}{suffix}"))
        or list(root.glob(f"{output_folder_name}*"))
    )

    for filename in ('model.best.pkl', 'model.pkl'):
        for parent in candidates:
            matches = sorted(parent.rglob(filename))
            if matches:
                return matches[0], candidates
    return None, candidates


def main():
    parser = argparse.ArgumentParser(
        description='Train a weak ERM via subpopbench.train wrapper.'
    )
    parser.add_argument('--dataset', type=str, required=True, choices=datasets.DATASETS)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--output_folder_name', type=str, default=None,
                        help="Default: weak_{dataset_lower}.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hparams_seed', type=int, default=0)
    parser.add_argument('--image_arch', type=str, default='resnet18_sup_in1k',
                        help='Default: resnet18_sup_in1k. Override to match the main ERM '
                             'arch if it differs (e.g. resnet_sup_in1k for ResNet-50).')
    parser.add_argument('--train_attr', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--steps', type=int, default=None,
                        help='Total training steps. Default: dataset.N_STEPS // 4.')
    # Anything else (e.g. --cmnist_*, --hparams) is forwarded to subpopbench.train.
    args, extra = parser.parse_known_args()

    if args.output_folder_name is None:
        args.output_folder_name = f"weak_{args.dataset.lower()}"

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
        '--seed', str(args.seed),
        '--hparams_seed', str(args.hparams_seed),
        '--image_arch', args.image_arch,
        '--train_attr', args.train_attr,
        '--steps', str(args.steps),
    ] + extra

    print(f"=== Train Weak ERM ===")
    print(f"  Dataset:           {args.dataset}")
    print(f"  Steps:             {args.steps}  (default: N_STEPS // 4)")
    print(f"  Arch:              {args.image_arch}")
    print(f"  train_attr:        {args.train_attr}")
    print(f"  Seed:              {args.seed}")
    print(f"  Output folder:     {args.output_folder_name} (subpopbench will append "
          f"_attr{'Yes' if args.train_attr == 'yes' else 'No'} + auto store_name)")
    print(f"  Forwarded extras:  {extra}")
    print(f"  Command:           {' '.join(cmd)}")
    print()

    subprocess.run(cmd, check=True)

    # Locate the produced checkpoint
    ckpt, searched = _locate_checkpoint(args.output_dir, args.output_folder_name, args.train_attr)
    print()
    if ckpt is not None:
        print(f"  Weak ERM checkpoint located:")
        print(f"    {ckpt}")
        print(f"  Use as:")
        print(f"    --weak_checkpoint {ckpt}")
    else:
        print(f"  ERROR: no model.best.pkl or model.pkl found.")
        print(f"  Searched under:")
        if searched:
            for d in searched:
                print(f"    {d}")
        else:
            print(f"    (no folder matching '{args.output_folder_name}*' "
                  f"under {args.output_dir})")
        sys.exit(1)


if __name__ == '__main__':
    main()
