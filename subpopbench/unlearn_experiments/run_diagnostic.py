"""Diagnostic: class-targeted forget + SalUn unlearning + CUPID-style eval.

Loads a trained ERM checkpoint, builds the forget set via
forget_regimes.class_forget(target_class), and runs the *exact* SalUn
pipeline already in this repo (subpopbench.unlearning.{generate_mask,
salun_unlearn}) by importing its primitives — no SalUn logic is
re-implemented here.

Usage:
    python -m subpopbench.unlearn_experiments.run_diagnostic \
        --dataset Waterbirds \
        --erm_checkpoint ./output/<folder>/<store>/model.best.pkl \
        --output_dir ./output/diagnostics/wb_class0 \
        --target_class 0 \
        --seed 0 \
        --data_dir /path/to/datasets
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from subpopbench.dataset import datasets
from subpopbench.learning import algorithms
from subpopbench.unlearning.generate_mask import compute_saliency_mask
from subpopbench.unlearning.salun_unlearn import RandomLabelDataset, run_salun
from subpopbench.unlearn_experiments.forget_regimes import class_forget
from subpopbench.unlearn_experiments.metrics import full_eval


def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic: class-forget unlearning + CUPID-style metrics'
    )
    # required
    parser.add_argument('--dataset', type=str, default='Waterbirds',
                        choices=['Waterbirds', 'CMNIST'])
    parser.add_argument('--erm_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--target_class', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    # data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--train_attr', type=str, default='yes', choices=['yes', 'no'],
                        help="'yes' so attributes are available for CUPID-style metrics on the train set.")
    parser.add_argument('--image_arch', type=str, default=None,
                        help='Override image_arch (default: read from checkpoint)')
    # SalUn hyperparameters (defaults match subpopbench.unlearning.salun_unlearn)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--unlearn_lr', type=float, default=0.01)
    parser.add_argument('--unlearn_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=108)
    # CMNIST params (only used if dataset == CMNIST)
    parser.add_argument('--cmnist_label_prob', type=float, default=0.5)
    parser.add_argument('--cmnist_attr_prob', type=float, default=0.5)
    parser.add_argument('--cmnist_spur_prob', type=float, default=0.2)
    parser.add_argument('--cmnist_flip_prob', type=float, default=0.25)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load checkpoint (same fields as generate_mask.py / salun_unlearn.py) ---
    ckpt = torch.load(args.erm_checkpoint, map_location='cpu', weights_only=False)
    hparams = ckpt['model_hparams']
    input_shape = ckpt['model_input_shape']
    num_labels = ckpt['num_labels']
    num_attributes = ckpt['num_attributes']

    ckpt_arch = hparams.get('image_arch')
    if args.image_arch is not None:
        if ckpt_arch is not None and args.image_arch != ckpt_arch:
            raise ValueError(
                f"image_arch mismatch: --image_arch={args.image_arch} vs checkpoint={ckpt_arch}"
            )
        hparams['image_arch'] = args.image_arch
    if 'image_arch' not in hparams:
        raise ValueError("image_arch missing in checkpoint hparams; pass --image_arch explicitly.")

    if args.dataset == 'CMNIST':
        hparams.update({
            'cmnist_label_prob': args.cmnist_label_prob,
            'cmnist_attr_prob': args.cmnist_attr_prob,
            'cmnist_spur_prob': args.cmnist_spur_prob,
            'cmnist_flip_prob': args.cmnist_flip_prob,
        })

    # --- 2. Build datasets (identical pattern to train.py / salun_unlearn.py) ---
    train_dataset = vars(datasets)[args.dataset](
        args.data_dir, 'tr', hparams, train_attr=args.train_attr)
    test_dataset = vars(datasets)[args.dataset](args.data_dir, 'te', hparams)

    # --- 3. Reconstruct ERM algorithm + load weights ---
    algorithm = algorithms.get_algorithm_class('ERM')(
        'images', input_shape, num_labels, num_attributes,
        len(train_dataset), hparams, grp_sizes=train_dataset.group_sizes,
    )
    algorithm.load_state_dict(ckpt['model_dict'])
    algorithm.to(device)

    # --- 4. Build forget / retain via class_forget ---
    forget_idx, retain_idx = class_forget(train_dataset, args.target_class, seed=args.seed)
    if len(forget_idx) == 0:
        raise ValueError(f"class_forget produced empty forget set for target_class={args.target_class}")
    if len(retain_idx) == 0:
        raise ValueError(f"class_forget produced empty retain set for target_class={args.target_class}")

    print(f"=== Diagnostic ({args.dataset}, class_forget target_class={args.target_class}) ===")
    print(f"  Train total:    {len(train_dataset)}")
    print(f"  Forget:         {len(forget_idx)}")
    print(f"  Retain:         {len(retain_idx)}")
    print(f"  Test total:     {len(test_dataset)}")
    print(f"  Arch:           {hparams['image_arch']}")
    print(f"  Device:         {device}")

    # --- 5. Saliency mask via the existing compute_saliency_mask() ---
    forget_subset = Subset(train_dataset, forget_idx.tolist())
    forget_loader = DataLoader(
        forget_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    print(f"  Computing saliency mask (alpha={args.alpha})...")
    mask = compute_saliency_mask(algorithm, forget_loader, device, alpha=args.alpha)

    # --- 6. Random-label forget + real-label retain ---
    forget_random = RandomLabelDataset(forget_subset, num_labels, seed=args.seed)
    retain_subset = Subset(train_dataset, retain_idx.tolist())

    # --- 7. Run SalUn unlearning via the shared loop ---
    run_salun(
        algorithm, mask, forget_random, retain_subset,
        lr=args.unlearn_lr,
        epochs=args.unlearn_epochs,
        batch_size=args.batch_size,
        device=device,
        verbose=True,
    )

    # --- 8. CUPID-style evaluation ---
    results = full_eval(algorithm, train_dataset, test_dataset,
                        forget_idx, retain_idx, device=device)

    # --- 9. Save JSON ---
    out_path = os.path.join(
        args.output_dir,
        f"diagnostic_class{args.target_class}_seed{args.seed}.json",
    )
    with open(out_path, 'w') as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2, default=str)

    # --- 10. Print summary table ---
    fa_ba = results['fa_bias_aligned']
    fa_bc = results['fa_bias_conflicting']
    dg = results['delta_gap']
    print("\n=== Results ===")
    print(f"  {'metric':<25s} {'value':>10s}")
    print(f"  {'-' * 37}")
    print(f"  {'FA (forget acc)':<25s} {results['fa']:>10.4f}")
    print(f"  {'  fa_bias_aligned':<25s} {(f'{fa_ba:.4f}' if fa_ba is not None else 'n/a'):>10s}")
    print(f"  {'  fa_bias_conflicting':<25s} {(f'{fa_bc:.4f}' if fa_bc is not None else 'n/a'):>10s}")
    print(f"  {'  delta_gap':<25s} {(f'{dg:.4f}' if dg is not None else 'n/a'):>10s}")
    print(f"  {'RA (retain acc)':<25s} {results['ra']:>10.4f}")
    print(f"  {'WGA (worst-group acc)':<25s} {results['wga']:>10.4f}")
    print(f"  {'avg_acc (test)':<25s} {results['avg_acc']:>10.4f}")
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
