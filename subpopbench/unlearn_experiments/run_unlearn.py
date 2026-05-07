"""Generalized SalUn-as-debiasing runner: any forget regime + CUPID metrics.

Dispatches to the six forget regimes in forget_regimes.py, runs the existing
SalUn pipeline (subpopbench.unlearning.{generate_mask, salun_unlearn}) without
re-implementing it, and writes a regime-tagged JSON of CUPID-style metrics.

Output filename:
  - regime == 'class':  class{target_class}_seed{seed}.json
  - other regimes:      {regime}_ratio{int(ratio*100)}_seed{seed}.json

Usage:
    python -m subpopbench.unlearn_experiments.run_unlearn \\
        --dataset Waterbirds \\
        --erm_checkpoint <path>/model.best.pkl \\
        --output_dir <out>/ \\
        --regime bias_conflicting \\
        --forget_ratio 0.1 \\
        --seed 0 \\
        --data_dir <path>
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
from subpopbench.unlearning.neggrad_unlearn import run_neggrad
from subpopbench.unlearning.salun_unlearn import RandomLabelDataset, run_salun
from subpopbench.unlearn_experiments.forget_regimes import (
    awm_fbc_forget,
    bias_aligned_forget,
    bias_conflicting_forget,
    class_forget,
    efbc_forget,
    fbc_band_forget,
    fbc_forget,
    group_uniform_forget,
    random_forget,
    ts_fbc_forget,
)
from subpopbench.unlearn_experiments.metrics import full_eval


REGIME_CHOICES = [
    'random', 'group_uniform', 'bias_aligned', 'bias_conflicting',
    'class', 'fbc', 'fbc_band', 'ts_fbc', 'awm_fbc', 'efbc',
]


def _build_forget_set(regime, train_dataset, algorithm, args, device, weak_model=None):
    """Dispatch to the correct forget-regime function. Returns (forget_idx, retain_idx)."""
    if regime == 'random':
        return random_forget(train_dataset, args.forget_ratio, seed=args.seed)
    if regime == 'group_uniform':
        return group_uniform_forget(train_dataset, args.forget_ratio, seed=args.seed)
    if regime == 'bias_aligned':
        return bias_aligned_forget(train_dataset, args.forget_ratio, seed=args.seed)
    if regime == 'bias_conflicting':
        return bias_conflicting_forget(train_dataset, args.forget_ratio, seed=args.seed)
    if regime == 'class':
        return class_forget(train_dataset, args.target_class, seed=args.seed)
    if regime == 'fbc':
        return fbc_forget(
            train_dataset, model=algorithm, ratio=args.forget_ratio,
            classwise=args.fbc_classwise,
            filter_correct=args.fbc_filter_correct,
            device=device, seed=args.seed,
        )
    if regime == 'fbc_band':
        return fbc_band_forget(
            train_dataset, model=algorithm, ratio=args.forget_ratio,
            low_pct=args.fbc_band_low, high_pct=args.fbc_band_high,
            classwise=args.fbc_classwise,
            filter_correct=args.fbc_filter_correct,
            device=device, seed=args.seed,
        )
    if regime == 'ts_fbc':
        return ts_fbc_forget(
            train_dataset, model=algorithm, ratio=args.forget_ratio,
            pool_multiplier=args.pool_multiplier,
            classwise=args.fbc_classwise,
            filter_correct=args.fbc_filter_correct,
            device=device, seed=args.seed,
        )
    if regime == 'awm_fbc':
        if weak_model is None:
            raise ValueError("awm_fbc regime requires weak_model to be loaded.")
        return awm_fbc_forget(
            train_dataset, weak_model=weak_model, ratio=args.forget_ratio,
            pool_multiplier=args.pool_multiplier,
            classwise=args.fbc_classwise,
            device=device, seed=args.seed,
        )
    if regime == 'efbc':
        return efbc_forget(
            train_dataset, model=algorithm, ratio=args.forget_ratio,
            classwise=args.fbc_classwise,
            device=device, seed=args.seed,
        )
    raise ValueError(f"Unknown regime: {regime}")


def _output_filename(args):
    method_prefix = f"{args.method}_" if args.method != 'salun' else ""

    if args.regime == 'class':
        return f"{method_prefix}class{args.target_class}_seed{args.seed}.json"

    suffix = ''
    if args.regime == 'ts_fbc':
        suffix = f"_M{args.pool_multiplier}"
    elif args.regime == 'fbc_band':
        suffix = f"_b{args.fbc_band_low}-{args.fbc_band_high}"
    elif args.regime == 'fbc':
        if not args.fbc_classwise or not args.fbc_filter_correct:
            suffix = f"_cw{args.fbc_classwise}_fc{args.fbc_filter_correct}"

    pct = int(round(args.forget_ratio * 100))
    return f"{method_prefix}{args.regime}{suffix}_ratio{pct}_seed{args.seed}.json"


def main():
    parser = argparse.ArgumentParser(
        description='Run SalUn unlearning under a chosen forget regime, then CUPID-style eval.'
    )
    # required core
    parser.add_argument('--dataset', type=str, default='Waterbirds',
                        choices=['Waterbirds', 'CMNIST'])
    parser.add_argument('--erm_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--method', type=str, default='salun',
                        choices=['salun', 'neggrad'],
                        help="Unlearning method.")
    parser.add_argument('--regime', type=str, required=True, choices=REGIME_CHOICES)
    parser.add_argument('--forget_ratio', type=float, default=0.1,
                        help='Fraction in (0, 1]. Used by all regimes except class.')
    parser.add_argument('--target_class', type=int, default=None,
                        help='Required iff --regime=class.')
    parser.add_argument('--weak_checkpoint', type=str, default=None,
                        help='Required iff --regime=awm_fbc. Path to a weak ERM model.pkl.')
    parser.add_argument('--seed', type=int, default=0)
    # data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--train_attr', type=str, default='yes', choices=['yes', 'no'],
                        help="'yes' so attributes are available for CUPID metrics on train.")
    parser.add_argument('--image_arch', type=str, default=None,
                        help='Override image_arch (default: read from checkpoint)')
    # SalUn hparams (defaults match subpopbench.unlearning.salun_unlearn)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='SalUn saliency mask threshold (ignored when --method=neggrad).')
    parser.add_argument('--unlearn_lr', type=float, default=0.01)
    parser.add_argument('--unlearn_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=108)
    # NegGrad+ hparams
    parser.add_argument('--neggrad_lambda', type=float, default=0.5,
                        help='NegGrad+ ascent weight on the forget term (ignored when --method=salun).')
    # FBC / FBC-band hparams (used when regime in {fbc, fbc_band})
    parser.add_argument('--fbc_classwise', action=argparse.BooleanOptionalAction,
                        default=True, help='Per-class budget split for FBC selection.')
    parser.add_argument('--fbc_filter_correct', action=argparse.BooleanOptionalAction,
                        default=True, help='Restrict FBC pool to correctly-classified samples.')
    parser.add_argument('--fbc_band_low', type=float, default=0.5,
                        help='Lower confidence quantile for fbc_band (in [0, 1]).')
    parser.add_argument('--fbc_band_high', type=float, default=0.8,
                        help='Upper confidence quantile for fbc_band (in [0, 1]).')
    parser.add_argument('--pool_multiplier', type=float, default=3.0,
                        help='ts_fbc: pool size = K * pool_multiplier; M=1 -> fbc, M=N/K -> random.')
    # CMNIST params (only used if dataset == CMNIST)
    parser.add_argument('--cmnist_label_prob', type=float, default=0.5)
    parser.add_argument('--cmnist_attr_prob', type=float, default=0.5)
    parser.add_argument('--cmnist_spur_prob', type=float, default=0.2)
    parser.add_argument('--cmnist_flip_prob', type=float, default=0.25)
    args = parser.parse_args()

    # --- Argument validation ---
    if args.regime == 'class' and args.target_class is None:
        parser.error("--target_class is required when --regime=class")
    if args.regime != 'class' and not (0.0 < args.forget_ratio <= 1.0):
        parser.error(f"--forget_ratio must be in (0, 1], got {args.forget_ratio}")
    if args.regime == 'awm_fbc' and args.weak_checkpoint is None:
        parser.error("--weak_checkpoint is required when --regime=awm_fbc")

    # --- Determinism ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load checkpoint ---
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

    # --- 2. Build datasets ---
    train_dataset = vars(datasets)[args.dataset](
        args.data_dir, 'tr', hparams, train_attr=args.train_attr)
    test_dataset = vars(datasets)[args.dataset](args.data_dir, 'te', hparams)

    # --- 3. Reconstruct algorithm + load weights ---
    algorithm = algorithms.get_algorithm_class('ERM')(
        'images', input_shape, num_labels, num_attributes,
        len(train_dataset), hparams, grp_sizes=train_dataset.group_sizes,
    )
    algorithm.load_state_dict(ckpt['model_dict'])
    algorithm.to(device)

    # --- 3b. (awm_fbc only) Load auxiliary weak ERM ---
    weak_model = None
    if args.regime == 'awm_fbc':
        weak_ckpt = torch.load(args.weak_checkpoint, map_location='cpu', weights_only=False)
        weak_hparams = weak_ckpt['model_hparams']
        weak_input_shape = weak_ckpt['model_input_shape']
        weak_num_labels = weak_ckpt['num_labels']
        weak_num_attributes = weak_ckpt['num_attributes']
        if 'image_arch' not in weak_hparams:
            raise ValueError("image_arch missing in weak_checkpoint hparams.")
        weak_model = algorithms.get_algorithm_class('ERM')(
            'images', weak_input_shape, weak_num_labels, weak_num_attributes,
            len(train_dataset), weak_hparams, grp_sizes=train_dataset.group_sizes,
        )
        weak_model.load_state_dict(weak_ckpt['model_dict'])
        weak_model.to(device)
        print(f"  Weak ERM:        {args.weak_checkpoint}  "
              f"(arch={weak_hparams['image_arch']})")

    # --- 4. Build forget / retain via the chosen regime ---
    forget_idx, retain_idx = _build_forget_set(
        args.regime, train_dataset, algorithm, args, device, weak_model=weak_model
    )
    if len(forget_idx) == 0:
        raise ValueError(f"Regime {args.regime} produced an empty forget set.")
    if len(retain_idx) == 0:
        raise ValueError(f"Regime {args.regime} produced an empty retain set.")

    # --- 5. Header (regime + sizes printed BEFORE mask, per spec) ---
    print(f"=== Run Unlearn ({args.dataset}, method={args.method}, regime={args.regime}) ===")
    print(f"  ERM checkpoint:  {args.erm_checkpoint}")
    print(f"  Arch:            {hparams['image_arch']}")
    print(f"  Device:          {device}")
    print(f"  Seed:            {args.seed}")
    print(f"  Train total:     {len(train_dataset)}")
    print(f"  Test total:      {len(test_dataset)}")
    print(f"  Method:          {args.method}")
    print(f"  Regime:          {args.regime}")
    if args.regime == 'class':
        print(f"  Target class:    {args.target_class}  (forget_ratio ignored)")
    else:
        print(f"  Forget ratio:    {args.forget_ratio}")
    print(f"  Forget set:      {len(forget_idx)}")
    print(f"  Retain set:      {len(retain_idx)}")

    # --- 5b. Truncation warning for bias_conflicting ---
    if args.regime == 'bias_conflicting':
        target = int(len(train_dataset) * args.forget_ratio)
        if len(forget_idx) < target:
            print(f"  WARNING: requested forget size {target}, but only {len(forget_idx)} "
                  f"bias-conflicting samples exist. Using all of them.")

    # --- 6. Subsets shared by both methods ---
    forget_subset = Subset(train_dataset, forget_idx.tolist())
    retain_subset = Subset(train_dataset, retain_idx.tolist())

    # --- 7. Method-specific unlearning ---
    if args.method == 'salun':
        forget_loader = DataLoader(
            forget_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        print(f"\n  Computing saliency mask (alpha={args.alpha})...")
        mask = compute_saliency_mask(algorithm, forget_loader, device, alpha=args.alpha)
        forget_random = RandomLabelDataset(forget_subset, num_labels, seed=args.seed)
        run_salun(
            algorithm, mask, forget_random, retain_subset,
            lr=args.unlearn_lr,
            epochs=args.unlearn_epochs,
            batch_size=args.batch_size,
            device=device,
            verbose=True,
        )
    elif args.method == 'neggrad':
        print(f"\n  Running NegGrad+ (lambda={args.neggrad_lambda})...")
        run_neggrad(
            algorithm, forget_subset, retain_subset,
            lr=args.unlearn_lr,
            epochs=args.unlearn_epochs,
            batch_size=args.batch_size,
            neggrad_lambda=args.neggrad_lambda,
            device=device,
            verbose=True,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # --- 9. CUPID-style evaluation ---
    results = full_eval(algorithm, train_dataset, test_dataset,
                        forget_idx, retain_idx, device=device)

    # --- 9b. FBC / FBC-band overlap with bias-aligned ground-truth (Waterbirds-only diagnostic) ---
    # Legacy: random-draw bias_aligned vs selected. Kept for backward compatibility.
    if (args.regime in {'fbc', 'fbc_band'}
            and args.dataset == 'Waterbirds'
            and args.train_attr == 'yes'):
        gt_forget, _ = bias_aligned_forget(train_dataset, args.forget_ratio, seed=args.seed)
        overlap_n = len(set(forget_idx.tolist()) & set(gt_forget.tolist()))
        results['fbc_n_selected'] = int(len(forget_idx))
        results['fbc_n_overlap_with_bias_aligned'] = int(overlap_n)
        results['fbc_overlap_ratio'] = float(overlap_n) / len(forget_idx)

    # --- 9c. FBC / FBC-band counts vs full bias-aligned / bias-conflicting pools ---
    # More informative than 9b: independent of any random ground-truth draw.
    if args.regime in ('fbc', 'fbc_band', 'ts_fbc', 'awm_fbc', 'efbc') and args.train_attr == 'yes':
        from subpopbench.unlearn_experiments.forget_regimes import _collect_groups
        y_all, a_all = _collect_groups(train_dataset)
        ba_pool = set(np.where(y_all == a_all)[0].tolist())
        bc_pool = set(np.where(y_all != a_all)[0].tolist())

        selected = set(forget_idx.tolist())
        n_ba = len(selected & ba_pool)
        n_bc = len(selected & bc_pool)

        results['fbc_n_selected'] = int(len(selected))
        results['fbc_n_bias_aligned'] = int(n_ba)
        results['fbc_n_bias_conflicting'] = int(n_bc)
        results['fbc_pct_bias_aligned'] = float(n_ba) / len(selected)

    # --- 10. Save JSON ---
    out_path = os.path.join(args.output_dir, _output_filename(args))
    with open(out_path, 'w') as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2, default=str)

    # --- 11. Summary table ---
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
