"""NegGrad+ unlearning: gradient ascent on forget + descent on retain.

Loss formulation per step:
    L = -lambda * CE(forget_batch) + (1 - lambda) * CE(retain_batch)

Simpler than SalUn: no saliency mask, no random labels. Real labels are used
on both sets; the negative coefficient on the forget term implements ascent.

Usage:
    python -m subpopbench.unlearning.neggrad_unlearn \\
        --model_path ./output/.../model.pkl \\
        --dataset CMNIST \\
        --train_attr no \\
        --forget_mode random --forget_ratio 0.1 --forget_seed 0 \\
        --neggrad_lambda 0.5 \\
        --unlearn_lr 0.013 --unlearn_epochs 10 --batch_size 108 \\
        --data_dir /path/to/datasets \\
        --output_dir ./output/cmnist_neggrad_random_10pct \\
        --seed 0
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from subpopbench.dataset import datasets
from subpopbench.learning import algorithms
from subpopbench.unlearning.forget_split import build_forget_indices
from subpopbench.utils.eval_helper import eval_metrics


# ---------------------------------------------------------------------------
# Evaluation helpers (parallel to salun_unlearn._make_eval_loader / _eval_all)
# ---------------------------------------------------------------------------
def _make_eval_loader(subset, num_labels, batch_size, num_workers=4):
    subset.num_labels = num_labels
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


def _eval_all(algorithm, loaders, device):
    results = {}
    for name, loader in loaders.items():
        results[name] = eval_metrics(algorithm, loader, device)
    return results


# ---------------------------------------------------------------------------
# Public unlearning loop
# ---------------------------------------------------------------------------
def run_neggrad(model, forget_set, retain_set, *,
                lr, epochs, batch_size, neggrad_lambda,
                device='cuda', verbose=False, on_epoch_end=None):
    """Apply NegGrad+ unlearning in-place on `model`.

    Each step:
        L = -lambda * CE(forget_batch) + (1 - lambda) * CE(retain_batch)
        loss.backward(); SGD step

    Iteration: drives by retain_loader (typically larger) and cycles forget_loader
    so every retain batch is paired with a fresh forget batch. Per-epoch step
    count == len(retain_loader).

    Args:
        model: SubpopBench algorithm exposing .predict(x)
        forget_set, retain_set: torch Datasets returning (i, x, y, a)
        lr, epochs, batch_size: SGD/loop hyperparameters
        neggrad_lambda: weight on the forget (ascent) term, in [0, 1]
        device: torch device string or object
        verbose: if True, print per-epoch (epoch_time, avg_loss)
        on_epoch_end: optional callable(epoch, avg_loss, epoch_time)

    Returns:
        The same `model` instance, modified in place.
    """
    forget_loader = torch.utils.data.DataLoader(
        forget_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        n_batches = 0

        forget_iter = iter(forget_loader)
        for retain_batch in retain_loader:
            try:
                forget_batch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                forget_batch = next(forget_iter)

            _, x_f, y_f, _ = forget_batch
            _, x_r, y_r, _ = retain_batch
            x_f = x_f.to(device)
            y_f = y_f.to(device, dtype=torch.long)
            x_r = x_r.to(device)
            y_r = y_r.to(device, dtype=torch.long)

            loss_f = criterion(model.predict(x_f), y_f)
            loss_r = criterion(model.predict(x_r), y_r)
            loss = -neggrad_lambda * loss_f + (1.0 - neggrad_lambda) * loss_r

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        if verbose:
            print(f"  Epoch {epoch}/{epochs - 1} "
                  f"({epoch_time:.1f}s) loss={avg_loss:.4f}")
        if on_epoch_end is not None:
            on_epoch_end(epoch, avg_loss, epoch_time)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='NegGrad+ unlearning')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=datasets.DATASETS)
    parser.add_argument('--image_arch', type=str, default=None,
                        help='Override image_arch (default: use checkpoint value)')
    parser.add_argument('--train_attr', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--forget_mode', type=str, required=True,
                        choices=['random', 'group_targeted', 'class_targeted', 'group_balanced'])
    parser.add_argument('--forget_ratio', type=float, required=True)
    parser.add_argument('--forget_seed', type=int, default=0)
    parser.add_argument('--target_group', type=int, default=None)
    parser.add_argument('--target_class', type=int, default=None)
    parser.add_argument('--neggrad_lambda', type=float, default=0.5)
    parser.add_argument('--unlearn_lr', type=float, default=0.01)
    parser.add_argument('--unlearn_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=108)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    # CMNIST params
    parser.add_argument('--cmnist_label_prob', type=float, default=0.5)
    parser.add_argument('--cmnist_attr_prob', type=float, default=0.5)
    parser.add_argument('--cmnist_spur_prob', type=float, default=0.2)
    parser.add_argument('--cmnist_flip_prob', type=float, default=0.25)
    args = parser.parse_args()

    # --- Deterministic setup ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load checkpoint ---
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    hparams = checkpoint['model_hparams']
    input_shape = checkpoint['model_input_shape']
    num_labels = checkpoint['num_labels']
    num_attributes = checkpoint['num_attributes']

    ckpt_arch = hparams.get('image_arch')
    if args.image_arch is not None:
        if ckpt_arch is not None and args.image_arch != ckpt_arch:
            raise ValueError(
                f"image_arch mismatch: --image_arch={args.image_arch} vs "
                f"checkpoint={ckpt_arch}."
            )
        hparams['image_arch'] = args.image_arch
    if 'image_arch' not in hparams:
        raise ValueError("image_arch not found in checkpoint hparams.")

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
    val_dataset = vars(datasets)[args.dataset](args.data_dir, 'va', hparams)
    test_dataset = vars(datasets)[args.dataset](args.data_dir, 'te', hparams)

    # --- 3. Reconstruct algorithm + load weights ---
    algorithm = algorithms.get_algorithm_class('ERM')(
        'images', input_shape, num_labels, num_attributes,
        len(train_dataset), hparams, grp_sizes=train_dataset.group_sizes
    )
    algorithm.load_state_dict(checkpoint['model_dict'])
    algorithm.to(device)

    # --- 4. Build forget/retain splits ---
    forget_idx, retain_idx, meta = build_forget_indices(
        train_dataset, mode=args.forget_mode, forget_ratio=args.forget_ratio,
        target_group=args.target_group, target_class=args.target_class,
        seed=args.forget_seed,
    )

    forget_subset = torch.utils.data.Subset(train_dataset, forget_idx.tolist())
    retain_subset = torch.utils.data.Subset(train_dataset, retain_idx.tolist())

    # --- 5. Eval loaders (with num_labels patched) ---
    forget_eval_subset = torch.utils.data.Subset(train_dataset, forget_idx.tolist())
    retain_eval_subset = torch.utils.data.Subset(train_dataset, retain_idx.tolist())
    eval_loaders = {
        'forget': _make_eval_loader(forget_eval_subset, num_labels, args.batch_size),
        'retain': _make_eval_loader(retain_eval_subset, num_labels, args.batch_size),
        'val':    _make_eval_loader(val_dataset, num_labels, args.batch_size),
        'test':   _make_eval_loader(test_dataset, num_labels, args.batch_size),
    }

    print(f"=== NegGrad+ Unlearning ===")
    print(f"  Model:          {args.model_path}")
    print(f"  Arch:           {hparams['image_arch']}")
    print(f"  Dataset:        {args.dataset} (train={len(train_dataset)}, "
          f"val={len(val_dataset)}, test={len(test_dataset)})")
    print(f"  Forget mode:    {args.forget_mode}")
    print(f"  Forget/retain:  {meta['n_forget']}/{meta['n_retain']}")
    print(f"  Lambda:         {args.neggrad_lambda}")
    print(f"  Unlearn LR:     {args.unlearn_lr}")
    print(f"  Unlearn epochs: {args.unlearn_epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Device:         {device}")

    # --- 6. Per-epoch eval hook ---
    all_results = []

    def _on_epoch_end(epoch, avg_loss, epoch_time):
        epoch_results = _eval_all(algorithm, eval_loaders, device)
        summary = {
            split: epoch_results[split]['overall']['accuracy']
            for split in epoch_results
        }
        epoch_results['epoch'] = epoch
        epoch_results['train_loss'] = avg_loss
        all_results.append(epoch_results)
        print(f"  Epoch {epoch}/{args.unlearn_epochs - 1} "
              f"({epoch_time:.1f}s) loss={avg_loss:.4f} | "
              f"forget_acc={summary['forget']:.4f} "
              f"retain_acc={summary['retain']:.4f} "
              f"val_acc={summary['val']:.4f} "
              f"test_acc={summary['test']:.4f}")

    # --- 7. Run NegGrad+ ---
    run_neggrad(
        algorithm, forget_subset, retain_subset,
        lr=args.unlearn_lr,
        epochs=args.unlearn_epochs,
        batch_size=args.batch_size,
        neggrad_lambda=args.neggrad_lambda,
        device=device,
        on_epoch_end=_on_epoch_end,
    )

    # --- 8. Save outputs ---
    model_path = os.path.join(args.output_dir, 'unlearned_model.pkl')
    torch.save(algorithm.state_dict(), model_path)
    print(f"\n  Saved unlearned model to: {model_path}")

    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved results to:         {results_path}")

    args_path = os.path.join(args.output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"  Saved args to:            {args_path}")


if __name__ == '__main__':
    main()
