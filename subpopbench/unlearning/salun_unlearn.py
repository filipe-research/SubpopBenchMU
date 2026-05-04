"""SalUn unlearning via Random Labeling (RL) with saliency-masked gradients.

Port of reference/Unlearn-Saliency/Classification/unlearn/RL.py
adapted to SubpopBench infrastructure.

The script:
  1. Loads a trained ERM model + precomputed saliency mask
  2. Constructs forget/retain splits
  3. Fine-tunes with random labels on the forget set while masking
     gradients to only update salient weights
  4. Evaluates using SubpopBench's standard metrics

Usage:
    python -m subpopbench.unlearning.salun_unlearn \
        --model_path ./output/.../model.pkl \
        --mask_path ./output/masks/cmnist_random_10pct_alpha0.5.pt \
        --dataset CMNIST \
        --train_attr no \
        --forget_mode random --forget_ratio 0.1 --forget_seed 0 \
        --unlearn_lr 0.013 --unlearn_epochs 10 --batch_size 108 \
        --data_dir /path/to/datasets \
        --output_dir ./output/cmnist_salun_random_10pct \
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
# Random label wrapper
# ---------------------------------------------------------------------------
class RandomLabelDataset(torch.utils.data.Dataset):
    """Wraps a dataset replacing labels with fixed random ones."""

    def __init__(self, dataset, num_labels, seed):
        self.dataset = dataset
        self.num_labels = num_labels
        gen = torch.Generator().manual_seed(seed)
        self.random_labels = torch.randint(
            0, num_labels, (len(dataset),), generator=gen
        )

    def __getitem__(self, index):
        i, x, _y, a = self.dataset[index]
        return i, x, self.random_labels[index], a

    def __len__(self):
        return len(self.dataset)


# ---------------------------------------------------------------------------
# Mask helpers (from reference RL.py)
# ---------------------------------------------------------------------------
def _apply_mask_to_grads(model, mask):
    """Zero out gradients on non-salient (masked-out) weights."""
    for name, param in model.named_parameters():
        if name in mask and param.grad is not None:
            param.grad *= mask[name].to(param.grad.device, dtype=param.grad.dtype)


def _restore_masked_params(model, mask, theta0, optimizer):
    """Restore non-salient weights to their pre-unlearning values (theta0)
    and clear momentum buffers on those coordinates.

    Needed because weight_decay modifies weights even when grad == 0.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in mask:
                continue
            m = mask[name].to(device=param.device, dtype=param.dtype)
            inv_m = 1 - m
            if torch.count_nonzero(inv_m) == 0:
                continue
            param.data.mul_(m).add_(theta0[name].to(param.device) * inv_m)
            state = optimizer.state.get(param, None)
            if state is not None and "momentum_buffer" in state:
                state["momentum_buffer"].mul_(m)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _make_eval_loader(subset, num_labels, batch_size, num_workers=4):
    """Build a DataLoader from a Subset, patching num_labels for eval_metrics."""
    subset.num_labels = num_labels
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


def _eval_all(algorithm, loaders, device):
    """Evaluate on all splits, return dict of results."""
    results = {}
    for name, loader in loaders.items():
        results[name] = eval_metrics(algorithm, loader, device)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='SalUn unlearning (Random Labeling)')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--mask_path', type=str, required=True)
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

    # Validate/resolve image_arch (same logic as generate_mask.py)
    ckpt_arch = hparams.get('image_arch')
    if args.image_arch is not None:
        if ckpt_arch is not None and args.image_arch != ckpt_arch:
            raise ValueError(
                f"image_arch mismatch: --image_arch={args.image_arch} vs "
                f"checkpoint={ckpt_arch}. "
                "Use the same arch as training, or omit --image_arch."
            )
        hparams['image_arch'] = args.image_arch
    if 'image_arch' not in hparams:
        raise ValueError("image_arch not found in checkpoint hparams. Pass --image_arch explicitly.")

    # --- 2. Load mask ---
    mask_data = torch.load(args.mask_path, map_location='cpu', weights_only=False)
    mask = mask_data['mask']
    mask_meta = mask_data['forget_metadata']

    # Validate mask metadata matches current args
    for key in ['mode', 'seed']:
        if mask_meta[key] != getattr(args, f'forget_{key}'):
            raise ValueError(
                f"Mask metadata mismatch on '{key}': mask has {mask_meta[key]}, "
                f"but args specify {getattr(args, f'forget_{key}')}. "
                "Ensure --forget_mode/--forget_seed match the mask."
            )
    if abs(mask_meta['forget_ratio'] - args.forget_ratio) > 1e-9:
        raise ValueError(
            f"Mask metadata mismatch on 'forget_ratio': mask has {mask_meta['forget_ratio']}, "
            f"but args specify {args.forget_ratio}."
        )

    # --- 3. Build datasets ---
    if args.dataset == 'CMNIST':
        hparams.update({
            'cmnist_label_prob': args.cmnist_label_prob,
            'cmnist_attr_prob': args.cmnist_attr_prob,
            'cmnist_spur_prob': args.cmnist_spur_prob,
            'cmnist_flip_prob': args.cmnist_flip_prob,
        })

    train_dataset = vars(datasets)[args.dataset](
        args.data_dir, 'tr', hparams, train_attr=args.train_attr)
    val_dataset = vars(datasets)[args.dataset](
        args.data_dir, 'va', hparams)
    test_dataset = vars(datasets)[args.dataset](
        args.data_dir, 'te', hparams)

    # --- 4. Reconstruct algorithm + load weights ---
    algorithm = algorithms.get_algorithm_class('ERM')(
        'images', input_shape, num_labels, num_attributes,
        len(train_dataset), hparams, grp_sizes=train_dataset.group_sizes
    )
    algorithm.load_state_dict(checkpoint['model_dict'])
    algorithm.to(device)

    # --- 5. Build forget/retain splits ---
    forget_idx, retain_idx, meta = build_forget_indices(
        train_dataset, mode=args.forget_mode, forget_ratio=args.forget_ratio,
        target_group=args.target_group, target_class=args.target_class,
        seed=args.forget_seed,
    )

    forget_subset = torch.utils.data.Subset(train_dataset, forget_idx.tolist())
    retain_subset = torch.utils.data.Subset(train_dataset, retain_idx.tolist())

    # --- 6. Training data: random labels on forget + real labels on retain ---
    forget_random = RandomLabelDataset(forget_subset, num_labels, seed=args.seed)
    concat_dataset = torch.utils.data.ConcatDataset([forget_random, retain_subset])
    train_loader = torch.utils.data.DataLoader(
        concat_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )

    # --- 7. Eval loaders (with num_labels patched) ---
    forget_eval_subset = torch.utils.data.Subset(train_dataset, forget_idx.tolist())
    retain_eval_subset = torch.utils.data.Subset(train_dataset, retain_idx.tolist())
    eval_loaders = {
        'forget': _make_eval_loader(forget_eval_subset, num_labels, args.batch_size),
        'retain': _make_eval_loader(retain_eval_subset, num_labels, args.batch_size),
        'val':    _make_eval_loader(val_dataset, num_labels, args.batch_size),
        'test':   _make_eval_loader(test_dataset, num_labels, args.batch_size),
    }

    # --- 8. Mask coverage log ---
    mask_param_count = 0
    mask_selected_count = 0
    model_param_names = {n for n, _ in algorithm.named_parameters()}
    mask_layer_names = set(mask.keys())
    free_layers = model_param_names - mask_layer_names

    for name, m in mask.items():
        mask_param_count += m.numel()
        mask_selected_count += m.sum().item()

    print(f"=== SalUn Unlearning (Random Labeling) ===")
    print(f"  Model:          {args.model_path}")
    print(f"  Mask:           {args.mask_path}")
    print(f"  Arch:           {hparams['image_arch']}")
    print(f"  Dataset:        {args.dataset} (train={len(train_dataset)}, "
          f"val={len(val_dataset)}, test={len(test_dataset)})")
    print(f"  Forget mode:    {args.forget_mode}")
    print(f"  Forget/retain:  {meta['n_forget']}/{meta['n_retain']}")
    print(f"  Mask alpha:     {mask_data.get('alpha', 'N/A')}")
    print(f"  Mask coverage:  {int(mask_selected_count)}/{mask_param_count} "
          f"({100 * mask_selected_count / mask_param_count:.1f}%) salient params")
    print(f"  Layers in mask: {len(mask_layer_names)}, free: {len(free_layers)}")
    print(f"  Unlearn LR:     {args.unlearn_lr}")
    print(f"  Unlearn epochs: {args.unlearn_epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Device:         {device}")

    # --- 9. Capture theta0 (pre-unlearning weights for masked params) ---
    with torch.no_grad():
        theta0 = {
            name: param.detach().clone()
            for name, param in algorithm.named_parameters()
            if name in mask
        }

    # --- 10. Create SGD optimizer ---
    optimizer = torch.optim.SGD(
        algorithm.parameters(),
        lr=args.unlearn_lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    criterion = nn.CrossEntropyLoss()

    # --- 11. Unlearning loop ---
    all_results = []

    for epoch in range(args.unlearn_epochs):
        epoch_start = time.time()
        algorithm.train()

        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            _, x, y, a = batch
            x, y = x.to(device), y.to(device, dtype=torch.long)

            output = algorithm.predict(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()

            # Gradient masking: zero gradients on non-salient weights
            _apply_mask_to_grads(algorithm, mask)

            optimizer.step()

            # Restore non-salient weights to theta0
            _restore_masked_params(algorithm, mask, theta0, optimizer)

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        # Evaluate
        epoch_results = _eval_all(algorithm, eval_loaders, device)

        # Extract summary accuracies
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

    # --- 12. Save outputs ---
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
