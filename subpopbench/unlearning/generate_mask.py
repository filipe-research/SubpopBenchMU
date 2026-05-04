"""Generate SalUn saliency mask for machine unlearning.

Port of reference/Unlearn-Saliency/Classification/generate_mask.py
adapted to SubpopBench infrastructure.

The core logic is preserved exactly:
  1. Accumulate gradients of NEGATIVE CE loss over forget set
  2. Take absolute value of accumulated gradients
  3. Select top-alpha fraction globally via argsort ranking

Usage:
    python -m subpopbench.unlearning.generate_mask \
        --model_path ./output/.../model.pkl \
        --dataset CMNIST \
        --forget_mode random --forget_ratio 0.1 --forget_seed 0 \
        --alpha 0.5 --batch_size 108 \
        --data_dir /path/to/datasets \
        --output_path ./output/masks/cmnist_random_10pct_alpha0.5.pt
"""
import argparse
import os

import torch
import torch.nn as nn

from subpopbench.dataset import datasets
from subpopbench.hparams_registry import default_hparams
from subpopbench.learning import algorithms
from subpopbench.unlearning.forget_split import build_forget_indices


def compute_saliency_mask(model, forget_loader, device, alpha=0.5):
    """Compute gradient-based saliency mask over forget set.

    Exactly mirrors SalUn's save_gradient_ratio():
    - Accumulate gradients of -CrossEntropy (gradient ascent direction)
    - Global top-alpha selection by |gradient| magnitude
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()

    # Initialize accumulator (same as original: gradients[name] = 0)
    gradients = {}
    for name, param in model.named_parameters():
        gradients[name] = 0

    # Accumulate gradients over forget set
    for batch in forget_loader:
        _, x, y, _ = batch  # SubpopBench: (i, x, y, a)
        x, y = x.to(device), y.to(device)

        output = model.predict(x)
        loss = -criterion(output, y)  # Negative loss (SalUn convention)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    # Absolute value of accumulated gradients
    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    # Global threshold via argsort ranking (exact SalUn logic)
    # Negate so argsort ascending = largest magnitude first
    all_elements = -torch.cat([t.flatten() for t in gradients.values()])
    threshold_index = int(len(all_elements) * alpha)

    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)

    # Build per-parameter binary mask
    mask = {}
    start_index = 0
    for name, grad_tensor in gradients.items():
        num_elements = grad_tensor.numel()
        tensor_ranks = ranks[start_index:start_index + num_elements]

        threshold_tensor = torch.zeros_like(tensor_ranks)
        threshold_tensor[tensor_ranks < threshold_index] = 1
        threshold_tensor = threshold_tensor.reshape(grad_tensor.shape)
        mask[name] = threshold_tensor
        start_index += num_elements

    return mask


def main():
    parser = argparse.ArgumentParser(description='Generate SalUn saliency mask')
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
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=108)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_path', type=str, required=True)
    # CMNIST params
    parser.add_argument('--cmnist_label_prob', type=float, default=0.5)
    parser.add_argument('--cmnist_attr_prob', type=float, default=0.5)
    parser.add_argument('--cmnist_spur_prob', type=float, default=0.2)
    parser.add_argument('--cmnist_flip_prob', type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. Load checkpoint ---
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    hparams = checkpoint['model_hparams']
    input_shape = checkpoint['model_input_shape']
    num_labels = checkpoint['num_labels']
    num_attributes = checkpoint['num_attributes']

    # Validate/resolve image_arch
    ckpt_arch = hparams.get('image_arch')
    if args.image_arch is not None:
        if ckpt_arch is not None and args.image_arch != ckpt_arch:
            raise ValueError(
                f"image_arch mismatch: --image_arch={args.image_arch} vs "
                f"checkpoint={ckpt_arch}. "
                "Use the same arch as training, or omit --image_arch."
            )
        hparams['image_arch'] = args.image_arch
    # If neither arg nor checkpoint has it, fail early
    if 'image_arch' not in hparams:
        raise ValueError("image_arch not found in checkpoint hparams. Pass --image_arch explicitly.")

    # --- 2. Build train dataset (needed for algorithm constructor + forget split) ---
    if args.dataset == 'CMNIST':
        hparams.update({
            'cmnist_label_prob': args.cmnist_label_prob,
            'cmnist_attr_prob': args.cmnist_attr_prob,
            'cmnist_spur_prob': args.cmnist_spur_prob,
            'cmnist_flip_prob': args.cmnist_flip_prob,
        })

    train_dataset = vars(datasets)[args.dataset](
        args.data_dir, 'tr', hparams, train_attr=args.train_attr)

    # --- 3. Reconstruct algorithm (positional args matching train.py) ---
    algorithm = algorithms.get_algorithm_class('ERM')(
        'images', input_shape, num_labels, num_attributes,
        len(train_dataset), hparams, grp_sizes=train_dataset.group_sizes
    )
    algorithm.load_state_dict(checkpoint['model_dict'])
    algorithm.to(device)

    # --- 4. Build forget set ---
    forget_idx, retain_idx, meta = build_forget_indices(
        train_dataset, mode=args.forget_mode, forget_ratio=args.forget_ratio,
        target_group=args.target_group, target_class=args.target_class,
        seed=args.forget_seed,
    )

    forget_subset = torch.utils.data.Subset(train_dataset, forget_idx.tolist())
    forget_loader = torch.utils.data.DataLoader(
        forget_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"=== Generating SalUn saliency mask ===")
    print(f"  Model:        {args.model_path}")
    print(f"  Arch:         {hparams['image_arch']}")
    print(f"  Dataset:      {args.dataset} (n={len(train_dataset)})")
    print(f"  Forget mode:  {args.forget_mode}")
    print(f"  Forget size:  {meta['n_forget']}")
    print(f"  Alpha:        {args.alpha}")
    print(f"  Device:       {device}")

    # --- 5. Compute mask ---
    mask = compute_saliency_mask(algorithm, forget_loader, device, alpha=args.alpha)

    # --- 6. Save ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save({
        'mask': mask,
        'alpha': args.alpha,
        'forget_metadata': meta,
        'config': vars(args),
    }, args.output_path)

    # --- 7. Print statistics ---
    total_params = 0
    selected_params = 0
    print(f"\n  Per-layer mask coverage:")
    for name, m in mask.items():
        n = m.numel()
        s = m.sum().item()
        total_params += n
        selected_params += s
        pct = 100 * s / n if n > 0 else 0
        print(f"    {name:50s} {s:>8}/{n:<8} ({pct:5.1f}%)")

    actual_sparsity = 100 * selected_params / total_params
    print(f"\n  Total selected: {selected_params}/{total_params} ({actual_sparsity:.2f}%)")
    print(f"  Target alpha:   {args.alpha * 100:.1f}%")
    print(f"  Saved to:       {args.output_path}")


if __name__ == '__main__':
    main()
