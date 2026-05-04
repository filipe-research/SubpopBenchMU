import numpy as np


def build_forget_indices(dataset, mode, forget_ratio, target_group=None,
                         target_class=None, seed=0):
    """Build forget/retain index split for machine unlearning.

    Args:
        dataset: SubpopDataset instance (split 'tr')
        mode: 'random' | 'group_targeted' | 'class_targeted' | 'group_balanced'
        forget_ratio: fraction of samples to forget. Interpretation depends on mode:
            - 'random': fraction of the TOTAL dataset
            - 'group_targeted': fraction of the TARGET GROUP
            - 'class_targeted': fraction of the TARGET CLASS
            - 'group_balanced': fraction of EACH GROUP (applied independently)
        target_group: int, group id (num_attributes * y + a) for 'group_targeted'
        target_class: int, class label for 'class_targeted'
        seed: random seed for reproducibility

    Returns:
        (forget_idx, retain_idx, metadata) where idx arrays are positions
        into the dataset (0..len(dataset)-1).

    Raises:
        ValueError: if forget_ratio is not in (0, 1] or mode is unknown.
    """
    if not (0 < forget_ratio <= 1.0):
        raise ValueError(f"forget_ratio must be in (0, 1], got {forget_ratio}")

    rng = np.random.default_rng(seed)
    n = len(dataset)
    all_idx = np.arange(n)

    y = np.array([dataset.y[dataset.idx[i]] for i in range(n)])
    a = np.array([dataset.a[dataset.idx[i]] for i in range(n)])
    groups = dataset.num_attributes * y + a

    if mode == 'random':
        n_forget = int(n * forget_ratio)
        forget_pos = rng.choice(all_idx, size=n_forget, replace=False)

    elif mode == 'group_targeted':
        assert target_group is not None, "target_group required for mode='group_targeted'"
        group_pos = np.where(groups == target_group)[0]
        n_forget = int(forget_ratio * len(group_pos))
        forget_pos = rng.choice(group_pos, size=n_forget, replace=False)

    elif mode == 'class_targeted':
        assert target_class is not None, "target_class required for mode='class_targeted'"
        class_pos = np.where(y == target_class)[0]
        n_forget = int(forget_ratio * len(class_pos))
        forget_pos = rng.choice(class_pos, size=n_forget, replace=False)

    elif mode == 'group_balanced':
        unique_groups = np.unique(groups)
        parts = []
        for g in unique_groups:
            g_pos = np.where(groups == g)[0]
            take = int(forget_ratio * len(g_pos))
            parts.append(rng.choice(g_pos, size=take, replace=False))
        forget_pos = np.concatenate(parts)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    forget_pos = np.sort(forget_pos)
    retain_pos = np.setdiff1d(all_idx, forget_pos)

    metadata = {
        'mode': mode,
        'forget_ratio': forget_ratio,
        'target_group': target_group,
        'target_class': target_class,
        'seed': seed,
        'n_total': n,
        'n_forget': len(forget_pos),
        'n_retain': len(retain_pos),
        'forget_group_counts': {int(g): int((groups[forget_pos] == g).sum())
                                for g in np.unique(groups)},
        'retain_group_counts': {int(g): int((groups[retain_pos] == g).sum())
                                for g in np.unique(groups)},
    }

    return forget_pos, retain_pos, metadata


if __name__ == '__main__':
    from subpopbench.dataset import datasets
    from subpopbench.hparams_registry import default_hparams

    hparams = default_hparams('ERM', 'CMNIST')
    hparams.update({'cmnist_label_prob': 0.5, 'cmnist_attr_prob': 0.5,
                    'cmnist_spur_prob': 0.2, 'cmnist_flip_prob': 0.25})

    ds = vars(datasets)['CMNIST']('./data', 'tr', hparams)

    # Test 1: random mode
    forget_idx, retain_idx, meta = build_forget_indices(ds, mode='random',
                                                        forget_ratio=0.1, seed=0)
    print("=== build_forget_indices test (CMNIST, random, 10%) ===")
    print(f"  Total:  {meta['n_total']}")
    print(f"  Forget: {meta['n_forget']}")
    print(f"  Retain: {meta['n_retain']}")
    print(f"  Forget per group: {meta['forget_group_counts']}")
    print(f"  Retain per group: {meta['retain_group_counts']}")

    # Test 2: group_targeted, target_group=0, forget_ratio=0.5
    # Expected: 50% of group 0 samples (12228 * 0.5 = 6114)
    forget_idx, retain_idx, meta = build_forget_indices(ds, mode='group_targeted',
                                                        forget_ratio=0.5,
                                                        target_group=0, seed=0)
    print("\n=== group_targeted, target_group=0, forget_ratio=0.5 ===")
    print(f"  Total:  {meta['n_total']}")
    print(f"  Forget: {meta['n_forget']}  (expected ~6114 = 50% of group 0)")
    print(f"  Retain: {meta['n_retain']}")
    print(f"  Forget per group: {meta['forget_group_counts']}")

    # Test 3: group_targeted, target_group=2, forget_ratio=0.3
    # Expected: 30% of group 2 samples (3002 * 0.3 = ~900)
    forget_idx, retain_idx, meta = build_forget_indices(ds, mode='group_targeted',
                                                        forget_ratio=0.3,
                                                        target_group=2, seed=0)
    print("\n=== group_targeted, target_group=2, forget_ratio=0.3 ===")
    print(f"  Total:  {meta['n_total']}")
    print(f"  Forget: {meta['n_forget']}  (expected ~900 = 30% of group 2)")
    print(f"  Retain: {meta['n_retain']}")
    print(f"  Forget per group: {meta['forget_group_counts']}")
