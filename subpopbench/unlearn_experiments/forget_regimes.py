"""Forget-set construction strategies for machine-unlearning-as-debiasing.

Six regimes:
  - random
  - group_uniform
  - bias_aligned
  - bias_conflicting
  - class
  - fbc (forget-by-confidence)

Adapted to SubpopBench's SubpopDataset: items are (i, x, y, a), and labels/
attributes are exposed via dataset.y / dataset.a / dataset.idx without needing
to materialize images.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _collect_groups(dataset):
    """Return (y_array, a_array) as numpy arrays of length len(dataset).

    SubpopDataset exposes self.y / self.a indexed by raw csv row, with
    self.idx mapping positional index -> raw idx. Reading directly avoids
    a full DataLoader pass that would decode every image just for labels.
    """
    n = len(dataset)
    y = np.array([int(dataset.y[dataset.idx[i]]) for i in range(n)])
    a = np.array([int(dataset.a[dataset.idx[i]]) for i in range(n)])
    return y, a


def random_forget(dataset, ratio, seed=0):
    rng = np.random.default_rng(seed)
    n = len(dataset)
    n_forget = int(n * ratio)
    perm = rng.permutation(n)
    return perm[:n_forget], perm[n_forget:]


def group_uniform_forget(dataset, ratio, seed=0):
    rng = np.random.default_rng(seed)
    y, a = _collect_groups(dataset)
    g = y * (a.max() + 1) + a
    n_forget_total = int(len(dataset) * ratio)
    groups = np.unique(g)
    per_group = n_forget_total // len(groups)
    forget_idx = []
    for gid in groups:
        members = np.where(g == gid)[0]
        take = min(per_group, len(members))
        forget_idx.append(rng.choice(members, size=take, replace=False))
    forget_idx = np.concatenate(forget_idx)
    retain_idx = np.setdiff1d(np.arange(len(dataset)), forget_idx)
    return forget_idx, retain_idx


def bias_aligned_forget(dataset, ratio, seed=0):
    rng = np.random.default_rng(seed)
    y, a = _collect_groups(dataset)
    aligned_idx = np.where(y == a)[0]
    n_forget = int(len(dataset) * ratio)
    if n_forget > len(aligned_idx):
        raise ValueError(
            f"Need {n_forget} bias-aligned samples, only {len(aligned_idx)} available."
        )
    forget_idx = rng.choice(aligned_idx, size=n_forget, replace=False)
    retain_idx = np.setdiff1d(np.arange(len(dataset)), forget_idx)
    return forget_idx, retain_idx


def bias_conflicting_forget(dataset, ratio, seed=0):
    rng = np.random.default_rng(seed)
    y, a = _collect_groups(dataset)
    conflict_idx = np.where(y != a)[0]
    n_forget_target = int(len(dataset) * ratio)
    n_forget = min(n_forget_target, len(conflict_idx))
    forget_idx = rng.choice(conflict_idx, size=n_forget, replace=False)
    retain_idx = np.setdiff1d(np.arange(len(dataset)), forget_idx)
    return forget_idx, retain_idx


def class_forget(dataset, target_class, seed=0):
    y, _ = _collect_groups(dataset)
    forget_idx = np.where(y == target_class)[0]
    retain_idx = np.where(y != target_class)[0]
    return forget_idx, retain_idx


def fbc_forget(dataset, model, ratio, classwise=True, filter_correct=True,
               device='cuda', seed=0, batch_size=256, num_workers=4):
    """Forget-by-confidence: pick the highest-confidence (and optionally only
    correctly-classified) samples per class as the forget set.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    confs, labels, correct = [], [], []
    with torch.no_grad():
        for batch in loader:
            # SubpopBench: batch is (i, x, y, a)
            x = batch[1].to(device, non_blocking=True)
            y = batch[2].to(device, non_blocking=True)
            logits = model.predict(x)
            probs = F.softmax(logits, dim=1)
            c, p = probs.max(dim=1)
            confs.append(c.cpu())
            labels.append(y.cpu())
            correct.append((p == y).cpu())
    confs = torch.cat(confs).numpy()
    labels = torch.cat(labels).numpy()
    correct = torch.cat(correct).numpy()

    n_forget = int(len(dataset) * ratio)
    eligible = correct.astype(bool) if filter_correct else np.ones_like(correct, dtype=bool)
    if classwise:
        n_classes = int(labels.max()) + 1
        per_class = n_forget // n_classes
        forget_idx_list = []
        for c in range(n_classes):
            mask = eligible & (labels == c)
            cand = np.where(mask)[0]
            order = np.argsort(-confs[cand])
            forget_idx_list.append(cand[order[:per_class]])
        forget_idx = np.concatenate(forget_idx_list)
    else:
        cand = np.where(eligible)[0]
        order = np.argsort(-confs[cand])
        forget_idx = cand[order[:n_forget]]
    retain_idx = np.setdiff1d(np.arange(len(dataset)), forget_idx)
    return forget_idx, retain_idx


def fbc_band_forget(dataset, model, ratio, low_pct=0.5, high_pct=0.8,
                    classwise=True, filter_correct=True, device='cuda', seed=0):
    """FBC variant: select samples whose confidence lies in the
    [low_pct, high_pct] quantile band of the eligible pool.

    Quantiles are computed over the eligible pool (per class when classwise=True,
    or globally otherwise). If the band yields more than the budget (per_class
    when classwise, n_forget otherwise), randomly subsample down to the budget.
    If the band yields fewer, take all of it (no replacement).
    """
    if not (0.0 <= low_pct <= high_pct <= 1.0):
        raise ValueError(
            f"Require 0 <= low_pct <= high_pct <= 1, got ({low_pct}, {high_pct})"
        )
    rng = np.random.default_rng(seed)
    model.eval()
    loader = DataLoader(dataset, batch_size=256, num_workers=4, shuffle=False)
    confs, labels, correct = [], [], []
    with torch.no_grad():
        for batch in loader:
            # SubpopBench: batch is (i, x, y, a)
            x = batch[1].to(device, non_blocking=True)
            y = batch[2].to(device, non_blocking=True)
            logits = model.predict(x)
            probs = F.softmax(logits, dim=1)
            c, p = probs.max(dim=1)
            confs.append(c.cpu())
            labels.append(y.cpu())
            correct.append((p == y).cpu())
    confs = torch.cat(confs).numpy()
    labels = torch.cat(labels).numpy()
    correct = torch.cat(correct).numpy()

    n_forget = int(len(dataset) * ratio)
    eligible = correct.astype(bool) if filter_correct else np.ones_like(correct, dtype=bool)

    if classwise:
        n_classes = int(labels.max()) + 1
        per_class = n_forget // n_classes
        forget_idx_list = []
        for c in range(n_classes):
            mask = eligible & (labels == c)
            cand = np.where(mask)[0]
            if len(cand) == 0:
                continue
            confs_cand = confs[cand]
            lo, hi = np.quantile(confs_cand, [low_pct, high_pct])
            in_band = (confs_cand >= lo) & (confs_cand <= hi)
            band = cand[in_band]
            if len(band) > per_class:
                band = rng.choice(band, size=per_class, replace=False)
            forget_idx_list.append(band)
        forget_idx = (np.concatenate(forget_idx_list)
                      if forget_idx_list else np.array([], dtype=int))
    else:
        cand = np.where(eligible)[0]
        if len(cand) == 0:
            forget_idx = np.array([], dtype=int)
        else:
            confs_cand = confs[cand]
            lo, hi = np.quantile(confs_cand, [low_pct, high_pct])
            in_band = (confs_cand >= lo) & (confs_cand <= hi)
            band = cand[in_band]
            if len(band) > n_forget:
                band = rng.choice(band, size=n_forget, replace=False)
            forget_idx = band

    retain_idx = np.setdiff1d(np.arange(len(dataset)), forget_idx)
    return forget_idx, retain_idx


REGIMES = {
    "random": random_forget,
    "group_uniform": group_uniform_forget,
    "bias_aligned": bias_aligned_forget,
    "bias_conflicting": bias_conflicting_forget,
    "class": class_forget,
    "fbc": fbc_forget,
    "fbc_band": fbc_band_forget,
}
