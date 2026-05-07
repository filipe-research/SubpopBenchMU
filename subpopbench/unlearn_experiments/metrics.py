"""CUPID-style evaluation metrics for unlearning-as-debiasing.

Reports:
  - avg_acc:             overall test accuracy
  - wga:                 worst-group accuracy on test
  - per_group_acc:       per-group accuracies on test (group = y * num_attr + a)
  - fa:                  forget-set accuracy (lower = more forgetting)
  - ra:                  retain-set accuracy (higher = preserved knowledge)
  - fa_bias_aligned:     forget acc on samples where y == a
  - fa_bias_conflicting: forget acc on samples where y != a
  - delta_gap:           |fa_bias_aligned - fa_bias_conflicting|

Adapted to SubpopBench: items are (i, x, y, a) and the algorithm exposes
.predict(x) rather than __call__.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def _predict(model, dataset, device='cuda', batch_size=256, num_workers=4):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    preds, ys, attrs = [], [], []
    for batch in loader:
        # SubpopBench: batch is (i, x, y, a)
        x = batch[1].to(device, non_blocking=True)
        y = batch[2]
        a = batch[3]
        logits = model.predict(x)
        preds.append(logits.argmax(1).cpu())
        ys.append(y)
        attrs.append(a)
    return (
        torch.cat(preds).numpy(),
        torch.cat(ys).numpy(),
        torch.cat(attrs).numpy(),
    )


def evaluate_test(model, test_dataset, device='cuda'):
    pred, y, a = _predict(model, test_dataset, device)
    correct = (pred == y).astype(float)
    avg_acc = float(correct.mean())
    n_attrs = int(a.max()) + 1
    g = y * n_attrs + a
    per_group = {int(gid): float(correct[g == gid].mean()) for gid in np.unique(g)}
    wga = float(min(per_group.values()))
    return {"avg_acc": avg_acc, "wga": wga, "per_group_acc": per_group}


def evaluate_forget_retain(model, train_dataset, forget_idx, retain_idx, device='cuda'):
    forget_set = Subset(train_dataset, forget_idx.tolist())
    retain_set = Subset(train_dataset, retain_idx.tolist())
    pred_f, y_f, a_f = _predict(model, forget_set, device)
    pred_r, y_r, _ = _predict(model, retain_set, device)
    fa = float((pred_f == y_f).mean())
    ra = float((pred_r == y_r).mean())
    aligned_mask = (y_f == a_f)
    fa_ba = float((pred_f[aligned_mask] == y_f[aligned_mask]).mean()) if aligned_mask.any() else None
    fa_bc = float((pred_f[~aligned_mask] == y_f[~aligned_mask]).mean()) if (~aligned_mask).any() else None
    delta_gap = float(abs(fa_ba - fa_bc)) if (fa_ba is not None and fa_bc is not None) else None
    return {
        "fa": fa,
        "ra": ra,
        "fa_bias_aligned": fa_ba,
        "fa_bias_conflicting": fa_bc,
        "delta_gap": delta_gap,
    }


def full_eval(model, train_dataset, test_dataset, forget_idx, retain_idx, device='cuda'):
    out = evaluate_test(model, test_dataset, device)
    out.update(evaluate_forget_retain(model, train_dataset, forget_idx, retain_idx, device))
    return out
