"""Shared MIA scoring and metrics (higher score = more likely member).
Hàm điểm MIA và metric dùng chung (điểm cao = nhiều khả năng là member).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve


def attack_scores_cls(logits: torch.Tensor, labels) -> dict:
    """Loss / Confidence / Entropy scores for classification / Điểm Loss/Conf/Entropy cho cls."""
    logp = F.log_softmax(logits.float(), -1)
    p = logp.exp()
    lab = torch.as_tensor(labels)[:, None]
    ce = -logp.gather(1, lab).squeeze(1)
    return {
        'loss': (-ce).numpy(),
        'conf': p.max(1).values.numpy(),
        'entropy': (p * logp).sum(1).numpy(),
    }


def tpr_at(fpr, tpr, t: float) -> float:
    m = fpr <= t
    return float(tpr[m].max()) if m.any() else 0.0


def mia_metrics(sm, sn):
    fpr, tpr, _ = roc_curve(np.r_[np.ones(len(sm)), np.zeros(len(sn))], np.r_[sm, sn])
    return tpr_at(fpr, tpr, 1e-3), tpr_at(fpr, tpr, 1e-2), float(sk_auc(fpr, tpr))


def bootstrap_ci(sm, sn, n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    res = np.array([
        mia_metrics(sm[rng.integers(0, len(sm), len(sm))], sn[rng.integers(0, len(sn), len(sn))])[:2]
        for _ in range(n)
    ])
    lo, hi = np.percentile(res, [2.5, 97.5], axis=0)
    return [float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]


def save_result_json(path: str, payload: dict) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f)
