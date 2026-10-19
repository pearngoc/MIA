#!/usr/bin/env python3
"""Temporal Feature MIA for iTransformer classification (Koren-style).
Attack Temporal Feature cho iTransformer classification (kiểu Koren).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from itransformer.mia import bootstrap_ci, mia_metrics
from itransformer.ptbxl import PTBXLDataset, load_split_meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ptbxl-dir', required=True)
    p.add_argument('--split-path', required=True)
    p.add_argument('--cache-dir', default='/workspace/cache/cls')
    p.add_argument('--ckpt-dir', default='/workspace/ckpt/itransformer_cls')
    p.add_argument('--results-dir', default='/workspace/results/itransformer/cls')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--eval-at', default='5,10,25,50')
    p.add_argument('--tag', default='itx_record_s42')
    return p.parse_args()


def per_epoch(logits, labels):
    logp = F.log_softmax(logits.float(), -1)
    lab = torch.as_tensor(labels)[:, None]
    ce = -logp.gather(1, lab).squeeze(1)
    ent = -(logp.exp() * logp).sum(1)
    true = logits.float().gather(1, lab).squeeze(1)
    other = logits.float().scatter(1, lab, float('-inf')).max(1).values
    return ce.numpy(), ent.numpy(), (true - other).numpy()


def slope(v, t):
    t = np.asarray(t, dtype=np.float64)
    t = t - t.mean()
    return (v * t).sum(1) / (t ** 2).sum()


def cross_fit(make_clf, X, y, n_folds, seed):
    score = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y):
        clf = make_clf().fit(X[tr], y[tr])
        score[te] = clf.predict_proba(X[te])[:, 1]
    return score


def main():
    args = parse_args()
    eval_at = [int(x) for x in args.eval_at.split(',')]
    os.makedirs(args.results_dir, exist_ok=True)

    _, meta = load_split_meta(args.split_path)
    member_ds = PTBXLDataset(args.ptbxl_dir, args.split_path, 'train', args.seq_len, args.cache_dir)
    nonmem_ds = PTBXLDataset(args.ptbxl_dir, args.split_path, 'test', args.seq_len, args.cache_dir)
    y_m = np.asarray(member_ds.labels).astype(int)
    y_n = np.asarray(nonmem_ds.labels).astype(int)

    lg = {
        ep: torch.load(f'{args.ckpt_dir}/{args.tag}_logits_ep{ep}.pt', map_location='cpu', weights_only=False)
        for ep in eval_at
    }
    assert all(len(lg[ep]['lm']) == len(y_m) and len(lg[ep]['ln']) == len(y_n) for ep in eval_at)

    def trajectory_features(key, labels):
        ce, ent, mar = (
            np.stack(z, 1)
            for z in zip(*[per_epoch(lg[ep][key], labels) for ep in eval_at])
        )
        dce = np.diff(ce, axis=1)
        dyn = np.stack([slope(ce, eval_at), slope(mar, eval_at)], 1)
        final = np.stack([ce[:, -1], ent[:, -1], mar[:, -1]], 1)
        return np.concatenate([ce, ent, mar, dce, dyn], 1), final, ce[:, -1]

    Xm, Fm, Lm = trajectory_features('lm', y_m)
    Xn, Fn, Ln = trajectory_features('ln', y_n)
    X = np.r_[Xm, Xn]
    Xf = np.r_[Fm, Fn]
    y = np.r_[np.ones(len(Xm)), np.zeros(len(Xn))]
    loss_all = -np.r_[Lm, Ln]
    print('features', X.shape, 'members', len(Xm), 'non-members', len(Xn))

    hgb = lambda: HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, random_state=args.seed,
    )
    makers = {
        'temporal': (hgb, X),
        'temporal_lr': (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X),
        'final_epoch_clf': (hgb, Xf),
        'loss_only_clf': (hgb, loss_all[:, None]),
    }
    S, M = {}, {}
    nm = len(Xm)
    for name, (mk, feats) in makers.items():
        S[name] = cross_fit(mk, feats, y, args.n_folds, args.seed)
        M[name] = mia_metrics(S[name][:nm], S[name][nm:])
    M['loss (raw -CE)'] = mia_metrics(loss_all[:nm], loss_all[nm:])
    print(pd.DataFrame(M, index=['TPR@0.1%', 'TPR@1%', 'AUC']).T.round(4))

    sm, sn = S['temporal'][:nm], S['temporal'][nm:]
    c01, c1 = bootstrap_ci(sm, sn, args.n_boot, args.seed)
    t01, t1, au = M['temporal']

    done_path = f'{args.ckpt_dir}/{args.tag}.done.json'
    hist = {}
    if os.path.exists(done_path):
        done = json.load(open(done_path))
        rows = [r for r in done.get('history', []) if r.get('epoch') == max(eval_at)]
        hist = rows[-1] if rows else {}

    out = {
        'model': 'itransformer', 'task': 'cls', 'attack': 'temporal',
        'owner_tv': 3, 'level': 'record', 'dataset': 'ptbxl', 'seed': args.seed,
        'n_member': int(len(sm)), 'n_nonmember': int(len(sn)),
        'scores_member': np.round(sm, 6).tolist(),
        'scores_nonmember': np.round(sn, 6).tolist(),
        'tpr_at_0.1fpr': t01, 'tpr_at_1fpr': t1, 'auc': au,
        'bootstrap_ci': {
            'n_resamples': args.n_boot, 'confidence': 0.95,
            'tpr_at_0.1fpr': c01, 'tpr_at_1fpr': c1,
        },
        'utility': {k: hist[k] for k in ('train_acc', 'test_acc', 'gap') if k in hist},
        'n_shadow': None,
        'extra': {
            'eval_epochs': eval_at, 'backbone': 'iTransformer',
            'variant': 'training-dynamics',
            'classifier': 'HistGradientBoosting', 'n_folds': args.n_folds,
            'features': 'per-epoch CE/entropy/margin + consecutive CE change + CE and margin slope',
            'threat_model': 'gray-box: intermediate checkpoints + labelled member/non-member subset',
            'ablation': {
                k: {'tpr_at_0.1fpr': v[0], 'tpr_at_1fpr': v[1], 'auc': v[2]} for k, v in M.items()
            },
            'split_sha256': meta['sha256'],
            'node': os.environ.get('NODE_NAME', ''),
        },
    }
    fn = f'{args.results_dir}/temporal.json'
    json.dump(out, open(fn, 'w'))
    print('saved', fn, '| TPR@0.1%', round(t01, 4), c01, '| TPR@1%', round(t1, 4), c1, '| AUC', round(au, 4))


if __name__ == '__main__':
    main()
