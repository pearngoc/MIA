#!/usr/bin/env python3
"""Temporal Feature MIA for iTransformer forecasting (Koren DFT+trend).
Attack Temporal Feature cho iTransformer forecasting.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
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
    p.add_argument('--cache-dir', default='/workspace/cache/fcst')
    p.add_argument('--ckpt-dir', default='/workspace/ckpt/itransformer_fcst')
    p.add_argument('--results-dir', default='/workspace/results/itransformer/fcst')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lookback', type=int, default=512)
    p.add_argument('--horizon', type=int, default=96)
    p.add_argument('--target-epoch', type=int, default=50)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--n-bands', type=int, default=8)
    p.add_argument('--tag', default='itx_fcst_record_s42')
    return p.parse_args()


def slope(v):
    t = np.arange(v.shape[-1], dtype=np.float32)
    t -= t.mean()
    return (v * t).sum(-1) / (t ** 2).sum()


def temporal_features(res, n_bands):
    n, c, h = res.shape
    mag = np.abs(np.fft.rfft(res, axis=-1))[..., 1:]
    bands = np.array_split(np.arange(mag.shape[-1]), n_bands)
    dft = np.stack([np.log1p(mag[..., b].mean(-1)) for b in bands], -1).reshape(n, -1)
    trend = slope(res)
    drift = slope(np.abs(res))
    lmse = np.log((res ** 2).mean(-1) + 1e-8)
    return np.concatenate([dft, trend, drift, lmse], 1)


def cross_fit(make_clf, X, y, n_folds, seed):
    score = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y):
        clf = make_clf().fit(X[tr], y[tr])
        score[te] = clf.predict_proba(X[te])[:, 1]
    return score


def main():
    args = parse_args()
    seq_len = args.lookback + args.horizon
    os.makedirs(args.results_dir, exist_ok=True)
    _, meta = load_split_meta(args.split_path)

    member_ds = PTBXLDataset(args.ptbxl_dir, args.split_path, 'train', seq_len, args.cache_dir)
    nonmem_ds = PTBXLDataset(args.ptbxl_dir, args.split_path, 'test', seq_len, args.cache_dir)
    pr = torch.load(
        f'{args.ckpt_dir}/{args.tag}_preds_ep{args.target_epoch}.pt',
        map_location='cpu', weights_only=False,
    )
    tgt_m = np.asarray(member_ds.timeseries[..., args.lookback:]).astype(np.float32)
    tgt_n = np.asarray(nonmem_ds.timeseries[..., args.lookback:]).astype(np.float32)
    res_m = tgt_m - pr['pm'].numpy()
    res_n = tgt_n - pr['pn'].numpy()

    Xm = temporal_features(res_m, args.n_bands)
    Xn = temporal_features(res_n, args.n_bands)
    X = np.r_[Xm, Xn]
    y = np.r_[np.ones(len(Xm)), np.zeros(len(Xn))]
    loss_all = -np.r_[(res_m ** 2).mean((1, 2)), (res_n ** 2).mean((1, 2))]
    print('features', X.shape)

    makers = {
        'temporal': (
            lambda: HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, random_state=args.seed,
            ),
            X,
        ),
        'temporal_lr': (
            lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            X,
        ),
        'loss_only_clf': (
            lambda: HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, random_state=args.seed,
            ),
            loss_all[:, None],
        ),
    }
    S, M = {}, {}
    nm = len(Xm)
    for name, (mk, feats) in makers.items():
        S[name] = cross_fit(mk, feats, y, args.n_folds, args.seed)
        M[name] = mia_metrics(S[name][:nm], S[name][nm:])
    M['loss (raw -MSE)'] = mia_metrics(loss_all[:nm], loss_all[nm:])
    print(pd.DataFrame(M, index=['TPR@0.1%', 'TPR@1%', 'AUC']).T.round(4))

    sm, sn = S['temporal'][:nm], S['temporal'][nm:]
    c01, c1 = bootstrap_ci(sm, sn, args.n_boot, args.seed)
    t01, t1, au = M['temporal']
    done_path = f'{args.ckpt_dir}/{args.tag}.done.json'
    hist = {}
    if os.path.exists(done_path):
        done = json.load(open(done_path))
        hist = done['history'][-1] if done.get('history') else {}

    out = {
        'model': 'itransformer', 'task': 'fcst', 'attack': 'temporal',
        'owner_tv': 3, 'level': 'record', 'dataset': 'ptbxl', 'seed': args.seed,
        'n_member': int(len(sm)), 'n_nonmember': int(len(sn)),
        'scores_member': np.round(sm, 6).tolist(),
        'scores_nonmember': np.round(sn, 6).tolist(),
        'tpr_at_0.1fpr': t01, 'tpr_at_1fpr': t1, 'auc': au,
        'bootstrap_ci': {
            'n_resamples': args.n_boot, 'confidence': 0.95,
            'tpr_at_0.1fpr': c01, 'tpr_at_1fpr': c1,
        },
        'utility': {k: hist[k] for k in ('train_mse', 'test_mse', 'gap') if k in hist},
        'n_shadow': None,
        'extra': {
            'eval_epoch': args.target_epoch, 'lookback': args.lookback, 'horizon': args.horizon,
            'backbone': 'iTransformer', 'classifier': 'HistGradientBoosting',
            'n_folds': args.n_folds, 'n_bands': args.n_bands,
            'features': 'log DFT band magnitude + residual trend + drift + per-lead log MSE',
            'ablation': {
                k: {'tpr_at_0.1fpr': v[0], 'tpr_at_1fpr': v[1], 'auc': v[2]} for k, v in M.items()
            },
            'split_sha256': meta['sha256'],
            'node': os.environ.get('NODE_NAME', ''),
        },
    }
    fn = f'{args.results_dir}/temporal.json'
    json.dump(out, open(fn, 'w'))
    print('saved', fn, '| TPR@0.1%', round(t01, 4), '| TPR@1%', round(t1, 4), '| AUC', round(au, 4))


if __name__ == '__main__':
    main()
