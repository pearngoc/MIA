#!/usr/bin/env python3
"""Train K shadow models + LiRA/RMIA for iTransformer forecasting (TV3).
Huấn luyện K shadow + LiRA/RMIA cho iTransformer forecasting (TV3).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from itransformer.mia import bootstrap_ci, mia_metrics
from itransformer.model import ITransformer
from itransformer.ptbxl import PTBXLDataset, load_split_meta


def control_randomness(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_atomic(obj, path: str) -> None:
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ptbxl-dir', required=True)
    p.add_argument('--split-path', required=True)
    p.add_argument('--shadow-mask', required=True)
    p.add_argument('--cache-dir', default='/workspace/cache/fcst')
    p.add_argument('--ckpt-dir', default='/workspace/ckpt/itransformer_fcst')
    p.add_argument('--results-dir', default='/workspace/results/itransformer/fcst')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--max-lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--target-epoch', type=int, default=50)
    p.add_argument('--lookback', type=int, default=512)
    p.add_argument('--horizon', type=int, default=96)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--e-layers', type=int, default=3)
    p.add_argument('--d-ff', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--rmia-gamma', type=float, default=2.0)
    p.add_argument('--rmia-a', type=float, default=0.3)
    p.add_argument('--ckpt-every', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    seq_len = args.lookback + args.horizon
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    split, meta = load_split_meta(args.split_path)
    member_ds = PTBXLDataset(args.ptbxl_dir, args.split_path, 'train', seq_len, args.cache_dir)
    nonmem_ds = PTBXLDataset(args.ptbxl_dir, args.split_path, 'test', seq_len, args.cache_dir)
    pool_ds = PTBXLDataset(args.ptbxl_dir, args.split_path, 'pool', seq_len, args.cache_dir)
    n_m = len(member_ds)
    n_v = len(split['val'])
    N = len(pool_ds)
    pool_ids = pool_ds.ids
    print('pool', N, 'member', n_m, 'val', n_v, 'nonmem', len(nonmem_ds))

    tag = f'itx_fcst_record_s{args.seed}'
    tgt_path = f'{args.ckpt_dir}/{tag}_preds_ep{args.target_epoch}.pt'
    assert os.path.exists(tgt_path), f'missing target preds {tgt_path}'
    tgt = torch.load(tgt_path, map_location='cpu', weights_only=False)

    target_key = (
        f"itx|fcst|record|{args.seed}|{args.epochs}|{args.batch}|{args.max_lr}|"
        f"{args.d_model}|L{args.lookback}|H{args.horizon}|{meta['sha256'][:12]}"
    )
    mk = np.load(args.shadow_mask)
    order = np.argsort(mk['ids'])
    assert len(mk['ids']) == N and (mk['ids'][order] == pool_ids).all()
    masks_pool = mk['masks'][:args.k][:, order]
    ref_pool = np.flatnonzero(np.isin(pool_ids, mk['ref_ids']))
    print('K', args.k, 'ref', len(ref_pool))

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    amp_ctx = lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=bf16)

    def split_xy(x):
        x = x.to(dev).float()
        return x[..., :args.lookback], x[..., args.lookback:]

    def build_model(name):
        model = ITransformer(
            'forecasting', args.lookback, pred_len=args.horizon, n_variates=12,
            d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers,
            d_ff=args.d_ff, dropout=args.dropout,
        ).to(dev)
        print(f'[{name}] {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
        return model

    @torch.no_grad()
    def get_mse(model, ds, bs=64):
        model.eval()
        out = []
        for x, _ in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0):
            x_in, y_tgt = split_xy(x)
            with amp_ctx():
                pred = model(x_in)
            out.append(((pred - y_tgt) ** 2).mean(dim=(1, 2)).cpu())
        return torch.cat(out)

    def train_shadow(j: int) -> None:
        shadow_tag = f'shadow_fcst_record_s{args.seed}_k{j}'
        out_path = f'{args.ckpt_dir}/{shadow_tag}_mse.pt'
        ckpt_path = f'{args.ckpt_dir}/{shadow_tag}.pt'
        key = f"{target_key}|shadow{j}|{meta['shadow']['sha256'][:12]}"
        if os.path.exists(out_path):
            prev = torch.load(out_path, map_location='cpu', weights_only=False)
            if prev.get('key') == key:
                print(f'[shadow {j}] skip'); return

        control_randomness(args.seed + 1 + j)
        model = build_model(f'shadow {j}')
        idx = np.flatnonzero(masks_pool[j])
        loader = DataLoader(
            Subset(pool_ds, idx), batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0,
        )
        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=args.max_lr / 25)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.max_lr, total_steps=args.epochs * len(loader),
        )
        mse = nn.MSELoss()
        start = 1
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if ck.get('key') == key:
                model.load_state_dict(ck['model'])
                opt.load_state_dict(ck['opt'])
                sched.load_state_dict(ck['sched'])
                start = ck['epoch'] + 1
            del ck

        for ep in range(start, args.epochs + 1):
            model.train()
            losses = []
            for x, _ in tqdm(loader, desc=f'shadow{j} ep{ep}', leave=False):
                opt.zero_grad(set_to_none=True)
                x_in, y_tgt = split_xy(x)
                with amp_ctx():
                    pred = model(x_in)
                loss = mse(pred.float(), y_tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                losses.append(loss.item())
            print(f'[shadow {j}] ep{ep} loss {np.mean(losses):.4f}')
            if ep % args.ckpt_every == 0 and ep < args.epochs:
                save_atomic({
                    'key': key, 'epoch': ep,
                    'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'model': {k: v.cpu() for k, v in model.state_dict().items()},
                }, ckpt_path)

        mse_all = get_mse(model, pool_ds)
        save_atomic({'key': key, 'mse': mse_all}, out_path)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        del model, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for j in range(args.k):
        t0 = time.time()
        train_shadow(j)
        print(f'shadow {j} done in {(time.time() - t0) / 60:.1f} min')

    EPS = 1e-8
    S_mse = np.stack([
        torch.load(f'{args.ckpt_dir}/shadow_fcst_record_s{args.seed}_k{j}_mse.pt', weights_only=False)['mse'].numpy()
        for j in range(args.k)
    ]).astype(np.float64)

    def rec_mse(preds, ds):
        tgt = np.asarray(ds.timeseries[..., args.lookback:]).astype(np.float64)
        return ((preds.numpy().astype(np.float64) - tgt) ** 2).mean(axis=(1, 2))

    T_mse = np.full(N, np.nan)
    idx_m, idx_n = np.arange(n_m), np.arange(n_m + n_v, N)
    T_mse[idx_m] = rec_mse(tgt['pm'], member_ds)
    T_mse[idx_n] = rec_mse(tgt['pn'], nonmem_ds)
    idx_t = np.r_[idx_m, idx_n]
    Min = masks_pool.astype(float)
    Mout = 1 - Min
    PHI_S, PHI_T = -np.log(S_mse + EPS), -np.log(T_mse + EPS)

    def group_stats(W, X):
        c = W.sum(0)
        mu = (W * X).sum(0) / c
        return mu, (W * (X - mu) ** 2).sum(0) / (c - 1)

    def lira(online: bool):
        mu_in, v_in = group_stats(Min, PHI_S)
        mu_out, v_out = group_stats(Mout, PHI_S)
        s_in, s_out = np.sqrt(v_in[idx_t].mean()), np.sqrt(v_out[idx_t].mean())
        x = PHI_T[idx_t]
        if online:
            return norm.logpdf(x, mu_in[idx_t], s_in) - norm.logpdf(x, mu_out[idx_t], s_out)
        return (x - mu_out[idx_t]) / s_out

    TAU0 = float(np.median(S_mse[Mout.astype(bool)]))

    def rmia(online: bool, tau_mult=1.0):
        tau = TAU0 * tau_mult
        P_S, P_T = np.exp(-S_mse / tau), np.exp(-T_mse / tau)
        p_in = (Min * P_S).sum(0) / Min.sum(0)
        p_out = (Mout * P_S).sum(0) / Mout.sum(0)
        p_bar = 0.5 * (p_in + p_out) if online else 0.5 * ((1 + args.rmia_a) * p_out + (1 - args.rmia_a))
        ratio = P_T / p_bar
        z = ratio[np.intersect1d(ref_pool, idx_t)]
        return np.searchsorted(np.sort(z), ratio[idx_t] / args.rmia_gamma, side='right') / len(z)

    scores = {
        'lira': lira(True), 'lira_offline': lira(False),
        'rmia': rmia(False), 'rmia_online': rmia(True),
    }
    mse_m, mse_n = float(T_mse[idx_m].mean()), float(T_mse[idx_n].mean())
    method = {
        'lira': 'LiRA online on -log(MSE), global variance',
        'lira_offline': 'LiRA offline z-score on -log(MSE)',
        'rmia': f'RMIA offline a={args.rmia_a} gamma={args.rmia_gamma} p=exp(-MSE/tau)',
        'rmia_online': f'RMIA online gamma={args.rmia_gamma}',
    }
    for a, s in scores.items():
        sm, sn = s[:n_m], s[n_m:]
        t01, t1, au = mia_metrics(sm, sn)
        c01, c1 = bootstrap_ci(sm, sn, args.n_boot, args.seed)
        out = {
            'model': 'itransformer', 'task': 'fcst', 'attack': a, 'owner_tv': 3,
            'level': 'record', 'dataset': 'ptbxl', 'seed': args.seed,
            'n_member': int(len(sm)), 'n_nonmember': int(len(sn)),
            'scores_member': np.round(sm, 6).tolist(),
            'scores_nonmember': np.round(sn, 6).tolist(),
            'tpr_at_0.1fpr': t01, 'tpr_at_1fpr': t1, 'auc': au,
            'bootstrap_ci': {
                'n_resamples': args.n_boot, 'confidence': 0.95,
                'tpr_at_0.1fpr': c01, 'tpr_at_1fpr': c1,
            },
            'utility': {'train_mse': mse_m, 'test_mse': mse_n, 'gap': mse_n - mse_m},
            'n_shadow': args.k,
            'extra': {
                'method': method[a], 'epochs': args.epochs, 'eval_epoch': args.target_epoch,
                'max_lr': args.max_lr, 'batch': args.batch, 'backbone': 'iTransformer',
                'lookback': args.lookback, 'horizon': args.horizon, 'tau': TAU0,
                'gamma': args.rmia_gamma, 'a': args.rmia_a,
                'split_sha256': meta['sha256'],
                'shadow_masks_sha256': meta['shadow']['sha256'],
                'node': os.environ.get('NODE_NAME', ''),
            },
        }
        fn = f'{args.results_dir}/{a}.json'
        json.dump(out, open(fn, 'w'))
        print('saved', fn, {'tpr@0.1': t01, 'tpr@1': t1, 'auc': au})


if __name__ == '__main__':
    main()
