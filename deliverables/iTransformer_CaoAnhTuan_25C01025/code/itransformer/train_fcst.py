#!/usr/bin/env python3
"""Train iTransformer forecasting on PTB-XL + Loss MIA.
Huấn luyện iTransformer forecasting trên PTB-XL + MIA Loss.
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
from torch.utils.data import DataLoader
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
    p.add_argument('--cache-dir', default='/workspace/cache/fcst')
    p.add_argument('--ckpt-dir', default='/workspace/ckpt/itransformer_fcst')
    p.add_argument('--results-dir', default='/workspace/results/itransformer/fcst')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--max-lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lookback', type=int, default=512)
    p.add_argument('--horizon', type=int, default=96)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--e-layers', type=int, default=3)
    p.add_argument('--d-ff', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--n-boot', type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    seq_len = args.lookback + args.horizon
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    control_randomness(args.seed)
    split, meta = load_split_meta(args.split_path)
    print('split', {k: len(v) for k, v in split.items()}, 'sha256', meta['sha256'][:12])

    member_ds = PTBXLDataset(args.ptbxl_dir, args.split_path, 'train', seq_len, args.cache_dir)
    val_ds = PTBXLDataset(args.ptbxl_dir, args.split_path, 'val', seq_len, args.cache_dir)
    nonmem_ds = PTBXLDataset(args.ptbxl_dir, args.split_path, 'test', seq_len, args.cache_dir)
    print('sizes', len(member_ds), len(val_ds), len(nonmem_ds), member_ds.timeseries.shape)

    eval_at = sorted({max(1, int(np.ceil(args.epochs * f))) for f in (0.1, 0.2, 0.5, 1.0)})
    print('EVAL_AT', eval_at)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    amp_ctx = lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=bf16)
    print('device', dev, 'bf16', bf16, 'gpu', torch.cuda.get_device_name(0) if dev == 'cuda' else None)

    tag = f'itx_fcst_record_s{args.seed}'
    ckpt_path = f'{args.ckpt_dir}/{tag}.pt'
    done_path = f'{args.ckpt_dir}/{tag}.done.json'
    key = (
        f"itx|fcst|record|{args.seed}|{args.epochs}|{args.batch}|{args.max_lr}|"
        f"{args.d_model}|L{args.lookback}|H{args.horizon}|{meta['sha256'][:12]}"
    )
    if os.path.exists(done_path) and json.load(open(done_path)).get('key') == key:
        print('already finished, skip'); return

    model = ITransformer(
        task='forecasting', seq_len=args.lookback, pred_len=args.horizon, n_variates=12,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers,
        d_ff=args.d_ff, dropout=args.dropout,
    ).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    def split_xy(x):
        x = x.to(dev).float()
        return x[..., :args.lookback], x[..., args.lookback:]

    @torch.no_grad()
    def get_preds(ds, bs=64):
        model.eval()
        out = []
        for x, _ in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0):
            x_in, _ = split_xy(x)
            with amp_ctx():
                out.append(model(x_in).float().cpu())
        return torch.cat(out)

    def targets_of(ds):
        return torch.as_tensor(np.asarray(ds.timeseries[..., args.lookback:])).float()

    def per_sample_mse(preds, ds):
        return ((preds - targets_of(ds)) ** 2).mean(dim=(1, 2)).numpy()

    loader = DataLoader(member_ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.max_lr / 25)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.max_lr, total_steps=args.epochs * len(loader))
    mse_loss = nn.MSELoss()
    history, start = [], 1

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if ck.get('key') == key:
            model.load_state_dict(ck['model'])
            opt.load_state_dict(ck['opt'])
            sched.load_state_dict(ck['sched'])
            history, start = ck['history'], ck['epoch'] + 1
            print(f'resumed after epoch {ck["epoch"]}')
        del ck

    t0 = time.time()
    for ep in range(start, args.epochs + 1):
        model.train()
        losses = []
        for x, _ in tqdm(loader, desc=f'ep{ep}/{args.epochs}', leave=False):
            opt.zero_grad(set_to_none=True)
            x_in, y_tgt = split_xy(x)
            with amp_ctx():
                pred = model(x_in)
            loss = mse_loss(pred.float(), y_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        if ep in eval_at:
            pm, pn, pv = get_preds(member_ds), get_preds(nonmem_ds), get_preds(val_ds)
            save_atomic({'pm': pm, 'pn': pn}, f'{args.ckpt_dir}/{tag}_preds_ep{ep}.pt')
            mse_m = float(per_sample_mse(pm, member_ds).mean())
            mse_n = float(per_sample_mse(pn, nonmem_ds).mean())
            mse_v = float(per_sample_mse(pv, val_ds).mean())
            sm = -per_sample_mse(pm, member_ds)
            sn = -per_sample_mse(pn, nonmem_ds)
            t01, t1, au = mia_metrics(sm, sn)
            row = {
                'epoch': ep, 'train_loss': float(np.mean(losses)),
                'train_mse': mse_m, 'val_mse': mse_v, 'test_mse': mse_n,
                'gap': mse_n - mse_m,
                'loss_tpr@0.1': t01, 'loss_tpr@1': t1, 'loss_auc': au,
            }
            history.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

        save_atomic({
            'key': key, 'epoch': ep, 'history': history,
            'opt': opt.state_dict(), 'sched': sched.state_dict(),
            'model': {k: v.cpu() for k, v in model.state_dict().items()},
        }, ckpt_path)

    final_ep = max(e for e in eval_at if e <= args.epochs)
    row = [r for r in history if r['epoch'] == final_ep][-1]
    pr = torch.load(f'{args.ckpt_dir}/{tag}_preds_ep{final_ep}.pt', weights_only=False)
    sm = -per_sample_mse(pr['pm'], member_ds)
    sn = -per_sample_mse(pr['pn'], nonmem_ds)
    c01, c1 = bootstrap_ci(sm, sn, args.n_boot, args.seed)
    out = {
        'model': 'itransformer', 'task': 'fcst', 'attack': 'loss',
        'owner_tv': 3, 'level': 'record', 'dataset': 'ptbxl', 'seed': args.seed,
        'n_member': int(len(sm)), 'n_nonmember': int(len(sn)),
        'scores_member': np.round(sm, 6).tolist(),
        'scores_nonmember': np.round(sn, 6).tolist(),
        'tpr_at_0.1fpr': row['loss_tpr@0.1'], 'tpr_at_1fpr': row['loss_tpr@1'], 'auc': row['loss_auc'],
        'bootstrap_ci': {
            'n_resamples': args.n_boot, 'confidence': 0.95,
            'tpr_at_0.1fpr': c01, 'tpr_at_1fpr': c1,
        },
        'utility': {'train_mse': row['train_mse'], 'test_mse': row['test_mse'], 'gap': row['gap']},
        'n_shadow': None,
        'extra': {
            'eval_epochs': eval_at, 'tuning': 'default recipe; tune on val_mse only',
            'epochs': args.epochs, 'eval_epoch': row['epoch'], 'max_lr': args.max_lr,
            'batch': args.batch, 'backbone': 'iTransformer',
            'lookback': args.lookback, 'horizon': args.horizon,
            'd_model': args.d_model, 'val_mse': row.get('val_mse'),
            'split_sha256': meta['sha256'], 'node': os.environ.get('NODE_NAME', ''),
        },
    }
    fn = f'{args.results_dir}/loss.json'
    json.dump(out, open(fn, 'w'))
    print('saved', fn)
    json.dump({'key': key, 'history': history}, open(done_path, 'w'))
    print(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
