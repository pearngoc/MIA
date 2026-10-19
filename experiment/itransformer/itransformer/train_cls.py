#!/usr/bin/env python3
"""Train iTransformer classification on PTB-XL + Loss/Conf/Entropy MIA (TV3).
Huấn luyện iTransformer classification trên PTB-XL + MIA Loss/Conf/Entropy (TV3).
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

from itransformer.mia import attack_scores_cls, bootstrap_ci, mia_metrics
from itransformer.model import ITransformer
from itransformer.ptbxl import PTBXLDataset, load_split_meta


def control_randomness(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_atomic(obj, path: str) -> None:
    # Atomic checkpoint write / Ghi checkpoint atomic
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ptbxl-dir', required=True)
    p.add_argument('--split-path', required=True)
    p.add_argument('--cache-dir', default='/tmp/itx_cache_cls')
    p.add_argument('--ckpt-dir', default='/workspace/ckpt/itransformer_cls')
    p.add_argument('--results-dir', default='/workspace/results/itransformer/cls')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--max-lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--e-layers', type=int, default=3)
    p.add_argument('--d-ff', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--n-boot', type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    control_randomness(args.seed)
    split, meta = load_split_meta(args.split_path)
    print('split', {k: len(v) for k, v in split.items()}, 'sha256', meta['sha256'][:12])

    member_ds = PTBXLDataset(args.ptbxl_dir, args.split_path, 'train', args.seq_len, args.cache_dir)
    val_ds = PTBXLDataset(args.ptbxl_dir, args.split_path, 'val', args.seq_len, args.cache_dir)
    nonmem_ds = PTBXLDataset(args.ptbxl_dir, args.split_path, 'test', args.seq_len, args.cache_dir)
    print('sizes', len(member_ds), len(val_ds), len(nonmem_ds), member_ds.timeseries.shape)

    # Eval epochs at 10/20/50/100% / Epoch đánh giá MIA theo tỉ lệ
    eval_at = sorted({max(1, int(np.ceil(args.epochs * f))) for f in (0.1, 0.2, 0.5, 1.0)})
    print('EVAL_AT', eval_at)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    amp_ctx = lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=bf16)
    print('device', dev, 'bf16', bf16, 'gpu', torch.cuda.get_device_name(0) if dev == 'cuda' else None)

    tag = f'itx_record_s{args.seed}'
    ckpt_path = f'{args.ckpt_dir}/{tag}.pt'
    done_path = f'{args.ckpt_dir}/{tag}.done.json'
    key = (
        f"itx|cls|record|{args.seed}|{args.epochs}|{args.batch}|{args.max_lr}|"
        f"{args.d_model}|{meta['sha256'][:12]}"
    )
    if os.path.exists(done_path) and json.load(open(done_path)).get('key') == key:
        print('already finished, skip'); return

    model = ITransformer(
        task='classification', seq_len=args.seq_len, n_variates=12, n_class=5,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers,
        d_ff=args.d_ff, dropout=args.dropout,
    ).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    loader = DataLoader(member_ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=2)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.max_lr / 25)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.max_lr, total_steps=args.epochs * len(loader))
    ce = nn.CrossEntropyLoss()
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

    @torch.no_grad()
    def get_logits(ds):
        model.eval()
        out = []
        for x, _ in DataLoader(ds, batch_size=64, shuffle=False, num_workers=2):
            with amp_ctx():
                out.append(model(x.to(dev).float()).float().cpu())
        return torch.cat(out)

    t0 = time.time()
    for ep in range(start, args.epochs + 1):
        model.train()
        losses = []
        for x, yb in tqdm(loader, desc=f'ep{ep}/{args.epochs}', leave=False):
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                logits = model(x.to(dev).float())
            loss = ce(logits.float(), yb.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        if ep in eval_at:
            lm, ln, lv = get_logits(member_ds), get_logits(nonmem_ds), get_logits(val_ds)
            save_atomic({'lm': lm, 'ln': ln}, f'{args.ckpt_dir}/{tag}_logits_ep{ep}.pt')
            acc_m = float((lm.argmax(1).numpy() == member_ds.labels).mean())
            acc_n = float((ln.argmax(1).numpy() == nonmem_ds.labels).mean())
            acc_v = float((lv.argmax(1).numpy() == val_ds.labels).mean())
            sm = attack_scores_cls(lm, member_ds.labels)
            sn = attack_scores_cls(ln, nonmem_ds.labels)
            row = {
                'epoch': ep, 'train_loss': float(np.mean(losses)),
                'train_acc': acc_m, 'val_acc': acc_v, 'test_acc': acc_n, 'gap': acc_m - acc_n,
            }
            for a in sm:
                t01, t1, au = mia_metrics(sm[a], sn[a])
                row.update({f'{a}_tpr@0.1': t01, f'{a}_tpr@1': t1, f'{a}_auc': au})
            history.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

        save_atomic({
            'key': key, 'epoch': ep, 'history': history,
            'opt': opt.state_dict(), 'sched': sched.state_dict(),
            'model': {k: v.cpu() for k, v in model.state_dict().items()},
        }, ckpt_path)

    final_ep = max(e for e in eval_at if e <= args.epochs)
    row = [r for r in history if r['epoch'] == final_ep][-1]
    lg = torch.load(f'{args.ckpt_dir}/{tag}_logits_ep{final_ep}.pt', weights_only=False)
    sm = attack_scores_cls(lg['lm'], member_ds.labels)
    sn = attack_scores_cls(lg['ln'], nonmem_ds.labels)
    for a in sm:
        c01, c1 = bootstrap_ci(sm[a], sn[a], args.n_boot, args.seed)
        out = {
            'model': 'itransformer', 'task': 'cls', 'attack': a,
            'owner_tv': 3, 'level': 'record', 'dataset': 'ptbxl', 'seed': args.seed,
            'n_member': int(len(sm[a])), 'n_nonmember': int(len(sn[a])),
            'scores_member': np.round(sm[a], 6).tolist(),
            'scores_nonmember': np.round(sn[a], 6).tolist(),
            'tpr_at_0.1fpr': row[f'{a}_tpr@0.1'], 'tpr_at_1fpr': row[f'{a}_tpr@1'], 'auc': row[f'{a}_auc'],
            'bootstrap_ci': {
                'n_resamples': args.n_boot, 'confidence': 0.95,
                'tpr_at_0.1fpr': c01, 'tpr_at_1fpr': c1,
            },
            'utility': {'train_acc': row['train_acc'], 'test_acc': row['test_acc'], 'gap': row['gap']},
            'n_shadow': None,
            'extra': {
                'eval_epochs': eval_at, 'tuning': 'default recipe; tune on val_acc only',
                'epochs': args.epochs, 'eval_epoch': row['epoch'], 'max_lr': args.max_lr,
                'batch': args.batch, 'backbone': 'iTransformer',
                'd_model': args.d_model, 'n_heads': args.n_heads, 'e_layers': args.e_layers,
                'd_ff': args.d_ff, 'val_acc': row.get('val_acc'),
                'split_sha256': meta['sha256'], 'node': os.environ.get('NODE_NAME', ''),
            },
        }
        fn = f'{args.results_dir}/{a}.json'
        json.dump(out, open(fn, 'w'))
        print('saved', fn)

    json.dump({'key': key, 'history': history}, open(done_path, 'w'))
    print(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
