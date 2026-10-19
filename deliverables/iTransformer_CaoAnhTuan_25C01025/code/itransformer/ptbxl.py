"""PTB-XL loader matching the shared split protocol (record-level, L/H windows).
Loader PTB-XL khớp protocol split chung (record-level, cửa sổ L/H).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

CLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']


def load_split_meta(split_path: str) -> tuple[dict, dict]:
    """Load split JSON + meta and verify sha256 / Tải split JSON + meta và kiểm tra sha256."""
    split = json.load(open(split_path))
    meta = json.load(open(split_path.replace('.json', '.meta.json')))
    raw = open(split_path, 'rb').read()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == meta['sha256'], f"split sha256 mismatch: {digest} vs {meta['sha256']}"
    return split, meta


def _single_label_table(ptbxl_dir: str) -> pd.DataFrame:
    """Keep only single-superclass records / Chỉ giữ bản ghi đúng 1 superclass."""
    df = pd.read_csv(os.path.join(ptbxl_dir, 'ptbxl_database.csv'), index_col='ecg_id')
    codes = pd.read_csv(os.path.join(ptbxl_dir, 'scp_statements.csv'), index_col=0)
    diag = codes[codes.diagnostic == 1.0].diagnostic_class
    df['scp_codes'] = df.scp_codes.apply(ast.literal_eval)
    df['superclass'] = df.scp_codes.apply(
        lambda d: sorted({diag[k] for k in d if k in diag.index and pd.notna(diag[k])})
    )
    df = df[df.superclass.apply(len) == 1].copy()
    df['superclass'] = df.superclass.apply(lambda c: c[0])
    df['label'] = df.superclass.map({c: i for i, c in enumerate(CLASSES)})
    return df.sort_index()


def _resample_zscore(sig: np.ndarray, seq_len: int) -> np.ndarray:
    """Interpolate each lead to seq_len then z-score / Nội suy từng lead rồi z-score."""
    # sig: [T, 12] from wfdb -> transpose to [12, T]
    x = sig.T.astype(np.float64)
    t_old = np.linspace(0.0, 1.0, x.shape[1])
    t_new = np.linspace(0.0, 1.0, seq_len)
    out = np.stack([interp1d(t_old, x[c], kind='linear')(t_new) for c in range(x.shape[0])])
    # Per-lead z-score on the full L+H window / Z-score theo từng lead trên cả cửa sổ
    mu = out.mean(axis=1, keepdims=True)
    sd = out.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((out - mu) / sd).astype(np.float32)


class PTBXLDataset(Dataset):
    """Cached PTB-XL windows for cls (L) or fcst (L+H).
    Cửa sổ PTB-XL đã cache cho cls (L) hoặc fcst (L+H).
    """

    def __init__(
        self,
        ptbxl_dir: str,
        split_path: str,
        phase: str,
        seq_len: int = 512,
        cache_dir: Optional[str] = None,
        load_cache: bool = True,
    ):
        assert phase in ('train', 'val', 'test', 'pool')
        self.ptbxl_dir = ptbxl_dir
        self.seq_len = seq_len
        split, meta = load_split_meta(split_path)
        self.meta = meta
        table = _single_label_table(ptbxl_dir)

        if phase == 'pool':
            # Full single-label pool in split-id order for shadow masks / Toàn bộ pool theo thứ tự id
            ids = sorted(int(i) for i in table.index.tolist())
        else:
            ids = [int(i) for i in split[phase]]
        self.ids = np.asarray(ids, dtype=np.int64)
        self.labels = table.loc[self.ids, 'label'].values.astype(np.int64)
        self.filenames = table.loc[self.ids, 'filename_lr'].astype(str).tolist()

        cache_dir = cache_dir or os.path.join(ptbxl_dir, '.cache_itransformer')
        os.makedirs(cache_dir, exist_ok=True)
        tag = hashlib.sha1(f'{phase}|{seq_len}|{meta["sha256"]}|{len(self.ids)}'.encode()).hexdigest()[:12]
        cache_x = os.path.join(cache_dir, f'{phase}_L{seq_len}_{tag}.npy')
        cache_y = os.path.join(cache_dir, f'{phase}_L{seq_len}_{tag}_y.npy')
        cache_ids = os.path.join(cache_dir, f'{phase}_L{seq_len}_{tag}_ids.npy')

        if load_cache and os.path.exists(cache_x) and os.path.exists(cache_y):
            self.timeseries = np.load(cache_x, mmap_mode='r')
            self.labels = np.load(cache_y)
            self.ids = np.load(cache_ids)
            assert len(self.timeseries) == len(self.labels) == len(self.ids)
            return

        # Build cache by reading wfdb records / Xây cache bằng cách đọc bản ghi wfdb
        import wfdb

        xs = []
        for rel in self.filenames:
            path = os.path.join(ptbxl_dir, rel)
            rec = wfdb.rdrecord(path)
            sig = rec.p_signal  # [1000, 12] at 100 Hz
            assert sig.shape[1] == 12, sig.shape
            xs.append(_resample_zscore(sig, seq_len))
        self.timeseries = np.stack(xs).astype(np.float32)  # [N, 12, seq_len]
        np.save(cache_x, self.timeseries)
        np.save(cache_y, self.labels)
        np.save(cache_ids, self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        # Return (x [12, T], y) to match MOMENT loaders / Trả (x [12, T], y) giống loader MOMENT
        x = torch.from_numpy(np.asarray(self.timeseries[idx])).float()
        y = int(self.labels[idx])
        return x, y
