"""Shared PTB-XL member / non-member split for all models (LSTM, TCN, PatchTST, iTransformer, MOMENT).

Uses only the 5-superclass, single-label records (same filter as the MOMENT tutorial), so every
team member gets identical ecg_id sets. Output keys follow the MOMENT dataset convention:

    train = member       (used to train the target model)
    test  = non-member   (never seen by the target model)
    val   = small held-out set carved out of the non-member pool (not used for the attack)

Usage:
    python make_split.py --ptbxl-dir /path/to/ptb-xl --level both --seed 42
"""
import argparse
import ast
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

CLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']   # sorted, same as sklearn LabelEncoder in momentfm


def load_single_label_records(ptbxl_dir):
    df = pd.read_csv(os.path.join(ptbxl_dir, 'ptbxl_database.csv'), index_col='ecg_id')
    codes = pd.read_csv(os.path.join(ptbxl_dir, 'scp_statements.csv'), index_col=0)
    diag = codes[codes.diagnostic == 1.0].diagnostic_class
    df['scp_codes'] = df.scp_codes.apply(ast.literal_eval)
    df['superclass'] = df.scp_codes.apply(lambda d: sorted({diag[k] for k in d if k in diag.index and pd.notna(diag[k])}))
    df = df[df.superclass.apply(len) == 1].copy()
    df['superclass'] = df.superclass.apply(lambda c: c[0])
    assert set(df.superclass) <= set(CLASSES), set(df.superclass)
    df['label'] = df.superclass.map({c: i for i, c in enumerate(CLASSES)})
    return df.sort_index()


def make_split(df, level, seed, n_val):
    ids = df.index.values
    if level == 'record':
        member, non = train_test_split(ids, test_size=0.5, stratify=df.label.values, random_state=seed)
    elif level == 'user':
        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        mi, ni = next(gss.split(ids, groups=df.patient_id.values))
        member, non = ids[mi], ids[ni]
        pid = df.patient_id
        assert not (set(pid.loc[member]) & set(pid.loc[non])), 'patient leak between member and non-member'
    else:
        raise ValueError(level)
    non = np.random.default_rng(seed).permutation(non)
    split = {'train': sorted(int(i) for i in member),
             'val': sorted(int(i) for i in non[:n_val]),
             'test': sorted(int(i) for i in non[n_val:])}
    assert not (set(split['train']) & set(split['val'])) and not (set(split['train']) & set(split['test']))
    assert not (set(split['val']) & set(split['test']))
    return split


def make_shadow_masks(df, level, seed, k):
    """Balanced in/out masks for LiRA / RMIA shadow models over the whole single-label pool.

    masks[j, i] = True  <=>  shadow j is trained on record ids[i]. Every unit (record, or patient at
    user level) is "in" for exactly k/2 shadows, so each sample has k/2 IN and k/2 OUT reference logits.
    Shadows use the same training recipe (epochs, optimizer, lr) as the target.
    """
    assert k % 2 == 0, 'k must be even'
    ids = df.index.values
    unit = ids if level == 'record' else df.patient_id.values
    units = np.unique(unit)
    rng = np.random.default_rng(seed + 1000)
    unit_masks = np.zeros((k, len(units)), dtype=bool)
    for u in range(len(units)):
        unit_masks[rng.permutation(k)[:k // 2], u] = True
    masks = unit_masks[:, np.searchsorted(units, unit)]
    ref_ids = np.sort(rng.choice(ids, size=min(5000, len(ids)), replace=False))   # RMIA population Z
    return ids, masks, ref_ids


def write_shadow(df, level, seed, k, out_dir):
    ids, masks, ref_ids = make_shadow_masks(df, level, seed, k)
    stem = f'ptbxl_{level}_s{seed}'
    path = os.path.join(out_dir, f'shadow_masks_{level}_K{k}_s{seed}.npz')
    np.savez_compressed(path, ids=ids, masks=masks, ref_ids=ref_ids)
    with open(path, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    meta_path = os.path.join(out_dir, stem + '.meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    meta['shadow'] = {'file': os.path.basename(path), 'K': k, 'sha256': digest, 'n_ref': int(len(ref_ids)),
                      'in_fraction': float(masks.mean()),
                      'rmia_defaults': {'gamma': 2.0, 'a': 0.3, 'offline': True}}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=1)
    print(' ', os.path.basename(path), masks.shape, 'sha256', digest[:12])


def write(df, split, level, seed, n_val, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stem = f'ptbxl_{level}_s{seed}'
    path = os.path.join(out_dir, stem + '.json')
    with open(path, 'w') as f:
        json.dump(split, f)
    with open(path, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    meta = {
        'level': level, 'seed': seed, 'n_val': n_val, 'classes': CLASSES, 'sha256': digest,
        'n_records_total': int(len(df)), 'n_patients_total': int(df.patient_id.nunique()),
        'sizes': {k: len(v) for k, v in split.items()},
        'class_counts': {k: np.bincount(df.label.loc[v].values, minlength=len(CLASSES)).tolist() for k, v in split.items()},
        'n_patients': {k: int(df.patient_id.loc[v].nunique()) for k, v in split.items()},
    }
    with open(os.path.join(out_dir, stem + '.meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(stem, meta['sizes'], 'sha256', digest[:12])
    if meta['sizes']['test'] < 10000:
        print(f"  WARNING: only {meta['sizes']['test']} non-members; TPR@0.1%FPR rests on ~{int(meta['sizes']['test'] * 0.001)} samples")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ptbxl-dir', required=True, help='folder containing ptbxl_database.csv')
    ap.add_argument('--level', choices=['record', 'user', 'both'], default='both')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-val', type=int, default=500)
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits'))
    ap.add_argument('--shadow-k', type=int, default=8, help='number of LiRA/RMIA shadow masks (even); 0 to skip')
    a = ap.parse_args()
    data = load_single_label_records(a.ptbxl_dir)
    for lv in (['record', 'user'] if a.level == 'both' else [a.level]):
        write(data, make_split(data, lv, a.seed, a.n_val), lv, a.seed, a.n_val, a.out_dir)
        if a.shadow_k:
            write_shadow(data, lv, a.seed, a.shadow_k, a.out_dir)
