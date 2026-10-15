"""TUH-EEG loading and the Johansson patient split."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import RobustScaler

from experiment.config import (
    CHANNEL_NAMES,
    HORIZON,
    LOOKBACK,
    N_SUBJECTS,
    SAMPLE_RATE,
    SERIES_LENGTH,
    SKIP_SECONDS,
)


@dataclass
class Recording:
    subject_id: str
    values: np.ndarray  # (T, M) float32


def patient_split(recordings: list[Recording], rng: np.random.Generator) -> dict[str, list[Recording]]:
    """40% audit / 40% auxiliary / 20% validation, then audit 50/50 into train/test.

    Matches Johansson et al. 2025, Table 3.1, for 100 subjects:
    20 train, 20 test, 20 val, 40 auxiliary.
    """
    order = rng.permutation(len(recordings))
    n = len(order)
    n_audit = int(round(0.4 * n))
    n_aux = int(round(0.4 * n))
    audit_idx = order[:n_audit]
    aux_idx = order[n_audit : n_audit + n_aux]
    val_idx = order[n_audit + n_aux :]
    half = n_audit // 2
    groups = {
        "train": [recordings[i] for i in audit_idx[:half]],
        "test": [recordings[i] for i in audit_idx[half:]],
        "val": [recordings[i] for i in val_idx],
        "aux": [recordings[i] for i in aux_idx],
    }
    if min(len(v) for v in groups.values()) < 1:
        raise ValueError(f"Need more subjects to split; got {n}")
    return groups


def robust_scale_inplace(groups: dict[str, list[Recording]]) -> RobustScaler:
    """IQR scaling fit on every selected recording, before the split is used.

    Johansson et al. scale in §3.1.1 and only then partition in §3.1.3.
    """
    stacked = np.concatenate([rec.values for group in groups.values() for rec in group], axis=0)
    scaler = RobustScaler()
    scaler.fit(stacked)
    iqr = scaler.scale_.copy()
    iqr[iqr < 1e-6] = 1.0
    scaler.scale_ = iqr
    for group in groups.values():
        for rec in group:
            rec.values = scaler.transform(rec.values).astype(np.float32)
    return scaler


def window_index(recordings: list[Recording], stride: int) -> np.ndarray:
    """Rows are (recording_index, start)."""
    rows = []
    need = LOOKBACK + HORIZON
    for i, rec in enumerate(recordings):
        n = (len(rec.values) - need) // stride + 1
        if n <= 0:
            continue
        starts = np.arange(n, dtype=np.int32) * stride
        rec_ids = np.full(n, i, dtype=np.int32)
        rows.append(np.column_stack([rec_ids, starts]))
    if not rows:
        raise ValueError("No complete windows. Check series length, lookback, and horizon.")
    return np.concatenate(rows, axis=0)


def gather_windows(recordings: list[Recording], index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.empty((len(index), LOOKBACK, recordings[0].values.shape[1]), dtype=np.float32)
    y = np.empty((len(index), HORIZON, recordings[0].values.shape[1]), dtype=np.float32)
    for row, (rec_i, start) in enumerate(index):
        series = recordings[int(rec_i)].values
        start = int(start)
        x[row] = series[start : start + LOOKBACK]
        y[row] = series[start + LOOKBACK : start + LOOKBACK + HORIZON]
    return x, y


def sample_index(index: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    if size <= 0 or size >= len(index):
        return index
    choice = rng.choice(len(index), size=size, replace=False)
    return index[choice]


def load_tuh_directory(root: Path, n_subjects: int = N_SUBJECTS) -> list[Recording]:
    """One recording per patient: skip 60 s, keep the next 15,000 samples at 250 Hz."""
    paths = sorted(root.rglob("*.edf"))
    if not paths:
        raise FileNotFoundError(
            f"No EDF files under {root}. TUH-EEG is distributed by NEDC after the "
            "access form at https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ . "
            "Place the corpus there, or pass a preprocessed npz via --npz."
        )
    by_patient: dict[str, Path] = {}
    for path in paths:
        patient = _patient_id(path)
        by_patient.setdefault(patient, path)
    recordings: list[Recording] = []
    skipped = 0
    for patient, path in by_patient.items():
        if len(recordings) >= n_subjects:
            break
        try:
            values = _read_edf_segment(path)
        except (OSError, ValueError, RuntimeError):
            skipped += 1
            continue
        if values is None:
            skipped += 1
            continue
        recordings.append(Recording(patient, values))
    if len(recordings) < 10:
        raise RuntimeError(
            f"Only {len(recordings)} usable recordings in {root} "
            f"({skipped} skipped). Need FP1, FP2, F3 and at least two minutes at 250 Hz."
        )
    return recordings


def load_npz(path: Path, n_subjects: int = N_SUBJECTS) -> list[Recording]:
    """Array `values` with shape (N, T, 3), optional `ids`."""
    blob = np.load(path, allow_pickle=True)
    values = blob["values"]
    ids = blob["ids"] if "ids" in blob.files else [f"s{i:04d}" for i in range(len(values))]
    recordings = []
    for subject_id, series in zip(ids, values):
        series = np.asarray(series, dtype=np.float32)
        if series.shape[0] < SERIES_LENGTH or series.shape[1] != 3:
            continue
        recordings.append(Recording(str(subject_id), series[:SERIES_LENGTH]))
        if len(recordings) >= n_subjects:
            break
    if len(recordings) < 10:
        raise RuntimeError(f"{path} does not contain enough (T>={SERIES_LENGTH}, M=3) series")
    return recordings


def synthetic_population(n_subjects: int, length: int, rng: np.random.Generator) -> list[Recording]:
    """Sinusoid mixture used only to exercise the pipeline. Not TUH-EEG."""
    t = np.arange(length, dtype=np.float32) / SAMPLE_RATE
    recordings = []
    for i in range(n_subjects):
        freq = 1.5 + (i % 7) * 0.35
        phase = rng.uniform(0, 2 * np.pi, size=3).astype(np.float32)
        amp = rng.uniform(0.4, 1.2, size=3).astype(np.float32)
        series = np.stack(
            [amp[c] * np.sin(2 * np.pi * freq * t + phase[c]) for c in range(3)],
            axis=1,
        )
        series += rng.normal(0, 0.05, size=series.shape).astype(np.float32)
        recordings.append(Recording(f"syn{i:03d}", series))
    return recordings


def _patient_id(path: Path) -> str:
    """TUH layout: <patient>/s001_<date>/<patient>_s001_t000.edf."""
    for parent in path.parents:
        if re.fullmatch(r"s\d{3}_.+", parent.name):
            return parent.parent.name.lower()
    return path.parent.name.lower()


def _read_edf_segment(path: Path) -> np.ndarray | None:
    import pyedflib

    reader = pyedflib.EdfReader(str(path))
    try:
        labels = reader.getSignalLabels()
        picks = [_match_channel(labels, name) for name in CHANNEL_NAMES]
        if any(p is None for p in picks):
            return None
        sfreq = float(reader.getSampleFrequency(picks[0]))
        if any(abs(reader.getSampleFrequency(p) - sfreq) > 1e-3 for p in picks):
            return None
        channels = [reader.readSignal(p).astype(np.float32) for p in picks]
    finally:
        reader.close()
    series = np.stack(channels, axis=1)
    if abs(sfreq - SAMPLE_RATE) > 1:
        series = _resample(series, sfreq, SAMPLE_RATE)
    skip = int(SKIP_SECONDS * SAMPLE_RATE)
    end = skip + SERIES_LENGTH
    if len(series) < end:
        return None
    return series[skip:end]


def _match_channel(labels: list[str], name: str) -> int | None:
    pattern = re.compile(rf"(^|[^a-z0-9]){re.escape(name.lower())}([^a-z0-9]|$)")
    for i, label in enumerate(labels):
        if pattern.search(label.lower()):
            return i
    return None


def _resample(series: np.ndarray, src_rate: float, dst_rate: int) -> np.ndarray:
    from scipy.signal import resample

    n_out = int(round(len(series) * dst_rate / src_rate))
    return resample(series, n_out, axis=0).astype(np.float32)
