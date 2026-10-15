"""Forecasting signals and the Koren temporal feature vector."""

from __future__ import annotations

import numpy as np

from experiment.config import ACF_LAGS, TREND_DEGREE


def mae(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(yhat - y), axis=(1, 2))


def mse(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return np.mean((yhat - y) ** 2, axis=(1, 2))


def smape(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Johansson et al. 2025, eq. 2.22, clipped to [0, 1)."""
    numer = np.abs(yhat - y)
    denom = np.abs(yhat) + np.abs(y) + eps
    value = np.mean(numer / denom, axis=(1, 2))
    return np.clip(value, 0.0, 1.0 - 1e-6)


def rsmape(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Rescaled SMAPE, Johansson et al. 2025, eq. 2.23."""
    s = smape(y, yhat)
    return np.log(s / (1.0 - s))


def _vandermonde(horizon: int, degree: int) -> np.ndarray:
    # Koren et al.: t = [0, 1, ..., H-1] / H. Highest power is `degree`.
    t = np.arange(horizon, dtype=np.float64) / horizon
    return np.stack([t**p for p in range(degree + 1)], axis=1)


def trend_coefficients(y: np.ndarray, degree: int = TREND_DEGREE) -> np.ndarray:
    """Least-squares coefficients, shape (N, degree+1, M)."""
    pinv = np.linalg.pinv(_vandermonde(y.shape[1], degree))
    return np.einsum("dh,nhm->ndm", pinv, y.astype(np.float64))


def dft_coefficients(y: np.ndarray) -> np.ndarray:
    """2D DFT over horizon and channels. Shape (N, H, M), complex."""
    return np.fft.fft2(y.astype(np.float64), axes=(1, 2))


def acf_features(y: np.ndarray, lags: tuple[int, ...] = ACF_LAGS) -> np.ndarray:
    """Normalized autocorrelation per channel. Shape (N, M, n_lags)."""
    horizon = y.shape[1]
    centered = y.astype(np.float64)
    centered = centered - centered.mean(axis=1, keepdims=True)
    energy = np.sum(centered**2, axis=1) / horizon
    columns = []
    for lag in lags:
        if lag >= horizon:
            columns.append(np.zeros(y.shape[0:1] + y.shape[2:3], dtype=np.float64))
            continue
        corr = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / horizon
        columns.append(corr / np.maximum(energy, 1e-8))
    return np.stack(columns, axis=-1)


def temporal_features(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """One vector per window: DFT, degree-4 trend, and ACF. No feature ablation.

    Coefficients follow Koren et al. 2025 §4.2 (true horizon versus forecast).
    Autocorrelation is the extra time-series feature in this study.
    """
    trend_gap = trend_coefficients(y) - trend_coefficients(yhat)
    spectrum_gap = dft_coefficients(y) - dft_coefficients(yhat)
    acf_gap = acf_features(y) - acf_features(yhat)
    trend_l2 = np.linalg.norm(trend_gap.reshape(len(y), -1), axis=1, keepdims=True)
    spectrum_l2 = np.linalg.norm(spectrum_gap.reshape(len(y), -1), axis=1, keepdims=True)
    parts = [
        spectrum_gap.real.reshape(len(y), -1),
        spectrum_gap.imag.reshape(len(y), -1),
        spectrum_l2,
        trend_gap.reshape(len(y), -1),
        trend_l2,
        acf_gap.reshape(len(y), -1),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)
