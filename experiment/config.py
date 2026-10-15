"""Protocol shared by Koren et al. (2025) and Johansson et al. (2025).

Both papers attack numeric forecasting on TUH-EEG and both include an LSTM.
They do not publish the same LSTM width or the same cohort size:

- Dataset intersection: TUH-EEG, 250 Hz, leads FP1, FP2, F3
  (Johansson et al. 2025, §3.1.1, "as done in Koren et al.").
- LSTM used for LiRA comparison: Johansson et al. 2025, §3.2.1 and Table 3.1
  (2 unidirectional layers, hidden 64, lookback 100, horizon 20, MAE, Adam
  1e-3, batch 1024, max 50 epochs, early stopping patience 3).
  Koren et al. use hidden 50 and MSE, and only 32 patients, with no published
  lookback. Hidden 50 cannot be matched at the same time as the LiRA setup,
  so the forecasting target follows Johansson.
- Temporal features: Koren et al. 2025, §4.2 — 2D-DFT of the horizon and a
  degree-4 polynomial trend — plus autocorrelation, concatenated into one
  vector. No per-feature ablation.
- LiRA: offline, K = 16, global OUT variance when K < 64 (Carlini et al.
  2022, §VI-B). Forecasting signal is rescaled SMAPE (Johansson et al.
  2025, eq. 2.23). Shadows train on 50% of Daux individuals (§3.1.3).
"""

LOOKBACK = 100
HORIZON = 20
N_CHANNELS = 3
CHANNEL_NAMES = ("FP1", "FP2", "F3")
SAMPLE_RATE = 250
SKIP_SECONDS = 60
SERIES_LENGTH = 15000

HIDDEN = 64
LSTM_LAYERS = 2
MAX_EPOCHS = 50
PATIENCE = 3
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024

N_SUBJECTS = 100
SHADOW_MODELS = 16
AUDIT_PER_CLASS = 2000
TREND_DEGREE = 4
ACF_LAGS = tuple(range(1, 11))
TEMPORAL_PER_SHADOW = 256
FPR_TARGET = 0.001
