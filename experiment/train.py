"""Target and shadow training. Same recipe for every LSTM (Johansson Table 3.1)."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiment.config import BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, PATIENCE
from experiment.model import LSTMForecaster
from experiment.signals import mae, mse, smape


def device_of() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_forecaster(
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    seed: int = 0,
) -> tuple[LSTMForecaster, list[dict]]:
    torch.manual_seed(seed)
    device = device_of()
    model = LSTMForecaster().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=True,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(xb) - yb).abs().mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(xb)
            seen += len(xb)
        val_mae = float(np.mean(mae(y_val, predict(model, x_val))))
        row = {"epoch": epoch, "train_mae": total / max(seen, 1), "val_mae": val_mae}
        history.append(row)
        print(f"    epoch {epoch:02d}  train_mae={row['train_mae']:.4f}  val_mae={val_mae:.4f}", flush=True)
        if val_mae < best_val - 1e-5:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    early stop at epoch {epoch}", flush=True)
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    outputs = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).to(device)
        outputs.append(model(xb).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def utility(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(mae(y, yhat))),
        "mse": float(np.mean(mse(y, yhat))),
        "smape": float(np.mean(smape(y, yhat))),
    }
