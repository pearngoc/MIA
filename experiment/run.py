"""Train LSTM-fcst and run Loss, offline LiRA, and Temporal Feature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiment import config
from experiment.attacks import (
    auc_roc,
    fit_temporal_attack,
    lira_offline_scores,
    loss_scores,
    roc_curve,
    temporal_scores,
    tpr_at_fpr,
)
from experiment.data import (
    gather_windows,
    load_npz,
    load_tuh_directory,
    patient_split,
    robust_scale_inplace,
    sample_index,
    synthetic_population,
    window_index,
)
from experiment.train import predict, train_forecaster, utility


def main() -> None:
    args = _parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    recordings = _load_recordings(args, rng)
    groups = patient_split(recordings, rng)
    robust_scale_inplace(groups)
    _print_split(groups, args)

    print("Training LSTM-fcst target", flush=True)
    target, history = _train_split(groups["train"], groups["val"], args, args.seed, out / "target.pt")
    utility_report = _utility_report(target, groups, args.stride)
    print(
        "  utility train/test MAE "
        f"{utility_report['train']['mae']:.4f} / {utility_report['test']['mae']:.4f}",
        flush=True,
    )

    member_x, member_y = _audit_windows(groups["train"], args.audit, args.stride, rng)
    nonmember_x, nonmember_y = _audit_windows(groups["test"], args.audit, args.stride, rng)
    y = np.concatenate([member_y, nonmember_y], axis=0)
    x = np.concatenate([member_x, nonmember_x], axis=0)
    n_member = len(member_y)
    print(f"Audit windows: {n_member} member + {len(nonmember_y)} non-member", flush=True)
    yhat_target = predict(target, x)

    attack_scores = {"loss": loss_scores(y, yhat_target)}

    shadow_forecasts = []
    temporal_y = []
    temporal_yhat = []
    temporal_labels = []
    for k in range(args.shadows):
        chosen = _half_subjects(groups["aux"], rng)
        print(f"Training shadow {k + 1}/{args.shadows} on {len(chosen)} aux subjects", flush=True)
        shadow, _ = _train_split(chosen, groups["val"], args, args.seed + 1 + k, out / f"shadow_{k:02d}.pt")
        shadow_forecasts.append(predict(shadow, x))
        y_s, yhat_s, labels_s = _shadow_membership_batch(shadow, groups["aux"], chosen, args, rng)
        temporal_y.append(y_s)
        temporal_yhat.append(yhat_s)
        temporal_labels.append(labels_s)
        del shadow

    attack_scores["lira"] = lira_offline_scores(y, yhat_target, np.stack(shadow_forecasts, axis=0))
    print("Fitting temporal-feature MLP on shadow forecasts", flush=True)
    scaler, clf = fit_temporal_attack(
        np.concatenate(temporal_y, axis=0),
        np.concatenate(temporal_yhat, axis=0),
        np.concatenate(temporal_labels, axis=0),
        seed=args.seed,
    )
    attack_scores["temporal"] = temporal_scores(y, yhat_target, scaler, clf)

    report = {
        "protocol": _protocol_note(args),
        "utility": utility_report,
        "generalization_gap_mae_train_minus_test": (
            utility_report["train"]["mae"] - utility_report["test"]["mae"]
        ),
        "target_epochs": history,
        "attacks": {},
    }
    labels = np.array([1] * n_member + [0] * (len(y) - n_member))
    for name, scores in attack_scores.items():
        member = scores[labels == 1]
        nonmember = scores[labels == 0]
        low = tpr_at_fpr(member, nonmember, config.FPR_TARGET)
        report["attacks"][name] = {
            "tpr_at_0.1pct_fpr": low["tpr"],
            "empirical_fpr": low["fpr_empirical"],
            "auc": auc_roc(member, nonmember),
            "roc": roc_curve(member, nonmember),
        }
        print(
            f"  {name:10s}  TPR@0.1%FPR={low['tpr']:.4f}  AUC={report['attacks'][name]['auc']:.4f}",
            flush=True,
        )

    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    np.savez(
        out / "scores.npz",
        labels=labels,
        loss=attack_scores["loss"],
        lira=attack_scores["lira"],
        temporal=attack_scores["temporal"],
    )
    print(f"Wrote {out / 'metrics.json'}", flush=True)


def _train_split(train_recs, val_recs, args, seed: int, checkpoint: Path):
    if checkpoint.exists() and not args.retrain:
        from experiment.model import LSTMForecaster
        from experiment.train import device_of

        model = LSTMForecaster().to(device_of())
        model.load_state_dict(torch.load(checkpoint, map_location=device_of(), weights_only=True))
        model.eval()
        print(f"  loaded {checkpoint.name}", flush=True)
        return model, []
    x, y = gather_windows(train_recs, window_index(train_recs, args.stride))
    x_val, y_val = gather_windows(val_recs, sample_index(window_index(val_recs, args.stride), args.val_windows, np.random.default_rng(seed)))
    model, history = train_forecaster(
        x, y, x_val, y_val, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, seed=seed
    )
    torch.save(model.state_dict(), checkpoint)
    return model, history


def _utility_report(model, groups, stride: int) -> dict:
    report = {}
    for name in ("train", "val", "test"):
        index = window_index(groups[name], stride)
        if len(index) > 8192:
            index = index[:: max(1, len(index) // 8192)][:8192]
        x, y = gather_windows(groups[name], index)
        report[name] = utility(y, predict(model, x))
    return report


def _audit_windows(recordings, size: int, stride: int, rng: np.random.Generator):
    index = sample_index(window_index(recordings, stride), size, rng)
    return gather_windows(recordings, index)


def _half_subjects(recordings, rng: np.random.Generator):
    choice = rng.choice(len(recordings), size=max(1, len(recordings) // 2), replace=False)
    return [recordings[i] for i in choice]


def _shadow_membership_batch(model, aux, chosen, args, rng: np.random.Generator):
    chosen_ids = {rec.subject_id for rec in chosen}
    inside = [rec for rec in aux if rec.subject_id in chosen_ids]
    outside = [rec for rec in aux if rec.subject_id not in chosen_ids]
    if not outside:
        outside = inside
    parts_y = []
    parts_yhat = []
    parts_label = []
    for label, pool in ((1, inside), (0, outside)):
        x, y = _audit_windows(pool, args.temporal_windows, args.stride, rng)
        parts_y.append(y)
        parts_yhat.append(predict(model, x))
        parts_label.append(np.full(len(y), label, dtype=np.int64))
    return np.concatenate(parts_y), np.concatenate(parts_yhat), np.concatenate(parts_label)


def _load_recordings(args, rng: np.random.Generator):
    if args.demo:
        print("DEMO: synthetic series, not TUH-EEG. Numbers are not comparable.", flush=True)
        return synthetic_population(args.subjects, args.length, rng)
    if args.npz is not None:
        return load_npz(args.npz, args.subjects)
    return load_tuh_directory(args.data, args.subjects)


def _print_split(groups, args) -> None:
    counts = {name: len(recs) for name, recs in groups.items()}
    print(
        "Split (patients) "
        + " ".join(f"{name}={count}" for name, count in counts.items())
        + f"  stride={args.stride}  shadows={args.shadows}",
        flush=True,
    )


def _protocol_note(args) -> dict:
    return {
        "dataset": "synthetic-demo" if args.demo else "TUH-EEG",
        "comparable_to": [
            "Johansson et al. 2025 Table 3.1 / §3.2.1 (LSTM-fcst, LiRA signal rSMAPE)",
            "Koren et al. 2025 §4.2 (2D-DFT + degree-4 trend) plus ACF",
            "Carlini et al. 2022 offline LiRA, global variance for K<64",
        ],
        "not_matched": "Koren LSTM hidden 50 and MSE; their cohort is 32 patients and lookback is unpublished.",
        "lookback": config.LOOKBACK,
        "horizon": config.HORIZON,
        "hidden": config.HIDDEN,
        "layers": config.LSTM_LAYERS,
        "loss": "MAE",
        "channels": list(config.CHANNEL_NAMES),
        "shadows": args.shadows,
        "stride": args.stride,
        "audit_per_class": args.audit,
        "demo": args.demo,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM-fcst Loss, LiRA, and Temporal Feature attacks")
    parser.add_argument("--data", type=Path, default=Path("data/tuh_eeg"))
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/lstm_fcst"))
    parser.add_argument("--demo", action="store_true", help="Synthetic smoke run. Not a paper result.")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subjects", type=int, default=config.N_SUBJECTS)
    parser.add_argument("--shadows", type=int, default=config.SHADOW_MODELS)
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--audit", type=int, default=config.AUDIT_PER_CLASS)
    parser.add_argument("--val-windows", type=int, default=4096)
    parser.add_argument("--temporal-windows", type=int, default=config.TEMPORAL_PER_SHADOW)
    parser.add_argument("--length", type=int, default=config.SERIES_LENGTH)
    args = parser.parse_args()
    if args.demo:
        args.subjects = min(args.subjects, 12)
        args.shadows = min(args.shadows, 2)
        args.epochs = min(args.epochs, 3)
        args.stride = max(args.stride, 20)
        args.audit = min(args.audit, 128)
        args.val_windows = min(args.val_windows, 256)
        args.temporal_windows = min(args.temporal_windows, 64)
        args.length = min(args.length, 800)
        args.batch_size = min(args.batch_size, 128)
        args.out = Path("results/lstm_fcst_demo")
    return args


if __name__ == "__main__":
    main()
