"""Attack scores and Carlini low-FPR metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from experiment.config import FPR_TARGET
from experiment.signals import mae, rsmape, temporal_features


def loss_scores(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Yeom et al. 2018. Training objective is MAE, so a lower loss scores higher."""
    return -mae(y, yhat)


def lira_offline_scores(y: np.ndarray, yhat_target: np.ndarray, yhat_shadows: np.ndarray) -> np.ndarray:
    """Offline LiRA with a global OUT variance.

    `yhat_shadows` has shape (K, N, H, M). Every shadow is OUT for the audit
    set, because shadows train on Daux (Johansson et al. 2025, §3.1.3).
    The membership direction is the negation of rSMAPE: a better forecast is
    a larger score, matching the direction of Carlini's logit.
    """
    target = -rsmape(y, yhat_target)
    shadow = np.stack([-rsmape(y, yhat_shadows[k]) for k in range(len(yhat_shadows))], axis=0)
    mean_out = shadow.mean(axis=0)
    sigma = float(np.std(shadow - mean_out, ddof=1))
    sigma = max(sigma, 1e-6)
    scores = -norm.logsf(target, loc=mean_out, scale=sigma)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)


def fit_temporal_attack(
    y_shadow: np.ndarray,
    yhat_shadow: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[StandardScaler, MLPClassifier]:
    features = temporal_features(y_shadow, yhat_shadow)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=200,
        early_stopping=True,
        random_state=seed,
    )
    clf.fit(scaled, labels)
    return scaler, clf


def temporal_scores(
    y: np.ndarray,
    yhat: np.ndarray,
    scaler: StandardScaler,
    clf: MLPClassifier,
) -> np.ndarray:
    scaled = scaler.transform(temporal_features(y, yhat))
    return clf.predict_proba(scaled)[:, 1]


def tpr_at_fpr(member_scores: np.ndarray, nonmember_scores: np.ndarray, fpr: float = FPR_TARGET) -> dict:
    """TPR at the most permissive threshold whose empirical FPR is still at most `fpr`."""
    if len(nonmember_scores) == 0:
        raise ValueError("non-member scores are empty")
    allowed = int(np.floor(fpr * len(nonmember_scores)))
    if allowed <= 0:
        threshold = float(np.max(nonmember_scores))
        positive = np.greater
    else:
        # k-th largest non-member. Fall back to a strict cut when ties would
        # push the empirical FPR above the budget.
        threshold = float(np.partition(nonmember_scores, -allowed)[-allowed])
        if np.mean(nonmember_scores >= threshold) <= fpr + 1e-12:
            positive = np.greater_equal
        else:
            positive = np.greater
    tpr = float(np.mean(positive(member_scores, threshold)))
    empirical_fpr = float(np.mean(positive(nonmember_scores, threshold)))
    return {
        "tpr": tpr,
        "fpr_target": fpr,
        "fpr_empirical": empirical_fpr,
        "threshold": threshold,
    }


def auc_roc(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    scores = np.concatenate([member_scores, nonmember_scores])
    return float(roc_auc_score(y, scores))


def roc_curve(member_scores: np.ndarray, nonmember_scores: np.ndarray, points: int = 200) -> dict:
    y = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    scores = np.concatenate([member_scores, nonmember_scores])
    thresholds = np.quantile(scores, np.linspace(0, 1, points))
    fpr = []
    tpr = []
    for threshold in thresholds:
        pred = scores >= threshold
        tpr.append(float(np.mean(pred[y == 1])))
        fpr.append(float(np.mean(pred[y == 0])))
    return {"fpr": fpr, "tpr": tpr}
