"""Five-fold training of the detection model (AD / CANCER / HC).

Input: ``Features`` over a candidate gene pool, one entry per sample and cell type: ``mu`` (mean
expression of the sample's cells), ``cells`` (m sampled cells) and ``mask`` (cell type present).
Per fold the top-k genes of each cell type are chosen by XGBoost total gain; the decision
threshold is Youden's J on the pooled out-of-fold predictions.
"""
from dataclasses import dataclass

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

from .detection_model import score, to_tensors, train_model

DECISION = "AD if argmax is AD; else CANCER if p_C/(p_C+p_HC) >= threshold; else HC"


@dataclass
class Features:
    genes: dict   # celltype -> gene names (columns of mu / cells)
    mu: dict      # celltype -> (n_samples, n_genes)
    cells: dict   # celltype -> (n_samples, m, n_genes)
    mask: np.ndarray  # (n_samples, n_celltypes)


def gene_gain(X, labels, seed, device="cuda"):
    """Total gain per column of a class-balanced, 150-round, 15-leaf XGBoost fit."""
    encoded = np.asarray([("AD", "CANCER", "HC").index(v) for v in labels], dtype=np.int32)
    weights = len(encoded) / (3 * np.bincount(encoded, minlength=3)[encoded])
    data = xgb.QuantileDMatrix(np.asarray(X), label=encoded, weight=weights, missing=np.nan, max_bin=64)
    params = {"tree_method": "hist", "device": device, "grow_policy": "lossguide", "max_leaves": 15,
              "max_depth": 0, "eta": 0.06, "lambda": 1.0, "objective": "multi:softprob",
              "num_class": 3, "max_bin": 64, "colsample_bynode": 1.0, "seed": seed}
    gain = np.zeros(X.shape[1], dtype=np.float64)
    for name, value in xgb.train(params, data, num_boost_round=150).get_score(importance_type="total_gain").items():
        gain[int(name.removeprefix("f"))] = float(value)
    return gain


def binary_score(proba):
    return proba[:, 1] / (proba[:, 1] + proba[:, 2])


def youden_threshold(labels, proba):
    keep = labels != "AD"
    fpr, tpr, thr = roc_curve((labels[keep] == "CANCER").astype(int), binary_score(proba[keep]))
    finite = np.flatnonzero(np.isfinite(thr))
    return float(thr[finite[int(np.argmax((tpr - fpr)[finite]))]])


def decide(proba, threshold):
    out = np.where(binary_score(proba) >= threshold, "CANCER", "HC").astype(object)
    out[np.argmax(proba, axis=1) == 0] = "AD"
    return out


def train(cfg, features, labels, strata, device="cuda", seed=0):
    """Returns (checkpoints, out-of-fold probabilities, fold of each sample)."""
    cts, labels = cfg["celltypes"], np.asarray(labels, dtype=str)
    folds = np.full(len(labels), -1)
    for k, (_, held) in enumerate(StratifiedKFold(cfg["n_folds"], shuffle=True, random_state=seed)
                                  .split(np.zeros(len(labels)), strata)):
        folds[held] = k
    oof = np.zeros((len(labels), 3))
    checkpoints = []
    for k in range(cfg["n_folds"]):
        inside = folds != k
        columns = {ct: np.argsort(-gene_gain(features.mu[ct][inside], labels[inside], seed, device),
                                  kind="stable")[:cfg["top_k"]] for ct in cts}
        tensors = to_tensors(features, cts, columns, device)
        dims = {ct: cfg["top_k"] for ct in cts}
        model = train_model(cts, dims, cfg["architecture"], tensors, labels, np.flatnonzero(inside),
                            cfg["train"], device, seed + k, cfg["class_order"])
        oof[~inside] = score(model, cts, tensors, np.flatnonzero(~inside), device)
        checkpoints.append({"class_order": cfg["class_order"], "celltypes": cts, "gene_dims": dims,
                            "selected_genes": {ct: [features.genes[ct][i] for i in columns[ct]] for ct in cts},
                            "architecture": cfg["architecture"],
                            "state_dict": {n: v.detach().cpu() for n, v in model.state_dict().items()}})
    threshold = youden_threshold(labels, oof)
    for cp in checkpoints:
        cp.update(threshold=threshold, decision=DECISION)
    return checkpoints, oof, folds
