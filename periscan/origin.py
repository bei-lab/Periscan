"""Training of the cancer-origin model: five folds per training seed, ensemble by mean softmax.

Input ``store``: per cell type, ``X`` (cells x genes, all cells of all samples) and
``ranges[sample] = (start, stop)`` giving each sample's rows.  During training each sample keeps at
most ``cap_quantile`` of the per-sample cell counts of that cell type (a fixed random subset).  Each
fold keeps the epoch with the best top-1 accuracy on its own validation split (ties: balanced
accuracy).  Only training samples are seen here: the held-out test set is passed to
``ensemble_predict`` after training, never to model selection.
"""
import hashlib

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from .origin_model import make_model


class CellDataset(Dataset):
    def __init__(self, store, samples, y, caps=None, seed=0):
        self.store, self.cts = store, list(store)
        self.samples, self.y, self.caps, self.seed = list(samples), list(y), caps, seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sid, mats = self.samples[i], []
        rng = np.random.default_rng(int(hashlib.sha256(f"{sid}|{self.seed}".encode()).hexdigest()[:8], 16))
        for ct in self.cts:
            a, b = self.store[ct]["ranges"].get(sid, (0, 0))
            rows = np.arange(a, b)
            if self.caps and len(rows) > self.caps[ct]:
                rows = np.sort(rng.choice(rows, self.caps[ct], replace=False))
            mats.append(self.store[ct]["X"][rows].toarray().astype(np.float32))
        return mats, self.y[i]


def collate(batch):
    xs, masks = [], []
    for i in range(len(batch[0][0])):
        n = max(1, max(item[0][i].shape[0] for item in batch))
        X = np.zeros((len(batch), n, batch[0][0][i].shape[1]), dtype=np.float32)
        M = np.zeros((len(batch), n), dtype=bool)
        for r, item in enumerate(batch):
            X[r, :len(item[0][i])], M[r, :len(item[0][i])] = item[0][i], True
        xs.append(torch.from_numpy(X))
        masks.append(torch.from_numpy(M))
    return xs, masks, torch.tensor([item[1] for item in batch])


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    return np.vstack([torch.softmax(model([x.to(device) for x in xs], [m.to(device) for m in ms]), 1).cpu().numpy()
                      for xs, ms, _ in loader])


def normalized_entropy(p, eps=1e-12):
    return -(p * np.log(np.clip(p, eps, 1.0))).sum(1) / np.log(p.shape[1])


def train(cfg, store, train_samples, train_labels, device="cuda", seeds=(0,), fold_seed=0):
    """Returns (one state dict per (training seed, fold), out-of-fold probabilities, fold of sample).

    ``train`` only ever sees ``train_samples``.  Each fold evaluates on its own validation split,
    which is also the source of the returned out-of-fold probabilities.
    """
    t = cfg["train"]
    classes = cfg["classes"]
    store = {ct: store[ct] for ct in cfg["celltypes"]}
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.asarray([c2i[v] for v in train_labels])
    dims = {ct: store[ct]["X"].shape[1] for ct in store}
    caps = {ct: int(np.quantile([store[ct]["ranges"][s][1] - store[ct]["ranges"][s][0]
                                 for s in train_samples if s in store[ct]["ranges"]], t["cap_quantile"]))
            for ct in store}
    splits = list(StratifiedKFold(cfg["n_folds"], shuffle=True, random_state=fold_seed)
                  .split(train_samples, y))
    folds = np.full(len(train_samples), -1)
    oof = np.zeros((len(train_samples), len(classes)))
    weights = []
    for seed in seeds:
        for k, (inside, outside) in enumerate(splits):
            torch.manual_seed(seed)
            np.random.seed(seed)
            loader = DataLoader(CellDataset(store, [train_samples[i] for i in inside], y[inside], caps, seed),
                                batch_size=t["batch_size"], shuffle=True, collate_fn=collate,
                                num_workers=t.get("num_workers", 0))
            valid = DataLoader(CellDataset(store, [train_samples[i] for i in outside], y[outside]),
                               batch_size=t["batch_size"], collate_fn=collate)
            model = make_model(dims, list(store), len(classes)).to(device)
            count = np.bincount(y[inside], minlength=len(classes))
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(inside) / (len(classes) * np.maximum(count, 1)),
                                                              dtype=torch.float32, device=device))
            opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
            best, best_state, best_proba = (-1.0, -1.0), None, None
            for _ in range(t["epochs"]):
                model.train()
                for xs, ms, yb in loader:
                    opt.zero_grad()
                    loss = loss_fn(model([x.to(device) for x in xs], [m.to(device) for m in ms]), yb.to(device))
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
                    opt.step()
                sched.step()
                proba = predict_proba(model, valid, device)
                pred = proba.argmax(1)
                result = (float((pred == y[outside]).mean()), float(balanced_accuracy_score(y[outside], pred)))
                if result > best:
                    best, best_proba = result, proba
                    best_state = {name: v.detach().cpu().clone() for name, v in model.state_dict().items()}
            weights.append(best_state)
            oof[outside] += best_proba
            folds[outside] = k
    return weights, oof / len(seeds), folds


def ensemble_predict(cfg, store, samples, weights, device="cuda"):
    """Mean softmax over ``weights``; returns (probabilities, predicted class, Unknown flag)."""
    store = {ct: store[ct] for ct in cfg["celltypes"]}
    dims = {ct: store[ct]["X"].shape[1] for ct in store}
    loader = DataLoader(CellDataset(store, samples, [0] * len(samples)), batch_size=4, collate_fn=collate)
    probs = []
    for state in weights:
        model = make_model(dims, list(store), len(cfg["classes"])).to(device)
        model.load_state_dict(state)
        probs.append(predict_proba(model, loader, device))
    p = np.mean(probs, axis=0)
    return p, np.asarray(cfg["classes"])[p.argmax(1)], normalized_entropy(p) >= cfg["entropy_threshold"]
