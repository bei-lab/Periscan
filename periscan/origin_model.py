"""Cancer-origin network (dynamic cell-type gate), eight cancer classes.

Per cell type: cells -> gene encoder -> masked attention pooling; a gate re-weights the nine
cell-type vectors before the MLP head.
"""
import torch
import torch.nn as nn

EMB = 32
HEAD_HID = 128
DROP = 0.5
NHEAD = 4
FINAL_CFG = dict(ctx=False, heads=False, dyn=True, zero=False)


class GeneEncoder(nn.Module):
    def __init__(self, i, e, p, zero=False):
        super().__init__()
        self.zero = zero
        self.net = nn.Sequential(nn.Linear(i * (2 if zero else 1), e), nn.LayerNorm(e), nn.ReLU(), nn.Dropout(p))

    def forward(self, x):
        if self.zero:
            x = torch.cat([x, (x > 0).float()], -1)
        return self.net(x)


class AttnPool(nn.Module):
    def __init__(self, e, p, ctx=False, nhead=1):
        super().__init__()
        self.ctx = ctx
        self.nhead = nhead
        self.hd = e // nhead
        din = e * (2 if ctx else 1)
        self.score = nn.Sequential(nn.LayerNorm(din), nn.Linear(din, e), nn.Tanh(), nn.Dropout(p), nn.Linear(e, nhead))

    def forward(self, h, mask):
        m = mask.unsqueeze(-1).float()
        if self.ctx:
            mu = (h * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
            z = torch.cat([h, mu.expand_as(h)], -1)
        else:
            z = h
        score = self.score(z).masked_fill(~mask.unsqueeze(-1), -1e4)
        attn = torch.softmax(score, 1) * m
        attn = attn / attn.sum(1, keepdim=True).clamp(min=1e-6)
        if self.nhead == 1:
            return (h * attn).sum(1)
        batch, length, emb = h.shape
        out = (h.view(batch, length, self.nhead, self.hd) * attn.unsqueeze(-1)).sum(1)
        return out.reshape(batch, emb)


class OriginNet(nn.Module):
    def __init__(self, feature_dims, celltypes, n_classes, cfg=None, emb=EMB, hidden=HEAD_HID, p=DROP):
        super().__init__()
        self.celltypes = list(celltypes)
        self.cfg = cfg or dict(FINAL_CFG)
        self.genc = nn.ModuleDict({c: GeneEncoder(feature_dims[c], emb, p, self.cfg["zero"]) for c in self.celltypes})
        self.pool = nn.ModuleDict({c: AttnPool(emb, p, self.cfg["ctx"], NHEAD if self.cfg["heads"] else 1)
                                   for c in self.celltypes})
        if self.cfg["dyn"]:
            self.gate = nn.Sequential(nn.Linear(emb * len(self.celltypes), emb), nn.ReLU(), nn.Dropout(p),
                                      nn.Linear(emb, len(self.celltypes)))
        else:
            self.ctw = nn.Parameter(torch.zeros(len(self.celltypes)))
        self.head = nn.Sequential(nn.Linear(emb * len(self.celltypes), hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(hidden, n_classes))

    def forward(self, xs, masks):
        stacked = torch.stack([self.pool[c](self.genc[c](xs[i]), masks[i]) for i, c in enumerate(self.celltypes)], 1)
        if self.cfg["dyn"]:
            weights = torch.softmax(self.gate(stacked.flatten(1)), -1) * len(self.celltypes)
            stacked = stacked * weights.unsqueeze(-1)
        else:
            weights = torch.softmax(self.ctw, 0) * len(self.celltypes)
            stacked = stacked * weights.view(1, -1, 1)
        return self.head(stacked.flatten(1))


def make_model(feature_dims, celltypes, n_classes):
    return OriginNet(feature_dims, celltypes, n_classes, cfg=dict(FINAL_CFG))
