"""Detection network (AD / CANCER / HC), its training loop and scoring.

Per cell type: cells -> encoder -> attention pooling -> LayerNorm, plus a gated
LayerNorm(encoder(mu)) residual; the nine cell-type vectors are concatenated into an MLP
head over AD / CANCER / HC.  Absent blocks contribute zeros.
"""
import math

import numpy as np
import torch
import torch.nn as nn

MASK_FILL = -1e9


def _encoder(in_dim, embed_dim, dropout):
    return nn.Sequential(nn.Linear(in_dim, embed_dim), nn.LayerNorm(embed_dim),
                         nn.ReLU(), nn.Dropout(dropout))


class DetectionNet(nn.Module):
    def __init__(self, gene_dims, celltypes, num_classes=3, embed_dim=128, attn_hidden=64,
                 head_hidden=128, dropout=0.4):
        super().__init__()
        self.celltypes = list(celltypes)
        cts = self.celltypes
        self.cell_encoders = nn.ModuleDict({ct: _encoder(gene_dims[ct], embed_dim, dropout) for ct in cts})
        self.scorers = nn.ModuleDict({ct: nn.Sequential(nn.Linear(embed_dim, attn_hidden), nn.Tanh(),
                                                        nn.Linear(attn_hidden, 1)) for ct in cts})
        self.mu_encoders = nn.ModuleDict({ct: _encoder(gene_dims[ct], embed_dim, dropout) for ct in cts})
        self.norm_attn = nn.ModuleDict({ct: nn.LayerNorm(embed_dim) for ct in cts})
        self.norm_mu = nn.ModuleDict({ct: nn.LayerNorm(embed_dim) for ct in cts})
        self.gate = nn.Parameter(torch.full((len(cts),), -3.0))
        self.head = nn.Sequential(nn.Linear(embed_dim * len(cts), head_hidden), nn.LayerNorm(head_hidden),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, num_classes))

    def forward(self, cells, mask, mu, block_mask):
        parts = []
        for i, ct in enumerate(self.celltypes):
            e = self.cell_encoders[ct](cells[ct])
            scores = self.scorers[ct](e).squeeze(-1).masked_fill(~mask[ct], MASK_FILL)
            pooled = torch.einsum("bm,bme->be", torch.softmax(scores, dim=1), e)
            h = self.norm_attn[ct](pooled) + torch.sigmoid(self.gate[i]) * self.norm_mu[ct](self.mu_encoders[ct](mu[ct]))
            parts.append(h * block_mask[:, i].unsqueeze(1).float())
        return self.head(torch.cat(parts, dim=1))


def to_tensors(features, celltypes, columns, device):
    """(cells, mask, mu, block_mask) tensors; ``columns[ct]`` picks the model's genes."""
    cells = {ct: torch.tensor(features.cells[ct][:, :, columns[ct]], dtype=torch.float32, device=device)
             for ct in celltypes}
    mu = {ct: torch.tensor(features.mu[ct][:, columns[ct]], dtype=torch.float32, device=device)
          for ct in celltypes}
    block = torch.tensor(features.mask, dtype=torch.bool, device=device)
    m = next(iter(cells.values())).shape[1]
    mask = {ct: block[:, i].unsqueeze(1).expand(-1, m).contiguous() for i, ct in enumerate(celltypes)}
    return cells, mask, mu, block


def batch(tensors, celltypes, sel):
    """Model arguments for rows ``sel``: (cells, mask, mu, block)."""
    cells, mask, mu, block = tensors
    return ({ct: cells[ct][sel] for ct in celltypes}, {ct: mask[ct][sel] for ct in celltypes},
            {ct: mu[ct][sel] for ct in celltypes}, block[sel])


def train_model(celltypes, gene_dims, arch, tensors, labels, rows, config, device, seed, class_order):
    """AdamW (gate not decayed), warmup then cosine, class-balanced cross-entropy."""
    torch.manual_seed(seed)
    model = DetectionNet(gene_dims, celltypes, len(class_order), **arch).to(device)
    index = {label: k for k, label in enumerate(class_order)}
    encoded = np.asarray([index[v] for v in np.asarray(labels)[rows]])
    counts = np.bincount(encoded, minlength=len(class_order)).astype(np.float64)
    weights = (1.0 / np.maximum(counts, 1)) ** config["class_weight_power"]
    weights = weights / weights.sum() * len(class_order)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    decayed = [p for name, p in model.named_parameters() if name != "gate"]
    gate = [p for name, p in model.named_parameters() if name == "gate"]
    optimizer = torch.optim.AdamW([{"params": decayed, "weight_decay": config["weight_decay"]},
                                   {"params": gate, "weight_decay": 0.0}], lr=config["lr"])
    warmup = int(config["warmup_epochs"])

    def schedule(epoch):
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(config["epochs"] - warmup, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    targets = torch.tensor(encoded, dtype=torch.long, device=device)
    order_rng = np.random.default_rng(seed)
    model.train()
    for _ in range(config["epochs"]):
        order = order_rng.permutation(len(rows))
        for start in range(0, len(order), config["batch_size"]):
            positions = order[start:start + config["batch_size"]]
            sel = torch.tensor(rows[positions], dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = criterion(model(*batch(tensors, celltypes, sel)), targets[positions])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
            optimizer.step()
        scheduler.step()
    return model


@torch.no_grad()
def score(model, celltypes, tensors, rows, device, batch_size=128):
    """Softmax probabilities (n, 3) in class order AD, CANCER, HC."""
    model.eval()
    parts = []
    for start in range(0, len(rows), batch_size):
        sel = torch.tensor(rows[start:start + batch_size], dtype=torch.long, device=device)
        args = batch(tensors, celltypes, sel)
        logits = model(*args)
        parts.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(parts)
