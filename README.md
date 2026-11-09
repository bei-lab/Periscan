# PERISCAN

## Overview

PERISCAN is an attention-based deep learning framework that reads peripheral immune dynamics from single-cell RNA sequencing of peripheral blood mononuclear cells (PBMCs). It detects cancer and predicts its tissue of origin. Cancer perturbs immune homeostasis throughout the body, so its signature can be read from circulating immune cells, using only a standard blood draw.

The framework has two models that share one input format and one code base:

- **Detection**: classifies a sample as cancer, autoimmune disease (AD) or healthy control (HC).
- **Origin**: assigns a cancer sample to one of eight cancer types: BLCA, CESC, ESCA, HNSC, LIHC, LUCA, NPC, PAAD. Samples whose prediction is too uncertain are reported as *Unknown*.

## Key Features

- **Cancer detection**: on the held-out test set (593 samples), cancer sensitivity 97.35% and specificity 95.15% (AUC 0.979).
  - Across 1,402 healthy controls from two independent public cohorts, 95.1% are correctly called healthy.
- **Tissue-of-origin prediction**: on the held-out test set (393 samples), top-1 accuracy 70.73% after removing 7 Unknown calls, top-3 accuracy 89.64%, and micro-averaged AUC 0.916.
- **Uncertainty-aware output**: the origin model abstains (*Unknown*) when its prediction entropy is high.
- **Cell-type resolution**: both models work on nine immune cell populations. Each population is encoded separately before fusion.
- **Clinical applicability**: requires only PBMCs from a standard blood draw.

## Input Data

Both models consume the same per-sample object: single-cell expression of nine immune cell
populations, projected onto one shared candidate gene pool. The nine populations are

- naive and central-memory CD4+ T cells (`CD4_Naive-CCR7`, `CD4_TCM-GPR183`);
- naive, effector-memory and terminal effector CD8+ T cells (`CD8_Naive-CCR7`, `CD8_TEM-GZMK`, `CD8_Temra-FGFBP2`);
- two cytotoxic NK subsets (`CD56lowCD16high NK-LAG3`, `CD56lowCD16high NK-TYROBP`);
- classical monocytes (`CD14+ Monocyte`);
- naive B cells (`NaiveB-TCL1A`).

**Detection** takes a `Features` object that holds, per sample and cell type:

| field | shape | meaning |
|---|---|---|
| `mu` | `n_samples × n_genes` | mean expression of the sample's cells of that type |
| `cells` | `n_samples × m × n_genes` | `m` sampled cells of that type, padded to a shared `m` |
| `mask` | `n_samples × 9` | cell type present; an absent block contributes zeros to the head |
| `genes` | per cell type | gene names of the columns, carried into every checkpoint |

Gene selection is part of the fold loop: the top 100/300 genes of each cell type are ranked
by total gain, and the network is then fitted on that gene set.

**Origin** takes a `store` dict: for each cell type a sparse matrix `X` holding the cells of all
samples, plus `ranges[sample] = (start, stop)` locating that sample's rows. During training a
sample keeps at most the 95th percentile of its per-cell-type cell counts, drawn as a fixed random
subset per sample; at prediction time all of its cells are used.

## Model Architecture

Both models process a sample at three levels:

- **Gene Encoder**: cell-type-specific linear projection, layer normalisation and nonlinear activation map each cell's expression vector to an embedding.
- **Cell Encoder**: attention-weighted pooling aggregates the cells of each immune cell type into a population-level representation.
  - The detection model adds a gated encoding of the cell type's mean expression profile.
- **Feature Fusion**: the cell-type representations are combined into a sample-level representation. Fully connected layers then produce class probabilities.
  - The origin model re-weights cell types per sample with a learned gate before fusion.

The origin model is an ensemble of five fold models whose probabilities are averaged.

## Code

```text
periscan/
  config.json         cell types, classes, architecture and training settings
  detection_model.py  detection network, training loop, scoring
  detection.py        five-fold training: per-fold gene selection, decision threshold
  origin_model.py     origin network
  origin.py           five-fold training per seed, epoch chosen on the fold's validation split
```

```python
from periscan import detection, origin

# training samples only; both return out-of-fold probabilities
checkpoints, det_oof, det_folds = detection.train(cfg["detection"], features, labels, strata)
weights, org_oof, org_folds = origin.train(cfg["origin"], store, train_ids, train_labels)

# held-out test samples are touched here, after training
proba, pred, unknown = origin.ensemble_predict(cfg["origin"], store, test_ids, weights)
```

Install the dependencies with `pip install -r requirements.txt` (Python 3.12).

## Reproducible Execution

Data preparation, the trained model weights, inference on example data and the scripts that reproduce the published tables are available in a fully configured environment on Code Ocean:

🔗 https://codeocean.com/capsule/3575314/

## Citation
If you use PERISCAN in your research, please cite:



## License
Repository: https://github.com/bei-lab/Periscan
MIT; see `LICENSE`.
Contact: Jin-Xin Bei — beijx@sysucc.org.cn