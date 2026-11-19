# Graph-JEPA for Protein Stability Prediction

We started this project at the Petnica Summer Institute (Scientific Machine Learning, August 2026),
and we liked it enough that we wanted to keep going. This repo is where we do that.

**Authors:** Fidan Karimli and Pau Martínez

## What we're trying to do

A single mutation can change one amino acid in a protein, and that can be enough to make the
protein fall apart. Measuring how much a mutation changes stability (ΔΔG) in the lab is slow and
expensive. Protein structures, on the other hand, are plentiful and have no labels at all.

So we asked a simple question: if a model first learns from lots of unlabeled structures, does it
need less labeled data to predict stability afterwards?

To try this, we:

1. Turn each protein into a graph. Residues are nodes, and residues that are close in 3D are connected.
2. Pretrain a graph neural network (GINE) on 6,782 unlabeled structures from SCOP, using a
   Joint-Embedding Predictive Architecture (JEPA). The model learns to predict the embedding of a
   hidden part of the protein from the rest.
3. Fine-tune on MegaScale (Tsuboyama et al., Nature 2023): 271,231 stability measurements across
   862 proteins. The train and test sets share no proteins.

## What we found so far

| Model | RMSE | Spearman | Spearman per protein |
|---|---|---|---|
| from scratch | 0.7084 ± 0.0140 | 0.5476 ± 0.0135 | 0.5946 ± 0.0125 |
| pretrained | 0.6798 ± 0.0122 | 0.5997 ± 0.0035 | 0.6307 ± 0.0098 |

Pretraining helped on every metric. We're glad about that, but we don't want to oversell it:

- We only pretrained for 20 epochs. That is short.
- The embeddings look partly collapsed. The effective rank was around 5–8 out of 128 dimensions,
  so the model is using only a small part of the space it has. The JEPA loss also went up during
  training instead of down, which is the opposite of what you'd normally hope for. We don't fully
  understand this yet.
- Our setup is small. For reference, ThermoMPNN (PNAS 2024) gets Spearman 0.642 from scratch and
  0.725 pretrained on this same data and split, but it pretrains on far more data. We don't expect
  to match that.
- TODO: how many seeds the ± values come from, and exact training settings.

## What we want to do next

- [ ] Reproduce the baseline cleanly, with fixed seeds and the exact protocol written down
- [ ] Understand and fix the collapse (and write up what we tried, including what didn't work)
- [ ] Compare architectures: GINEConv vs GAT vs Graph Transformer
- [ ] Use the full backbone geometry (N, Cα, C, O atoms) instead of just residues
- [ ] See how much pretraining data matters (a scaling curve)
- [ ] See how much labeled data pretraining saves us (a label-efficiency curve)
- [ ] Write a blog post about all of it

## Running it

This project builds on the course library, which we don't copy here:

```bash
pip install -r requirements.txt
```

The data downloads itself the first time you use it.

## Credits and thanks

Thank you to our mentor and the Petnica team for the course materials and code this builds on:
https://github.com/zenwor/sciml26_gjepa

Our team repo from the course: https://github.com/paumartinez160/sciml26_gjepa

Ideas we're building on: LeCun (2022), Assran et al. (2023, I-JEPA), Hu et al. (2020, GINE),
Tsuboyama et al. (2023, MegaScale).

## License

Our own code here is MIT licensed (see LICENSE). The course library keeps its own terms.