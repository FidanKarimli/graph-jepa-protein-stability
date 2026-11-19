# Graph-JEPA for Protein Stability Prediction

We started this project at the Petnica Summer Institute (Scientific Machine Learning, August 2026),
and we liked it enough that we wanted to keep going. This repo is where we do that.

**Authors:** Fidan Karimli and Pau Martínez

[![Slides](https://img.shields.io/badge/Slides-Canva-00C4CC?logo=canva&logoColor=white)](https://canva.link/7dgujdjexcuhrbe)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Why one tiny change can break a protein

Imagine a necklace made of hundreds of beads, where every bead is one of 20 types. Some beads
are sticky, some repel water, some carry a charge, some are bulky and some are tiny. Now let the
necklace crumple up on its own. The beads pull and push on each other until the chain settles
into one precise 3D shape. That is a protein, the beads are amino acids, and the shape is
everything: it is what lets an enzyme grip its target or an antibody recognize a virus.

Now swap a single bead. Most of the time the necklace doesn't care. But swap the wrong bead in
the wrong place, say a tiny one for a bulky one in the tightly packed core, and the whole shape
can loosen and fall apart. That swap is a **mutation**, and the number that tells us how much
it shakes the protein is **ΔΔG**: the stability of the mutant minus the stability of the original
(ΔΔG = ΔG<sub>mutant</sub> − ΔG<sub>wild-type</sub>). In MegaScale, a negative value means the
mutation destabilizes the protein.

Measuring ΔΔG in a lab means making the mutant and testing it, one at a time. Predicting it from
the structure alone would save a huge amount of work. That is the game we are playing here.

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

To make this concrete, here is one real protein from SCOP, first as a fold and then as the graph
the model actually sees (233 residues, 4,288 edges, about 18 neighbours per residue):

![A protein as a fold and as a graph](figures/protein_as_graph.png)

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

These are our pretraining diagnostics. The JEPA loss (left) rises after the first epochs,
the embedding spread stays above zero (middle), but the effective rank (right) stays low:

![Pretraining diagnostics: JEPA loss, embedding std, effective rank](figures/pretraining_diagnostics.png)

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