# Progress report

Running record of what we built, what we found and what we decided. Updated as we go.
Last updated 2026-11-20. Plan of record: `../Project_IDEA.md`.

---

## 1. The question

A model can learn about proteins without any lab measurements, by hiding part of a
structure and guessing what was there. We are testing **what kind of guess matters**:

- **Letter:** which of the 20 amino acids was hidden?
- **Vector:** what embedding would a slowly updated copy of the model give that position?
- **Both.**

Everything else is held fixed: the same graph, the same encoder, the same corpus, the same
masking rate, the same optimiser, the same seeds. Then each pretrained model is used to
predict how much a mutation destabilises a protein (ΔΔG), and we measure two things: how
well it predicts, and what its embeddings look like.

| Arm | Pretraining |
|---|---|
| A | none, trained from labels only |
| B | vector only (JEPA, EMA teacher) |
| C | letter only (masked residue) |
| D | letter + 0.45 × vector |

To our knowledge there is no controlled letter vs vector vs both comparison for a
**structure** graph network on stability. ProteinJEPA did it on sequences;
ProteinWorkshop pretrained on structure graphs but with corruption tasks, no JEPA and no
stability task.

---

## 2. Headline results so far

One seed, SCOPe corpus. Seeds 1 and 2 still to run.

| Arm | pooled Spearman | per-protein median | RMSE | effective rank |
|---|---|---|---|---|
| A — none (full) | 0.604 ± 0.006 | 0.642 ± 0.008 | 0.808 | — |
| B — vector (frozen) | 0.520 | 0.571 | 0.897 | **84.3** |
| B — vector (full) | 0.560 | 0.617 | 0.864 | |
| C — letter (frozen) | 0.641 | 0.696 | 0.778 | 61.6 |
| C — letter (full) | 0.631 | 0.675 | 0.786 | |
| **D — both (frozen)** | **0.643** | **0.700** | **0.758** | 62.6 |

Reference points, not targets: ThermoMPNN reports 0.725 on this test set with much heavier
pretraining, and 0.642 from scratch. The earlier Petnica project got 0.548 from scratch and
0.600 pretrained.

**Three findings.**

1. **Pretraining helps, but only with the right target.** Letter pretraining beats the
   no-pretraining baseline by 0.037 pooled and 0.054 per-protein, roughly three times the
   tie threshold. It does so with a **frozen** encoder that never saw a single ΔΔG value,
   against a baseline that trained its whole network on 215,000 labelled mutations.

2. **The wrong target actively hurts.** Vector-only pretraining scores 0.084 *below* random
   initialisation. Fine-tuned in full, under exactly the protocol the baseline uses, it is
   still 0.044 below. This is negative transfer: a vector-pretrained encoder is a worse
   starting point than no pretraining at all, and training only partly undoes it.

3. **Effective rank points the wrong way.** The arm with the **highest** rank performs
   **worst**. Ranking the arms by embedding rank gives exactly the reverse of ranking them
   by ΔΔG accuracy.

4. **A harmful objective becomes harmless in company.** Alone, the vector loss costs 0.084.
   Added at λ = 0.45 alongside the letter loss it costs nothing: arm D matches arm C within
   0.002 pooled and 0.004 per-protein, both far inside the tie threshold. It also prevents
   the collapse arm B fell into — at epoch 1 arm B's rank was 7.7 while arm D's was 47.3.
   The letter loss constrains the embedding space enough that the latent task cannot take
   the degenerate shortcut. ProteinJEPA found the combination best; we find it tied, so the
   same direction with a weaker effect.

   The final ordering is **D ≈ C > A > B**.

---

## 3. The rank result, in detail

This is the part we did not expect, and the part most worth writing about.

Effective rank (the RankMe quantity) measures how many of the 128 embedding dimensions a
model actually uses. It is the standard label-free diagnostic in self-supervised learning:
a collapsed model, where every input maps to nearly the same vector, scores near 1, and
the usual assumption is that higher rank means a better representation.

What we observed during arm B's pretraining:

| epoch | vector loss | effective rank |
|---|---|---|
| 0 | 0.285 | 12.3 |
| 1 | **0.0073** | **7.7** |
| 10 | 0.049 | 34.0 |
| 30 | 0.110 | 70.1 |
| 99 | 0.131 | 84.3 |

At epoch 1 the model **collapses**: it maps every residue to nearly the same vector, and
the loss falls almost to zero because predicting a constant is trivial. That rank of 7.7 is
the same 5–8 the earlier Petnica project reported and could not explain.

Then it **climbs back out**. Because the teacher is an EMA copy that trails the student, a
constant is not a stable solution; the representation spreads back out, the task gets
genuinely harder, and the loss **rises**.

**This explains the open question in the old project's README.** It recorded that the JEPA
loss "went up during training instead of down, which is the opposite of what you'd normally
hope for. We don't fully understand this yet." The answer: with a moving target, loss is
not comparable across epochs. A low loss can mean a useless representation. A rising loss
alongside a rising rank is the signature of **escaping** collapse.

And yet arm B, which ends with the highest rank of any arm, transfers worst. RankMe's own
paper warns that rank is only comparable between runs of the same method; this is a
concrete demonstration of why. A representation can spread across 84 dimensions and carry
nothing about stability. **Escaping collapse is not the same as learning something useful.**

A related observation from arm D: at epoch 1 its rank was 47.3, not 7.7. **The letter loss
prevents the collapse from happening at all** — which is ProteinJEPA's argument for the
combination, visible in our own run.

---

## 4. Hypotheses, as stated in advance

| ID | Prediction | Outcome |
|---|---|---|
| H1 | Vector-only is worst on ΔΔG | **Holds**, and more strongly than expected: worse than no pretraining |
| H2 | Both ≥ letter-only | **Holds, as a tie.** 0.643 vs 0.641 pooled, inside the seed spread |
| H3 | Vector-only has the lowest effective rank | **Fails.** It has the highest, 84.3 vs 61.6 |

H3 failing with a clear mechanism behind it is the most interesting outcome in the project.

---

## 5. What we built

### Data, and why we checked it so hard

- **Labels:** ThermoMPNN's MegaScale files — 271,231 single substitutions across 298
  proteins, split 239 / 31 / 28 by protein at 25% sequence identity. The ThermoMPNN paper
  quotes 272,712; the difference is 1,481 wild-type reference rows, which we drop.
- **Structures:** the Tsuboyama AlphaFold models from Zenodo, 862 files of which 298 are
  labelled, 32–72 residues, median 48.
- **Sign convention:** positive = stabilising, settled from the data rather than from a
  paper. In the validation file, mutations the lab itself flagged as stabilising have mean
  ΔΔG +1.40 and 77% of all values are negative, which matches the biology.

Labels and structures come from different sources, so we verified they agree before
training anything: all 298 structure sequences match the label file exactly, and all
271,231 mutations point at the right residue. A silent misalignment here would have ruined
every number downstream without raising an error.

### The graph

One node per residue at its Cα atom, joined to its 16 nearest neighbours in 3D, undirected.
Measured over all 298 proteins: 15,193 residues, 288,828 edges, average degree 19.

**62.5% of edges join residues more than 4 apart in the chain**, and only 10.3% are chain
neighbours. Most of the graph is contact information a sequence model cannot see. That is
the justification for using a graph at all.

Node features are backbone torsions (φ, ψ, ω) and two virtual Cα angles, each as
(sin, cos). Edge features are a distance RBF plus bucketed sequence separation.

**No feature may reveal which amino acid a residue is.** Only backbone atoms are read,
never side chains, whose shape is effectively the residue's fingerprint. A test enforces
this: relabel a residue, rebuild the graph, and every geometric feature must be
bit-identical.

### The encoder

4 layers of GINEConv, 128 dimensions, pre-norm residual blocks and a final layer norm.
Residue identity and backbone geometry are embedded separately and added, so masking only
has to replace the identity term and the geometry stays visible.

### Pretraining corpus

ASTRAL SCOPe 2.08, already filtered to 40% sequence identity: 15,177 domains → 11,791 after
a 30–256 residue length filter → **11,566** after removing anything resembling a MegaScale
protein. The leakage filter uses MMseqs2 at its most sensitive setting and removes *every*
hit at e-value ≤ 1e-3, not only those above 30% identity, because a remote homologue can
share a fold. It removed 225 domains, 1.9%, of which 218 were already above 30% identity —
so the stricter rule cost 7 extra domains.

The kept domains have a median of 126 residues against MegaScale's 48, so we pretrain on
somewhat larger proteins than we fine-tune on. Worth stating; not worth fixing.

### Fine-tuning

One item is one protein carrying all of its mutations, so a batch encodes a handful of
structures and scores every mutation on them. That turns 215,000 training examples into 239
encoder passes per epoch.

---

## 6. What counts as a real difference

Before running any pretrained arm, we ran the baseline three times with different random
seeds and measured the spread. That fixes the bar in advance, so it cannot be moved later
to suit a result.

| metric | arm A, 3 seeds |
|---|---|
| pooled Spearman | 0.604 ± 0.006 |
| per-protein median | 0.642 ± 0.008 |

**A difference under 0.012 pooled, or 0.016 per-protein, is a tie.** The experiment can
detect moderate effects. It cannot detect an effect of 0.01, and we should not claim one.

---

## 7. Bugs we found, and how

Worth writing about: each was caught by a test that checked a known quantity, not by
inspection.

**A metric that divided by the wrong number.** The first pretraining run reported a loss of
0.0046 with 0.000 accuracy — impossible together. The number of masked residues was stored
in the same dictionary as the metrics, so every quantity was weighted by itself and divided
by the sum of squares. Caught by writing `test_untrained_loss_is_near_chance`: an untrained
20-way classifier must start at ln(20) ≈ 3.0.

**An encoder whose outputs grew with depth.** That same test then failed at 4.34 instead of
3.0. Each residual block adds to the hidden state, and there was no final layer norm, so
the scale grew with depth. Adding one barely moved the mean score but **cut the per-protein
seed spread from 0.023 to 0.008**, three times more sensitive. It mattered more than it
looked, because arm B's cosine loss and the rank diagnostics both read those embeddings.

**A rate-limit lesson.** The AlphaFold download ran at 3.2 structures/s. More than half of
Swiss-Prot accessions have no AlphaFold model, and our code retried each 404 three times
with sleeps — most of the run was threads sleeping over answers already received. Fixing
that took it to 68/s, at which point AlphaFold DB blocked our IP for hours. The honest
summary: the slowness was our bug, but the server's sustainable rate is only a few
requests per second.

Tests now standing: 42 across data alignment, graph construction, the leak rule, masking,
batching, metrics, the EMA teacher and overfitting checks.

---

## 8. Decisions and why

| Decision | Choice | Reason |
|---|---|---|
| ΔΔG sign | positive = stabilising | read off the data, kept as the file has it, never flipped |
| Splits | ThermoMPNN's, unchanged | comparable with published numbers, no split design of our own to defend |
| k neighbours | 16, no distance cutoff | matches ProteinWorkshop; uniform degree; only ~1% of edges exceed 14.8 Å |
| Length filter | 30–256 | the plan first said 50–400, but MegaScale domains are 32–72, so 50 would exclude the size we care about |
| Leakage filter | every MMseqs2 hit, not only ≥30% | remote homologues share folds; it cost 7 extra domains |
| Pilot corpus | SCOPe before AlphaFold | minutes to obtain, proves the pipeline before spending hours |
| Tie threshold | 0.012 / 0.016 | two standard deviations of arm A, fixed before seeing any pretrained result |
| Final layer norm | added | loss started at 4.34 instead of ln(20); cut seed spread threefold |

---

## 9. Open, and what is next

**Still to run**

- Seeds 1 and 2 for every arm. Everything above is one seed. The gaps are large relative to
  the baseline spread, so they will very likely hold — but "very likely" is not a result.
  About 6 hours; `scripts/run_grid.sh` runs it with resume.
- `scripts/make_report.py` for the tables and figures.
- A README that describes this experiment rather than the earlier Petnica project.

**Known limitations to state plainly**

- 28 test proteins. Small, and the per-protein median over so few is noisy.
- One encoder, one stability dataset, one corpus, three seeds. A controlled small
  experiment, not a ranking of methods in general.
- The effective-rank sample is about 1,400 residues, not the 25,600 RankMe used — the test
  split simply does not contain that many. The sample size belongs in every table that
  reports rank.
- Pretraining structures are larger than the fine-tuning ones (median 126 vs 48).

**AlphaFold**

Downloading in the background; 6,254 structures so far out of a target of 50,000.
AlphaFold DB rate-limits sustained access, so this is a trickle rather than a sprint. It is
**not required for the main result** — SCOPe is a complete corpus and the findings above
stand on it. What it would add is a second, larger and more diverse corpus, and with it the
scaling curve the plan wants: does more pretraining data help, and does the gap between
arms change with corpus size? If the final grid runs on Colab, downloading there would
sidestep the rate limit entirely.
