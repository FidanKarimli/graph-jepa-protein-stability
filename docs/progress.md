# Progress report

A running record of what I built, what I found and what I decided. Updated as I go.
Last updated 2026-11-20.

**Status.** The experiment is **complete**: 4 arms × 3 seeds on SCOPe, and the three
pretrained arms × 3 seeds again on a second corpus 2.25 times larger, all on one A100. Every
finding from the Mac pilot held, arm A reproduced across two different machines, and the
ordering survived the change of corpus. What remains is the write-up.

---

## 1. The question

A model can learn about proteins without any lab measurements, by hiding part of a structure
and guessing what was there. What I am testing is **what kind of guess matters**:

- **Letter:** which of the 20 amino acids was hidden?
- **Vector:** what embedding would a slowly updated copy of the model give that position?
- **Both.**

Everything else is held fixed: the same graph, the same encoder, the same corpus, the same
masking rate, the same optimiser, the same seeds. Then each pretrained model is used to
predict how much a mutation destabilises a protein (ΔΔG), and I measure two things: how well
it predicts, and what its embeddings look like.

| Arm | Pretraining |
|---|---|
| A | none, trained from labels only |
| B | vector only (JEPA, EMA teacher) |
| C | letter only (masked residue) |
| D | letter + 0.45 × vector |

As far as I know there is no controlled letter vs vector vs both comparison for a
**structure** graph network on stability. ProteinJEPA did it on sequences. ProteinWorkshop
pretrained on structure graphs but with corruption tasks, no JEPA and no stability task.

---

## 2. Headline results

**Three seeds per arm, two pretraining corpora, all on one A100.** Arms B, C and D fine-tune
a frozen encoder. Arm A has nothing to freeze, so it trains everything. Effective rank is
measured on a fixed 80,000 held-out pretraining residues in both corpora.

### SCOPe, 11,566 domains

| Arm | pooled Spearman | per-protein median | RMSE | effective rank |
|---|---|---|---|---|
| A (none) | 0.605 ± 0.008 | 0.643 ± 0.011 | 0.802 ± 0.011 | n/a |
| B (vector) | **0.506 ± 0.023** | 0.539 ± 0.018 | 0.903 ± 0.015 | **74.2 ± 2.3** |
| C (letter) | 0.640 ± 0.005 | 0.688 ± 0.005 | 0.767 ± 0.006 | 61.9 ± 0.5 |
| D (both) | **0.643 ± 0.008** | **0.692 ± 0.010** | 0.768 ± 0.005 | 62.6 ± 0.3 |

### AlphaFold, 26,057 structures (2.25 times larger)

| Arm | pooled Spearman | per-protein median | RMSE | effective rank |
|---|---|---|---|---|
| B (vector) | 0.543 ± 0.012 | 0.563 ± 0.029 | 0.894 ± 0.021 | **108.2 ± 2.4** |
| C (letter) | 0.639 ± 0.003 | 0.698 ± 0.004 | 0.784 ± 0.025 | 64.6 ± 0.8 |
| D (both) | 0.636 ± 0.002 | 0.705 ± 0.005 | 0.781 ± 0.012 | 65.3 ± 0.6 |

**More than twice the pretraining data changed almost nothing.** Arms C and D moved by 0.001
and 0.007 pooled, both inside the tie threshold. Arm B improved by 0.037, which is a real
gain by that threshold, and is still **0.062 below using no pretraining at all**. The
ordering C ≈ D > A > B holds on a second corpus that is larger and built differently.

Two checks the single-seed runs could not give me:

**The baseline reproduces across hardware.** Arm A scored 0.604 ± 0.006 on an Apple M4 Pro
and 0.605 ± 0.008 on an A100. Different chips, different floating-point arithmetic, same
answer. That is a strong check on the whole pipeline, and the reason I re-ran everything
instead of mixing machines.

**Arm B is not only worse, it is less stable.** Its seed spread is ±0.023 on SCOPe against
arm C's ±0.005, and its per-protein spread is ±0.029 on AlphaFold against ±0.004. The vector
objective produces features that are both weaker and more dependent on initialisation.

The Mac pilot, kept for comparison, also ran each arm with a full (unfrozen) encoder:

| Arm, pilot, seed 0 | frozen | full |
|---|---|---|
| A, none | n/a | 0.609 |
| B, vector | 0.520 | 0.560 |
| C, letter | 0.641 | 0.631 |

Arm C is better frozen and arm B better unfrozen. That is itself evidence that the
difference between them is feature quality rather than protocol: good features are worth
keeping, bad ones are worth overwriting.

Reference points, not targets: ThermoMPNN reports 0.725 on this test set with much heavier
pretraining, and 0.642 from scratch. The earlier Petnica project got 0.548 from scratch and
0.600 pretrained.

**Six findings.**

1. **Pretraining helps, but only with the right target.** Letter pretraining beats the
   no-pretraining baseline by 0.035 pooled and 0.045 per-protein, several times the tie
   threshold. It does this with a **frozen** encoder that never saw a single ΔΔG value,
   against a baseline that trained its whole network on 215,000 labelled mutations.

2. **The wrong target hurts.** Vector-only pretraining scores 0.099 *below* random
   initialisation. In the pilot, fine-tuned in full under the same protocol the baseline
   uses, it was still 0.044 below. This is negative transfer. A vector-pretrained encoder is
   a worse starting point than no pretraining at all, and training only partly undoes it.

3. **Effective rank points the wrong way.** The arm with the **highest** rank performs
   **worst**. Ranking the arms by embedding rank gives exactly the reverse of ranking them by
   ΔΔG accuracy.

4. **A harmful objective becomes harmless in company.** Alone, the vector loss costs 0.084.
   Added at λ = 0.45 alongside the letter loss it costs nothing: arm D matches arm C within
   0.002 pooled and 0.004 per-protein, both far inside the tie threshold. It also prevents
   the collapse arm B fell into. At epoch 1 arm B's rank was 7.7 while arm D's was 47.3. The
   letter loss constrains the embedding space enough that the latent task cannot take the
   degenerate shortcut. ProteinJEPA found the combination best. I find it tied, so the same
   direction with a weaker effect.

   The final ordering is **D ≈ C > A > B**.

5. **More data does not rescue the latent objective, and does not help the letter one
   either.** On a corpus 2.25 times larger, arms C and D are unchanged within the tie
   threshold, and arm B gains 0.037 while remaining 0.062 below no pretraining at all. The
   usual defence of JEPA methods, that they just need more data, does not hold here at this
   scale.

6. **The pretext task improved a lot while the downstream task did not move at all.** On the
   larger corpus arm C's held-out letter accuracy rose from **0.267 to 0.353**, a 32%
   relative improvement at the task it was trained on. Its ΔΔG score went from 0.640 to
   0.639. Arm D is the same: 0.266 to 0.350, and 0.643 to 0.636.

   So the model got much better at predicting hidden residues and no better at predicting
   stability. Pretext-task performance is not a proxy for downstream value. Together with
   finding 3, that means neither of the two things you would normally measure without labels,
   pretext loss and embedding rank, predicted what I cared about.

---

## 3. The rank result, in detail

This is the part I did not expect, and the part most worth writing about.

Effective rank (the RankMe quantity) measures how many of the 128 embedding dimensions a
model actually uses. It is the standard label-free diagnostic in self-supervised learning. A
collapsed model, where every input maps to nearly the same vector, scores near 1, and the
usual assumption is that higher rank means a better representation.

What I saw during arm B's pretraining:

| epoch | vector loss | effective rank |
|---|---|---|
| 0 | 0.285 | 12.3 |
| 1 | **0.0073** | **7.7** |
| 10 | 0.049 | 34.0 |
| 30 | 0.110 | 70.1 |
| 99 | 0.131 | 84.3 |

At epoch 1 the model **collapses**. It maps every residue to nearly the same vector, and the
loss falls almost to zero because predicting a constant is easy. That rank of 7.7 is the
same 5 to 8 the earlier Petnica project reported and could not explain.

Then it **climbs back out**. Because the teacher is an EMA copy that trails the student, a
constant is not a stable solution. The representation spreads back out, the task gets
genuinely harder, and the loss **rises**.

**This explains the open question in the old project's README.** It recorded that the JEPA
loss "went up during training instead of down, which is the opposite of what you'd normally
hope for. We don't fully understand this yet." The answer is that with a moving target, loss
is not comparable across epochs. A low loss can mean a useless representation. A rising loss
alongside a rising rank is the signature of **escaping** collapse.

And yet arm B, which ends with the highest rank of any arm, transfers worst. RankMe's own
paper warns that rank is only comparable between runs of the same method, and this is a
concrete demonstration of why. A representation can spread across 84 dimensions and carry
nothing about stability. **Escaping collapse is not the same as learning something useful.**

A related observation from arm D: at epoch 1 its rank was 47.3, not 7.7. **The letter loss
prevents the collapse from happening at all**, which is ProteinJEPA's argument for the
combination, visible here in my own run.

**The larger corpus makes the reversal starker.** On AlphaFold, arm B collapses again at
epoch 1 (rank 10.7, so this is a property of the method rather than of SCOPe) and then climbs
to **106 of 128 dimensions**, far higher than on SCOPe. It is still the worst arm by 0.1
Spearman. Meanwhile arm C's rank *falls* during training, from 71.5 at epoch 1 down to 65.3.
The letter objective concentrates the representation, the latent objective spreads it. One of
those produces features that transfer, and it is not the one that looks healthier by the
standard diagnostic.

---

## 4. Hypotheses, as stated in advance

| ID | Prediction | Outcome |
|---|---|---|
| H1 | Vector-only is worst on ΔΔG | **Holds**, and more strongly than I expected: 0.099 below no pretraining at all |
| H2 | Both ≥ letter-only | **Holds, as a tie.** 0.643 vs 0.640, inside the seed spread |
| H3 | Vector-only has the lowest effective rank | **Fails.** It has the highest, 74.2 vs 61.9 |

H3 failing, with a clear mechanism behind it, is the most interesting outcome in the project.

---

## 5. What I built

### Two datasets, and why both are needed

The premise of the whole project is an asymmetry: protein **shapes** are abundant and free,
protein **stability measurements** are scarce and expensive. If practising on the first makes
a model better at the second, that is worth knowing.

| | MegaScale | SCOPe / AlphaFold |
|---|---|---|
| what | 298 proteins with lab-measured ΔΔG | 11,566 / 25,000 structures with no labels |
| role | the task itself: train, validate, test | pretraining: learn what proteins look like |
| why this size | every value cost lab work | free, and limited only by download time |

**MegaScale** (Tsuboyama et al. 2023) measured the stability of every single-residue mutant
of hundreds of small protein domains, 271,231 measurements in all. That is the task.

**SCOPe** is a classification of real experimentally determined protein domains. I use the
ASTRAL subset already filtered to 40% sequence identity, so the corpus is not dominated by
near-duplicates of the same fold.

**AlphaFold DB** is DeepMind's prediction of the 3D shape of essentially every known protein,
about 214 million of them, published free. I sample from Swiss-Prot, the hand-curated part of
UniProt, and keep structures AlphaFold itself was confident about (mean pLDDT ≥ 70, which it
writes into the B-factor column of every file).

A detail worth noticing: the MegaScale structures are *also* AlphaFold predictions. They are
simply a small set that happens to have lab measurements attached.

### Data, and why I checked it so hard

- **Labels:** ThermoMPNN's MegaScale files, 271,231 single substitutions across 298 proteins,
  split 239 / 31 / 28 by protein at 25% sequence identity. The ThermoMPNN paper quotes
  272,712. The difference is 1,481 wild-type reference rows, which I drop.
- **Structures:** the Tsuboyama AlphaFold models from Zenodo, 862 files of which 298 are
  labelled, 32 to 72 residues, median 48.
- **Sign convention:** positive = stabilising, settled from the data rather than from a paper.
  In the validation file, mutations the lab itself flagged as stabilising have mean ΔΔG +1.40,
  and 77% of all values are negative, which matches the biology.

Labels and structures come from different sources, so I verified they agree before training
anything. All 298 structure sequences match the label file exactly, and all 271,231 mutations
point at the right residue. A silent misalignment here would have ruined every number
downstream without raising an error.

### The graph

One node per residue at its Cα atom, joined to its 16 nearest neighbours in 3D, undirected.
Measured over all 298 proteins: 15,193 residues, 288,828 edges, average degree 19.

**62.5% of edges join residues more than 4 apart in the chain**, and only 10.3% are chain
neighbours. Most of the graph is contact information a sequence model cannot see. That is the
justification for using a graph at all.

Node features are backbone torsions (φ, ψ, ω) and two virtual Cα angles, each as (sin, cos).
Edge features are a distance RBF plus bucketed sequence separation.

**No feature may reveal which amino acid a residue is.** Only backbone atoms are read, never
side chains, whose shape is effectively the residue's fingerprint. A test enforces this:
relabel a residue, rebuild the graph, and every geometric feature must come out
bit-identical.

### The encoder

4 layers of GINEConv, 128 dimensions, pre-norm residual blocks and a final layer norm.
Residue identity and backbone geometry are embedded separately and added, so masking only has
to replace the identity term and the geometry stays visible.

### Pretraining corpus

ASTRAL SCOPe 2.08, already filtered to 40% sequence identity: 15,177 domains, then 11,791
after a 30 to 256 residue length filter, then **11,566** after removing anything resembling a
MegaScale protein. The leakage filter uses MMseqs2 at its most sensitive setting and removes
*every* hit at e-value ≤ 1e-3, not only those above 30% identity, because a remote homologue
can share a fold. It removed 225 domains, 1.9%, of which 218 were already above 30% identity.
So being strict cost me 7 extra domains.

The kept domains have a median of 126 residues against MegaScale's 48, so I pretrain on
somewhat larger proteins than I fine-tune on. Worth stating, not worth fixing.

### Fine-tuning

One item is one protein carrying all of its mutations, so a batch encodes a handful of
structures and scores every mutation on them. That turns 215,000 training examples into 239
encoder passes per epoch.

---

## 6. What counts as a real difference

Before running any pretrained arm, I ran the baseline three times with different random seeds
and measured the spread. That fixes the bar in advance, so it cannot be moved later to suit a
result.

| metric | arm A, 3 seeds |
|---|---|
| pooled Spearman | 0.604 ± 0.006 |
| per-protein median | 0.642 ± 0.008 |

**A difference under 0.012 pooled, or 0.016 per-protein, is a tie.** The experiment can detect
moderate effects. It cannot detect an effect of 0.01, and I should not claim one.

---

## 7. Bugs I found, and how

Worth writing about, because each one was caught by a test that checked a known quantity, not
by reading the code.

**A metric that divided by the wrong number.** The first pretraining run reported a loss of
0.0046 with 0.000 accuracy, which cannot both be true. The number of masked residues was
stored in the same dictionary as the metrics, so every quantity was weighted by itself and
divided by the sum of squares. Caught by writing `test_untrained_loss_is_near_chance`: an
untrained 20-way classifier must start at ln(20) ≈ 3.0.

**An encoder whose outputs grew with depth.** That same test then failed at 4.34 instead of
3.0. Each residual block adds to the hidden state, and there was no final layer norm, so the
scale grew with depth. Adding one barely moved the mean score but **cut the per-protein seed
spread from 0.023 to 0.008**, three times more sensitive. It mattered more than it looked,
because arm B's cosine loss and the rank diagnostics both read those embeddings.

**A rate limit, misdiagnosed twice.** The AlphaFold download crawled at 3.2 structures/s with
over half of all attempts coming back as skips. I blamed missing models. The code retried
every failure three times with sleeps, so a protein AlphaFold simply does not have cost three
seconds of waiting. Taking that retry out pushed me to 68 requests a second, and AlphaFold DB
thanked me by blocking my IP for several hours, because 24 parallel requests is far past what
it allows. Fair, in hindsight.

The final clean run shows the first diagnosis was also wrong. Of 30,000 attempts, **87%
succeeded**. Only 3.8% were missing models, 6.4% failed the pLDDT ≥ 70 filter and 2.9% were
rate-limited. The original 55% skip rate was mostly the server throttling me, which I had been
counting as "no model here".

Two lessons worth keeping. A 404 and a 429 look alike in a skip counter and mean opposite
things. And a burst benchmark tells you what a server *can* serve, not what it will tolerate.
The sustainable rate was about 3 requests per second.

**A .gitignore line that hid a whole package.** `data/` had been quietly excluding
`src/pobj/data/` as well, so the entire data package was never on GitHub. I found this out on
the A100, mid-run, when the import failed. The fix is one slash: `/data/`.

Tests now standing: 45 across data alignment, graph construction, the leak rule, masking,
batching, metrics, the EMA teacher and overfitting checks.

---

## 8. Decisions and why

| Decision | Choice | Reason |
|---|---|---|
| ΔΔG sign | positive = stabilising | read off the data, kept as the file has it, never flipped |
| Splits | ThermoMPNN's, unchanged | comparable with published numbers, and no split design of my own to defend |
| k neighbours | 16, no distance cutoff | matches ProteinWorkshop; uniform degree; only ~1% of edges exceed 14.8 Å |
| Length filter | 30 to 256 | the plan first said 50 to 400, but MegaScale domains are 32 to 72, so 50 would exclude the size I care about |
| Leakage filter | every MMseqs2 hit, not only ≥30% | remote homologues share folds; it cost 7 extra domains |
| Pilot corpus | SCOPe before AlphaFold | minutes to obtain, proves the pipeline before spending hours |
| Tie threshold | 0.012 / 0.016 | two standard deviations of arm A, fixed before seeing any pretrained result |
| Final layer norm | added | loss started at 4.34 instead of ln(20); cut seed spread threefold |

---

## 9. Open, and what is next

### Done (2026-11-20 to 21, A100 80 GB)

The whole experiment was re-run on one machine. The same seed gives slightly different results
on Apple MPS and on CUDA, so mixing hardware inside a comparison would put noise into exactly
the thing being measured. The Mac runs became the pilot, and re-running them doubled as a
reproducibility check, which arm A passed.

SCOPe grid, 4 arms × 3 seeds, about 2 hours. 6 to 7.7 seconds per pretraining epoch against 24
to 33 on the Mac, so roughly 4 times faster. The JEPA arms cost about 1.2 times the letter
arm, from the teacher's extra forward pass.

AlphaFold grid, arms B/C/D × 3 seeds on 26,057 structures, 253 minutes. About 19 s per epoch
against SCOPe's 7.8, scaling with the corpus as expected.

### Still to do

- The blog post.

**Known limitations to state plainly**

- 28 test proteins. That is not many, and a median over so few is noisy.
- One encoder, one stability dataset, one corpus, three seeds. A controlled small experiment,
  not a ranking of methods in general.
- Effective rank is measured on 80,275 held-out **pretraining** residues, not on MegaScale
  test residues as the plan first proposed. The 28 test proteins hold only about 1,400
  residues, far fewer than the 25,600 RankMe used. Measuring on the pretraining validation set
  is consistent across arms and gives a large enough sample, but it describes the embedding
  space the model was trained in rather than the one it transfers to. The sample size belongs
  in every table that reports rank.
- Pretraining structures are larger than the fine-tuning ones (median 126 vs 48).

### The second corpus

AlphaFold is **not required for the main result**. SCOPe is a complete corpus and every
finding above rests on it. What AlphaFold adds is the one question a single corpus cannot
answer: does the ordering survive a different, larger corpus?

**Collected: 26,057 structures**, from 30,000 attempts (87% success; 6.4% below the pLDDT
threshold, 3.8% with no model, 2.9% rate-limited). That is **2.25 times SCOPe's 11,566**.

| | SCOPe | AlphaFold |
|---|---|---|
| structures | 11,566 | 26,057 |
| kind | real, experimentally determined domains | predicted structures of curated proteins |
| median length | 126 | 157 |
| steps per epoch at batch 32 | 344 | 774 |

**Four things change at once**, and that limits what a difference between the two runs can be
attributed to: corpus size, corpus type, protein length, and the number of optimiser steps,
which rises 2.25 times as well because epochs are held at 100 while the corpus is larger. A
result here says "the ordering held, or did not, on a second corpus". It is **not** a scaling
curve, which would need the same corpus at two sizes.

Two of three preparations were made before running it, and one was deliberately skipped:

1. **The corpus is pinned.** `data/processed/alphafold_ids.txt` lists all 26,057 accessions,
   with a summary in `alphafold_corpus.md`. Without it the corpus would have been only
   "whatever is in the shards", and the 878 rate-limited accessions are marked retryable, so a
   later download run would have grown it silently.
2. **The effective-rank sample is fixed at 80,000 residues** in both corpora (`eval/rank.py`),
   chosen so SCOPe's already-reported ranks are unaffected, since its validation slice held
   80,275.
3. **Epochs were held at 100 rather than matched on steps**, so the AlphaFold runs took 2.25
   times more optimiser updates. Since arms C and D did not improve despite both more data and
   more training, that confound does not threaten the conclusion. It would matter if I were
   claiming a gain. A step-matched run of arm B (about 44 epochs) would settle whether its
   +0.037 comes from the data or the extra steps. Not run.

### Ideas I have not run

Listed so they are not mistaken for things I tried and hid: a λ sweep for arm D (0.1 against
0.45), label efficiency at 1% and 10% of the training mutations, a frozen random encoder as
arm E, longer pretraining (arm B's rank was still rising at epoch 99), and linear probes for
burial or secondary structure to say *what* the letter arm learned that the vector arm did
not.
