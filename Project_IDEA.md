# Does it matter what you predict? Latent vs masked-residue pretraining for protein stability

Status: in progress. Step 1 of the build order (section 12) is underway: labels and structures are downloaded and verified, the loader and its tests are written. Nothing trained yet. Last edited 2026-11-20.

How to read this file:

- **(suggested)** marks a default I chose. It is not taken from a paper. Change it if you have a reason.
- **TODO** marks a decision or a fact to check before relying on it.
- Numbers from papers are listed with their source in the References section at the end.

---

## 1. The idea in one paragraph

A graph network can learn about protein structures without any stability labels by hiding part of a protein and guessing it. We compare two kinds of guess:

- **Letter (masked residue):** which amino acid was hidden?
- **Summary vector (JEPA):** what vector would a slowly updated teacher copy of the network assign to the hidden residue?

We also test both together. Every variant shares the same graph, encoder, data and settings, so any difference comes from the pretraining target alone. Each variant is then used to predict mutation stability (ΔΔG), and we measure two things: how well it predicts ΔΔG, and what its internal embeddings look like.

## 2. Research question and expectations

**Question.** For a structure graph network, does the pretraining target (letter, vector, or both) change ΔΔG accuracy and the quality of the embeddings?

| ID | Expectation | Reason |
|---|---|---|
| H1 | Vector-only is worst on ΔΔG | ProteinJEPA (sequences) found JEPA without the residue loss collapsed |
| H2 | Both together matches or beats letter-only | the same paper found the combination worked best |
| H3 | Vector-only uses the fewest embedding dimensions (lowest effective rank) | JEPA is prone to dimensional collapse |

These are predictions, not facts. If they fail, that is still a result.

**Novelty claim, worded carefully.** ProteinJEPA tested this on sequences with frozen linear probes on averaged embeddings. ProteinWorkshop tested pretraining on structure graphs with corruption-style tasks, with no JEPA and no stability task. We found no controlled letter vs vector vs both comparison for a structure graph network on ΔΔG. We cannot prove none exists, so write "to our knowledge".

## 3. Experiment at a glance

| Arm | Pretraining | Purpose |
|---|---|---|
| A | none, train from scratch | baseline |
| B | vector only (JEPA, EMA teacher) | the latent idea |
| C | letter only | the standard alternative |
| D | letter + λ × vector | the combination |

Optional extras, only after A to D work:

- E: frozen random encoder (shows what pretraining adds at all)
- full fine-tuning of the encoder, not only a frozen encoder
- 1% and 10% of the labels
- a λ sweep

Fixed across all arms: graph construction, encoder, masking rate, pretraining data, pretraining length, ΔΔG head, optimizer, seeds.

## 4. Repository layout (suggested)

```
protein-objective/
├── README.md
├── pyproject.toml            # or requirements.txt
├── configs/
│   ├── base.yaml             # shared settings
│   ├── arm_a.yaml            # one small file per arm, only differences
│   ├── arm_b.yaml
│   ├── arm_c.yaml
│   └── arm_d.yaml
├── data/                     # gitignored
│   ├── raw/                  # downloads
│   └── processed/            # cached graphs
├── src/pobj/
│   ├── data/
│   │   ├── megascale.py      # load csv, splits, sign convention
│   │   ├── pretrain_set.py   # structure subset + similarity filter
│   │   ├── graph.py          # coordinates -> graph
│   │   ├── features.py       # node and edge features
│   │   └── masking.py        # choose masked nodes, apply mask
│   ├── models/
│   │   ├── encoder.py        # GNN encoder
│   │   ├── heads.py          # letter head, predictor, ddG head
│   │   └── ema.py            # teacher update
│   ├── losses.py
│   ├── train/
│   │   ├── pretrain.py
│   │   └── finetune.py
│   └── eval/
│       ├── metrics.py        # Spearman, per-protein Spearman, RMSE
│       ├── rank.py           # effective rank
│       └── probes.py         # optional
├── scripts/
│   ├── download_megascale.sh
│   ├── build_pretrain_set.sh
│   ├── run_grid.sh
│   └── make_report.py
├── tests/
├── results/                  # csv per run, figures
└── docs/
    └── blog_outline.md
```

## 5. Environment (suggested)

- Python 3.10 or newer.
- PyTorch and PyTorch Geometric (provides `GINEConv` and kNN graph utilities).
- numpy, scipy, pandas, scikit-learn, matplotlib, pyyaml, tqdm.
- Biopython or a small PDB parser for coordinates.
- MMseqs2 for the sequence similarity filter (command line tool).
- Pin versions in `pyproject.toml` after the first working run.
- Set a global seed function that seeds Python, numpy and torch, and log the seed in every result row.

## 6. Data

### 6.1 Fine-tuning and testing: MegaScale ΔΔG

- Use the **ThermoMPNN version** of MegaScale. It has 272,712 single mutations across 298 proteins (ThermoMPNN preprint).
- ThermoMPNN provides train, validation and test splits made at a 25% sequence identity cutoff, roughly 216k training and 28k test mutations (as described in the HERMES paper). Use these splits as they are. They keep test proteins fixed and make our numbers comparable with published ones.
- Both sources confirmed working on 2026-11-20:
  - **Labels and splits:** `Kuhlman-Lab/ThermoMPNN` on github. `data_all/training/mega_train.csv`, `data_all/training/mega_val.csv`, `data_all/testing/mega_test.csv` (note the test file is in a different folder). See `scripts/download_megascale.sh`. We do not use `mega_splits.pkl`; the three csv files already define the split, and loading a pickle runs whatever code is inside it.
  - **Structures:** Zenodo record 7992926 (version v2_230420 of concept doi 10.5281/zenodo.7401274), file `AlphaFold_model_PDBs.zip`, 13 MB. It holds 862 pdb files, of which 298 are the labelled proteins, plus a `__MACOSX/` folder of mac metadata that `scripts/download_structures.sh` deletes.
- The proteins are short domains, so graphs are small and many mutations share one structure. Measured: **32 to 72 residues, median 48**, single chain A, no missing N/CA/C/O backbone atoms. This matters for 6.2: a pretraining length filter starting at 50 residues would exclude proteins the size of most of MegaScale.

**TODO checks before any training:**

1. done. 271,231 single mutations + 1,481 wt rows = 272,712; 298 proteins.
2. done. positive = stabilizing
3. done. Splits share no proteins (239 / 31 / 28).
4. done. All 298 structure sequences equal `wt_seq`, and all 271,231 wild-type letters match (1-based positions, node index = pos - 1).

### 6.2 Pretraining: unlabeled structures

- Source: the AlphaFold database representative structures. The database has about 214.7 million structures, clustered by shape into about 2.3 million non-singleton clusters (Barrio-Hernandez et al.). ProteinWorkshop distributes a 2.27 million structure representative set with data loaders.

**What we actually did (2026-11-20), and how it differs from the above.** We use two corpora, a pilot and the main one.

- **Pilot: SCOPe.** ASTRAL SCOPe 2.08, the subset already filtered to 40% sequence identity: 15,177 domains, a 992 MB tarball of plain pdb files plus a fasta of their sequences, from `scop.berkeley.edu`. Real experimental structures, domain-sized, and no new python dependencies. We rejected `proteinshake` (which the earlier project used) because it pulls in rdkit, statsmodels and a dozen other packages for one download, and we want the dependency list of this repo to stay short. See `scripts/download_scope.sh`.
- **Main: AlphaFold, sampled from Swiss-Prot.** Rather than the AFDB cluster representatives, we sample random reviewed UniProt entries of 30-256 residues and fetch their AlphaFold models from `alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb`. The files endpoint returns 403 without a User-Agent header; any identifier works. Mean pLDDT is read from the B-factor column of the file itself, so the quality filter needs no extra request. Each file is parsed to backbone coordinates immediately and the text discarded, which keeps 50,000 structures to a few hundred MB instead of several GB. Resumable, in shards of 2,000. See `scripts/download_alphafold.py`.

  Honest caveat for the write-up: Swiss-Prot is less redundant than raw AFDB but **more** redundant than cluster representatives, so our pretraining corpus contains related proteins that a cluster-representative set would have collapsed. We chose it because it is far lighter to obtain and needs no 2.27M-entry index. The leakage filter still runs on it.

- Downloading 50,000 is not the same decision as pretraining on 50,000. The download is one-off; the training subset is a flag, and the plan already wants a scaling curve, which having the larger set on disk makes possible.
- **Subset (suggested):** sample about 20,000 structures at random as the first version. Scale up only if time allows.
- **Quality filter:** mean pLDDT of at least 70. Length range **30 to 256**, not the 50 to 400 first suggested here: MegaScale domains are 32-72 residues (median 48), so a lower bound of 50 would have excluded structures the size of most of the downstream task. The upper bound keeps batches from going ragged. The same range is applied to both corpora.
- **Leakage filter (my recommendation, not from a paper):** remove every pretraining structure whose sequence is similar to any MegaScale protein (train, validation or test). Without this, the model may have seen test proteins during pretraining and the scores will look too good.

  Implemented in `scripts/build_pretrain_set.py` with MMseqs2 19-ab57c (`brew install mmseqs2`): `easy-search` at its most sensitive setting (`-s 7.5`), coverage 0.5, e-value 1e-3. We remove **every domain with any hit**, not only those at 30% identity or above, because remote homologues can share a fold while falling below 30%. The script prints both counts so the difference is visible, and writes `removed_similar.txt` next to `pretrain_ids.txt` so the decision can be audited. `tests/test_pretrain_filter.py` checks that no removed domain survived.
- **Fallback:** the 6,782 SCOP structures from the earlier project can serve as a tiny pilot set.
- Record the final list of pretraining structure IDs in `data/processed/pretrain_ids.txt` and commit it.

## 7. Pipeline

### 7.1 Graph construction (`data/graph.py`)

- Nodes: one per residue, using the Cα position (suggested, following ProteinWorkshop).
- Edges: connect each residue to its 16 nearest neighbors by Cα distance, then make the graph undirected (suggested k = 16, which ProteinWorkshop also used).

**Measured on all 298 MegaScale proteins (2026-11-20, `scripts/inspect_graph.py`):** 15,193 residues and 288,828 edges in total; 32-72 residues per protein (median 48), 630-1,396 edges (median 915), average degree 18.1-20.6 (median 19.0, which is k=16 plus the extra from symmetrising). Edge lengths: median 8.5 Å, 99th percentile 14.8 Å, max 25.0 Å. **62.5% of edges join residues more than 4 apart in the chain** and only 10.3% are chain neighbours, so most of the graph is contact information a sequence model cannot see. This is the justification for the graph.

Decision: **no distance cutoff.** On a 32-residue protein k=16 reaches halfway across the molecule, so about 1% of edges are longer than 14.8 Å and fall past the end of the distance RBF (2-22 Å), leaving them with near-zero distance features. They still carry their sequence-separation feature, they are rare, and a fixed k keeps every node's degree uniform. Revisit only if long edges turn out to matter.
- Do not use side chain information. Side chain angles reveal the residue type and would leak the answer in the letter arm.

### 7.2 Features (`data/features.py`)

Node features (all rotation invariant):

| Feature | Shape | Note |
|---|---|---|
| residue type embedding | 20 classes to 128 | replaced by a learned mask vector at masked nodes |
| backbone torsions (φ, ψ, ω) as sin and cos | 6 | backbone only |
| virtual Cα bond angle and torsion as sin and cos | 4 | defined on the Cα trace |

Edge features:

| Feature | Shape | Note |
|---|---|---|
| distance between Cα atoms, radial basis expansion | 16 (suggested) | |
| sequence separation, clipped at a maximum, one hot | about 8 (suggested) | |

**Leak rule.** At a masked node, no input feature may depend on that residue's identity. Backbone torsions and distances do not, but check anyway (see tests).

### 7.3 Encoder (`models/encoder.py`)

- Message passing network built from `GINEConv`, which updates a node as `h_i' = MLP((1 + ε) h_i + Σ_j ReLU(h_j + e_ji))`.
- Size: 4 layers, 128 hidden units, residual connections, layer norm, dropout 0.1. Pre-norm residual blocks (`h = h + relu(conv(norm(h))))`) **and a final layer norm on the output**. The final norm was added on 2026-11-20 after `test_untrained_loss_is_near_chance` showed the letter loss starting at 4.34 rather than ln(20) = 3.0: every block adds to h, so without it the output scale grows with depth. It matters beyond tidiness, since arm B's cosine loss and the effective-rank diagnostics both read these embeddings. Arm A was re-run after the change.
- Residue identity and backbone geometry are embedded separately and added, so masking only replaces the identity term and the geometry stays visible.
- Output: a matrix of shape `[num_residues, 128]` per protein.
- Same code and size for every arm.

### 7.4 Masking (`data/masking.py`)

- Mask 20% of the residues of each protein (suggested; ProteinJEPA masked 20%, ProteinWorkshop corrupted 25%).
- Masked nodes get the learned mask vector in place of the residue type embedding.
- Sample masks with the run seed, new mask every epoch.
- Guarantee at least one masked node per protein.

### 7.5 Pretraining arms (`train/pretrain.py`, `losses.py`)

Let `z_i` be the student encoder output at node i, run on the masked graph. Let `M` be the set of masked nodes.

**Arm C, letter only**

- Head: linear layer from 128 to 20.
- Loss: cross-entropy between the predicted and true letter at masked nodes.

```
L_letter = mean over i in M of  -log softmax(W z_i)[true_letter_i]
```

**Arm B, vector only (JEPA)**

- Teacher: an EMA copy of the encoder. It runs on the full, unmasked graph with gradients blocked. Its output at node i is `t_i`.
- Predictor: a small MLP applied to `z_i` at masked nodes.
- Loss: one minus cosine similarity between predictor output and `t_i` (suggested, as ProteinJEPA used cosine).

```
L_vector = mean over i in M of  1 - cos( predictor(z_i), stopgrad(t_i) )
teacher  = m * teacher + (1 - m) * student     # EMA update each step
```

- EMA momentum `m` rises from 0.99 to 0.9999 over training (suggested, taken from the earlier project and common in JEPA work).
- Log effective rank of the student embeddings every epoch (see 8.2). If it collapses to about 1, stop and investigate.
- Optional ablation if collapse is severe: add a variance regularizer, or use a detached target without EMA plus a regularizer as ProteinJEPA did. Do not add it by default, because it changes the arm.

**Arm D, both**

```
L = L_letter + lambda * L_vector
```

- Start with `lambda = 0.45` (the value ProteinJEPA used). ProteinWorkshop used 0.1 for its auxiliary loss, so 0.1 is a sensible second value.
- Same teacher, predictor and letter head as in B and C.

**Arm A**

- No pretraining. Encoder starts from random weights.

**Shared pretraining settings (suggested):**

| Setting | Value |
|---|---|
| optimizer | AdamW, learning rate 3e-4, weight decay 0.01 |
| warm-up | 5% of steps, then cosine decay |
| epochs | one fixed number for all arms, for example 100 (TODO: pick after timing one epoch) |
| batch | proteins per batch fixed (for example 32) |
| precision | float32 first, mixed precision only after it works |

**Compute fairness.** JEPA costs more per step (ProteinJEPA measured about 1.8 times). Log wall-clock time per arm. Report results matched by epochs, and also note the time. If time permits, add a version matched by wall-clock.

### 7.6 Fine-tuning on ΔΔG (`train/finetune.py`)

- Load the pretrained encoder (arms B, C, D) or a random one (arm A). Drop the pretraining heads and the teacher.
- **Encoder input at fine-tuning:** the full wild-type structure with all real letters visible, no masking.
- One encoder pass per protein gives a matrix of residue embeddings. For each mutation at position p, take `z_p`.
- **ΔΔG head:** an MLP taking `[ z_p , embedding(wild_type_letter) , embedding(mutant_letter) ]` and outputting one number (suggested: 2 hidden layers of 256).
- Loss: mean squared error on ΔΔG (suggested; try Huber if outliers hurt).
- **Main setting: frozen encoder.** Only the head is trained. This measures embedding quality directly. Arm A has nothing to freeze, so for arm A train the encoder and head together (the baseline is "learn everything from labels"). Write this difference in the results table notes.
- Optional second setting: fine-tune everything with a smaller learning rate for the encoder.
- Early stopping on validation Spearman, patience 10 epochs (suggested). Evaluate the test set once, at the best validation checkpoint.
- Batch by protein so all mutations of one protein reuse one encoder pass.

### 7.7 Evaluation and embedding analysis

See sections 8 and 9.

## 8. Metrics

### 8.1 Stability prediction (`eval/metrics.py`)

| Metric | Definition | Role |
|---|---|---|
| Pooled Spearman | Spearman correlation across all test mutations together | main number |
| Per-protein Spearman, median | Spearman within each test protein, median over proteins (require at least 10 mutations per protein, suggested) | matters for design use |
| RMSE | root mean squared error in kcal/mol | secondary |

Rules:

- Report mean ± standard deviation over seeds.
- Treat a difference smaller than the seed spread as a tie. Measured on arm A with 3 seeds (see step 2): **under 0.012 pooled, or under 0.016 per-protein median, is a tie.** Fixed before any pretrained arm was run.
- Never compare pooled and per-protein numbers with each other. Papers report both, and they are not interchangeable.
- Reference points, not targets: ThermoMPNN reports Spearman 0.725 on the MegaScale test set, and unmodified ProteinMPNN 0.487. We do not aim to beat them.

### 8.2 Embedding quality (`eval/rank.py`)

**Effective rank** (the RankMe quantity). For an embedding matrix `Z` of size `N × K` with singular values `σ_k`:

```
p_k = σ_k / sum(σ)                      # normalize to sum to 1
effective_rank(Z) = exp( - sum_k p_k * log p_k )
```

- Sample size: **TODO, and the plan's original number is impossible.** RankMe used N = 25,600, but the 28 MegaScale test proteins hold only about **1,400 residues**, and all 298 proteins together only 15,193. Options: (a) use every test residue, about 1,400, which is still 11x the 128 embedding dimensions; (b) add the validation proteins, about 3,000; (c) compute it on held-out pretraining structures instead, where residues are plentiful. Decide at step 5 and **state the sample size in every table**, since effective rank is not comparable across different N. Use the wild-type structure with real letters, no masking.
- Compute it on the frozen encoder output after pretraining, and also log it during pretraining.
- **Caveat from the RankMe paper:** compare effective rank only between runs of the same method, because rank is not the only factor affecting performance. Here it is a warning light across arms, not a quality score.
- Also log per-dimension variance. A few dimensions with large variance and the rest near zero means collapse.

**Optional probes (`eval/probes.py`):** fit a linear model from frozen embeddings to something computable from structure, such as relative burial (number of neighbors within 10 Å) or secondary structure. Probing for the residue letter is only a sanity check, because arm C trains on it directly.

**Optional similarity between arms:** linear CKA between embeddings of the same residues from different arms (Kornblith et al. 2019).

### 8.3 Cost

- Wall-clock time per pretraining epoch and total.
- Peak GPU memory.

## 9. Safety checks and tests (`tests/`)

Write these before the real runs. They are cheap and catch the mistakes that ruin results.

| Test | What it checks |
|---|---|
| `test_no_letter_leak` | take a graph, mask node i, change its true letter, run the encoder. The embedding at node i must be identical. Also check edge features and kNN edges do not depend on letters |
| `test_split_disjoint` | no protein appears in two of train, validation and test |
| `test_mutation_alignment` | wild-type letter in the csv equals the letter in the structure at that position, for every mutation |
| `test_pretrain_filter` | no pretraining structure is in the removed-similar list |
| `test_effective_rank` | rank of an identity matrix is K, rank of a matrix with one repeated row is close to 1 |
| `test_ema_update` | teacher moves toward the student by the expected amount and receives no gradients |
| `test_overfit_batch` | each arm can drive its loss near zero on a tiny batch. If not, there is a bug |
| `test_letter_baseline` | letter accuracy is compared with always guessing the most common amino acid, so a trivial model is not mistaken for learning |
| `test_seed_repro` | same seed gives the same loss in the first steps |

Other habits:

- Print one example batch (shapes, number of masked nodes) at the start of every run.
- Save every config next to its results.
- Never tune hyperparameters on the test set.

## 10. Experiment grid and results format

**Grid for the first full run:** 4 arms × 3 seeds (0, 1, 2) × frozen encoder = 12 fine-tuning runs, and 3 arms × 3 seeds = 9 pretraining runs (arm A needs none).

**Results file:** one row per run in `results/runs.csv` with these columns (suggested):

```
arm, seed, lambda, pretrain_epochs, pretrain_time_s, finetune_mode,
val_spearman, test_pooled_spearman, test_per_protein_median_spearman,
test_rmse, effective_rank, git_commit, config_path
```

**Report tables to generate (`scripts/make_report.py`):**

1. Main table: arm × (pooled Spearman, per-protein median Spearman, RMSE), mean ± std over seeds.
2. Embedding table: arm × effective rank.
3. Figure: effective rank during pretraining for arms B, C and D.
4. Cost table: time per arm.

Template for the main table (fill after the runs):

| Arm | Pooled Spearman | Per-protein median Spearman | RMSE | Effective rank |
|---|---|---|---|---|
| A. none | | | | |
| B. vector | | | | |
| C. letter | | | | |
| D. both | | | | |

## 11. How to read the outcomes

| Result | Meaning |
|---|---|
| D ≥ C > B, and B has low rank | letters matter, and vector-only collapses on structures too |
| B ≥ C | latent prediction is enough here, unlike on sequences |
| A, B, C, D within seed spread | little pretraining benefit at this scale and setup |
| arm E (if run) close to B | the encoder design explains more than what it learned |

## 12. Build order with acceptance checks

Do these in order. Each step ends with something you can look at, and a check that says it is done.

1. **Data and graphs. DONE 2026-11-20.**
   Download MegaScale and splits, load them, build the graph of one protein and plot it.
   Done when: counts printed and compared with 272,712 and 298, sign convention chosen and written down, `test_split_disjoint` and `test_mutation_alignment` pass.

   Built: `scripts/download_megascale.sh`, `scripts/download_structures.sh`, `src/pobj/data/megascale.py` (labels + pdb parsing), `src/pobj/data/features.py`, `src/pobj/data/graph.py`, `scripts/inspect_graph.py`. Tests: `tests/test_megascale.py` (8 passing, including `test_split_disjoint` and `test_mutation_alignment`) and `tests/test_graph.py` (8 passing, including `test_no_letter_leak` and rotation/translation invariance, both written early rather than at step 4). Figure: `results/figures/example_graph.png`.

2. **Arm A end to end. DONE 2026-11-20, 3 seeds.**

   | metric | mean ± sd |
   |---|---|
   | test pooled Spearman | **0.604 ± 0.006** |
   | test per-protein median Spearman (28 proteins) | **0.642 ± 0.008** |
   | test RMSE | **0.808 ± 0.009** |

   Best epoch 28 ± 8, about 2 minutes per run on the Mac. For comparison, the earlier Petnica project got 0.548 from scratch and 0.600 pretrained, and ThermoMPNN reports 0.642 and 0.725. Our baseline is above the old from-scratch number and above its pretrained one, which raises the bar pretraining has to clear here. RMSE is not comparable with the old project, which used the deduplicated HuggingFace copy.

   These are the numbers after the final layer norm was added to the encoder (see 7.3). Before it, the same three seeds gave 0.594 ± 0.008 pooled and 0.641 ± **0.023** per-protein. The means barely moved, but the per-protein spread fell to a third of what it was, which is the real gain: it makes the experiment sensitive to effects three times smaller.

   **Tie threshold, fixed before seeing any pretrained result.** Roughly two standard deviations: a difference under **0.012 pooled** or under **0.016 per-protein median** is a tie. The effect the old project reported (+0.052 pooled) would clear that comfortably.

   Built: `src/pobj/models/encoder.py` (GINE, 4 layers, dim 128; residue identity and backbone geometry are embedded separately and added, so masking only has to replace the identity), `src/pobj/models/heads.py`, `src/pobj/data/dataset.py` (one item per protein carrying all its mutations, 239 encoder passes per epoch instead of 215k), `src/pobj/eval/metrics.py`, `src/pobj/train/finetune.py`, `src/pobj/utils.py`, `configs/base.yaml`, `configs/arm_a.yaml`. Tests: `tests/test_metrics.py`, `tests/test_dataset.py`, `tests/test_overfit.py`, 11 passing. The one that matters most is `test_batching_shifts_node_indices`, which asserts `batch.x[batch.mut_node] == batch.mut_wt`, so a mutation can never be scored against the wrong residue after batching.

   Timing for planning: fine-tuning is cheap, about 2.5 minutes per run, so the whole 12-run fine-tuning grid is roughly half an hour on the Mac. Pretraining will be the expensive part.

   Original step 2 text:
   Encoder and ΔΔG head trained from scratch, evaluated on the test split.
   Done when: you have a first pooled Spearman and RMSE, and `test_overfit_batch` passes for the ΔΔG head.

3. **Pretraining data. SCOPe pilot DONE 2026-11-20; AlphaFold downloading.**
   Build the subset and run the similarity filter.
   Done when: `pretrain_ids.txt` exists and `test_pretrain_filter` passes.

   SCOPe: 15,177 ASTRAL domains -> 11,791 after the 30-256 length filter (98 too short, 3,288 too long, 0 unreadable) -> **11,566 after the leakage filter**. MMseqs2 removed only 225 domains, 1.9%, of which 218 were already at 30% identity or above, so the stricter "remove every hit" rule cost 7 extra domains. The low number is expected, since many MegaScale proteins are de novo designs with no natural relatives. `tests/test_pretrain_filter.py` passes, 4 tests.

   Size mismatch to state in the write-up: the kept SCOPe domains have a median of 126 residues against MegaScale's 48, so we pretrain on somewhat larger proteins than we fine-tune on.

   AlphaFold: 241,239 Swiss-Prot entries fall in the length range, which is the pool 50,000 are sampled from with seed 0.

4. **Arm C, letter pretraining. Pilot DONE 2026-11-20, seed 0 on SCOPe.**

   Pretraining, 100 epochs on 11,566 SCOPe domains, 40 min on the Mac at 461 structures/s: letter accuracy on held-out structures **0.269** against a most-common-residue baseline of 0.095, loss 2.94 -> 2.35, flat over the last 20 epochs, so 100 epochs is enough rather than cut short. Effective rank **61.6 of 128**, stable, no collapse. That rank is the reference arm B will be read against.

   Fine-tuning, frozen encoder, seed 0: test pooled Spearman **0.641**, per-protein median **0.696**, RMSE **0.778**. Against arm A's 0.604 ± 0.006 and 0.642 ± 0.008 that is **+0.037 and +0.054**, both about three times the tie threshold. One seed only, so the claim waits on seeds 1 and 2, but the pilot question is answered: pretraining helps here.

   Worth stating plainly in the write-up: arm C won with a frozen encoder that never saw a ddG value, against an arm A that trained its whole network on 215k labelled mutations. For scale, 0.641 is level with ThermoMPNN's from-scratch 0.642, though they reach 0.725 with far heavier pretraining.

   Original step 4 text:
   Add masking, the letter head and loss, then fine-tune.
   Done when: `test_no_letter_leak` passes, letter accuracy on held-out masked residues beats the most-common-letter baseline, fine-tuning runs.

5. **Arm B, JEPA pretraining. Seed 0 run 2026-11-20 on SCOPe.**

   **Collapse happens, and then the model climbs back out.** Epoch 0 starts at vector loss 0.285 with rank 12.3; by epoch 1 the loss is 0.0073 and the rank is **7.7**, which is the trivial solution of mapping every residue to nearly the same vector. That is the same 5-8 rank the earlier Petnica project reported. From there the EMA teacher makes the constant unstable, the representation spreads back out, and by epoch 53 the rank is **79.5** while the loss has risen to 0.129.

   **This explains the open question in the old project's README.** It said the JEPA loss "went up during training instead of down, which is the opposite of what you'd normally hope for. We don't fully understand this yet." With an EMA teacher the target moves, so the loss is not comparable across epochs: a collapsed representation makes prediction trivially easy and the loss tiny. A **rising loss alongside a rising rank is the signature of escaping collapse**, not of failure. Loss alone is not a progress signal for this arm, which is exactly why rank is logged beside it.

   **H3 looks wrong.** The plan predicted vector-only would use the fewest dimensions. It uses the most: rank 79.5 against arm C's 61.6, and still rising at epoch 53. A failed prediction with a mechanism behind it is still a result. High rank does not imply a useful representation, so the ddG number decides.

   Final state after 100 epochs: rank **84.3**, vector loss plateaued at 0.131, both flat over the last ten epochs. The collapse at epoch 1 was escaped and never returned.

   **ddG result, seed 0, frozen encoder: pooled Spearman 0.520, per-protein median 0.571, RMSE 0.897.** That is 0.084 *below* arm A's 0.604, seven times the tie threshold. Pretraining this way left the encoder worse than random initialisation: negative transfer, which Hu et al. 2020 warn about.

   **And it is not an artefact of freezing.** Fine-tuned in full, the same protocol arm A uses, arm B reaches 0.560: better than frozen, but still 0.044 below random initialisation. A vector-pretrained encoder is a worse starting point than no pretraining at all, and training only partly undoes it.

   The two arms respond to freezing in opposite directions, which is itself evidence the difference is about feature quality rather than protocol:

   | arm | frozen | full |
   |---|---|---|
   | A, none | - | 0.604 |
   | B, vector | 0.520 | 0.560 |
   | C, letter | 0.641 | 0.631 |

   Arm C is better frozen, so its features are worth keeping. Arm B is better unfrozen, so its features are worth overwriting.

   **The two hypotheses split, and the combination is the interesting part.** H1 holds: vector-only is worst on ddG. H3 fails, and fails hard: vector-only has the **highest** effective rank, not the lowest. Ranking the arms by rank gives exactly the reverse of ranking them by ddG.

   | arm | effective rank (of 128) | pooled Spearman |
   |---|---|---|
   | B, vector | 84.3 | 0.520 |
   | C, letter | 61.6 | 0.641 |

   This matters beyond our setup. Effective rank, via RankMe, is a standard label-free proxy for how good a pretrained representation is. Here it points the wrong way. RankMe's own paper says rank is only comparable between runs of the same method, and this is a concrete demonstration of why: a representation can spread across 84 dimensions and still carry nothing about stability. Escaping collapse is not the same as learning something useful.

   Cost: 33 s per epoch against arm C's 24 s, so the teacher's extra forward pass costs about 1.4x. ProteinJEPA measured 1.8x.

   Original step 5 text:
   Add teacher, predictor, vector loss, rank logging.
   Done when: `test_ema_update` passes, effective rank is logged every epoch and does not collapse to about 1, fine-tuning runs.

6. **Arm D, both.**
   Combine losses with the chosen λ.
   Done when: both losses decrease in the log.

7. **Full grid.**
   Run all arms with 3 seeds, collect `runs.csv`.
   Done when: every cell of the grid has a row and the report script produces all tables and the figure.

8. **Write the blog post.** Outline in section 14.

## 13. Open decisions

Fill these in before step 2.

| Decision | Options | My default | Chosen |
|---|---|---|---|
| Compute | GPU or laptop | decides subset size and epochs | **Mac M4 Pro (MPS) for development; final grid on a GPU, probably Colab A100.** Run the whole grid on one machine, since the same seed gives slightly different numbers on MPS and CUDA. Pick the device as cuda, else mps, else cpu. |
| Pretraining subset size | 5k, 20k, more | 20,000 | TODO |
| Pretraining epochs | 50, 100, more | 100 after timing one epoch | Timed 2026-11-20 on the Mac: **468 structures/s**, so 25 s per epoch on the 11,566 SCOPe domains and about 107 s on 50,000 AlphaFold ones. 100 epochs is therefore 41 min per SCOPe run, and the 3 arms x 3 seeds pilot is about 6 hours. TODO: confirm 100 is enough once arm C's letter accuracy curve is visible. |
| Masking | learned mask vector, or random letter | learned mask vector | TODO |
| Teacher | EMA, or detached target plus regularizer | EMA | TODO |
| λ for arm D | 0.1, 0.45, both | 0.45 first | TODO |
| ΔΔG sign | positive = stabilizing | check the file | **positive = stabilizing**, verified 2026-11-20. In `mega_val.csv` the lab's own `Stabilizing_mut == True` rows have mean ddG_ML **+1.40** (min +1.01), `False` rows mean **-0.91**, and 77% of all values are negative, which matches the biology. Kept as the file has it and never flipped. Recorded in `configs/base.yaml` as `ddg_sign: positive_stabilizing`. |
| Fine-tuning mode | frozen only, or also full | frozen only | TODO |

## 14. Blog outline

1. **The question.** What should a model guess when it practices on proteins: the letter or a summary?
2. **A picture.** One hidden amino acid and the two kinds of guess.
3. **The setup.** The four arms in a small table, and what stays the same.
4. **The result.** The main table.
5. **Inside the model.** The effective rank figure.
6. **What it means, and what we did not test.** One encoder, one stability dataset, a small pretraining set, three seeds. Say plainly that this is a controlled small experiment, not a ranking of methods in general.

Honesty rules for the post: report seed spread, call differences within the spread ties, say which ΔΔG sign and which data file were used, and state the leakage filter.

## 15. Notes for working with Claude Code

Paste or point Claude Code at this file at the start of a session. Suggested ground rules:

- Follow the build order in section 12. Do not skip ahead. After each step, run the tests listed for it and show the output.
- Change only one thing between arms, the pretraining loss. Do not alter shared settings (section 7) without asking.
- Never touch the test split during development. Use validation for all decisions.
- Prefer small, readable functions. Add a shape comment on every tensor that crosses a function boundary.
- Every item marked TODO or (suggested) is a decision for me. Ask before choosing, and then record the choice in `configs/base.yaml` and in section 13.
- Log everything needed to reproduce a run: config, seed, git commit, data file hashes.
- If a result looks too good, assume a leak first and look for it before celebrating.

Example prompts for each milestone:

```
Step 1: Implement data/megascale.py to load the ThermoMPNN MegaScale csv and
split files, convert ddG using the ddg_sign in configs/base.yaml, and print
counts of mutations and proteins per split. Then write tests/test_split_disjoint.py
and tests/test_mutation_alignment.py.

Step 4: Implement masking.py and the letter head. Write tests/test_no_letter_leak.py
first, so the leak rule in section 7.2 is enforced before any training run.

Step 5: Implement the EMA teacher and the vector loss from section 7.5. Log
effective rank every epoch to results/. Stop and report if rank falls below 2.
```

## 16. References

Facts used in this plan and where they come from:

- Tsuboyama et al. 2023. Mega-scale experimental analysis of protein folding stability in biology and design. Nature 620:434-444. doi:10.1038/s41586-023-06328-6. The MegaScale data.
- Dieckhaus et al. 2024. Transfer learning to leverage larger datasets for improved prediction of protein stability changes (ThermoMPNN). PNAS. bioRxiv 10.1101/2023.07.27.550881. Code: github.com/Kuhlman-Lab/ThermoMPNN. The 272,712 mutations, 298 proteins, and the 0.725 and 0.487 Spearman values.
- Beltran et al. 2025. Site-saturation mutagenesis of 500 human protein domains. Nature 637:885-894. doi:10.1038/s41586-024-08370-4. Optional second test set (563,534 variants, 522 domains).
- Ofer, Linial, Shahaf 2026. ProteinJEPA: Latent prediction complements protein language models. arXiv:2605.07554. The sequence-level letter vs vector vs both comparison, λ = 0.45, 20% masking, cosine loss.
- Jamasb et al. 2024. Evaluating representation learning on the protein structure universe (ProteinWorkshop). ICLR 2024. arXiv:2406.13864. Code: github.com/a-r-j/ProteinWorkshop. Structure pretraining tasks, k = 16 neighbors, backbone features, 2.27 million AlphaFold representative structures.
- Barrio-Hernandez et al. 2023. Clustering predicted structures at the scale of the known protein universe. Nature 622:637-645. doi:10.1038/s41586-023-06510-w. AlphaFold DB size and clustering.
- Garrido et al. 2023. RankMe: Assessing the downstream performance of pretrained self-supervised representations by their rank. ICML 2023. arXiv:2210.02885. Effective rank formula and its caveats.
- Roy and Vetterli 2007. The effective rank: a measure of effective dimensionality. EUSIPCO. Original effective rank definition.
- Kornblith et al. 2019. Similarity of neural network representations revisited. ICML 2019. arXiv:1905.00414. CKA.
- Hu et al. 2020. Strategies for pre-training graph neural networks. ICLR 2020. arXiv:1905.12265. Source of `GINEConv`, and a warning that naive pretraining can cause negative transfer.
- Skenderi et al. 2023. Graph-level representation learning with joint-embedding predictive architectures (Graph-JEPA). arXiv:2309.16014.
- Assran et al. 2023. Self-supervised learning from images with a joint-embedding predictive architecture (I-JEPA). arXiv:2301.08243.
- Zhang et al. 2023. Protein representation learning by geometric structure pretraining (GearNet). ICLR 2023. arXiv:2203.06125.
- Dauparas et al. 2022. Robust deep learning based protein sequence design using ProteinMPNN. Science 378:49-56.
- Zaidi et al. 2023. Pre-training via denoising for molecular property prediction. ICLR 2023. arXiv:2206.00133.
- Littwin et al. 2024. How JEPA avoids noisy features. arXiv:2407.03475.

Things I did not verify, so treat them as TODO: current download URLs for the MegaScale structures, the split files and the AlphaFold representative set, and the exact MMseqs2 options for the identity filter.