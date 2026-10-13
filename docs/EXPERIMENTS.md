# Experiments

Last updated: 2026-10-13 (Phase A refactor). Status marks: ✅ implemented and tested, 🟡 implemented but to be validated, ⬜ not implemented.

**About file paths:** sections 2 to 5 were written before the refactor and name files where they were then (`priors/`, `train_mixed.py`, `checkpoints/`, and so on). Section 1 maps them to the new layout; all pre-refactor outputs moved to `legacy_results/` (not in git).

## 0. The question

**RQ1: does the tradeoff curve exist?** As the finance prior's share r goes from 0% to 100%, how much do finance tables gain and how much do general tables lose?

Approach: train one nanoTabPFN per r, test it on two benchmark suites, and draw two curves (x: r; y: AUC / ECE / Brier).

Since 2026-10-12 the focus is how a domain prior's value changes with the training budget and with the strength of the generic prior (see [RESEARCH.md](RESEARCH.md)).

## 1. Pipeline and code (after the 2026-10-13 refactor)

```
prior (priors/registry.py) - scripts/make_dump.py - pools artifacts/dumps/*.h5 -+
                                                                                +- PoolMixtureLoader(mixture) - train() - runs/<experiment>/<run>/
                                     configs/base.yaml + experiments/*.yaml ----+   draw by share, impute, split       checkpoints, synthetic validation, suites
```

| Step | Files | How to run |
|---|---|---|
| Priors | `src/tfmprior/priors/`: `generic.py` (graph_scm, mix_scm), `fintfm.py`, `registry.py` | Called by pool generation |
| Pool generation | `src/tfmprior/data/dump.py`, [scripts/make_dump.py](../scripts/make_dump.py) | `python scripts/make_dump.py --all`, or download with [scripts/download_pools.py](../scripts/download_pools.py) |
| Mixture sampling and batches | `PoolMixtureLoader` in `src/tfmprior/data/loader.py` | Called by training |
| Training | `src/tfmprior/train/loop.py`, `src/tfmprior/run.py`, [scripts/train.py](../scripts/train.py) | `python scripts/train.py --config configs/experiments/<experiment>.yaml` |
| Suites and evaluation | `src/tfmprior/eval/`: `benchmarks.py`, `metrics.py`, `evaluate.py`, `synthetic.py`, `health.py` | Automatic after training; `python scripts/evaluate.py <run dir>` |
| Golden answers | `golden/`, [scripts/verify_golden.py](../scripts/verify_golden.py) | `python scripts/verify_golden.py --device mps` |
| Diagnostics | `scripts/diagnostics/`: `prior_stats.py`, `prior_learnability.py`, `format_check.py` | |

Old files and their new homes: `GenericPrior` in `priors/mixture.py` → `priors/generic.py`; `DumpMixedDataLoader` in `priors/batching.py` → `PoolMixtureLoader` in `data/loader.py` (`impute_with_train_mean` → `data/preprocess.py`); `train()` in `train.py` → `train/loop.py`; `train_mixed.py`, `run_grid.py` → `scripts/train.py` (parallel `sweep.py` comes in Phase C); `benchmark/` → `eval/benchmarks.py`, `eval/metrics.py`; `evaluate.py` → `eval/evaluate.py`, `scripts/evaluate.py`; `make_prior_dump.py` → `data/dump.py`, `scripts/make_dump.py`; `check_priors.py` → `scripts/diagnostics/prior_stats.py`; `prior_dumps/` → `artifacts/dumps/`; `benchmark/cache/` → `artifacts/benchmark_cache/`.

## 2. Priors and training

### 2.1 The two priors ✅

| | What | Code |
|---|---|---|
| Generic prior | **TabICLv2's prior** (`graph_scm`); TabICL's official script `scripts/train_v2_clf_stage1.sh` uses `--prior_type graph_scm`. `mix_scm` gives the v1 prior | [priors/mixture.py](priors/mixture.py) `GenericPrior`; the generator lives in the installed package `.venv/.../tabicl/prior/` (tabicl 2.2.0) |
| Finance prior | FinTFM `sample_financial_task()`: financial statements obeying accounting identities, financial-ratio features, default rates from 0.1% to 30% | [priors/fintfm_financial.py](priors/fintfm_financial.py) (copied verbatim from FinTFM commit `7c297d0`, one import line changed) |

FinTFM's own mixer is not used: its generic part is the author's own SCM, and its tree prior takes its share out of the finance budget, so `p_financial` is not the real finance share.

### 2.2 Definition of r ✅

**Each table** comes from the finance prior with probability r, otherwise from the generic prior. r counts tables, matching r in data mixing laws. Code: `DumpMixedDataLoader` in [priors/batching.py](priors/batching.py).

### 2.2b Pools ✅ (2026-10-11)

| Pool | File | Size | Content |
|---|---|---|---|
| Generic | `prior_dumps/generic_graph_scm_100k_256x24_seed0.h5` | 0.80 GB | 100k tables, tabicl 2.2.0, median positive rate 0.496, no missing values |
| Finance | `prior_dumps/finance_100k_256x24_seed0.h5` | 1.55 GB | 100k tables, FinTFM `7c297d0`, median positive rate 0.047, 5.2% missing |

- Generated in chunks of 1000 tables with fixed seeds, so the file is identical whatever the number of processes (checksums of 1 and 4 workers verified identical)
- Missing values are kept and imputed at training time, once the split is known
- Sampling: each pool is shuffled once per seed and read in order, so no table repeats within a run; for one seed, the random numbers that decide the source are shared across r, so runs with different r differ only through r
- 1 to 3 ms per batch (on-the-fly generation took 280 ms)

### 2.3 Table format 🟡

| Item | Setting | Reason |
|---|---|---|
| Rows | 256 | The finance prior needs enough rows to produce low default rates |
| Features | At most 24, zero-padded to 24 | FinTFM's default; an all-zero column stays zero after the model's normalization |
| Classes | Binary | The finance suite is default prediction |
| Train/test split | Drawn per batch from {80, 96, ..., 224} | Multiples of 16 only, so there are just 10 batch shapes and the Mac GPU cache does not blow up |
| Missing values | Column mean of the train rows | `impute_with_train_mean()`, also used for real tables |

At 256 x 24 and batch 8, lr 4e-3 collapses at random and 1e-3 is stable (4.4, 4.5).

### 2.4 Model and training 🟡

| Item | Setting |
|---|---|
| Model | nanoTabPFN: embedding 96, 4 heads, MLP 192, 3 layers, 2 outputs |
| Optimizer | Schedule-free AdamW, **lr 1e-3** (the original 4e-3 was tuned for batch 32; at batch 8, 4e-3 collapses at random, see 4.5) |
| Batch | 8 on the Mac (16 GB memory limit), 32 planned on a GPU |
| Steps | 3000 (plan (a): breast cancer barely improves after step 1000) |
| Speed (Mac) | About 0.39 s per step reading from pools, about 20 minutes for 3000 steps (0.67 s with on-the-fly generation) |
| Randomness | Fully reproducible for a given seed: fixed pool files; python / numpy / torch seeds set; `PYTHONHASHSEED` fixed to 0 automatically ([priors/__init__.py](priors/__init__.py)) |

**Two changes to the original [train.py](train.py):**
1. Clear the GPU cache after every step on the Mac
2. **Bug fix:** `train()` did not switch the schedule-free optimizer to eval mode at the end, so it returned and saved the training weights instead of the averaged evaluation weights. The schedulefree documentation requires eval mode when saving checkpoints. **Every result before 2026-10-10 is affected**, including the 600-step numbers in 4.3.

## 3. Benchmark suites ✅

Both suites come from the binary tasks of TabArena v0.1 (OpenML suite 457). "Finance" means **credit risk, default and bankruptcy**, which is what the FinTFM finance prior simulates.

| Suite | Datasets |
|---|---|
| **Finance (7)** | polish_companies_bankruptcy, taiwanese_bankruptcy_prediction, credit-g, credit_card_clients_default, GiveMeSomeCredit, heloc, Is-this-a-good-customer |
| **General (17)** | Amazon_employee_access, Bioresponse, Diabetes130US, E-CommereShippingData, Fitness_Club, HR_Analytics, NATICUSdroid, blood-transfusion, churn, customer_satisfaction_in_airline, diabetes, hazelnut, in_vehicle_coupon, jm1, online_shoppers_intention, qsar-biodeg, seismic-bumps |
| In neither (4) | Bank_Customer_Churn, bank-marketing, coil2000_insurance_policies, Marketing_Campaign: banking or insurance, but not credit risk; **to be decided** |
| Skipped (2) | APSFailure, kddcup09_appetency: minority class under 2.5%, too few positives in 256 rows for 5 folds |

**Evaluation protocol** (adapted from the nanoTabPFN paper's [experiment.ipynb](../upstream_nanotabpfn/experiment.ipynb)):
- Per dataset, a stratified subsample of 256 rows, repeated 3 times (different seeds)
- Categorical columns ordinal-encoded, constant columns dropped; datasets with more than 24 features get a random 24
- 5-fold cross-validation, missing values imputed per fold with train-row means
- Metrics: AUC, ECE (10 equal-width bins), Brier. Averaged over the 3 repeats within a dataset, then over datasets
- Subsamples are cached in `benchmark/cache/`, so every model sees exactly the same tables

**Reference lines** (`python evaluate.py --baselines`, 45 seconds):

| | Finance AUC | Finance ECE | Finance Brier | General AUC | General ECE | General Brier |
|---|---|---|---|---|---|---|
| Random forest | 0.779 | 0.044 | 0.108 | 0.734 | 0.074 | 0.144 |
| Logistic regression | 0.738 | 0.057 | 0.116 | 0.730 | 0.063 | 0.146 |

## 4. Findings so far

**4.1 How learnable each prior's tables are** (random forest AUC on 40 tables, [diagnostics/prior_learnability.py](diagnostics/prior_learnability.py))

| Prior | Mean AUC | Share of tables with AUC > 0.7 |
|---|---|---|
| TabICL `mlp_scm` | 0.551 | 7% |
| TabICL `tree_scm` | 0.674 | 38% |
| TabICL `mix_scm` (v1) | 0.590 | 23% |
| TabICL `graph_scm` (v2, in use) | 0.707 | 47% |
| FinTFM finance | 0.658 | 44% |
| Original nanoTabPFN h5 dump | 0.595 | 20% |

**4.2 The two priors' tables look very different** ([check_priors.py](check_priors.py), graph_scm, medians)

| | Generic | Finance |
|---|---|---|
| Features | 12 | 18 |
| Positive rate | 0.49 | 0.039 (56% of tables under 5%) |
| Missing values | 0 | 8.7% |
| Largest absolute value | 4 | 5e7 (up to 1e12) |

**4.3 Training stability** ([diagnostics/table_size.py](diagnostics/table_size.py))

600 steps, batch 8, `mix_scm`, breast cancer AUC, **affected by the bug in 2.4**:

| Table size | Seed 0 | Seed 1 |
|---|---|---|
| 150 x 5 | 0.947 | 0.939 |
| 256 x 24 | 0.226 | 0.877 |

Plan (a) (done 2026-10-10): 256 x 24, `graph_scm`, 3000 steps, batch 8, bug fixed; log `logs/plan_a_graph_scm_256x24_3000steps.log`.

| | Seed 0 | Seed 1 |
|---|---|---|
| Breast cancer AUC at steps 500 / 1000 / 1500 / 2000 / 2500 / 3000 | 0.968 / 0.982 / 0.971 / 0.974 / 0.979 / 0.977 | 0.801 / 0.972 / 0.988 / 0.987 / 0.990 / 0.982 |
| AUC on 100 unseen graph_scm tables | 0.691 | 0.703 |

The conclusion then was "256 x 24 trains stably"; **overturned on 2026-10-11** (4.4): these two runs just happened not to collapse. The two seeds differ by 0.005 to 0.012. On unseen tables of the same kind the model reaches 0.69 to 0.70, close to the random forest's 0.707 on such tables (4.1), so it does learn from the prior. Breast cancer barely improves after step 1000. About 35 minutes per seed.
Note: this run changed both the generic prior (mix_scm to graph_scm) and the schedule-free bug at once, so the source of the stability cannot be separated.

**4.4 Training collapses at random (2026-10-11)**

r = 0 and r = 1, 3000 steps each (on-the-fly generation, batch 8, lr 4e-3, seed 0), archived in `checkpoints/onthefly_prehashfix_2026-10-11/` and `results/onthefly_prehashfix_2026-10-11/`:

| Model | Finance AUC | Finance ECE | General AUC | General ECE |
|---|---|---|---|---|
| r = 0 | 0.506 | 0.320 | 0.501 | 0.221 |
| r = 1 | 0.744 | 0.058 | 0.717 | 0.093 |

r = 0 collapsed between steps 500 and 1000: the loss rose from 0.57 to 0.69 and stayed there; breast cancer AUC 0.50. The settings matched plan (a)'s seed 0 exactly, yet the result differed, because TabICL depends on Python's hash seed and the two runs saw different data (fixed).

All 256 x 24, batch 8, lr 4e-3 runs so far:

| Run | Result |
|---|---|
| 10/9, 600 steps, mix_scm | 0.226, inverted |
| Plan (a) seed 0 | 0.977 |
| Plan (a) seed 1 | 0.982 |
| 10/11 r = 0 | 0.500, collapsed |

Likely cause: lr 4e-3 was tuned for batch 32 in the original, and at batch 8 the gradient noise is about twice as large; generic-prior tables are mostly balanced with weak signal, so "always predict 50/50" is a local optimum with loss 0.69 that a bad batch can push the model into for good. Finance tables mostly have low default rates, where predicting 50/50 costs a high loss, so training does not get stuck there.

**Criteria for "collapsed"** (used for the learning-rate check):

| | Healthy | Collapsed |
|---|---|---|
| AUC on 100 unseen synthetic tables | ≥ 0.65 | < 0.55 |
| Breast cancer AUC | ≥ 0.9 | < 0.6 |
| Training loss (r = 0) | Clearly below 0.69 | Stuck near 0.69 (ln 2) |

**4.5 Learning-rate check (2026-10-11)**

r = 0, reading from pools, batch 8, 1000 steps, [diagnostics/lr_stability.py](diagnostics/lr_stability.py); results in `results/lr_stability/summary.json`, log `logs/lr_stability.log`:

| lr | Seed | Verdict | Synthetic AUC | Synthetic log loss | Breast cancer AUC every 250 steps |
|---|---|---|---|---|---|
| 4e-3 | 0 | Healthy | 0.714 | 0.408 | 0.95 / 0.98 / 0.97 / 0.97 |
| 4e-3 | 1 | Healthy | 0.667 | 0.475 | 0.79 / 0.88 / 0.93 / 0.91 |
| 4e-3 | 2 | **Collapsed** | 0.500 | 0.693 | 0.27 / 0.48 / 0.54 / 0.38 |
| 1e-3 | 0 | Healthy | 0.661 | 0.470 | 0.91 / 0.95 / 0.96 / 0.95 |
| 1e-3 | 1 | Healthy | 0.656 | 0.471 | 0.89 / 0.95 / 0.96 / 0.96 |
| 1e-3 | 2 | Unclear | 0.649 | 0.475 | 0.90 / 0.93 / 0.96 / 0.94 |

- 4e-3 failed 3 of 7 times overall; 1e-3 learned in all three runs, with seeds within 0.012
- 1e-3 seed 2 is "unclear" only by 0.001 below the threshold; its log loss matches the two healthy runs; the threshold was fixed in advance and is not changed after the fact
- At 1000 steps 1e-3 learns a bit more slowly than the two healthy 4e-3 runs (0.655 vs 0.667 / 0.714); formal runs use 3000 steps
- Limited evidence (3 seeds per group); every formal run is checked against the 4.4 criteria

**4.6 First formal comparison: r = 0 vs r = 1 (2026-10-11)**

lr 1e-3, 3000 steps, batch 8, seed 0, reading from pools. Models in `checkpoints/lr1e-3/`, scores in `results/lr1e-3/`, log `logs/grid_lr1e-3_r0_r1.log`. Both models healthy (breast cancer 0.976 / 0.990).

| Model | Finance AUC | Finance ECE | Finance Brier | General AUC | General ECE | General Brier |
|---|---|---|---|---|---|---|
| r = 0 | 0.677 | 0.034 | 0.124 | 0.674 | 0.053 | 0.158 |
| r = 1 | 0.732 | 0.040 | 0.116 | 0.701 | 0.059 | 0.155 |
| Random forest | 0.779 | 0.044 | 0.108 | 0.734 | 0.074 | 0.144 |
| Logistic regression | 0.738 | 0.057 | 0.116 | 0.730 | 0.063 | 0.146 |

- **No crossover**: r = 1 is better on 7/7 finance tables (mean +0.055, largest heloc +0.18) and 13/17 general tables (mean +0.027; worse on Amazon_employee_access −0.085, E-CommereShippingData −0.064, NATICUSdroid −0.026, Bioresponse −0.010)
- The gain on finance tables is about twice the gain on general tables, so part of it is finance-specific
- On ECE, r = 0 is slightly better on both suites
- Single seed; needs more seeds

**Format check** ([diagnostics/format_check.py](diagnostics/format_check.py), `results/lr1e-3/format_check.json`): evaluation repeated after a per-column quantile transform to a normal distribution, which removes scale and tail differences.

| Model | Preprocessing | Finance AUC | General AUC |
|---|---|---|---|
| r = 0 | none / quantile | 0.677 / 0.699 | 0.674 / 0.665 |
| r = 1 | none / quantile | 0.732 / 0.728 | 0.701 / 0.690 |

r = 1's lead shrinks from 0.055 to 0.029 on finance tables (about half of it comes from heavy-tailed value scales) and from 0.027 to 0.025 on general tables (hardly any). The cause of the lead on general tables is unclear; candidates: the finance prior teaches handling class imbalance; the pure generic model is simply weak (0.674 on general tables, below logistic regression's 0.730).

**4.7 Seven values of r, seed 0 (2026-10-11)**

lr 1e-3, 3000 steps, batch 8. Log `logs/grid_lr1e-3_seed0.log`, figure [figures/rq1_curves_lr1e-3_seed0.png](../figures/rq1_curves_lr1e-3_seed0.png), health check `results/lr1e-3/health.json` ([diagnostics/checkpoint_health.py](diagnostics/checkpoint_health.py)).

| r | 0 | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1 |
|---|---|---|---|---|---|---|---|
| Finance AUC | 0.677 | 0.761 | 0.750 | 0.663 | 0.662 | 0.677 | 0.732 |
| General AUC | 0.674 | 0.732 | 0.724 | 0.655 | 0.647 | 0.648 | 0.701 |
| Finance minus general | 0.002 | 0.029 | 0.026 | 0.009 | 0.015 | 0.028 | 0.030 |
| Breast cancer AUC | 0.976 | 0.985 | 0.986 | 0.951 | 0.959 | 0.960 | 0.990 |
| Generic synthetic AUC | 0.681 | 0.710 | 0.715 | 0.669 | 0.647 | 0.657 | 0.685 |
| Finance synthetic AUC | 0.582 | 0.638 | 0.652 | 0.595 | 0.604 | 0.624 | 0.651 |

- **The curve is mostly training noise**: the finance and general lines rise and fall together; the three runs at r = 0.5 to 0.9 are weaker on every metric, even on finance synthetic tables, where they trail r = 0.3 (r = 0.9 saw three times as many finance tables as r = 0.3). Differences between runs reach 0.1 AUC, two to three times the r effect being measured (0.03 to 0.05)
- "Finance minus general" cancels out how well a single run trained and rises roughly with r (0.002 to about 0.03), a hint of a domain effect, but noisy
- Diagnosis: not enough training (24k tables per run, versus 80k in the original nanoTabPFN), so results depend heavily on randomness. Remedy: a GPU, batch 32, longer training

**4.8 First curve at batch 32 (2026-10-12, Mac)**

Batch 32, lr 4e-3, 3000 steps, seed 0, r = 0 / 0.3 / 0.5 / 0.7 / 1 (about 85 minutes each). Log `logs/grid_bs32_lr0.004_steps3000.log`, scores `results/bs32_lr0.004_steps3000/`, figure [figures/rq1_curves_bs32_lr0.004_steps3000.png](../figures/rq1_curves_bs32_lr0.004_steps3000.png). All 5 models healthy.

| r | 0 | 0.3 | 0.5 | 0.7 | 1 | Random forest |
|---|---|---|---|---|---|---|
| Finance AUC | **0.781** | 0.778 | 0.777 | 0.767 | 0.761 | 0.779 |
| General AUC | **0.738** | 0.733 | 0.734 | 0.729 | 0.727 | 0.734 |
| Finance ECE | 0.036 | 0.040 | 0.044 | 0.040 | **0.081** | 0.044 |
| General ECE | 0.045 | 0.047 | 0.043 | 0.045 | **0.111** | 0.074 |
| Generic synthetic AUC | 0.724 | 0.726 | 0.724 | 0.717 | 0.707 | |
| Finance synthetic AUC | 0.630 | 0.655 | 0.649 | 0.642 | 0.672 | |

- **With enough training the model is much stronger**: r = 0 matches or beats the random forest on both suites (finance 0.781 vs 0.779, general 0.738 vs 0.734); at batch 8 it reached only 0.677 / 0.674
- **No tradeoff on real tables**: as r grows, finance and general tables **both** get worse (finance −0.020, general −0.012). The pure generic prior is best even on finance tables
- **A tradeoff on synthetic tables**: as r grows, finance synthetic tables improve (0.630 → 0.672) and generic synthetic tables get worse (0.724 → 0.707). The model does learn the finance prior's patterns, but they do not transfer to real finance tables
- **Pure finance (r = 1) is badly calibrated**: more than twice the ECE of the other ratios; mixing in at least 30% generic tables fixes it (same direction as FinTFM's finding)
- **The finance suite splits** (r = 1 minus r = 0): consumer credit credit-g −0.098, credit_card_clients_default −0.041, Is-this-a-good-customer −0.025; ratio-based bankruptcy data polish +0.001 and taiwanese +0.003, as well as GiveMeSomeCredit +0.017 and heloc +0.003, are barely affected. The FinTFM prior simulates company statements and financial ratios, unlike consumer credit tables (demographics, categorical features)
- **Reverses the batch-8 conclusion**: at batch 8, r = 1 won on both suites (4.6); that came from an undertrained, weak pure generic model
- Single seed; on Colab, the same setting (r = 0) gave breast cancer AUC 0.983 to 0.987 over 3 seeds, so the seed-to-seed spread looks smaller than the r effect here, but confirming it needs the std on synthetic tables and suites

## 5. Decisions (2026-10-10)

1. Generic prior: `graph_scm` (TabICLv2)
2. Run plan (a) first to check training stability
3. TabArena's finance tables go into the finance suite and out of the general suite
4. No log transform for extreme values in finance tables for now
5. The confound between "finance patterns" and "format differences": ignored in the pilot, ablation after the curve exists

**Added 2026-10-11:**

6. Pre-generate pools and draw from them by r during training (2.2b)
7. Check the learning rate first (4e-3 vs 1e-3, r = 0, 3 seeds each, 1000 steps), then the formal runs
8. Formal runs use lr 1e-3 (4.5)

**Added 2026-10-13:**

9. At batch 32, back to the original lr 4e-3 (3 stable seeds on Colab, see the RESEARCH.md log of 2026-10-11 evening); decision 8 applies to batch 8 only
10. Refactor the repo following Jerry's task spec (Phase A), then prepare E1 to E6 (Phase B), run them on the company's SageMaker (Phase C)
11. Repo content is English only; code and pools are published at github.com/X66YSH/PriorResearch (pools as release `pools-v1`)

## 6. Open decisions

1. **The 4 banking-but-not-credit datasets** (section 3): become a third suite, `near_domain`, in Phase B.

## 7. File list

See section 1 and the "Layout" section of the [README](../README.md).
