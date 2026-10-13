# TFM prior research: overview

Last updated: 2026-10-13.

This is the entry point for the whole project: topic, research questions, plan, progress and literature. Experiment design, code and numbers are in [EXPERIMENTS.md](EXPERIMENTS.md).
Both documents are updated with every step forward: new progress goes into the log in section 5, experiment details into EXPERIMENTS.md.

## 1. Background

- **People:** Jerry Li (ML Engineer, BMO Institute for Applied AI & Quantum), reporting to Bo Zhao (Tech Lead).
- **Two goals:**
  1. **Product:** Bo's "drag-a-table": any BMO team drops in a table and gets predictions from a tabular foundation model (TFM).
  2. **Paper:** a top-conference paper through the team's research mandate. BMO allows publishing as long as there is no external collaboration.
- **Data rule:** the paper uses public data only. BMO internal tables may validate the product but cannot appear in the paper.
- **2026-10-08:** Jerry presented the deck to Bo. Bo likes the topic and thinks it has a real chance; his intuition is that, like a person, a model needs both general and domain knowledge to do best. He asked to move fast.

## 2. Research topic

**In one line:** bring LLM data mixing laws to TFM priors. Study the tradeoff curve of mixing a finance prior with a generic prior, fit a law for it, and find a metric that predicts it before training.

| | Question | Approach |
|---|---|---|
| **RQ1** | Does the curve exist? | Finance share r from 0% to 100%: how much do finance tables gain and general tables lose |
| **RQ2** | Can a law predict it? | Fit on small models and a few ratios (following Ye et al.'s L = c + k·exp(t·r)), extrapolate to larger models and unseen ratios |
| **RQ3** | Can it be predicted without training? | Estimate the best ratio before training from two metrics: how finance-like and how diverse the prior is |

**Bar for a main conference:** RQ1 alone is workshop level (like the Hutter group and CSIRO). RQ2 and RQ3 reach the level of Ye and Apple. Main-conference papers in this area all have three things: a predictive formula, extrapolation validated on unseen settings, and a concrete saving.

**Caution:** "mixing beats pure" is intuitive, so reviewers may find it obvious. The paper's value must lie in (1) where the optimum is and how to compute it (RQ2), and (2) non-obvious findings, such as FinTFM's tree prior improving general tables while hurting finance tables.

**How we differ from existing work:**

| Work | Its limitation |
|---|---|
| FinTFM | Only 3 ratios, 1 seed, no law |
| RAD-TFM | Measures in-domain gains only, never the out-of-domain cost |
| MITRA | Picks one generic mixture by heuristic criteria and does not predict domain tradeoffs; FinTFM found its criteria predicted the wrong direction on credit data |
| Ye, Apple | Have a law, but for LLMs only; LLM data is finite, while our priors generate unlimited tables |

**What it means for the product:** the answer decides whether drag-a-table ships one generic model or also a finance version, and how much finance that version mixes in.

**Risks:** Bristol found that the gap between synthetic and real data has no clear effect on performance; the Hutter group found that similarity explains performance only partly. A flat curve would itself be a finding, but the main-conference target would need rethinking.

## 3. Method in brief

A **prior** is a program that generates random practice tasks: each time it draws a random rule (a small random model that is never trained) and builds a table from it. A TFM practices on millions of such tables: "look at a few labelled example rows, guess the rule, predict new rows", i.e. in-context learning. Different priors are different assumptions about what real-world patterns look like.

**Our experiments (the RQ1 pilot), details in [EXPERIMENTS.md](EXPERIMENTS.md):**

| Item | Setting |
|---|---|
| Model | nanoTabPFN, about 0.36M parameters, fixed architecture, trained from scratch every time |
| Generic prior | TabICLv2's prior (`graph_scm`) |
| Finance prior | FinTFM's finance prior (financial statements, financial ratios, low default rates) |
| The only variable | Finance share r ∈ {0, 0.1, 0.3, 0.5, 0.7, 0.9, 1}, 3 seeds per r, 21 runs in total |
| Benchmark suites | Binary tasks from TabArena: 7 finance (credit, default, bankruptcy), 17 general |
| Metrics | AUC (ranking), ECE and Brier (calibration) |
| Hardware | So far Jerry's Mac (16 GB, MPS), about 35 minutes per run |

## 4. Timeline

**Deadlines (checked 2026-10-08):** ICLR 2027 (Sep 25) and AISTATS 2027 (Oct 6) have passed. ICML 2027 expected around 2027-01-22 (abstract about a week earlier; not yet announced; held in South America). NeurIPS 2027 expected around 2027-05-21. KDD 2027 cycle 2 around February 2027, UAI 2027 around 2027-02-12. ICML/ICLR workshops are the fallback.

**Working back from ICML:**

| When | What |
|---|---|
| October | Pilot: RQ1 |
| November to mid-December | RQ2 |
| Mid-December to mid-January | Write the paper; RQ3 as far as possible |
| Around January 22 | Submit |

**arXiv strategy:** post once RQ1 and RQ2 both hold (around December) to establish priority; post earlier if a very close competitor appears. Do not post with RQ1 only: too thin, and it would reveal the direction.

## 5. Progress log

### 2026-10-08
- Presented the deck (21 slides) and talking script to Bo; Bo agreed with the direction and asked to move fast.
- Wrote a handoff document and continued in a new conversation.

### 2026-10-09
- Read the source of FinTFM, TabICL and TFM-Playground and settled how to plug in the priors: TabICL for the generic prior, only the finance part of FinTFM for the finance prior, and our own mixer. FinTFM's own mixer was not used because its tree prior takes its share out of the finance budget, so `p_financial` is not the real finance share.
- Wrote the prior mixing, batch assembly and training entry point; ran end to end on the Mac.
- Pitfalls: TabICL uses Python's `random` module, which broke reproducibility (fixed); the Mac GPU caches memory per batch shape and ran out of memory (fixed by limiting batch shapes).
- Finding: the two priors' tables look very different. 56% of finance tables have under 5% positives, finance tables have 8% missing values and values up to 1e12; generic tables are nearly balanced with no missing values.
- Finding: 256-row, 24-feature tables trained for 600 steps were very unstable (AUC 0.226 and 0.877 for two seeds).

### 2026-10-10
- Jerry approved 5 decisions (EXPERIMENTS.md section 5), including switching the generic prior to TabICLv2's `graph_scm`.
- **Fixed a bug in the original train.py:** the schedule-free optimizer was not switched to eval mode at the end of training, so saving and evaluation used the training weights. All earlier results were affected.
- Built the two benchmark suites and the evaluation script. Random forest reference: finance AUC 0.779, general AUC 0.734.
- Wrote `run_grid.py` (batch training plus evaluation) and `plot_curves.py` (tradeoff curves).
- **Plan (a) passed:** 256 x 24, 3000 steps, breast cancer AUC 0.977 and 0.982 for two seeds, stable training; AUC 0.69 to 0.70 on unseen synthetic tables, close to the random forest's 0.707.

### 2026-10-11 (early morning)
- Found and fixed `run_grid.py` reusing a stale checkpoint.
- Started r = 0 and r = 1, then stopped them because the machine had to be shut down; rerun the next morning.
- Created this document; first git commit of the code.

### 2026-10-11 (morning)
- Ran r = 0 and r = 1 (tables generated on the fly, 3000 steps, batch 8, lr 4e-3). **The r = 0 run collapsed midway:** loss stuck at 0.69 (ln 2), AUC 0.50 on every suite. r = 1 was fine: finance AUC 0.744, general AUC 0.717 (random forest 0.779 / 0.734). Not reproducible; archived as `checkpoints/onthefly_prehashfix_2026-10-11/` and not used for conclusions.
- **Found a reproducibility bug:** TabICL's output depends on Python's per-process hash seed, so the same seed produced different tables in different processes. Fixed (`PYTHONHASHSEED` is fixed automatically). Plan (a)'s success and this collapse had therefore seen different data.
- **Unstable training:** 256 x 24 tables, batch 8, lr 4e-3: 2 of 4 runs failed. Likely cause: lr 4e-3 was tuned for batch 32 in the original nanoTabPFN and was not lowered when the batch went down to 8.
- **Switched to pre-generated pools (Jerry's suggestion):** 100k tables each of generic (graph_scm) and finance, stored as h5 (`make_prior_dump.py`). Training now draws from the pools by ratio (`DumpMixedDataLoader`): 1 to 3 ms per batch instead of 280 ms, fully reproducible across processes, no table repeated within a run, and common random numbers across r.
- **Learning-rate check:** r = 0, 1000 steps, 3 seeds each. 4e-3 failed once (3 of 7 overall); 1e-3 learned in all three runs. Formal runs switched to 1e-3.
- **First formal comparison (r = 0 vs r = 1, lr 1e-3, seed 0): no crossover; the pure finance model was better on both suites.** Finance AUC 0.732 vs 0.677 (better on 7/7 datasets), general AUC 0.701 vs 0.674 (13/17). ECE slightly better for pure generic. A format check showed that about half of the lead on finance tables came from heavy-tailed value scales, but not the lead on general tables. Details in EXPERIMENTS.md 4.6.
- **Seven r values (seed 0): the curve is mostly training noise.** Finance and general lines rose and fell together; the three runs at r = 0.5 to 0.9 were weaker across the board; differences between runs reached 0.1 AUC, two to three times the effect being measured. "Finance minus general" rose roughly with r (0.002 to about 0.03), a hint of a domain effect. Conclusion: not enough training on the Mac; move to a GPU (batch 32, longer). Details in EXPERIMENTS.md 4.7.

### 2026-10-11 (evening) to 2026-10-12 (morning)
- **Learning-rate check on a Colab T4 (batch 32, r = 0, 3000 steps):** lr 4e-3 completed 3 seeds with breast cancer AUC 0.987 / 0.986 / 0.983 and smooth training, no collapse (seed 2, which collapsed on the Mac at batch 8, was fine). Conclusion: back to the original 4e-3 at batch 32; 1e-3 no longer tested. About 37 minutes per run on a T4. The health-check table is in Drive under `tfm_prior/results/` and has not been retrieved.
- Colab and Mac agree: the same model gets identical evaluation scores, and the first training steps have identical losses.
- Colab's free quota ran out. **Started part of the formal experiment on the Mac on 2026-10-12 at 08:10:** batch 32, lr 4e-3, 3000 steps, seed 0, r = 0, 0.3, 0.5, 0.7, 1 (about 85 minutes each on the Mac), with automatic evaluation, health checks and plots.
- **First curve at batch 32 (seed 0, r = 0 / 0.3 / 0.5 / 0.7 / 1): no tradeoff on real tables; the pure generic prior is best on both suites.** Finance AUC 0.781 → 0.761, general AUC 0.738 → 0.727 as r goes from 0 to 1; r = 0 matches or beats the random forest. On synthetic tables the tradeoff is clear: finance synthetic 0.630 → 0.672, generic synthetic 0.724 → 0.707. The model learns the FinTFM prior's patterns, but they do not transfer to real finance tables. Pure finance has more than twice the ECE of the others. Within the finance suite, consumer credit suffers most (credit-g −0.098) while ratio-based bankruptcy data barely changes. This reverses the batch-8 conclusion that r = 1 wins everywhere, which came from undertraining. Details in EXPERIMENTS.md 4.8.

### 2026-10-13
- **Change of focus** (Jerry's task spec): the new hypothesis is that a domain prior's value depends on the training budget and on the strength of the generic prior. The paper's main line becomes: how does a domain prior's value change with training budget and generic-prior strength (E1 to E5), can a mixing law predict it (RQ2, E6), and can it be predicted before training (RQ3).
- **A direct competitor appeared, CreditICL** (github.com/andreasgoethals/CreditICL, KU Leuven, MIT license): sweeps credit_fraction ∈ {0, 0.5, 1} on TabICLv2, no results yet. We differ in budget dependence, the law and the predictive metric.
- **Phase A refactor done:** library code in `src/tfmprior/`, one YAML per experiment (`configs/`), one directory per training run. Golden answers were recorded with the old code before the refactor (`golden/`) and the new code reproduces all of them: pools byte-identical; source and pool index of all 640 sampled tables identical; evaluation differences 0 (model) and 4e-17 (random forest); every step's loss and the final parameters of a 200-step run identical bit for bit on both MPS and CPU.
- Two side findings: MPS reproduces itself (two runs on the same device are identical bit for bit), but MPS and CPU differ from step 1 (7th decimal) and by 0.017 in loss after 200 steps, so bit-identity is only checked on the same device. Switching schedule-free between train and eval mode is not exactly invertible in floating point, so the pre-refactor runs that evaluated every 500 steps were slightly perturbed; the new code restores a snapshot of the training weights after each checkpoint, so checkpoints no longer change the trajectory (covered by a test).
- Code pushed to github.com/X66YSH/PriorResearch (public); the prior pools are assets of its release `pools-v1`. Repo content is English only.

## 6. Next steps

**Phase B (after Jerry confirms Phase A):** three evaluation improvements (mutual-information feature selection, near-domain suite, per-dataset wins and losses), new priors (format_aug, credit_icl), synthetic validation sets, pool sizes, E1 to E6 configs, aggregation and plots.

**Phase C:** `docs/RUN_ON_SAGEMAKER.md`, parallel runs with `sweep.py`.

**To raise with Bo:**
- How long BMO's external publication approval takes (arXiv needs it too); more urgent now that CreditICL exists
- Whether moving code to SageMaker through a personal GitHub repo is compliant

**Offered earlier, not done yet:**
- A one-page two-week plan for Bo
- A weekly arXiv watch for competing work (keywords: tabular foundation model prior, domain prior, prior mixture, CreditICL)
- Fix the venues in the reading list (RAD-TFM and Explaining Data Mixing Scaling Laws are both ICML 2026 workshop papers)

## 7. Literature

Reading list (30 sources, 5 tables): https://claude.ai/code/artifact/dffb466c-9356-481c-9271-2ad03385c84f

### 7.1 Prior basics

| Paper | Key point |
|---|---|
| Müller et al., ICLR 2022 | PFN theory: training on prior-generated tasks amounts to learning Bayesian inference |
| TabPFN, ICLR 2023 / Nature 2025 | First to turn this idea into a tabular model |
| Prior vs synthetic data generation (CTGAN, TabDDPM) | One "writes new problems", the other "copies". No paper directly compares pretraining a TFM on CTGAN data with pretraining on a prior |
| Hutter group, ICML 2026 workshop | On the same real tables, one task per table ranks last; random targets with many tasks per table do much better |
| Ma et al., arXiv 2511.09665 | One real table turned into many tasks (different y and feature subsets) already teaches generalization |

### 7.2 Does a prior that looks like real data help?

| Paper | How "looks like" is measured | Finding | Limitation |
|---|---|---|---|
| CSIRO, arXiv 2609.06912 | 21 handcrafted statistics, k-NN coverage | TabICLv2 covers real tables best and performs best; "closer is better" on 72% to 86% of datasets | Correlation only; prior and architecture change together |
| Hutter group, arXiv 2606.29241 | Handcrafted statistics, similarity between priors | Fixed nanoTabPFN with 11 priors: the prior matters, but modestly; similarity only partly predicts performance | Small model, 16 datasets |
| Bristol, arXiv 2605.06343 | Discriminator AUC, k-NN coverage | TabICL's prior is almost perfectly separable from real tables, and 86k configurations cannot close the gap; yet the gap shows no clear link to performance | Distribution analysis only, no new models trained |
| MIT, arXiv 2603.04692 | TabPFN 2.5 embeddings plus an XGBoost discriminator | Continued training on the 200 most engineering-like synthetic tables improved 29 of 35 engineering datasets | No random-selection control; general ability not tested |

### 7.3 Domain priors and their cost

| Work | Key point | Limitation |
|---|---|---|
| FinTFM (github.com/kabartay/fintfm) | 100% finance AUC 0.757 / ECE 0.0106; 70/30 mix AUC 0.751 / ECE 0.0042; generic only AUC 0.708. Adding a tree prior: general +0.0094, credit −0.0221. A MITRA-style scorer predicted the wrong direction on finance | One-person project, weak model (rank 93 to 94 of 95 on TabArena), many comparisons with a single seed |
| RAD-TFM (IBM / MIT, ICML 2026 workshop) | Only re-weights the knobs of a generic generator: 64 meta-features plus Wasserstein alignment to a target domain, then sampling weighted by how far it trails XGBoost. Normalized AUC up to +0.12, raw AUC only +0.3% to +2.1% | The "domain" is the test benchmark itself; out-of-domain cost never measured; the two steps are not ablated |
| **CreditICL** (KU Leuven, github.com/andreasgoethals/CreditICL, MIT license; found 2026-10-13) | Adds a credit PD prior to TabICLv2 and sweeps credit_fraction ∈ {0, 0.5, 1}; **direct competitor**, no results yet | Only 3 ratios, no training-budget dimension, no law |
| KnowsTFM, arXiv 2606.30258 | Knowledge-graph attention bias at fine-tuning: small models 0.741 → 0.768; large models get worse with any fine-tuning (TabPFN v2.6: 0.892 → 0.862) | Fine-tuning, not a prior; biomedical domain |

### 7.4 LLM data mixing laws

| Paper | Key point |
|---|---|
| Ye et al. (Fudan / Shanghai AI Lab), ICLR 2025, arXiv 2403.16952 | L_i(r) = c_i + k_i·exp(Σ_j t_ij·r_j); about 20 mixtures on 70M to 410M models, chained with step and size scaling laws to extrapolate to 1B / 100B tokens; the optimized mixture equals the default trained for 48% more steps; predicts how much old data to keep in continued pretraining |
| Apple (Shukor et al.), NeurIPS 2025, arXiv 2507.09404 | One formula over mixture, model size and data size; about 10 to 20 runs; extrapolation error about 1% to 2% for LLMs; assumes no data repetition (which our priors satisfy) |
| Explaining Data Mixing Scaling Laws (Dai, Zheng), ICML 2026 workshop | Why the laws hold: with fixed capacity, domains compete for it. Abstract only |
| Specialist vs generalist | Text: the generalist mostly wins (BloombergGPT vs GPT-4); retina: the specialist wins narrowly; PPG: the specialist +27%. Tables: unknown |

## 8. Related files

| File | Where |
|---|---|
| Experiment details | [EXPERIMENTS.md](EXPERIMENTS.md) |
| Deck for Bo (21 slides) | `TFM_Prior_deck.pptx` (repo root, never in git: BMO internal material) |
| Talking script | `Bo_meeting_script.md / .pdf` (Jerry has it) |
| Handoff document | `TFM prior handoff` in the claude.ai project "Research & Product" |
| Reading list | Link at the top of section 7 |
