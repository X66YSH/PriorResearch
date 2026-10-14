# PriorResearch

RQ1: how does the finance-prior share in nanoTabPFN pretraining affect performance on finance and general tabular benchmarks?

Run [rq1_bs32_curve.ipynb](rq1_bs32_curve.ipynb). The notebook holds the experiment settings and analysis; `src/tfmprior/` supplies the model, pool loader, and training loop.

## SageMaker setup

Use a GPU instance and Python 3.11. From the project root:

```bash
uv sync --locked
uv run python -m ipykernel install --user --name priorresearch --display-name "PriorResearch (uv)"
```

Open the notebook, select **PriorResearch (uv)**, then use **Restart Kernel and Run All** after preparing the data. The first cell requires CUDA and stops on a CPU-only environment.

`pyproject.toml` declares dependencies, `uv.lock` fixes their versions, and `.python-version` selects Python 3.11. nanoTabPFN is included in `src/tfmprior/model/nanotabpfn.py`; no separate TabPFN package is needed. Create the environment on SageMaker rather than copying a Mac `.venv/`.

## Input data

The benchmark cache is included in Git and arrives with `git clone`. The large prior pools are excluded from Git; download them from the release below or copy this directory from the prepared machine:

```text
artifacts/dumps/
```

The notebook checks for all four pools:

| File in `artifacts/dumps/` | Tables | Purpose |
|---|---:|---|
| `generic_graph_scm_100k_256x24_seed0.h5` | 100,000 | Generic training |
| `finance_100k_256x24_seed0.h5` | 100,000 | Finance training |
| `generic_graph_scm_1k_256x24_seed1.h5` | 1,000 | Held-out; currently only checked for presence |
| `finance_1k_256x24_seed1.h5` | 1,000 | Held-out; currently only checked for presence |

Alternatively, download these pools and `SHA256SUMS` from the [pools-v1 release](https://github.com/X66YSH/PriorResearch/releases/tag/pools-v1) into `artifacts/dumps/`. On SageMaker, verify them with:

```bash
cd artifacts/dumps
sha256sum -c SHA256SUMS
cd ../..
```

`artifacts/checksums.json` preserves the reference content checksums and pool generation settings.

The versioned benchmark cache contains 26 task files (about 1.4 MiB): 24 usable datasets (7 finance and 17 general), plus 2 skip markers. Each usable dataset has 3 fixed subsamples. Use these committed files to preserve the same evaluation samples across machines; no OpenML download is needed. If cache files are accidentally removed, restore them from the repository:

```bash
git restore artifacts/benchmark_cache/
```

The notebook stops when expected files are missing; it does not download them. Its cache reader assigns all non-finance files to General, so keep this directory limited to the reference tasks. The two held-out pools are not yet used for synthetic validation scoring in this notebook.

## Experiment and results

- Finance shares: `0`, `0.3`, `0.5`, `0.7`, `1`; seeds: `0`, `1`, `2` (15 runs).
- Each run: 3,000 steps, batch size 32, learning rate 0.004; 96,000 tables drawn from the two pools without repetition within a run.
- Baselines: Random Forest and Logistic Regression. All models use the same cached samples and stratified five-fold evaluation.
- Metrics: AUC, ECE, Brier. Average repeats within datasets, datasets within suites, then report mean and standard deviation across training seeds.

Results go to `runs/<EXPERIMENT_NAME>/r<share>_seed<seed>/`: `checkpoint.pt`, `training_log.csv`, and `results.csv`. Completed runs are reused; a checkpoint without results is evaluated again. Training interrupted before its final checkpoint starts over.

Use a new `EXPERIMENT_NAME` when changing input data, benchmark membership, preprocessing, evaluation code, or experiment settings. Resume checks do not fingerprint all data and code. Retain `runs/` on storage that persists for your SageMaker setup.

Baseline summaries and plots stay in memory unless saved explicitly. Save the notebook after execution to retain displayed outputs.

## Layout

```text
PriorResearch/
├── rq1_bs32_curve.ipynb
├── src/tfmprior/
├── pyproject.toml
├── uv.lock
├── .python-version
├── README.md
├── LICENSE
├── CITATION.bib
├── .gitignore
└── artifacts/
    ├── checksums.json
    ├── dumps/
    └── benchmark_cache/
```

Git metadata and the local `.venv/` may also exist. `runs/` is created by the experiment and excluded from Git.

## Sources

- nanoTabPFN: model and training loop; see [LICENSE](LICENSE) and [CITATION.bib](CITATION.bib).
- TabICL 2.2.0: generic prior, installed as a dependency.
- FinTFM: finance prior vendored in `src/tfmprior/priors/_fintfm_financial.py`; its Apache-2.0 source and commit attribution are preserved.
