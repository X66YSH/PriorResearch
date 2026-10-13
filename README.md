# PriorResearch: mixing domain and generic priors for tabular foundation models

How does mixing a domain prior (for example finance) into a generic prior (TabICLv2's `graph_scm`) when pretraining a tabular foundation model (nanoTabPFN) change its performance on real tables, and can that effect be predicted with a formula? Background and progress: [docs/RESEARCH.md](docs/RESEARCH.md). Experiment design and every result so far: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## 1. Install

Python 3.11. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # dependencies from uv.lock plus this project (src/tfmprior)
source .venv/bin/activate
```

Without uv (for example on SageMaker):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps
```

Use Python 3.11 to match the reference environment: `requirements.txt` resolves numpy to a different version on 3.12.

## 2. Smoke test (no data needed, about 2 minutes)

```bash
pytest
```

The tests build tiny prior pools of a few dozen tables, train, checkpoint and evaluate on the CPU, and check key properties (pools are reproducible, mixture shares are right, checkpoints hold the schedule-free evaluation weights, and so on). The evaluation part needs the benchmark cache and is skipped without it.

## 3. Data

**Prior pools** (2.4 GB, too large for git; stored as assets of the GitHub release `pools-v1`). Either:

```bash
python scripts/download_pools.py                  # download the reference pools and check their sha256 (recommended)
python scripts/make_dump.py --all --workers 8     # or generate them: about 10 minutes on an M-series Mac, hours on slow CPUs
python scripts/make_dump.py --verify              # either way, compare content checksums with artifacts/checksums.json
```

**Benchmark suites** (real tables from TabArena):

```bash
python scripts/evaluate.py --baselines            # the first run downloads them from OpenML into artifacts/benchmark_cache/
```

## 4. Run an experiment

One experiment is one file in `configs/experiments/`, overriding `configs/base.yaml`:

```bash
python scripts/train.py --config configs/experiments/smoke.yaml              # tiny model, 10 steps, a few minutes
python scripts/train.py --config configs/experiments/smoke.yaml --list       # only list the runs it would start
python scripts/train.py --config configs/experiments/smoke.yaml train.steps=20 --overwrite   # override on the command line
```

Each run gets its own directory `runs/<experiment>/<run name>/` with the resolved config, git commit, environment, training log, checkpoints and evaluation results; see the docstring at the top of [src/tfmprior/run.py](src/tfmprior/run.py). An existing run directory is an error; `--resume` skips finished runs and `--overwrite` reruns.

## 5. Layout

```
configs/               base.yaml and one file per experiment
src/tfmprior/          library: priors/, data/ (pool generation, mixture loader), model/, train/, eval/, utils/
scripts/               entry points: make_dump.py, download_pools.py, train.py, evaluate.py, verify_golden.py, diagnostics/
tests/                 pytest
golden/                answers recorded before the refactor; scripts/verify_golden.py checks that results did not change
docs/                  RESEARCH.md, EXPERIMENTS.md
artifacts/             checksums.json is in git; dumps/ and benchmark_cache/ are not
upstream_nanotabpfn/   the original nanoTabPFN README and notebook, for reference only
```

## 6. Sources and licenses

- nanoTabPFN (Pfefferle et al., 2025): original model and training loop; see [LICENSE](LICENSE) and [CITATION.bib](CITATION.bib)
- TabICL (BSD-3): generic priors, installed as a dependency
- FinTFM (Apache-2.0): finance prior, copied verbatim into `src/tfmprior/priors/_fintfm_financial.py`; the file header names the source commit
