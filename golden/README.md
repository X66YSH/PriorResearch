# Golden answers

Results recorded with the code before the refactor (commit `6332ed5`). The refactored code must reproduce them exactly, which shows that the refactor changed no result.

## Files

| File | Content | Settings |
|---|---|---|
| `pools.json` | sha256, content checksum and table count of the 4 pool files, plus hashes of the first 5 tables of each | |
| `loader.json` | Table sampling: the source (generic / finance) and pool index of every table in the first 20 batches, and each batch's train/test split | r = 0.3, seed 0, batch 32 |
| `train_<device>.json` | Training: the loss of every step (all 200), the sha256 of the final parameters, the number of finance tables drawn | r = 0.3, seed 0, batch 32, lr 4e-3, 200 steps, graph_scm |
| `eval_<device>.json` | Evaluation: every dataset x repeat x metric (AUC / ECE / Brier) for a trained model and for the random forest | model: `checkpoints/bs32_lr0.004_steps3000/r0.00_seed0.pth` (moved to `legacy_results/checkpoints/...` by the refactor) |

`<device>` is `mps` (Apple GPU) or `cpu`. Training losses are only required to match bit for bit on the same device. `train_mps_run2.json` is a second MPS run, recorded to confirm that MPS reproduces itself.

## How they were recorded (old code)

```bash
git checkout 6332ed5            # last commit before the refactor
python golden/make_golden_legacy.py --device mps --part all
python golden/make_golden_legacy.py --device mps --part train --suffix _run2
python golden/make_golden_legacy.py --device cpu --part train
```

The script only calls functions that already existed in the old code (`train.py`, `priors/`, `benchmark/`, `evaluate.py`) and never changes them; the training loss is recorded by wrapping `CrossEntropyLoss`.

## How to check them (new code)

```bash
python scripts/verify_golden.py --device mps      # pools, loader, evaluation and training
python scripts/verify_golden.py --device cpu --part train
```

Pass criteria: pools and loader identical; on the same device, the training loss sequence and the final parameters identical bit for bit; evaluation metrics within 1e-6.
