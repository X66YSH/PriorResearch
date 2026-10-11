# 在 GPU 机器上运行

写于 2026-10-11，从 Mac（16 GB，MPS）转到 GPU。所有命令都在 repo 根目录下运行。

## 1. 带过去的东西

整个包解压后就是一个完整的 repo，不需要再联网下载数据：

| 内容 | 位置 | 大小 | 说明 |
|---|---|---|---|
| 代码和文档 | repo 根目录，含 `.git` | 约 1 MB | 可以在 GPU 机器上继续 commit |
| 两个题库 | `prior_dumps/*_100k_*.h5` | 2.3 GB | **必须带**：保证和 Mac 上用的是完全相同的题 |
| 两个小题库（健康检查用） | `prior_dumps/*_1k_*_seed1.h5` | 30 MB | 模型没见过的合成表，notebook 的健康检查用 |
| 两套考卷 | `benchmark/cache/*.npz` | 1.5 MB | 带上就不用连 OpenML |
| Mac 上的 seed 0 结果 | `checkpoints/lr1e-3/`、`results/lr1e-3/` | 10 MB | 用来对照 |

**没带的：** `.venv/`（要在 GPU 机器上重装）、`300k_150x5_2.h5`（原版 nanoTabPFN 的数据，我们不用）、`TFM_Prior_deck.pptx`、`logs/`。

## 2. 装环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements-gpu.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
最后一行要打印出 `True` 和 GPU 型号。如果是 `False`，按 `requirements-gpu.txt` 开头的注释装 CUDA 版的 torch。

## 推荐：用 notebook 一步一步跑

打开 **[rq1_gpu.ipynb](rq1_gpu.ipynb)**，从上往下一个 cell 一个 cell 地跑，下面第 3 到 5 节的事它都包含了：

1. 先把第一个 cell 里的 `QUICK_TEST` 改成 `True`，从头跑一遍（几分钟），确认环境没问题
2. 改回 `False`，正式跑。第 3 步要根据第 2 步的结果手动填学习率和步数
3. **每一步的结果都会自动写进 `results/gpu_report.txt`**，跑完把这一个文件发回来就行。某一步重跑，只替换那一段

notebook 训练和健康检查都只读预先造好的题库，所以在 Jupyter 里也完全可复现（不受 `PYTHONHASHSEED` 影响）。
notebook 由 [build_notebook.py](build_notebook.py) 生成，要改 notebook 的内容，改这个脚本再运行 `python build_notebook.py`。

下面第 3 到 5 节是同样的事情，用脚本在终端里跑的版本。

## 3. 确认和 Mac 上一致（约 5 分钟）

```bash
# 3a. 考卷和评估一致：应该和 Mac 完全一样，金融 AUC 0.7791，通用 AUC 0.7341
python evaluate.py --baselines --out_dir results/gpu_check

# 3b. 用 Mac 训好的模型在 GPU 上评估：应该和 Mac 上的 0.6767 / 0.6744 非常接近（GPU 浮点运算可能差在小数点后第 3 到 4 位）
python evaluate.py checkpoints/lr1e-3/r0.00_seed0.pth --out_dir results/gpu_check

# 3c. 训练能跑：50 步，几十秒
python train_mixed.py --r 0.3 --seed 0 --steps 50 --batch_size 32 --lr 4e-3 --steps_per_eval 25 --out_dir checkpoints/gpu_check
```

## 4. 第一个任务：batch 32 下的学习率和训练噪声

Mac 上的问题：batch 8 时，学习率 4e-3 会随机崩溃；1e-3 不崩，但不同训练之间的差异可达 0.1 AUC，比要测的效应大（EXPERIMENT.md 4.5、4.7）。
GPU 上用回原版 nanoTabPFN 的 batch 32，先测：

```bash
python -m diagnostics.lr_stability --lrs 4e-3 1e-3 --seeds 0 1 2 3 4 --steps 3000 --batch_size 32 \
    --out results/lr_stability_bs32/summary.json
```

- 每次训练用 3000 x 32 = 96,000 张表，题库有 10 万张，够用
- 要看的：哪个学习率 5 个 seed 都健康，以及**健康的那组 seed 之间的合成表 AUC 差多少**。差距要明显小于 0.03，正式实验才有意义
- 如果需要更多步数，先造更大的题库，否则会报 "pool is used up"：
  `python make_prior_dump.py --prior generic --n_tables 300000 --workers 16`（金融同理，`--prior finance`）

## 5. 正式实验

学习率和步数定下来之后（下面的 `<lr>` 和 `<steps>` 替换成选定的值）：

```bash
python run_grid.py --rs 0 0.1 0.3 0.5 0.7 0.9 1 --seeds 0 1 2 --steps <steps> --batch_size 32 --lr <lr> \
    --out_dir checkpoints/bs32_lr<lr>
python -m diagnostics.checkpoint_health checkpoints/bs32_lr<lr>/*.pth
python plot_curves.py --results_dir results/bs32_lr<lr> --out figures/rq1_curves_bs32_lr<lr>.png
```

## 6. 长时间运行

Mac 上用的 `caffeinate` 和 `osascript` 是 macOS 专用的。Linux 上用：
```bash
nohup python run_grid.py ... > logs/grid.log 2>&1 &
tail -f logs/grid.log
```
在 Jupyter 里，用 [rq1_gpu.ipynb](rq1_gpu.ipynb) 就行，它只读题库。只有现场造题（`make_prior_dump.py`、`MixedPriorDataLoader`）需要固定 `PYTHONHASHSEED`，这些请在终端里用脚本跑，[priors/__init__.py](priors/__init__.py) 会自动处理；在 notebook 内核里做不到，只会给出警告。

## 7. 注意

- 论文只能用公开数据。这个包里只有公开数据（TabArena）和合成数据，可以放心带
- 跑完后**把 `results/gpu_report.txt` 发回来**；需要更细的分析时，再带回 `results/`、`figures/`
