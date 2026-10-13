# 标准答案（golden）

重构之前（commit `6332ed5`）用旧代码记录的结果。重构后的代码必须完全复现它们，以证明重构没有改变任何结果。

## 文件

| 文件 | 内容 | 设置 |
|---|---|---|
| `pools.json` | 4 个题库文件的 sha256、内容校验值、表数，以及各自前 5 张表的哈希值 | |
| `loader.json` | 抽题：前 20 个 batch 里每张表的来源（generic / finance）和它在题库中的下标，以及每个 batch 的 train/test 切分点 | r = 0.3，seed 0，batch 32 |
| `train_<device>.json` | 训练：每一步的 loss（完整 200 步）、最终参数的 sha256、抽到的金融表数 | r = 0.3，seed 0，batch 32，lr 4e-3，200 步，graph_scm |
| `eval_<device>.json` | 评估：每个数据集、每次重复、每个指标（AUC / ECE / Brier），包括一个已训好的模型和随机森林 | 模型：`checkpoints/bs32_lr0.004_steps3000/r0.00_seed0.pth`（重构后移到 `legacy_results/checkpoints/...`） |

`<device>` 是 `mps`（Mac GPU）或 `cpu`。训练 loss 只在同一种设备上要求逐位一致。`train_mps_run2.json` 是在 MPS 上重跑的第二次，用来确认 MPS 本身是否可复现。

## 怎么生成（旧代码）

```bash
git checkout 6332ed5            # 重构前的最后一个 commit
python golden/make_golden_legacy.py --device mps --part all
python golden/make_golden_legacy.py --device mps --part train --suffix _run2
python golden/make_golden_legacy.py --device cpu --part train
```

脚本只调用旧代码里已有的函数（`train.py`、`priors/`、`benchmark/`、`evaluate.py`），不修改它们；训练 loss 是通过包一层 `CrossEntropyLoss` 记录下来的。

## 怎么检查（新代码）

```bash
python scripts/verify_golden.py --device mps      # 题库、抽题、评估、训练四项
python scripts/verify_golden.py --device cpu --part train
```

通过标准：题库和抽题完全一致；同一设备上训练 loss 序列和最终参数逐位一致；评估指标差异小于 1e-6。
