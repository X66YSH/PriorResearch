# 实验说明

最后更新：2026-10-13（Phase A 重构）。状态标记：✅ 已实现并测过，🟡 已实现但待验证，⬜ 未实现。

**关于文件路径：** 第 2 到 5 节是重构前写的，里面的文件路径（`priors/`、`train_mixed.py`、`checkpoints/` 等）是当时的位置。重构后的对应关系见第 1 节；重构前的输出都移到了 `legacy_results/`（不进 git）。

## 0. 要回答的问题

**RQ1：取舍曲线存在吗？** 金融 prior 的比例 r 从 0% 加到 100%，金融表上提升多少，通用表上损失多少？

做法：每个 r 训一个 nanoTabPFN，在两套考卷上测，画出两条曲线（x 轴 r，y 轴 AUC / ECE / Brier）。

2026-10-12 之后研究重心改为：领域 prior 的价值怎样随训练量和通用 prior 的强弱变化（见 [RESEARCH.md](RESEARCH.md)）。

## 1. 整体流程和对应的代码（2026-10-13 重构后）

```
prior (priors/registry.py) ─ scripts/make_dump.py ─ 题库 artifacts/dumps/*.h5 ─┐
                                                                               ├─ PoolMixtureLoader(mixture) ─ train() ─ runs/<实验>/<run>/
                                         configs/base.yaml + experiments/*.yaml ┘   按比例抽题、填 NaN、切分        存档、合成验证集、两套考卷
```

| 步骤 | 文件 | 运行方式 |
|---|---|---|
| prior | `src/tfmprior/priors/`：`generic.py`（graph_scm、mix_scm）、`fintfm.py`、`registry.py` | 被造题库调用 |
| 造题库 | `src/tfmprior/data/dump.py`、[scripts/make_dump.py](../scripts/make_dump.py) | `python scripts/make_dump.py --all` |
| 按比例抽题、拼 batch | `src/tfmprior/data/loader.py` 的 `PoolMixtureLoader` | 被训练调用 |
| 训练 | `src/tfmprior/train/loop.py`、`src/tfmprior/run.py`、[scripts/train.py](../scripts/train.py) | `python scripts/train.py --config configs/experiments/<实验>.yaml` |
| 考卷和评估 | `src/tfmprior/eval/`：`benchmarks.py`、`metrics.py`、`evaluate.py`、`synthetic.py`、`health.py` | 训练结束自动评估；`python scripts/evaluate.py <run 目录>` |
| 标准答案 | `golden/`、[scripts/verify_golden.py](../scripts/verify_golden.py) | `python scripts/verify_golden.py --device mps` |
| 诊断 | `scripts/diagnostics/`：`prior_stats.py`、`prior_learnability.py`、`format_check.py` | |

旧文件和新位置的对应：`priors/mixture.py` 的 `GenericPrior` → `priors/generic.py`；`priors/batching.py` 的 `DumpMixedDataLoader` → `data/loader.py` 的 `PoolMixtureLoader`（`impute_with_train_mean` → `data/preprocess.py`）；`train.py` 的 `train()` → `train/loop.py`；`train_mixed.py`、`run_grid.py` → `scripts/train.py`（并行的 `sweep.py` 在 Phase C）；`benchmark/` → `eval/benchmarks.py`、`eval/metrics.py`；`evaluate.py` → `eval/evaluate.py`、`scripts/evaluate.py`；`make_prior_dump.py` → `data/dump.py`、`scripts/make_dump.py`；`check_priors.py` → `scripts/diagnostics/prior_stats.py`；`prior_dumps/` → `artifacts/dumps/`；`benchmark/cache/` → `artifacts/benchmark_cache/`。

## 2. 造题和训练

### 2.1 两个 prior ✅

| | 用什么 | 代码 |
|---|---|---|
| 通用 prior | **TabICLv2 的 prior**（`graph_scm`），依据是 TabICL 官方脚本 `scripts/train_v2_clf_stage1.sh` 用的是 `--prior_type graph_scm`。用 `--generic mix_scm` 可以换回 v1 的 prior | [priors/mixture.py](priors/mixture.py) `GenericPrior`；生成代码在安装包 `.venv/.../tabicl/prior/`（tabicl 2.2.0） |
| 金融 prior | FinTFM `sample_financial_task()`：符合会计恒等式的财报、财务比率特征、违约率 0.1% 到 30% | [priors/fintfm_financial.py](priors/fintfm_financial.py)（从 FinTFM commit `7c297d0` 原样复制，只改 1 行 import） |

不用 FinTFM 自带的混合器：它的通用部分是作者自己写的 SCM；它的 tree prior 从金融预算里扣，`p_financial` 不等于真实的金融占比。

### 2.2 r 的定义 ✅

**每张表**以概率 r 来自金融 prior，否则来自通用 prior。r 按表的数量算，对应 data mixing law 里的 r。代码：[priors/batching.py](priors/batching.py) `DumpMixedDataLoader`。

### 2.2b 题库 ✅（2026-10-11）

| 题库 | 文件 | 大小 | 内容 |
|---|---|---|---|
| 通用 | `prior_dumps/generic_graph_scm_100k_256x24_seed0.h5` | 0.80 GB | 10 万张，tabicl 2.2.0，正例比例中位数 0.496，无缺失 |
| 金融 | `prior_dumps/finance_100k_256x24_seed0.h5` | 1.55 GB | 10 万张，FinTFM `7c297d0`，正例比例中位数 0.047，缺失 5.2% |

- 分块生成，每块 1000 张、seed 固定，所以不管用几个进程，文件都完全一样（已验证 1 个和 4 个 worker 的校验值相同）
- 缺失值原样保留，训练时按切分点填补
- 抽题规则：每个题库按 seed 打乱一次，依次往下取，一次训练内不重复；同一个 seed 下，决定来源的随机数在不同 r 之间共用，所以不同 r 只差在 r 本身
- 每批取题 1 到 3 毫秒（现场造题是 280 毫秒）

### 2.3 表的格式 🟡

| 项目 | 设定 | 理由 |
|---|---|---|
| 行数 | 256 | 金融 prior 需要足够多的行才能造出低违约率 |
| 特征数 | 最多 24，不足的补零到 24 | 和 FinTFM 默认值一致；全 0 列经模型标准化后仍为 0 |
| 类别 | 二分类 | 金融考卷是违约预测 |
| train/test 切分 | 每个 batch 从 {80, 96, ..., 224} 里随机选 | 只取 16 的倍数，batch 形状只有 10 种，避免 Mac GPU 缓存爆掉 |
| 缺失值 | 用 train 行的列均值填补 | `impute_with_train_mean()`，评估真实表时用同一个函数 |

256 乘 24、batch 8 时，学习率 4e-3 会随机崩溃，1e-3 稳定（见 4.4、4.5）。

### 2.4 模型和训练 🟡

| 项目 | 设定 |
|---|---|
| 模型 | nanoTabPFN：embedding 96，4 heads，MLP 192，3 层，2 个输出 |
| 优化器 | schedule-free AdamW，**学习率 1e-3**（原版 4e-3 是配 batch 32 的；batch 8 时 4e-3 会随机崩溃，见 4.5） |
| batch | Mac 上 8（16 GB 内存的限制），GPU 上计划 32 |
| 步数 | 3000（方案 (a)：1000 步后 breast cancer 基本不再上升） |
| 速度（Mac） | 从题库读题后每步约 0.39 秒，3000 步约 20 分钟（现场造题时是 0.67 秒） |
| 随机性 | 同一个 seed 完全可复现：题库文件固定；python / numpy / torch 三个 seed 都设了；`PYTHONHASHSEED` 自动固定为 0（[priors/__init__.py](priors/__init__.py)） |

**对原版 [train.py](train.py) 的两处修改：**
1. 每步之后在 Mac 上清一次 GPU 缓存
2. **修了一个 bug**：`train()` 结束时没有把 schedule-free 优化器切到 eval 模式，所以返回和保存的是"训练用参数"，不是"平均后的评估参数"。schedulefree 官方文档要求保存 checkpoint 时必须是 eval 模式。**2026-10-10 之前的所有结果都受影响**，包括 4.3 里 600 步的那组数字。

## 3. 考卷 ✅

两套考卷都来自 TabArena v0.1（OpenML suite 457）的二分类任务。"金融"定义为 **信贷风险、违约、破产**，也就是 FinTFM 金融 prior 模拟的东西。

| 套 | 数据集 |
|---|---|
| **金融（7 个）** | polish_companies_bankruptcy、taiwanese_bankruptcy_prediction、credit-g、credit_card_clients_default、GiveMeSomeCredit、heloc、Is-this-a-good-customer |
| **通用（17 个）** | Amazon_employee_access、Bioresponse、Diabetes130US、E-CommereShippingData、Fitness_Club、HR_Analytics、NATICUSdroid、blood-transfusion、churn、customer_satisfaction_in_airline、diabetes、hazelnut、in_vehicle_coupon、jm1、online_shoppers_intention、qsar-biodeg、seismic-bumps |
| 两边都不放（4 个） | Bank_Customer_Churn、bank-marketing、coil2000_insurance_policies、Marketing_Campaign：银行或保险相关，但不是信贷风险，**待决定** |
| 跳过（2 个） | APSFailure、kddcup09_appetency：少数类低于 2.5%，256 行里正例太少，没法做 5 折 |

**评估流程**（改自 nanoTabPFN 论文的 [experiment.ipynb](experiment.ipynb)）：
- 每个数据集分层抽样 256 行，重复 3 次（不同随机种子）
- 类别特征 ordinal 编码，去掉常数列；超过 24 个特征的随机选 24 个
- 5 折交叉验证，每折用 train 行均值填缺失值
- 指标：AUC、ECE（10 个等宽 bin）、Brier。先在每个数据集内对 3 次重复取平均，再对数据集取平均
- 抽好的数据缓存在 `benchmark/cache/`，保证所有模型考的是同一批表

**参考线**（`python evaluate.py --baselines`，45 秒）：

| | 金融 AUC | 金融 ECE | 金融 Brier | 通用 AUC | 通用 ECE | 通用 Brier |
|---|---|---|---|---|---|---|
| Random Forest | 0.779 | 0.044 | 0.108 | 0.734 | 0.074 | 0.144 |
| Logistic Regression | 0.738 | 0.057 | 0.116 | 0.730 | 0.063 | 0.146 |

## 4. 目前的实验发现

**4.1 每种 prior 造的表有多好学**（随机森林 AUC，40 张表，[diagnostics/prior_learnability.py](diagnostics/prior_learnability.py)）

| prior | 平均 AUC | AUC > 0.7 的表占比 |
|---|---|---|
| TabICL `mlp_scm` | 0.551 | 7% |
| TabICL `tree_scm` | 0.674 | 38% |
| TabICL `mix_scm`（v1） | 0.590 | 23% |
| TabICL `graph_scm`（v2，现在用的） | 0.707 | 47% |
| FinTFM 金融 | 0.658 | 44% |
| nanoTabPFN 原版 h5 | 0.595 | 20% |

**4.2 两个 prior 的表长得很不一样**（[check_priors.py](check_priors.py)，graph_scm，中位数）

| | 通用 | 金融 |
|---|---|---|
| 特征数 | 12 | 18 |
| 正例比例 | 0.49 | 0.039（56% 的表低于 5%） |
| 缺失值 | 0 | 8.7% |
| 最大绝对值 | 4 | 5e7（最大到 1e12） |

**4.3 训练稳定性**（[diagnostics/table_size.py](diagnostics/table_size.py)）

600 步、batch 8、`mix_scm`、breast cancer AUC，**受 2.4 的 bug 影响**：

| 表的尺寸 | seed 0 | seed 1 |
|---|---|---|
| 150 乘 5 | 0.947 | 0.939 |
| 256 乘 24 | 0.226 | 0.877 |

方案 (a)（2026-10-10 完成）：256 乘 24、`graph_scm`、3000 步、batch 8、bug 已修，日志在 `logs/plan_a_graph_scm_256x24_3000steps.log`。

| | seed 0 | seed 1 |
|---|---|---|
| breast cancer AUC，第 500 / 1000 / 1500 / 2000 / 2500 / 3000 步 | 0.968 / 0.982 / 0.971 / 0.974 / 0.979 / 0.977 | 0.801 / 0.972 / 0.988 / 0.987 / 0.990 / 0.982 |
| 100 张没见过的 graph_scm 表的 AUC | 0.691 | 0.703 |

当时的结论是"256 乘 24 可以稳定训练"，**2026-10-11 被推翻**（见 4.4）：这两次只是碰巧没崩。两个 seed 差 0.005 到 0.012。在没见过的同类表上，模型达到 0.69 到 0.70，接近随机森林在这类表上的 0.707（见 4.1），说明模型确实从 prior 里学到了东西。breast cancer 在 1000 步后基本不再上升。每个 seed 约 35 分钟。
注意：这次同时改了通用 prior（mix_scm 换成 graph_scm）和 schedule-free 的 bug，分不清稳定性来自哪一处。

**4.4 训练会随机崩溃（2026-10-11）**

r = 0 和 r = 1 各训 3000 步（现场造题，batch 8，学习率 4e-3，seed 0），结果归档在 `checkpoints/onthefly_prehashfix_2026-10-11/` 和 `results/onthefly_prehashfix_2026-10-11/`：

| 模型 | 金融 AUC | 金融 ECE | 通用 AUC | 通用 ECE |
|---|---|---|---|---|
| r = 0 | 0.506 | 0.320 | 0.501 | 0.221 |
| r = 1 | 0.744 | 0.058 | 0.717 | 0.093 |

r = 0 在第 500 到 1000 步之间崩了：loss 从 0.57 升到 0.69 并一直停在那里，breast cancer AUC 0.50。和方案 (a) 的 seed 0 设置完全相同却结果不同，原因是 TabICL 依赖 Python 的哈希种子，两次看到的是不同的数据（已修）。

到目前为止，256 乘 24、batch 8、学习率 4e-3 的 4 次训练：

| 运行 | 结果 |
|---|---|
| 10/9，600 步，mix_scm | 0.226，学反 |
| 方案 (a) seed 0 | 0.977 |
| 方案 (a) seed 1 | 0.982 |
| 10/11 r = 0 | 0.500，崩 |

推测：学习率 4e-3 是原版配 batch 32 调的，batch 8 时梯度噪声大约是 2 倍；通用 prior 的表大多类别平衡、规律很弱，"永远输出 50/50"是个 loss 0.69 的局部最优，被坏 batch 推进去就出不来。金融表大多违约率低，输出 50/50 的 loss 很高，所以不会停在那里。

**判断"崩了"的标准**（用于学习率验证）：

| | 健康 | 崩了 |
|---|---|---|
| 100 张没见过的合成表 AUC | ≥ 0.65 | < 0.55 |
| breast cancer AUC | ≥ 0.9 | < 0.6 |
| 训练 loss（r = 0） | 明显低于 0.69 | 停在 0.69（ln 2）附近 |

**4.5 学习率验证（2026-10-11）**

r = 0，从题库读题，batch 8，1000 步，[diagnostics/lr_stability.py](diagnostics/lr_stability.py)，结果在 `results/lr_stability/summary.json`，日志 `logs/lr_stability.log`：

| 学习率 | seed | 判定 | 合成表 AUC | 合成表 log loss | breast cancer AUC，每 250 步 |
|---|---|---|---|---|---|
| 4e-3 | 0 | 健康 | 0.714 | 0.408 | 0.95 / 0.98 / 0.97 / 0.97 |
| 4e-3 | 1 | 健康 | 0.667 | 0.475 | 0.79 / 0.88 / 0.93 / 0.91 |
| 4e-3 | 2 | **崩了** | 0.500 | 0.693 | 0.27 / 0.48 / 0.54 / 0.38 |
| 1e-3 | 0 | 健康 | 0.661 | 0.470 | 0.91 / 0.95 / 0.96 / 0.95 |
| 1e-3 | 1 | 健康 | 0.656 | 0.471 | 0.89 / 0.95 / 0.96 / 0.96 |
| 1e-3 | 2 | 不确定 | 0.649 | 0.475 | 0.90 / 0.93 / 0.96 / 0.94 |

- 4e-3 累计 7 次坏了 3 次；1e-3 三次都学会了，seed 之间只差 0.012
- 1e-3 seed 2 的"不确定"只差阈值 0.001，log loss 和另两个健康的几乎一样；阈值是事先定的，不事后修改
- 1e-3 在 1000 步时比 4e-3 健康的那两次学得慢一些（0.655 vs 0.667 / 0.714），正式实验训 3000 步
- 证据有限（每组 3 个 seed），正式实验每个模型训完都要按 4.4 的标准再检查一次

**4.6 第一次正式对比：r = 0 vs r = 1（2026-10-11）**

学习率 1e-3，3000 步，batch 8，seed 0，从题库读题。模型在 `checkpoints/lr1e-3/`，成绩在 `results/lr1e-3/`，日志 `logs/grid_lr1e-3_r0_r1.log`。两个模型都健康（breast cancer 0.976 / 0.990）。

| 模型 | 金融 AUC | 金融 ECE | 金融 Brier | 通用 AUC | 通用 ECE | 通用 Brier |
|---|---|---|---|---|---|---|
| r = 0 | 0.677 | 0.034 | 0.124 | 0.674 | 0.053 | 0.158 |
| r = 1 | 0.732 | 0.040 | 0.116 | 0.701 | 0.059 | 0.155 |
| 随机森林 | 0.779 | 0.044 | 0.108 | 0.734 | 0.074 | 0.144 |
| Logistic Regression | 0.738 | 0.057 | 0.116 | 0.730 | 0.063 | 0.146 |

- **没有出现预期的交叉**：r = 1 在金融表 7/7 更好（平均 +0.055，最大 heloc +0.18），在通用表 13/17 更好（平均 +0.027；变差的是 Amazon_employee_access −0.085、E-CommereShippingData −0.064、NATICUSdroid −0.026、Bioresponse −0.010）
- 金融表上的提升约是通用表上的两倍，说明有金融特有的部分
- ECE 方面 r = 0 在两套考卷上都略好
- 单个 seed，还需要补 seed 确认

**格式检查**（[diagnostics/format_check.py](diagnostics/format_check.py)，`results/lr1e-3/format_check.json`）：评估时先把每列 quantile 变换成正态分布，抹掉尺度和长尾差异。

| 模型 | 预处理 | 金融 AUC | 通用 AUC |
|---|---|---|---|
| r = 0 | 无 / quantile | 0.677 / 0.699 | 0.674 / 0.665 |
| r = 1 | 无 / quantile | 0.732 / 0.728 | 0.701 / 0.690 |

r = 1 的领先：金融表从 0.055 缩小到 0.029（约一半来自长尾格式），通用表从 0.027 到 0.025（几乎不是长尾格式）。通用表上的领先原因未明，候选：金融 prior 教会了处理类别不平衡；纯通用模型本身偏弱（通用表 0.674，低于 Logistic Regression 的 0.730）。

**4.7 七个 r，seed 0（2026-10-11）**

学习率 1e-3，3000 步，batch 8。日志 `logs/grid_lr1e-3_seed0.log`，图 [figures/rq1_curves_lr1e-3_seed0.png](figures/rq1_curves_lr1e-3_seed0.png)，健康检查 `results/lr1e-3/health.json`（[diagnostics/checkpoint_health.py](diagnostics/checkpoint_health.py)）。

| r | 0 | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1 |
|---|---|---|---|---|---|---|---|
| 金融 AUC | 0.677 | 0.761 | 0.750 | 0.663 | 0.662 | 0.677 | 0.732 |
| 通用 AUC | 0.674 | 0.732 | 0.724 | 0.655 | 0.647 | 0.648 | 0.701 |
| 金融 − 通用 | 0.002 | 0.029 | 0.026 | 0.009 | 0.015 | 0.028 | 0.030 |
| breast cancer AUC | 0.976 | 0.985 | 0.986 | 0.951 | 0.959 | 0.960 | 0.990 |
| 通用合成表 AUC | 0.681 | 0.710 | 0.715 | 0.669 | 0.647 | 0.657 | 0.685 |
| 金融合成表 AUC | 0.582 | 0.638 | 0.652 | 0.595 | 0.604 | 0.624 | 0.651 |

- **曲线主要是训练噪声**：金融和通用两条线同起同落；r = 0.5 到 0.9 三次训练在所有指标上都整体偏弱，连在金融合成表上都不如 r = 0.3（r = 0.9 练的金融题是 r = 0.3 的 3 倍）。训练之间的差异可达 0.1 AUC，比要测的 r 效应（0.03 到 0.05）大 2 到 3 倍
- "金融 − 通用"能抵消单次训练好坏的影响，大体随 r 上升（0.002 到约 0.03），是领域效应的线索，但也很吵
- 判断原因：训练量不足（每次 2.4 万张表，原版 nanoTabPFN 8 万张），结果对随机性敏感。对策是上 GPU：batch 32、训更久

**4.8 batch 32 下的第一条曲线（2026-10-12，Mac）**

batch 32、学习率 4e-3、3000 步、seed 0，r = 0 / 0.3 / 0.5 / 0.7 / 1（每个约 85 分钟）。日志 `logs/grid_bs32_lr0.004_steps3000.log`，成绩 `results/bs32_lr0.004_steps3000/`，图 [figures/rq1_curves_bs32_lr0.004_steps3000.png](figures/rq1_curves_bs32_lr0.004_steps3000.png)。5 个模型全部健康。

| r | 0 | 0.3 | 0.5 | 0.7 | 1 | 随机森林 |
|---|---|---|---|---|---|---|
| 金融 AUC | **0.781** | 0.778 | 0.777 | 0.767 | 0.761 | 0.779 |
| 通用 AUC | **0.738** | 0.733 | 0.734 | 0.729 | 0.727 | 0.734 |
| 金融 ECE | 0.036 | 0.040 | 0.044 | 0.040 | **0.081** | 0.044 |
| 通用 ECE | 0.045 | 0.047 | 0.043 | 0.045 | **0.111** | 0.074 |
| 通用合成表 AUC | 0.724 | 0.726 | 0.724 | 0.717 | 0.707 | |
| 金融合成表 AUC | 0.630 | 0.655 | 0.649 | 0.642 | 0.672 | |

- **训练量够了以后，模型强了很多**：r = 0 在两套考卷上都追平或超过随机森林（金融 0.781 vs 0.779，通用 0.738 vs 0.734）；batch 8 时只有 0.677 / 0.674
- **在真实表上没有取舍**：r 越大，金融表和通用表**都**变差（金融 −0.020，通用 −0.012）。纯通用 prior 在金融表上也是最好的
- **在合成表上有取舍**：r 越大，金融合成表变好（0.630 → 0.672）、通用合成表变差（0.724 → 0.707）。也就是说，模型确实学到了金融 prior 的规律，但这些规律没有迁移到真实的金融表上
- **纯金融（r = 1）的概率很不准**：ECE 是其他比例的 2 倍多；只要混入 30% 以上的通用题就恢复正常（和 FinTFM 的发现方向一致）
- **金融表内部有分化**（r = 1 减 r = 0）：消费信贷 credit-g −0.098、credit_card_clients_default −0.041、Is-this-a-good-customer −0.025；而财务比率型的破产数据 polish +0.001、taiwanese +0.003，以及 GiveMeSomeCredit +0.017、heloc +0.003 基本不受影响。FinTFM 的 prior 模拟的是公司财报和财务比率，和消费信贷的表（人口信息、类别特征）不像
- **和 batch 8 的结论相反**：batch 8 时 r = 1 在两套考卷上都赢（4.6），现在看是因为那时的纯通用模型训练不足、偏弱
- 单个 seed；Colab 上同样设置（r = 0）3 个 seed 的 breast cancer AUC 是 0.983 到 0.987，seed 间差异看起来比这里的 r 效应小，但还需要合成表和考卷上的 std 才能确认

## 5. 已做的决定（2026-10-10）

1. 通用 prior 用 `graph_scm`（TabICLv2）
2. 先跑方案 (a) 验证训练稳定性
3. TabArena 里的金融表划进金融考卷，从通用考卷里拿掉
4. 金融表里的极端值暂时不做 log 变换
5. "金融规律"和"格式差异"分不开的问题：pilot 先不管，曲线出来后再加 ablation

**2026-10-11 追加：**

6. 改为预先造好题库，训练时从题库按 r 抽题（2.2b）
7. 先验证学习率（4e-3 vs 1e-3，r = 0，各 3 个 seed，1000 步），再跑正式实验
8. 正式实验用学习率 1e-3（4.5）

**2026-10-13 追加：**

9. batch 32 下用回原版学习率 4e-3（Colab 上 3 个 seed 都稳定，见 RESEARCH.md 2026-10-11 晚上的日志）；8 号决定只适用于 batch 8
10. 按 Jerry 的任务说明重构 repo（Phase A），之后准备 E1 到 E6（Phase B），在公司 SageMaker 上跑（Phase C）

## 6. 待决定

1. **4 个"银行但非信贷"的数据集**（第 3 节）：Phase B 里作为第三套考卷 `near_domain`。

## 7. 文件清单

见第 1 节和 [README](../README.md) 的"结构"一节。
