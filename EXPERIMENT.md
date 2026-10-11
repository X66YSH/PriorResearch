# RQ1 实验说明：金融 prior 和通用 prior 的取舍曲线

最后更新：2026-10-11。状态标记：✅ 已实现并测过，🟡 已实现但待验证，⬜ 未实现。

## 0. 要回答的问题

**RQ1：取舍曲线存在吗？** 金融 prior 的比例 r 从 0% 加到 100%，金融表上提升多少，通用表上损失多少？

做法：每个 r 训一个 nanoTabPFN，在两套考卷上测，画出两条曲线（x 轴 r，y 轴 AUC / ECE / Brier）。

## 1. 整体流程和对应的代码

```
通用 prior (TabICLv2) ─ make_prior_dump.py ─ 通用题库 10 万张 ─┐
                                                              ├─ DumpMixedDataLoader(r) ─ train() ─ checkpoint ─ evaluate.py ─ plot_curves.py
金融 prior (FinTFM)   ─ make_prior_dump.py ─ 金融题库 10 万张 ─┘   按 r 抽题、填 NaN、切分                    两套考卷        取舍曲线
        ✅                      ✅                    ✅                       ✅                  🟡                  ✅              ✅
                                                        一键跑全部：run_grid.py ✅
```
（2026-10-11 之前是训练时现场造题：`MixedPrior` 加 `MixedPriorDataLoader`，现在用 `--on_the_fly` 仍可切回。）

| 步骤 | 文件 | 运行方式 |
|---|---|---|
| 造题 | [priors/mixture.py](priors/mixture.py)、[priors/fintfm_financial.py](priors/fintfm_financial.py) | 被造题库的脚本调用 |
| 造题库 | [make_prior_dump.py](make_prior_dump.py) | `python make_prior_dump.py --prior generic --n_tables 100000`（金融用 `--prior finance`） |
| 按 r 抽题、拼 batch | [priors/batching.py](priors/batching.py) `DumpMixedDataLoader` | 被训练脚本调用 |
| 训练一个模型 | [train_mixed.py](train_mixed.py)（调用 [train.py](train.py) 的 `train()`） | `python train_mixed.py --r 0.3 --seed 0 --steps 3000 --batch_size 8` |
| 考卷 | [benchmark/datasets.py](benchmark/datasets.py)、[benchmark/metrics.py](benchmark/metrics.py) | 被评估脚本调用 |
| 评估 | [evaluate.py](evaluate.py) | `python evaluate.py checkpoints/*.pth --baselines` |
| 画曲线 | [plot_curves.py](plot_curves.py) | `python plot_curves.py` |
| 跑完整网格 | [run_grid.py](run_grid.py) | `python run_grid.py --steps 3000 --batch_size 8 --lr 1e-3 --out_dir checkpoints/lr1e-3`，结果在 `results/lr1e-3/` |

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

## 6. 待决定

1. **4 个"银行但非信贷"的数据集**（第 3 节）放哪边？现在两边都不放。
3. **转到 GPU**（Jerry 有 GPU，2026-10-11）：先在 GPU 上验证 batch 32 下的学习率和训练噪声，再定步数和 seed 数，见 [RUN_ON_GPU.md](RUN_ON_GPU.md)。

## 7. 文件清单

```
EXPERIMENT.md               本文件
priors/
  fintfm_financial.py       金融 prior（FinTFM 原样复制）
  _task.py                  金融 prior 需要的数据类
  mixture.py                GenericPrior / FinancePrior / MixedPrior(r)
  batching.py               impute_with_train_mean / MixedPriorDataLoader
benchmark/
  datasets.py               两套考卷的数据集清单和抽样流程
  metrics.py                AUC / ECE / Brier
  cache/                    抽好的考卷（不进 git）
train_mixed.py              训练一个模型
train.py                    原版训练循环（改了 2 处，见 2.4）
model.py                    原版 nanoTabPFN 模型（未改）
evaluate.py                 在两套考卷上评估
plot_curves.py              画取舍曲线
run_grid.py                 跑完整的 7 个 r 乘 3 个 seed
check_priors.py             对比两个 prior 的输出
make_prior_dump.py          造题库
prior_dumps/                两个题库和两个健康检查用的小题库（.h5，不进 git）
rq1_gpu.ipynb               GPU 上一步一步跑的 notebook，结果写进 results/gpu_report.txt
build_notebook.py           生成 rq1_gpu.ipynb
RUN_ON_GPU.md               GPU 上怎么跑
requirements-gpu.txt        GPU 上的依赖
diagnostics/
  prior_learnability.py     4.1
  table_size.py             4.3、方案 (a)
  lr_stability.py           4.5
  format_check.py           4.6
  checkpoint_health.py      4.7
checkpoints/                训练好的模型（不进 git）
results/                    评估结果 json（不进 git）
logs/                       训练日志（不进 git）
```

依赖：`uv add` 加了 `tabicl` 和 `xgboost`（见 [pyproject.toml](pyproject.toml)）。
