# RQ1 实验说明：金融 prior 和通用 prior 的取舍曲线

最后更新：2026-10-10。状态标记：✅ 已实现并测过，🟡 已实现但待验证，⬜ 未实现。

## 0. 要回答的问题

**RQ1：取舍曲线存在吗？** 金融 prior 的比例 r 从 0% 加到 100%，金融表上提升多少，通用表上损失多少？

做法：每个 r 训一个 nanoTabPFN，在两套考卷上测，画出两条曲线（x 轴 r，y 轴 AUC / ECE / Brier）。

## 1. 整体流程和对应的代码

```
通用 prior (TabICLv2) ─┐
                       ├─ MixedPrior(r) ─ MixedPriorDataLoader ─ train() ─ checkpoint ─ evaluate.py ─ plot_curves.py
金融 prior (FinTFM)   ─┘   每张表按 r 选来源   填 NaN、补零、切分                       两套考卷        取舍曲线
        ✅                       ✅                   ✅                  🟡                     ✅              ✅
                                        一键跑全部：run_grid.py ✅
```

| 步骤 | 文件 | 运行方式 |
|---|---|---|
| 造题 | [priors/mixture.py](priors/mixture.py)、[priors/fintfm_financial.py](priors/fintfm_financial.py) | 被训练脚本调用 |
| 拼 batch | [priors/batching.py](priors/batching.py) | 被训练脚本调用 |
| 训练一个模型 | [train_mixed.py](train_mixed.py)（调用 [train.py](train.py) 的 `train()`） | `python train_mixed.py --r 0.3 --seed 0 --steps 3000 --batch_size 8` |
| 考卷 | [benchmark/datasets.py](benchmark/datasets.py)、[benchmark/metrics.py](benchmark/metrics.py) | 被评估脚本调用 |
| 评估 | [evaluate.py](evaluate.py) | `python evaluate.py checkpoints/*.pth --baselines` |
| 画曲线 | [plot_curves.py](plot_curves.py) | `python plot_curves.py` |
| 跑完整网格 | [run_grid.py](run_grid.py) | `python run_grid.py --steps 3000 --batch_size 8` |

## 2. 造题和训练

### 2.1 两个 prior ✅

| | 用什么 | 代码 |
|---|---|---|
| 通用 prior | **TabICLv2 的 prior**（`graph_scm`），依据是 TabICL 官方脚本 `scripts/train_v2_clf_stage1.sh` 用的是 `--prior_type graph_scm`。用 `--generic mix_scm` 可以换回 v1 的 prior | [priors/mixture.py](priors/mixture.py) `GenericPrior`；生成代码在安装包 `.venv/.../tabicl/prior/`（tabicl 2.2.0） |
| 金融 prior | FinTFM `sample_financial_task()`：符合会计恒等式的财报、财务比率特征、违约率 0.1% 到 30% | [priors/fintfm_financial.py](priors/fintfm_financial.py)（从 FinTFM commit `7c297d0` 原样复制，只改 1 行 import） |

不用 FinTFM 自带的混合器：它的通用部分是作者自己写的 SCM；它的 tree prior 从金融预算里扣，`p_financial` 不等于真实的金融占比。

### 2.2 r 的定义 ✅

**每张表**以概率 r 来自金融 prior，否则来自通用 prior。r 按表的数量算，对应 data mixing law 里的 r。代码：[priors/mixture.py](priors/mixture.py) `MixedPrior.sample()`。

### 2.3 表的格式 🟡

| 项目 | 设定 | 理由 |
|---|---|---|
| 行数 | 256 | 金融 prior 需要足够多的行才能造出低违约率 |
| 特征数 | 最多 24，不足的补零到 24 | 和 FinTFM 默认值一致；全 0 列经模型标准化后仍为 0 |
| 类别 | 二分类 | 金融考卷是违约预测 |
| train/test 切分 | 每个 batch 从 {80, 96, ..., 224} 里随机选 | 只取 16 的倍数，batch 形状只有 10 种，避免 Mac GPU 缓存爆掉 |
| 缺失值 | 用 train 行的列均值填补 | `impute_with_train_mean()`，评估真实表时用同一个函数 |

🟡：256 乘 24 在训练量少时不稳定，方案 (a) 正在验证（见 4.3）。

### 2.4 模型和训练 🟡

| 项目 | 设定 |
|---|---|
| 模型 | nanoTabPFN：embedding 96，4 heads，MLP 192，3 层，2 个输出 |
| 优化器 | schedule-free AdamW，lr 4e-3 |
| batch | Mac 上 8（16 GB 内存的限制），GPU 上计划 32 |
| 步数 | 待方案 (a) 定 |
| 速度（Mac） | 每步约 0.67 秒 = 模型 0.39 秒 + 造题 0.28 秒（造题是单线程 CPU，GPU 上可以并行） |
| 随机性 | 同一个 seed 完全可复现（python / numpy / torch 三个 seed 都设了） |

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

结论：**256 乘 24 可以稳定训练**，两个 seed 差 0.005 到 0.012，没有再出现学反的情况。在没见过的同类表上，模型达到 0.69 到 0.70，接近随机森林在这类表上的 0.707（见 4.1），说明模型确实从 prior 里学到了东西。breast cancer 在 1000 步后基本不再上升。每个 seed 约 35 分钟。
注意：这次同时改了通用 prior（mix_scm 换成 graph_scm）和 schedule-free 的 bug，分不清稳定性来自哪一处。

## 5. 已做的决定（2026-10-10）

1. 通用 prior 用 `graph_scm`（TabICLv2）
2. 先跑方案 (a) 验证训练稳定性
3. TabArena 里的金融表划进金融考卷，从通用考卷里拿掉
4. 金融表里的极端值暂时不做 log 变换
5. "金融规律"和"格式差异"分不开的问题：pilot 先不管，曲线出来后再加 ablation

## 6. 待决定

1. **4 个"银行但非信贷"的数据集**（第 3 节）放哪边？现在两边都不放。
2. **训练步数**：等方案 (a) 的结果。
3. **GPU 预算**：等步数定了再找 Bo。按 Mac 的速度，3000 步一次约 35 分钟，21 次约 12 小时；如果需要更多步数或 batch 32，Mac 就不够用了。

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
diagnostics/
  prior_learnability.py     4.1
  table_size.py             4.3、方案 (a)
checkpoints/                训练好的模型（不进 git）
results/                    评估结果 json（不进 git）
logs/                       训练日志（不进 git）
```

依赖：`uv add` 加了 `tabicl` 和 `xgboost`（见 [pyproject.toml](pyproject.toml)）。
