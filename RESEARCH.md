# TFM Prior 研究总文档

最后更新：2026-10-11。

这是整个研究的总入口：课题、研究问题、计划、进展和文献。RQ1 实验的具体设计、代码和数字在 [EXPERIMENT.md](EXPERIMENT.md)。
每次有进展，两份文档都要更新：新进展写进第 5 节的日志，实验细节写进 EXPERIMENT.md。

## 1. 背景

- **人：** Jerry Li（ML Engineer，BMO Institute for Applied AI & Quantum），汇报给 Bo Zhao（赵博，Tech Lead）。
- **两个目标：**
  1. **产品：** Bo 的 "drag-a-table"：BMO 任何团队拖一张表进来，就能用 tabular foundation model（TFM）得到预测。
  2. **论文：** 通过团队的研究职能发一篇顶会论文。BMO 允许发表，前提是不和外部合作。
- **数据规则：** 论文只能用公开数据。BMO 内部的表可以用来验证产品，但不能写进论文。
- **2026-10-08：** Jerry 给 Bo 讲了 deck。Bo 喜欢这个课题，认为有机会；他的直觉是模型要像人一样，同时有通用知识和领域知识才能做到最好。要求动作要快。

## 2. 研究课题

**一句话：** 把 LLM 的 data mixing law 搬到 TFM 的 prior 上。研究金融 prior 和通用 prior 混多少的取舍曲线，给出一个公式，并找一个训练前就能算的指标来预测它。

| | 问题 | 做法 |
|---|---|---|
| **RQ1** | 曲线存在吗？ | 金融 prior 的比例 r 从 0% 到 100%，金融表提升多少，通用表损失多少 |
| **RQ2** | 能用公式预测吗？ | 用小模型和少数几个比例拟合（参考 Ye et al. 的 L = c + k·exp(t·r)），外推到大模型和没试过的比例 |
| **RQ3** | 能不训练就预测吗？ | 用"像不像金融"和"够不够多样"两个指标，训练前估出最优比例 |

**冲主会的标准：** 只做 RQ1 是 workshop 级别（和 Hutter 组、CSIRO 一样）。做到 RQ2 和 RQ3 才是 Ye、Apple 那一层。这个方向的主会论文都有三样东西：能预测的公式、在没试过的设置上验证过的外推、实际节省了多少。

**要小心的地方：** "混合比纯用好"太直觉，审稿人可能觉得显然。论文的价值必须在 (1) 最优点在哪、怎么算（RQ2），(2) 反直觉的发现，比如 FinTFM 的 tree prior 让通用表变好、金融表变差。

**和已有工作的区别：**

| 工作 | 它的局限 |
|---|---|
| FinTFM | 只试了 3 个比例、1 个 seed，没有公式 |
| RAD-TFM | 只看领域内的收益，没测领域外的代价 |
| MITRA | 凭经验标准挑一个通用混合，不预测领域取舍；FinTFM 发现它的标准在信贷数据上预测反了 |
| Ye、Apple | 有公式，但只针对 LLM；LLM 数据有限，我们的 prior 可以无限造题 |

**对产品的意义：** 答案直接决定 drag-a-table 是只做一个通用模型，还是再加金融版本，以及金融版本混多少。

**风险：** Bristol 发现合成数据和真实数据的差距对性能没有明显影响；Hutter 组发现相似度只能部分解释性能。如果曲线是平的，这本身也是发现，但要重新考虑冲主会的目标。

## 3. 方法概要

**Prior** 是一段随机造练习题的程序：每次随机抽一条规则（一个从不训练的随机小模型），按规则造一张表。TFM 在几百万张这样的表上练习"看几行带答案的例子，猜出规则，预测新行"，即 in-context learning。不同的 prior 就是对"真实世界的规律长什么样"的不同假设。

**我们的实验（RQ1 pilot），细节见 [EXPERIMENT.md](EXPERIMENT.md)：**

| 项目 | 设定 |
|---|---|
| 模型 | nanoTabPFN，约 36 万参数，固定不变，每次从头训练 |
| 通用 prior | TabICLv2 的 prior（`graph_scm`） |
| 金融 prior | FinTFM 的金融 prior（财报、财务比率、低违约率） |
| 唯一的变量 | 金融题比例 r ∈ {0, 0.1, 0.3, 0.5, 0.7, 0.9, 1}，每个 r 跑 3 个 seed，共 21 次训练 |
| 考卷 | TabArena 的二分类任务：金融 7 个（信贷、违约、破产），通用 17 个 |
| 指标 | AUC（排序能力）、ECE 和 Brier（概率准不准） |
| 硬件 | 目前用 Jerry 的 Mac（16 GB，MPS），每次训练约 35 分钟 |

## 4. 时间线

**截稿日期（2026-10-08 查）：** ICLR 2027（9 月 25 日）和 AISTATS 2027（10 月 6 日）已过。ICML 2027 预计约 2027-01-22（摘要约早一周；尚未官宣；会议在南美）。NeurIPS 2027 预计约 2027-05-21。KDD 2027 cycle 2 约 2027 年 2 月，UAI 2027 约 2027-02-12。ICML/ICLR 的 workshop 是保底。

**按 ICML 倒推：**

| 时间 | 任务 |
|---|---|
| 10 月 | pilot：RQ1 |
| 11 月到 12 月中 | RQ2 |
| 12 月中到 1 月中 | 写论文，RQ3 尽量做 |
| 约 1 月 22 日 | 投稿 |

**arXiv 策略：** RQ1 和 RQ2 都成立后（约 12 月）挂 arXiv 占坑；如果出现很接近的竞争工作就提前挂。只有 RQ1 时不挂：太薄，还会暴露方向。

## 5. 进展日志

### 2026-10-08
- 给 Bo 讲了 deck（21 页）和讲稿，Bo 同意方向，要求快。
- 写了 handoff 文档，开新对话接手。

### 2026-10-09
- 读了 FinTFM、TabICL、TFM-Playground 的源码，确定 prior 的接法：通用用 TabICL，金融只拿 FinTFM 的金融部分，混合器自己写。不用 FinTFM 自带的混合器，因为它的 tree prior 从金融预算里扣，`p_financial` 不等于真实的金融占比。
- 写了 prior 混合、拼 batch、训练入口；Mac 上跑通。
- 踩坑：TabICL 用了 Python 的 `random` 模块，导致不可复现，已修；Mac GPU 按 batch 形状缓存显存会爆，已修（限制 batch 形状）。
- 发现：两个 prior 的表很不一样，金融表 56% 正例低于 5%、有 8% 缺失值、数值可到 1e12；通用表几乎平衡、无缺失。
- 发现：256 行乘 24 特征的大表训 600 步时极不稳定（两个 seed 的 AUC 是 0.226 和 0.877）。

### 2026-10-10
- Jerry 批准 5 项决定（见 EXPERIMENT.md 第 5 节），其中通用 prior 换成 TabICLv2 的 `graph_scm`。
- **修了原版 train.py 的 bug：** schedule-free 优化器训练结束时没切到 eval 模式，保存和评估的都是训练参数。之前的结果都受影响。
- 做好两套考卷和评估脚本。随机森林参考线：金融 AUC 0.779，通用 AUC 0.734。
- 写好 `run_grid.py`（批量训练加评估）和 `plot_curves.py`（画取舍曲线）。
- **方案 (a) 通过：** 256 乘 24、3000 步，两个 seed 的 breast cancer AUC 是 0.977 和 0.982，训练稳定；在没见过的合成表上 AUC 0.69 到 0.70，接近随机森林的 0.707。

### 2026-10-11（凌晨）
- 发现并修复 `run_grid.py` 会误用旧 checkpoint 的问题。
- r = 0 和 r = 1 的实验启动后，因为要关机停掉了，明早重跑。
- 建立本文档（RESEARCH.md），代码做了第一次 git commit。

### 2026-10-11（上午）
- 跑了 r = 0 和 r = 1（现场造题，3000 步，batch 8，学习率 4e-3）。**r = 0 训练中途崩了**：loss 卡在 0.69（ln 2），所有考卷 AUC 0.50。r = 1 正常：金融 AUC 0.744，通用 AUC 0.717（随机森林 0.779 / 0.734）。这组结果不可复现，已归档到 `checkpoints/onthefly_prehashfix_2026-10-11/`，不作结论。
- **发现可复现性 bug：** TabICL 的输出依赖 Python 每个进程的哈希种子，同一个 seed 在不同进程里造出不同的表。已修（`priors/__init__.py` 自动固定 `PYTHONHASHSEED`）。所以方案 (a) 的成功和今天的崩溃其实看到的是不同数据。
- **训练不稳定：** 256 乘 24 的表、batch 8、学习率 4e-3，4 次训练里坏了 2 次。推测原因：学习率 4e-3 是原版配 batch 32 的，batch 降到 8 没跟着降。
- **改成预先造好题库（Jerry 的建议）：** 通用（graph_scm）和金融各 10 万张表，存成 h5（`make_prior_dump.py`）。训练改为从题库按比例抽题（`DumpMixedDataLoader`）：每批取题从 280 毫秒降到 1 到 3 毫秒，跨进程完全可复现，一次训练内不重复，不同 r 之间共用随机数。

- **学习率验证：** r = 0、1000 步、各 3 个 seed。4e-3 坏了 1 次（累计 7 次坏 3 次），1e-3 三次都学会了。正式实验改用 1e-3。
- **第一次正式对比（r = 0 vs r = 1，学习率 1e-3，seed 0）：没有交叉，纯金融模型在两套考卷上都更好。** 金融 AUC 0.732 vs 0.677（7/7 个数据集更好），通用 AUC 0.701 vs 0.674（13/17）。ECE 是纯通用略好。格式检查显示金融表上的领先约一半来自长尾数值格式，通用表上的领先不是。如果多个 seed 下仍成立，RQ1 的故事要从"领域和通用的取舍"改成"领域 prior 为什么连领域外都更好、混合能否两头都赢"。细节见 EXPERIMENT.md 4.6。

- **七个 r（seed 0）跑完：曲线主要是训练噪声。** 金融和通用两条线同起同落，r = 0.5 到 0.9 三次训练整体偏弱；训练间差异可达 0.1 AUC，比要测的效应大 2 到 3 倍。"金融 − 通用"大体随 r 上升（0.002 到约 0.03），是领域效应的线索。结论：Mac 上训练量不够，转到 GPU（batch 32、训更久）。细节见 EXPERIMENT.md 4.7。

## 6. 下一步

**马上要做：**
1. **转到 GPU，用 Colab 的 T4**（[RUN_ON_GPU.md](RUN_ON_GPU.md)、[rq1_gpu.ipynb](rq1_gpu.ipynb)；公司邮件发不出代码包）：先确认环境和数据一致，再在 batch 32 下验证学习率、测训练噪声有多大。
2. 噪声降到明显小于 r 效应后，跑 7 个 r 乘 3 个 seed。
3. 查通用表上 r = 1 为什么也赢：候选是类别不平衡，或纯通用模型偏弱；对应的 ablation 是给通用 prior 加上不平衡、缺失和长尾。

**要找 Bo 的：**
- GPU 预算：Mac 上完整的 21 次约 12 小时，而且 batch 只能开到 8；GPU 上可以更快、训得更充分
- BMO 对外发表的审批要多久（arXiv 也要过）

**待决定：**
- 4 个"银行但非信贷"的 TabArena 数据集放哪边（现在两边都不放）
- "金融规律"和"格式差异"（不平衡、缺失、尺度）分不开，曲线出来后要设计 ablation

**之前提过但还没做的：**
- 给 Bo 的 2 周计划一页纸
- 每周 arXiv 监控竞争工作（关键词：tabular foundation model prior、domain prior、prior mixture）
- 修 reading list 的 venue（RAD-TFM、Explaining Data Mixing Scaling Laws 都是 ICML 2026 workshop）

## 7. 文献要点

Reading list（30 篇，5 张表）：https://claude.ai/code/artifact/dffb466c-9356-481c-9271-2ad03385c84f

### 7.1 Prior 基础

| 论文 | 要点 |
|---|---|
| Müller et al., ICLR 2022 | PFN 理论：在 prior 生成的任务上训练 = 学习贝叶斯推断 |
| TabPFN, ICLR 2023 / Nature 2025 | 第一个把这个思路做成表格模型 |
| Prior vs synthetic data generation（CTGAN、TabDDPM） | 一个"出题"，一个"复印"。没有论文直接比较用 CTGAN 数据和用 prior 预训练 TFM |
| Hutter 组，ICML 2026 workshop | 同样的真实表，每张只出 1 种题排最后；随机换 y 出很多题好很多 |
| Ma et al., arXiv 2511.09665 | 一张真实表换不同的 y 和特征子集造很多任务，也能学到泛化 |

### 7.2 prior 像真实数据，有没有用

| 论文 | 怎么测"像不像" | 发现 | 局限 |
|---|---|---|---|
| CSIRO, arXiv 2609.06912 | 21 个手工统计量，k-NN 覆盖率 | TabICLv2 覆盖最好、表现最好；72% 到 86% 的数据集上"越像越好" | 只是相关，prior 和模型结构一起变 |
| Hutter 组, arXiv 2606.29241 | 手工统计量，prior 之间的相似度 | 固定 nanoTabPFN、换 11 个 prior：prior 有影响但不大；相似度只能部分预测性能 | 小模型，16 个数据集 |
| Bristol, arXiv 2605.06343 | 判别器 AUC，k-NN 覆盖 | TabICL 的 prior 和真实表几乎 100% 能分开；调 8.6 万组参数也拉不近；但差距和性能没有明显关系 | 只分析分布，没训新模型 |
| MIT, arXiv 2603.04692 | TabPFN 2.5 的 embedding 加 XGBoost 判别器 | 挑 200 张最像工程领域的合成表继续训，35 个工程数据集里 29 个变好 | 没有随机挑选的对照，没测通用能力 |

### 7.3 领域 prior 和它的代价

| 工作 | 要点 | 局限 |
|---|---|---|
| FinTFM（github.com/kabartay/fintfm） | 100% 金融 AUC 0.757 / ECE 0.0106；70/30 混合 AUC 0.751 / ECE 0.0042；纯通用 AUC 0.708。加 tree prior：通用 +0.0094，信贷 −0.0221。MITRA 式打分在金融上预测反了 | 个人项目，模型很弱（TabArena 第 93 到 94 名，共 95），很多对比只有 1 个 seed |
| RAD-TFM（IBM / MIT，ICML 2026 workshop） | 只调通用 generator 的旋钮：64 个元特征加 Wasserstein 对齐目标领域，再按"比 XGBoost 差多少"加权抽题。normalized AUC 最多 +0.12，原始 AUC 只有 +0.3% 到 +2.1% | "领域"就是测试 benchmark 本身；没测领域外代价；两步没拆开 |
| KnowsTFM, arXiv 2606.30258 | 微调时用知识图谱加 attention bias：小模型 0.741 → 0.768；大模型一微调就变差（TabPFN v2.6：0.892 → 0.862） | 是微调不是 prior；生物医学领域 |

### 7.4 LLM 的 data mixing law

| 论文 | 要点 |
|---|---|
| Ye et al.（复旦 / 上海 AI Lab），ICLR 2025，arXiv 2403.16952 | L_i(r) = c_i + k_i·exp(Σ_j t_ij·r_j)；用 70M 到 410M 的小模型跑约 20 种比例，串联步数和模型大小的 scaling law，外推到 1B / 100B token；优化后的比例相当于默认比例多训 48% 的步数；能预测继续预训练时要保留多少旧数据 |
| Apple（Shukor et al.），NeurIPS 2025，arXiv 2507.09404 | 一个公式同时管混合比例、模型大小、数据量；约 10 到 20 次训练；外推误差 LLM 约 1% 到 2%；假设数据不重复（我们的 prior 正好满足） |
| Explaining Data Mixing Scaling Laws（Dai, Zheng），ICML 2026 workshop | 解释为什么成立：模型容量固定，各领域抢容量。只读了摘要 |
| 专才 vs 通才 | 文本上通才多数赢（BloombergGPT vs GPT-4）；视网膜上专才小胜；PPG 上专才 +27%。表格：未知 |

## 8. 相关文件

| 文件 | 在哪 |
|---|---|
| RQ1 实验说明 | [EXPERIMENT.md](EXPERIMENT.md) |
| 给 Bo 讲的 deck（21 页） | `TFM_Prior_deck.pptx`（repo 根目录，不进 git，因为是 BMO 内部材料） |
| 讲稿 | `Bo_meeting_script.md / .pdf`（Jerry 手上） |
| handoff 文档 | claude.ai 项目 "Research & Product" 里的 `TFM prior handoff` |
| Reading list | 见第 7 节开头的链接 |
