# tfm-prior-mixing

研究在预训练 tabular foundation model（nanoTabPFN）时，把领域 prior（比如金融）和通用 prior（TabICLv2 的 graph_scm）按比例混合，对真实表格的成绩有什么影响，以及这个影响能否用公式预测。研究背景和进展见 [docs/RESEARCH.md](docs/RESEARCH.md)，实验设计和所有结果见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 1. 安装

Python 3.11。推荐用 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync                      # 按 uv.lock 装依赖和本项目（src/tfmprior）
source .venv/bin/activate
```

没有 uv 的机器（比如 SageMaker）：

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps
```

## 2. 冒烟测试（不需要任何数据，约 2 分钟）

```bash
pytest
```

测试会自己造几十张表的小题库，在 CPU 上跑通训练、存档、评估，并检查几条关键性质（题库可复现、抽题比例、存档用的是 schedule-free 的评估参数等）。评估部分需要考卷缓存，没有时那一项会跳过。

## 3. 准备数据

```bash
python scripts/make_dump.py --all --workers 8     # 题库，写到 artifacts/dumps/；Mac 上约 10 分钟
python scripts/make_dump.py --verify              # 和 artifacts/checksums.json 里的参考校验值对比
python scripts/evaluate.py --baselines            # 第一次运行时从 OpenML 下载考卷到 artifacts/benchmark_cache/
```

## 4. 跑一个实验

一个实验 = `configs/experiments/` 里的一个文件，在 `configs/base.yaml` 的基础上改设置：

```bash
python scripts/train.py --config configs/experiments/smoke.yaml              # 小模型，10 步，几分钟
python scripts/train.py --config configs/experiments/smoke.yaml --list       # 只列出会跑哪些 run
python scripts/train.py --config configs/experiments/smoke.yaml train.steps=20 --overwrite   # 命令行覆盖设置
```

每个 run 一个目录 `runs/<实验名>/<run 名>/`，里面有完整配置、git commit、环境、训练日志、checkpoint 和评估结果，详见 [src/tfmprior/run.py](src/tfmprior/run.py) 开头的说明。目录已存在时会报错，`--resume` 跳过已完成的 run，`--overwrite` 重跑。

## 5. 结构

```
configs/          base.yaml 和每个实验的配置
src/tfmprior/     库代码：priors/、data/（造题库、按比例抽题）、model/、train/、eval/、utils/
scripts/          入口：make_dump.py、train.py、evaluate.py、verify_golden.py、diagnostics/
tests/            pytest
golden/           重构前记录的标准答案，scripts/verify_golden.py 用它检查结果没有变
docs/             RESEARCH.md、EXPERIMENTS.md
artifacts/        checksums.json 进 git；dumps/、benchmark_cache/ 不进 git
upstream_nanotabpfn/   原版 nanoTabPFN 的 README 和 notebook，仅供参考
```

## 6. 来源和许可

- nanoTabPFN（Pfefferle et al., 2025）：模型和训练循环的原始代码，见 [LICENSE](LICENSE) 和 [CITATION.bib](CITATION.bib)
- TabICL（BSD-3）：通用 prior，作为依赖安装
- FinTFM（Apache-2.0）：金融 prior，原样复制在 `src/tfmprior/priors/_fintfm_financial.py`，文件头写明来源 commit
