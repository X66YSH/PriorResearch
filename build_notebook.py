"""Writes rq1_gpu.ipynb, the step-by-step version of the RQ1 experiment. Edit the cells here, then rerun:
    python build_notebook.py
"""

import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# RQ1：金融 prior 和通用 prior 的取舍曲线（GPU 版）

从上往下一个 cell 一个 cell 地跑。每一步训好的模型和成绩都存在硬盘上（`checkpoints/`、`results/`），中途停了重新打开，已经训好的会自动跳过。

| 步骤 | 做什么 | 大约多久 |
|---|---|---|
| 0 | 准备：导入、检查 GPU 和数据 | 1 分钟 |
| 1 | 确认和 Mac 上一致 | 2 分钟 |
| 2 | batch 32 下验证学习率，测训练噪声 | 看 GPU |
| 3 | 选定学习率和步数（手动填） | |
| 4 | 正式实验：7 个 r 乘 3 个 seed | 看 GPU |
| 5 | 画取舍曲线 | 1 分钟 |

设计和之前的结果见 `EXPERIMENT.md`，环境安装见 `RUN_ON_GPU.md`。
""")

md("""
## Colab：先跑这一个 cell

在 Colab 上用之前（只需一次）：
1. 在 Google Drive 里建文件夹 `tfm_prior`（也就是 `MyDrive/tfm_prior`），把 `tfm_prior_gpu_code_*.zip` 传进去
2. 在 Colab 里打开这个 notebook，菜单 **Runtime → Change runtime type → T4 GPU**

下面这个 cell 会：连上 Drive；把代码解压到 Colab 本地；补装 `tabicl` 等几个包；把 `checkpoints/`、`results/`、`figures/` 放到 Drive 上（断线也不丢，重新跑会自动跳过已训好的）；准备题库。

题库：第一次在 Colab 上造，约 30 分钟（Colab 只有 2 个 CPU 核），造好后备份到 `MyDrive/tfm_prior/prior_dumps/`，以后直接拷回来。也可以把 Mac 上的 4 个题库文件预先传到这个文件夹里，就不用造了，而且和 Mac 上的题完全相同。

不在 Colab 上时，这个 cell 什么都不做。
""")

code("""
import glob
import os
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
DRIVE_DIR = "/content/drive/MyDrive/tfm_prior"  # the code zip goes here; outputs and pool backups are kept here
REPO_DIR = "/content/nanoTabPFN-main"


def colab_setup(drive_dir, repo_dir, mount=True, install=True):
    \"\"\"Unzips the code next to the runtime, keeps outputs on Drive, and puts the prior pools on local disk.\"\"\"
    shell = get_ipython().system  # streams command output into the notebook
    if mount:
        from google.colab import drive
        drive.mount(os.path.dirname(os.path.dirname(drive_dir)))
    if not os.path.exists(repo_dir):
        zips = sorted(glob.glob(f"{drive_dir}/tfm_prior_gpu_code_*.zip"))
        assert zips, f"upload tfm_prior_gpu_code_*.zip to {drive_dir}"
        subprocess.run(["unzip", "-q", zips[-1], "-d", os.path.dirname(repo_dir)], check=True)
    os.chdir(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    if install:
        shell(f"{sys.executable} -m pip install -q tabicl==2.2.0 schedulefree==1.4.1 h5py xgboost")
    # outputs live on Drive so a disconnect loses nothing; the Mac reference results in the zip are copied over once
    for folder in ["checkpoints", "results", "figures", "logs"]:
        target = f"{drive_dir}/{folder}"
        os.makedirs(target, exist_ok=True)
        if os.path.isdir(folder) and not os.path.islink(folder):
            subprocess.run(["cp", "-Rn", f"{folder}/.", target], check=True)
            subprocess.run(["rm", "-rf", folder], check=True)
        if not os.path.exists(folder):
            os.symlink(target, folder)
    # pools: read from local disk (fast), backed up on Drive between sessions
    backup = f"{drive_dir}/prior_dumps"
    os.makedirs("prior_dumps", exist_ok=True)
    os.makedirs(backup, exist_ok=True)
    shell(f"cp -n {backup}/*.h5 prior_dumps/ 2>/dev/null; true")
    shell(f"{sys.executable} make_prior_dump.py --all --workers {os.cpu_count()}")  # skips pools that exist
    shell(f"cp -n prior_dumps/*.h5 {backup}/")
    print("ready, working directory", os.getcwd())


if IN_COLAB:
    colab_setup(DRIVE_DIR, REPO_DIR)
""")

md("""
## 0. 准备

**第一次在新机器上跑之前，先在终端里造题库**（repo 根目录下，约 10 分钟）：

```bash
python make_prior_dump.py --all
```

造好的题库会算出校验值，并和 Mac 上的对比，结果也会写进报告。不完全一样也能用：GPU 上所有实验都用同一份题库，彼此可比；只是和 Mac 上的结果不能逐数字对照。

`QUICK_TEST = True` 时，所有步骤都用极小的规模（几步训练、1 个 seed），几分钟就能把整个 notebook 跑一遍，用来检查环境。正式跑之前改成 `False`。
""")

code("""
QUICK_TEST = False  # True: tiny settings that only check the notebook runs end to end

import glob
import json
import os
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Image, display
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark.datasets import SUITES, load_suite
from evaluate import evaluate, load_checkpoint
from model import NanoTabPFNModel
from priors.batching import DumpMixedDataLoader, impute_with_train_mean
from train import eval as breast_cancer_eval
from train import get_default_device, set_randomness_seed, train

device = get_default_device()
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

GENERIC_DUMP = "prior_dumps/generic_graph_scm_100k_256x24_seed0.h5"   # training pools (EXPERIMENT.md 2.2b)
FINANCE_DUMP = "prior_dumps/finance_100k_256x24_seed0.h5"
HELDOUT_DUMPS = {"generic": "prior_dumps/generic_graph_scm_1k_256x24_seed1.h5",  # seed 1: never seen in training
                 "finance": "prior_dumps/finance_1k_256x24_seed1.h5"}
missing = [path for path in [GENERIC_DUMP, FINANCE_DUMP, *HELDOUT_DUMPS.values()] if not os.path.exists(path)]
if missing:
    raise FileNotFoundError(f"missing {missing}: in a terminal at the repo root, run  python make_prior_dump.py --all  (about 10 minutes)")

SUITE_DATA = {suite: load_suite(suite) for suite in SUITES}  # read from benchmark/cache, no download
for suite, datasets in SUITE_DATA.items():
    print(f"{suite}: {len(datasets)} datasets")
TAG = "_quicktest" if QUICK_TEST else ""
""")

md("""
### 函数

- `train_one`：用比例 r 训一个模型，存成 checkpoint。和 `train_mixed.py` 做的事一样，存的格式也一样
- `health`：健康检查，标准和 EXPERIMENT.md 4.4 一样。注意这里的"没见过的合成表"来自 `*_1k_*_seed1.h5`，和 Mac 上脚本现场造的那批不同，所以数字不能和 EXPERIMENT.md 4.5、4.7 直接比
- `evaluate_and_save`：在两套考卷上评估，存成 json
""")

code("""
def train_one(r, seed, lr, steps, batch_size, out_dir):
    \"\"\"Trains one model with finance share r on the pools and saves it; skips if the checkpoint exists.\"\"\"
    path = os.path.join(out_dir, f"r{r:.2f}_seed{seed}.pth")
    if os.path.exists(path):
        print("skip, exists:", path)
        return path
    set_randomness_seed(seed)
    model = NanoTabPFNModel(embedding_size=96, num_attention_heads=4, mlp_hidden_size=192, num_layers=3, num_outputs=2)
    prior = DumpMixedDataLoader(r, steps, batch_size, GENERIC_DUMP, FINANCE_DUMP, seed=seed)
    print(f"--- train r={r} seed={seed} lr={lr:g} steps={steps} batch={batch_size} ---")
    start = time.time()
    model, history = train(model, prior, lr=lr, device=device, steps_per_eval=max(steps // 6, 1), eval_func=breast_cancer_eval)
    args = dict(r=r, seed=seed, lr=lr, steps=steps, batch_size=batch_size, embedding_size=96, num_layers=3,
                generic="graph_scm", on_the_fly=False, generic_dump=GENERIC_DUMP, finance_dump=FINANCE_DUMP)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": args, "history": history}, path)
    print(f"saved {path}: {(time.time() - start) / 60:.1f} min, finance share {prior.n_finance / prior.n_tables:.3f}")
    return path


def load_heldout(prior, n_tables=100):
    \"\"\"The first n_tables held-out tables that have both classes in each half.\"\"\"
    tables = []
    with h5py.File(HELDOUT_DUMPS[prior], "r") as f:
        for i in range(f["y"].shape[0]):
            X, y = f["X"][i, :, :f["num_features"][i]], f["y"][i].astype(np.int64)
            k = len(y) // 2
            if len(np.unique(y[:k])) == 2 and len(np.unique(y[k:])) == 2:
                tables.append((X, y))
            if len(tables) == n_tables:
                return tables
    return tables


HELDOUT_TABLES = {prior: load_heldout(prior) for prior in HELDOUT_DUMPS}


def heldout_auc(classifier, tables):
    aucs = []
    for X, y in tables:
        k = len(y) // 2
        X = impute_with_train_mean(X, k)
        classifier.fit(X[:k], y[:k])
        aucs.append(roc_auc_score(y[k:], classifier.predict_proba(X[k:])[:, 1]))
    return float(np.mean(aucs))


def health(path):
    \"\"\"Breast cancer AUC, held-out prior AUCs and the EXPERIMENT.md 4.4 label.\"\"\"
    classifier = load_checkpoint(path, device)
    bc = breast_cancer_eval(classifier)["roc_auc"]
    generic = heldout_auc(classifier, HELDOUT_TABLES["generic"])
    finance = heldout_auc(classifier, HELDOUT_TABLES["finance"])
    label = "healthy" if generic >= 0.65 and bc >= 0.9 else "collapsed" if generic < 0.55 or bc < 0.6 else "unclear"
    return dict(breast_cancer_auc=bc, generic_prior_auc=generic, finance_prior_auc=finance, label=label)


def evaluate_and_save(path, results_dir):
    \"\"\"Both suites (AUC / ECE / Brier); saved as results_dir/<checkpoint name>.json.\"\"\"
    result = evaluate(load_checkpoint(path, device), SUITE_DATA)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, os.path.basename(path).replace(".pth", ".json")), "w") as f:
        json.dump(result, f, indent=2)
    return result


def suite_means(result):
    return {f"{suite} {metric}": result[suite]["mean"][metric] for suite in SUITES for metric in ["auc", "ece", "brier"]}


REPORT_TXT, REPORT_JSON = f"results/gpu_report{TAG}.txt", f"results/gpu_report{TAG}.json"


def report(section, text):
    \"\"\"Saves one section of the report and rewrites results/gpu_report.txt; rerunning a step replaces its section.\"\"\"
    os.makedirs("results", exist_ok=True)
    sections = json.load(open(REPORT_JSON)) if os.path.exists(REPORT_JSON) else {}
    sections[section] = f"(written {time.strftime('%Y-%m-%d %H:%M')})\\n{text}"
    with open(REPORT_JSON, "w") as f:
        json.dump(sections, f, indent=1, ensure_ascii=False)
    with open(REPORT_TXT, "w") as f:
        f.write("RQ1 GPU report: send this file back as is\\n\\n")
        for name in sorted(sections):
            f.write(f"===== {name} =====\\n{sections[name]}\\n\\n")
    print(f"report section '{section}' saved to {REPORT_TXT}")


def git_commit():
    \"\"\"Current commit read from .git directly, so it works where the git command is not on the path.\"\"\"
    try:
        head = open(".git/HEAD").read().strip()
        if not head.startswith("ref: "):
            return head[:7]
        ref = head[5:]
        if os.path.exists(f".git/{ref}"):
            return open(f".git/{ref}").read().strip()[:7]
        for line in open(".git/packed-refs"):
            if line.strip().endswith(ref):
                return line[:7]
    except OSError:
        pass
    return "unknown"


from make_prior_dump import checksum_note

report("0 environment", "\\n".join([
    f"device: {device} {torch.cuda.get_device_name(0) if device == 'cuda' else ''}",
    f"torch {torch.__version__}, python {sys.version.split()[0]}, git commit {git_commit()}, QUICK_TEST {QUICK_TEST}",
    f"pools: {GENERIC_DUMP}, {FINANCE_DUMP}; held-out: {', '.join(HELDOUT_DUMPS.values())}",
    "suites: " + ", ".join(f"{s} {len(d)} datasets" for s, d in SUITE_DATA.items()),
    *[checksum_note(path) for path in [GENERIC_DUMP, FINANCE_DUMP, *HELDOUT_DUMPS.values()]],
]))
""")

md("""
## 1. 确认和 Mac 上一致

- 随机森林：应该**完全等于** Mac 上的金融 AUC 0.7791、通用 AUC 0.7341（同样的考卷、同样的随机种子）
- Mac 上训好的 r = 0 模型：应该**非常接近** 0.6767 / 0.6744。GPU 和 Mac 的浮点运算有细微差别，可能差在小数点后第 3 到 4 位
""")

code("""
rf = evaluate(RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1), SUITE_DATA)
mac_model = evaluate(load_checkpoint("checkpoints/lr1e-3/r0.00_seed0.pth", device), SUITE_DATA)
check = pd.DataFrame({
    "here": [rf["finance"]["mean"]["auc"], rf["general"]["mean"]["auc"], mac_model["finance"]["mean"]["auc"], mac_model["general"]["mean"]["auc"]],
    "Mac": [0.7791, 0.7341, 0.6767, 0.6744],
}, index=["random forest, finance AUC", "random forest, general AUC", "Mac r=0 model, finance AUC", "Mac r=0 model, general AUC"]).round(4)
report("1 consistency with Mac", check.to_string())
check
""")

md("""
## 2. batch 32 下验证学习率，测训练噪声

Mac 上 batch 8：学习率 4e-3 会随机崩溃；1e-3 不崩，但不同训练之间差异可达 0.1 AUC，比要测的 r 效应（0.03 到 0.05）还大（EXPERIMENT.md 4.5、4.7）。

这里用原版 nanoTabPFN 的 batch 32，r = 0（纯通用，最容易崩），每个学习率跑 5 个 seed。要看两件事：
1. 哪个学习率 5 个 seed **都健康**
2. 健康的那组，**seed 之间的差异**有多大（下面表格的 std）。要明显小于 0.03，正式实验才能分出 r 的效应

3000 步 x batch 32 = 96,000 张表，题库有 10 万张，够用。要更多步数，先用 `make_prior_dump.py` 造更大的题库（见 RUN_ON_GPU.md 第 4 节）。
""")

code("""
LR_CHECK = dict(lrs=[4e-3, 1e-3], seeds=[0, 1, 2, 3, 4], steps=3000, batch_size=32, r=0.0)
if QUICK_TEST:
    LR_CHECK.update(lrs=[1e-3], seeds=[0], steps=5)

lr_rows = []
for lr in LR_CHECK["lrs"]:
    for seed in LR_CHECK["seeds"]:
        out_dir = f"checkpoints/lr_check_bs{LR_CHECK['batch_size']}_lr{lr:g}{TAG}"
        path = train_one(LR_CHECK["r"], seed, lr, LR_CHECK["steps"], LR_CHECK["batch_size"], out_dir)
        lr_rows.append(dict(lr=lr, seed=seed, **health(path)))
lr_table = pd.DataFrame(lr_rows)
os.makedirs("results", exist_ok=True)
lr_table.to_csv(f"results/lr_check_bs{LR_CHECK['batch_size']}{TAG}.csv", index=False)
lr_table.round(3)
""")

code("""
# per learning rate: how many runs are healthy, and how much the healthy runs differ
summary = lr_table.groupby("lr").agg(
    healthy=("label", lambda s: f"{(s == 'healthy').sum()}/{len(s)}"),
    generic_auc_mean=("generic_prior_auc", "mean"),
    generic_auc_std=("generic_prior_auc", "std"),
    finance_auc_std=("finance_prior_auc", "std"),
    breast_cancer_std=("breast_cancer_auc", "std"),
)
report("2 learning rate check", f"settings: {LR_CHECK}\\n\\nper run:\\n{lr_table.round(4).to_string(index=False)}\\n\\n"
       f"per learning rate:\\n{summary.round(4).to_string()}")
summary.round(3)
""")

code("""
# breast cancer AUC during training, one line per run: a collapse shows as a line stuck near 0.5
fig, axes = plt.subplots(1, len(LR_CHECK["lrs"]), figsize=(6 * len(LR_CHECK["lrs"]), 4), squeeze=False)
for ax, lr in zip(axes[0], LR_CHECK["lrs"]):
    for seed in LR_CHECK["seeds"]:
        path = f"checkpoints/lr_check_bs{LR_CHECK['batch_size']}_lr{lr:g}{TAG}/r{LR_CHECK['r']:.2f}_seed{seed}.pth"
        history = torch.load(path, map_location="cpu", weights_only=False)["history"]
        ax.plot([t / 60 for t, _ in history], [s["roc_auc"] for _, s in history], marker="o", label=f"seed {seed}")
    ax.set_title(f"lr {lr:g}, batch {LR_CHECK['batch_size']}")
    ax.set_xlabel("training time (min)")
    ax.set_ylabel("breast cancer AUC")
    ax.set_ylim(0, 1)
    ax.legend()
plt.show()
""")

md("""
## 3. 选定学习率和步数（手动填）

根据第 2 步：
- 选 5 个 seed **都健康**、而且 **std 最小**的学习率
- 如果 std 还不够小（不明显小于 0.03），可以加步数；步数 x batch 超过 10 万时要先造更大的题库

填好后再往下跑。
""")

code("""
LR = 1e-3        # <- fill in from step 2
STEPS = 3000     # <- fill in from step 2
BATCH_SIZE = 32
RS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
SEEDS = [0, 1, 2]
if QUICK_TEST:
    STEPS, RS, SEEDS = 5, [0.0, 1.0], [0]
RUN_NAME = f"bs{BATCH_SIZE}_lr{LR:g}_steps{STEPS}{TAG}"
CHECKPOINT_DIR, RESULTS_DIR = f"checkpoints/{RUN_NAME}", f"results/{RUN_NAME}"
report("3 chosen settings", f"LR {LR}, STEPS {STEPS}, BATCH_SIZE {BATCH_SIZE}, RS {RS}, SEEDS {SEEDS}, RUN_NAME {RUN_NAME}")
print(RUN_NAME)
""")

md("""
## 4. 正式实验：7 个 r 乘 3 个 seed

seed 在外层循环：先把 seed 0 的 7 个 r 都跑完，中途停了也能先看到一条完整的曲线。每个模型训完立刻做健康检查和考卷评估，成绩存在 `results/<RUN_NAME>/`。
""")

code("""
grid_rows = []
for seed in SEEDS:
    for r in RS:
        path = train_one(r, seed, LR, STEPS, BATCH_SIZE, CHECKPOINT_DIR)
        result = evaluate_and_save(path, RESULTS_DIR)
        grid_rows.append(dict(r=r, seed=seed, **health(path), **suite_means(result)))
        print(grid_rows[-1])
grid_table = pd.DataFrame(grid_rows)
grid_table.to_csv(f"{RESULTS_DIR}/grid.csv", index=False)
report("4a grid, every run", grid_table.round(4).to_string(index=False))
grid_table.round(3)
""")

code("""
# reference lines for the plot
for name, baseline in [("baseline_random_forest", RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)),
                       ("baseline_logistic_regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))]:
    with open(f"{RESULTS_DIR}/{name}.json", "w") as f:
        json.dump(evaluate(baseline, SUITE_DATA), f, indent=2)
baselines = {name: json.load(open(f"{RESULTS_DIR}/{name}.json")) for name in ["baseline_random_forest", "baseline_logistic_regression"]}
report("4b grid, baselines", pd.DataFrame({name: suite_means(b) for name, b in baselines.items()}).T.round(4).to_string())
""")

md("""
## 5. 取舍曲线

- 两条线往**相反方向**走，才是"金融变好、通用变差"的取舍；两条线**同起同落**，说明主要是训练好坏的差别（Mac 上 seed 0 就是这样，见 EXPERIMENT.md 4.7）
- "金融 − 通用"这一列能抵消单次训练好坏的影响，是看领域效应最干净的指标
""")

code("""
grid_table["finance - general"] = grid_table["finance auc"] - grid_table["general auc"]
by_r = grid_table.groupby("r")
curve = pd.DataFrame({
    "finance auc": by_r["finance auc"].mean(), "finance auc std": by_r["finance auc"].std(),
    "general auc": by_r["general auc"].mean(), "general auc std": by_r["general auc"].std(),
    "finance - general": by_r["finance - general"].mean(), "finance - general std": by_r["finance - general"].std(),
    "finance ece": by_r["finance ece"].mean(), "general ece": by_r["general ece"].mean(),
    "healthy runs": by_r["label"].apply(lambda s: f"{(s == 'healthy').sum()}/{len(s)}"),
})
report("5 curve, mean over seeds", curve.round(4).to_string())

# per-dataset AUC, mean over seeds: rows = datasets, columns = r
per_dataset = []
for path in glob.glob(f"{RESULTS_DIR}/r*_seed*.json"):
    r = float(os.path.basename(path)[1:5])
    for suite, block in json.load(open(path)).items():
        for dataset, scores in block["datasets"].items():
            per_dataset.append(dict(r=r, suite=suite, dataset=dataset, **scores))
per_dataset = pd.DataFrame(per_dataset)
for metric in ["auc", "ece"]:
    table = per_dataset.pivot_table(index=["suite", "dataset"], columns="r", values=metric, aggfunc="mean")
    report(f"6 per dataset {metric}, mean over seeds", table.round(4).to_string())
curve.round(4)
""")

code("""
figure = f"figures/rq1_curves_{RUN_NAME}.png"
!{sys.executable} plot_curves.py --results_dir {RESULTS_DIR} --out {figure}
display(Image(figure))
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, cell in enumerate(notebook["cells"]):
    cell["id"] = f"cell-{i}"
    cell["source"] = cell["source"].splitlines(keepends=True)
with open("rq1_gpu.ipynb", "w") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print("wrote rq1_gpu.ipynb with", len(cells), "cells")
