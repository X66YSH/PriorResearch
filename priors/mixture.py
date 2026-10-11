"""Ratio-r mixture of a generic prior (TabICL, graph_scm by default) and a finance prior (FinTFM).

r is the probability that one table comes from the finance prior, so it counts tables,
not rows or training steps. This matches the r in the data mixing law L = c + k * exp(t * r).
"""

import random

import numpy as np
import torch
from tabicl.prior import PriorDataset

from priors.fintfm_financial import sample_financial_task


class GenericPrior:
    """TabICL's prior. 'graph_scm' is the TabICLv2 prior (scripts/train_v2_clf_stage1.sh in the TabICL repo);
    'mix_scm' is the TabICLv1 prior: each table from an MLP-SCM (70%) or a Tree-SCM (30%)."""

    def __init__(self, n_rows: int, max_features: int, max_classes: int, prior_type: str = "graph_scm"):
        self.dataset = PriorDataset(
            batch_size=1,
            batch_size_per_gp=1,
            min_features=2,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=n_rows,
            max_seq_len=n_rows + 1,  # TabICL draws np.random.randint(min, max), max exclusive
            prior_type=prior_type,
            n_jobs=1,
        )

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        X, y, d, _, _ = self.dataset.get_batch(1)
        n_active = int(d[0])  # TabICL zero-pads X up to max_features; keep only the real columns
        return X[0, :, :n_active].numpy(), y[0].numpy().astype(np.int64)


class FinancePrior:
    """FinTFM's financial prior: balance sheets that obey accounting identities, ratio features, rare defaults."""

    def __init__(self, n_rows: int, max_features: int, rng: np.random.Generator):
        self.n_rows = n_rows
        self.max_features = max_features
        self.rng = rng

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        task = sample_financial_task(self.rng, self.n_rows, max_features=self.max_features)
        return task.X, task.y.astype(np.int64)


class MixedPrior:
    """Draws each table from the finance prior with probability r, otherwise from the generic prior."""

    def __init__(self, r: float, n_rows: int = 256, max_features: int = 24, max_classes: int = 2, seed: int = 0,
                 generic_type: str = "graph_scm"):
        assert 0.0 <= r <= 1.0
        self.r = r
        self.rng = np.random.default_rng(seed)
        # TabICL draws from the global python / numpy / torch generators, so seed those too
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.generic = GenericPrior(n_rows, max_features, max_classes, generic_type)
        self.finance = FinancePrior(n_rows, max_features, self.rng)

    def sample(self) -> tuple[np.ndarray, np.ndarray, str]:
        if self.rng.random() < self.r:
            X, y = self.finance.sample()
            return X, y, "finance"
        X, y = self.generic.sample()
        return X, y, "generic"
