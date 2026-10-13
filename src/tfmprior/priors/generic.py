"""Generic priors from TabICL: graph_scm (the TabICLv2 prior) and mix_scm (the TabICLv1 prior)."""

import random

import numpy as np
import torch
from tabicl.prior import PriorDataset


class GenericPrior:
    """TabICL's prior. 'graph_scm' is the TabICLv2 prior (scripts/train_v2_clf_stage1.sh in the TabICL repo);
    'mix_scm' is the TabICLv1 prior: each table from an MLP-SCM (70%) or a Tree-SCM (30%).
    Runs on the CPU only: in tabicl 2.2.0, graph_scm creates some tensors without a device."""

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


def chunk_sampler(prior_type: str):
    """Returns make(seed, chunk, n_rows, max_features) -> sample(); TabICL draws from the global generators."""

    def make(seed: int, chunk: int, n_rows: int, max_features: int):
        chunk_seed = seed * 1_000_003 + chunk
        random.seed(chunk_seed)
        np.random.seed(chunk_seed)
        torch.manual_seed(chunk_seed)
        return GenericPrior(n_rows, max_features, 2, prior_type).sample

    return make
