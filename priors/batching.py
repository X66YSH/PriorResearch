"""Turns tables from MixedPrior into training batches in the same format as train.PriorDumpDataLoader."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from priors.mixture import MixedPrior


def impute_with_train_mean(X: np.ndarray, n_train: int) -> np.ndarray:
    """Fills NaN with the column mean of the first n_train rows (0 if that column has no value there).

    The model normalizes each column by its mean and std, so a single NaN would turn the whole column into NaN.
    Only the train rows are used so that test rows never leak into the fill value.
    Evaluation on real tables must call this same function.
    """
    X = X.copy()
    with np.errstate(all="ignore"):
        col_mean = np.nanmean(X[:n_train], axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = col_mean[cols]
    return X


class MixedPriorDataLoader(DataLoader):
    """Generates batches on the fly; each table in a batch independently comes from finance with probability r.

    All tables in a batch share one train/test split index (the model needs that), drawn uniformly
    from the multiples of split_step between min_train_frac and max_train_frac of the rows.
    Every table is zero-padded to max_features; a zero column stays zero after the model's normalization.
    Both choices keep the number of distinct batch shapes small: the MPS backend caches memory per shape
    and ran out of memory after ~150 steps when the shape changed every batch.
    """

    def __init__(self, r: float, num_steps: int, batch_size: int, n_rows: int = 256, max_features: int = 24,
                 min_train_frac: float = 0.3, max_train_frac: float = 0.9, split_step: int = 16, seed: int = 0,
                 generic_type: str = "graph_scm"):
        self.prior = MixedPrior(r=r, n_rows=n_rows, max_features=max_features, max_classes=2, seed=seed,
                                generic_type=generic_type)
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.n_rows = n_rows
        self.max_features = max_features
        lo, hi = int(min_train_frac * n_rows), int(max_train_frac * n_rows)
        self.split_choices = np.arange(-(-lo // split_step) * split_step, hi + 1, split_step)
        self.rng = np.random.default_rng(seed + 1)  # separate stream, so the split does not shift which prior is drawn
        self.n_finance = 0
        self.n_tables = 0

    def __iter__(self):
        for _ in range(self.num_steps):
            n_train = int(self.rng.choice(self.split_choices))
            tables = []
            for _ in range(self.batch_size):
                X, y, source = self.prior.sample()
                tables.append((impute_with_train_mean(X, n_train), y))
                self.n_finance += source == "finance"
                self.n_tables += 1
            x = np.zeros((self.batch_size, self.n_rows, self.max_features), dtype=np.float32)
            y = np.zeros((self.batch_size, self.n_rows), dtype=np.float32)
            for i, (X_i, y_i) in enumerate(tables):
                x[i, :, :X_i.shape[1]] = X_i
                y[i] = y_i
            yield dict(x=torch.from_numpy(x), y=torch.from_numpy(y), train_test_split_index=n_train)

    def __len__(self):
        return self.num_steps
