"""Training batches drawn from pre-generated pools in given proportions.

mixture is an ordered {prior: share} with shares summing to 1. The first entry is the base: each table draws
u ~ U(0, 1); the later entries own consecutive intervals of u ([0, s2), [s2, s2 + s3), ...) and the base takes
the rest. With {graph_scm: 0.7, fintfm: 0.3} a table comes from fintfm exactly when u < 0.3, as in the original
two-prior loader.

Common random numbers across mixtures: for one seed, the u sequence, the train/test split sequence and the
order inside each pool do not depend on the shares, so runs with different shares differ only through the shares.
Random streams (numpy SeedSequence keys): u -> [seed, 0], splits -> [seed, 1], pool order -> [seed, 2 + stream_id].

Each pool is read in its seed-specific order without replacement. The exact number of tables a run takes from
each pool is known before training (the u sequence is fixed), and a run that would need more tables than a pool
holds is refused up front: data mixing laws assume no table is seen twice.
"""

import h5py
import numpy as np
import torch

from tfmprior.data.preprocess import impute_with_train_mean
from tfmprior.priors.registry import get_prior


class PoolTooSmallError(ValueError):
    pass


def check_mixture(mixture: dict) -> None:
    if not mixture:
        raise ValueError("empty mixture")
    for name, share in mixture.items():
        get_prior(name)
        if share < 0:
            raise ValueError(f"negative share for {name}: {share}")
    if abs(sum(mixture.values()) - 1.0) > 1e-9:
        raise ValueError(f"mixture shares must sum to 1, got {sum(mixture.values())}: {mixture}")


def assign_sources(u: np.ndarray, mixture: dict) -> np.ndarray:
    """Index into list(mixture) for each draw u; see the module docstring for the rule."""
    names = list(mixture)
    source = np.zeros(len(u), dtype=np.int64)  # base by default
    lo = 0.0
    for k, name in enumerate(names[1:], start=1):
        hi = lo + mixture[name]
        source[(u >= lo) & (u < hi)] = k
        lo = hi
    return source


def required_tables(mixture: dict, num_steps: int, batch_size: int, seed: int) -> dict:
    """Exact number of tables a run takes from each pool."""
    check_mixture(mixture)
    u = np.random.default_rng([seed, 0]).random(num_steps * batch_size)
    counts = np.bincount(assign_sources(u, mixture), minlength=len(mixture))
    return {name: int(c) for name, c in zip(mixture, counts)}


class PoolMixtureLoader:
    """Yields dict(x, y, train_test_split_index, sources) like the original nanoTabPFN prior loader."""

    def __init__(self, mixture: dict, pools: dict, num_steps: int, batch_size: int, split_sizes, seed: int):
        """pools: {prior name: path to its pool}; only priors with a positive count are ever opened."""
        self.mixture = dict(mixture)
        self.names = list(self.mixture)
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.split_sizes = np.asarray(split_sizes)
        self.required = required_tables(self.mixture, num_steps, batch_size, seed)
        self.paths, self.order = {}, {}
        for name, need in self.required.items():
            if need == 0:
                continue
            path = pools[name]
            with h5py.File(path, "r") as f:
                size = int(f["X"].shape[0])
                self.n_rows, self.max_features = f["X"].shape[1:]
            if need > size:
                raise PoolTooSmallError(f"{name}: this run needs {need} tables but {path} holds {size}; "
                                        f"generate at least {need} tables with tfmprior.data.dump.make_dump")
            self.paths[name] = path
            self.order[name] = np.random.default_rng([seed, 2 + get_prior(name).stream_id]).permutation(size)
        self.position = {name: 0 for name in self.paths}
        self.source_rng = np.random.default_rng([seed, 0])
        self.split_rng = np.random.default_rng([seed, 1])
        self.counts = {name: 0 for name in self.names}

    def _next_index(self, name: str) -> int:
        index = int(self.order[name][self.position[name]])
        self.position[name] += 1
        return index

    def __iter__(self):
        files = {name: h5py.File(path, "r") for name, path in self.paths.items()}
        try:
            for _ in range(self.num_steps):
                n_train = int(self.split_rng.choice(self.split_sizes))
                x = np.zeros((self.batch_size, self.n_rows, self.max_features), dtype=np.float32)
                y = np.zeros((self.batch_size, self.n_rows), dtype=np.float32)
                sources = assign_sources(self.source_rng.random(self.batch_size), self.mixture)
                for i, k in enumerate(sources):
                    name = self.names[k]
                    index = self._next_index(name)
                    x[i] = impute_with_train_mean(files[name]["X"][index], n_train)
                    y[i] = files[name]["y"][index]
                    self.counts[name] += 1
                yield dict(x=torch.from_numpy(x), y=torch.from_numpy(y), train_test_split_index=n_train,
                           sources=[self.names[k] for k in sources])
        finally:
            for f in files.values():
                f.close()

    def __len__(self):
        return self.num_steps
