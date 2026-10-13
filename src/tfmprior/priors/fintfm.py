"""FinTFM's financial prior: balance sheets that obey accounting identities, ratio features, rare defaults."""

import numpy as np

from tfmprior.priors._fintfm_financial import sample_financial_task

FINTFM_COMMIT = "7c297d02a29c44d16a579d67e7b648cf61a631ad"


def chunk_sampler(seed: int, chunk: int, n_rows: int, max_features: int):
    """Returns sample() for one chunk of a pool; chunk c uses its own generator seeded with (seed, c)."""
    rng = np.random.default_rng([seed, chunk])

    def sample():
        task = sample_financial_task(rng, n_rows, max_features=max_features)
        return task.X, task.y

    return sample
