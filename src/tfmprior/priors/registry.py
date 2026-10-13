"""Prior names used in configs -> how to generate their tables and how their pools are named.

stream_id fixes the random stream that shuffles each pool when training (see data/loader.py). It is part of the
reproducibility contract: changing it changes which tables every run sees. graph_scm = 0 and fintfm = 1 reproduce
the pre-refactor runs; new priors take the next free id.
"""

from dataclasses import dataclass
from importlib.metadata import version
from typing import Callable

from tfmprior.priors import fintfm, generic


@dataclass(frozen=True)
class PriorSpec:
    name: str
    stream_id: int
    pool_stem: str  # pool file: <pool_stem>_<n_tables // 1000>k_<n_rows>x<max_features>_seed<seed>.h5
    make_sampler: Callable  # (seed, chunk, n_rows, max_features) -> sample() -> (X, y)
    provenance: Callable[[], dict]  # stored in the pool file's attributes


PRIORS = {
    "graph_scm": PriorSpec("graph_scm", 0, "generic_graph_scm", generic.chunk_sampler("graph_scm"),
                           lambda: {"tabicl_version": version("tabicl")}),
    "fintfm": PriorSpec("fintfm", 1, "finance", fintfm.chunk_sampler, lambda: {"fintfm_commit": fintfm.FINTFM_COMMIT}),
    "mix_scm": PriorSpec("mix_scm", 2, "generic_mix_scm", generic.chunk_sampler("mix_scm"),
                         lambda: {"tabicl_version": version("tabicl")}),
}


def get_prior(name: str) -> PriorSpec:
    if name not in PRIORS:
        raise KeyError(f"unknown prior {name!r}; known: {sorted(PRIORS)}")
    return PRIORS[name]


def pool_filename(name: str, n_tables: int, n_rows: int, max_features: int, seed: int) -> str:
    size = f"{n_tables // 1000}k" if n_tables >= 1000 and n_tables % 1000 == 0 else str(n_tables)
    return f"{get_prior(name).pool_stem}_{size}_{n_rows}x{max_features}_seed{seed}.h5"
